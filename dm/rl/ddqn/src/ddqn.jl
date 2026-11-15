# Gymnasium envs driven from Julia via PyCall, learned with DQN: a Q-network, ε-greedy with
# annealing, experience replay, a target network, Huber loss, Double DQN and n-step returns.
# Which env: EnvName in envs.jl (LUNAR_LANDER, CARTPOLE, ACROBOT, MOUNTAIN_CAR); the code here only sees d and K.
#
# LunarLander-v3, obs ∈ ℝ⁸: x, y, vx, vy, angle, angular velocity, left-leg contact, right-leg contact.
#   a ∈ {0 noop, 1 left engine, 2 main engine, 3 right engine}
# CartPole-v1,    obs ∈ ℝ⁴: cart x, cart v, pole angle, pole angular velocity.
#   a ∈ {0 push left, 1 push right}

using PyCall
using Lux, Random
using Reactant, Enzyme, Optimisers       # trace → MLIR → XLA; Enzyme for ∇ of the loss; Adam
using Functors                           # fmap over the (nested NamedTuple) parameter trees
using ProgressMeter                      # live-updating progress bar for train!()

# Every array the network touches (p, p⁻, Adam's moments, each minibatch) lives on the XLA backend
# chosen here, and every computation on them is a compiled XLA program (see `compiled!`).
#   DDQN_BACKEND=cpu (default)  XLA's CPU client. The net is 8→64→4: a GPU step would be launch latency.
#   DDQN_BACKEND=gpu            CUDA. XLA preallocates 75% of the GPU; XLA_REACTANT_GPU_PREALLOCATE=false
#                               to share it with another job.
# Must be set before the first array is moved: arrays belong to the client they were made on.
const BACKEND = get(ENV, "DDQN_BACKEND", "cpu")
Reactant.set_default_backend(BACKEND)
const cdev = cpu_device()
const rdev = reactant_device()

include("utils.jl")             # PlotStyle, learning_curve (Unicode), learning_curve_figure (Makie), show_curve
include("replay.jl")            # Replay ring buffer: push!, sample
include("envs.jl")              # EnvName, SPECS, get_env, get_model, outcome

# ---- the loop ----
# learn=false: act only (no buffer, no gradient steps), e.g. for evaluation.
function run_episode(env, learner; seed=nothing, max_steps=1000, learn=true)
    obs, _ = env.reset(; seed)
    obs = Float32.(obs)
    G = 0.0
    ℒ = 0.0                      # sum of batch losses over the episode's gradient steps
    nℒ = 0                       # number of gradient steps this episode
    for t ∈ 1:max_steps
        a = action(learner, obs)                                   # act: no gradient
        obs′, r, terminated, truncated, _ = env.step(a - 1)        # Python, host side
        obs′ = Float32.(obs′)
        timeout = truncated || t == max_steps                      # our own cap ends the episode too
        ℓ = learn ? on_step!(learner, obs, a, r, obs′, terminated, timeout) : NaN   # store; maybe one batch step
        G += r
        isnan(ℓ) || (ℒ += ℓ; nℒ += 1)
        obs = obs′
        if terminated || truncated
            o = outcome(learner.envname, r, terminated, truncated)     # envs.jl
            return (; G, steps=t, terminated, loss=ℒ / max(nℒ, 1), outcome=o, x=obs′[1])
        end
    end
    return (; G, steps=max_steps, terminated=false, loss=ℒ / max(nℒ, 1), outcome=:timeout, x=obs[1])
end

# Fields are grouped by upgrade. Hyperparameters carry their defaults here; the rest (network
# parameters, buffer, optimiser state) are built by the Learner(model; ...) constructor below.
#
# The online parameters θ, the layer states and Adam's moments live in `ts`, a Lux TrainState; read
# θ as `online(l)`. Never keep θ in a variable across a training step: the step may reuse (donate)
# its buffers. `ts` is untyped because its type changes once the first step caches the compiled program.
Base.@kwdef mutable struct Learner{M, P<:NamedTuple}
    envname::EnvName = LUNAR_LANDER   # which problem: sets d, K (envs.jl), outcome labels, plot
    model::M                      # Q_θ : ℝᵈ → ℝᴷ, must match spec(envname)
    γ::Float32 = 0.99f0
    η::Float32 = 1f-2             # Adam step size (fixed once the first training step compiles)
    ts::Training.TrainState       # online parameters θ (trained, used to act), states, Adam's state

	# epsilon annealing
    ε::Float32 = NaN32            # ε-greedy: 1 = always random, 0 = always greedy (current value; starts at ε₀)
    ε₀::Float32 = 1f0             # ε anneals linearly ε₀ → ε_min over the first ε_episodes episodes
    ε_min::Float32 = 0.05f0
    ε_episodes::Int = 200

	# replay
    buffer::Replay                # every transition seen, last `capacity` of them
    batch::Int = 256              # B: transitions per gradient step
    warmup::Int = 1_000           # no training until the buffer holds this many

	# DQN
    p⁻::P                         # target parameters θ⁻: frozen copy of p, used only for the TD target
    train_every::Int = 4          # env steps per gradient step
    target_every::Int = 1_000     # C: env steps between syncs θ⁻ ← θ (hard copy; ignored when polyak)
    polyak::Bool = false          # true: soft target, θ⁻ ← τθ + (1 − τ)θ⁻ every env step, instead of the hard copy
    τ::Float32 = 2f-3             # Polyak rate. Mean lag of θ⁻ ≈ 1/τ steps; hard copy's is C/2, so τ = 2/C matches (C=1000 → 0.002)

	# n-step returns
    n::Int = 10                   # n-step returns: look n real rewards ahead before bootstrapping (1 = plain 1-step)
                                  # queue: last ≤ n (s, a, r) of this episode, oldest first
    queue::Vector{Tuple{Vector{Float32}, Int, Float32}} = Tuple{Vector{Float32}, Int, Float32}[]

	# other
    κ::Float32 = 30f0             # Huber threshold: quadratic for |δ| ≤ κ, linear beyond (large κ, e.g. 1f6 = plain squared loss)
    steps::Int = 0                # env steps seen so far
    returns::Vector{Float64} = Float64[]   # every training episode's return G, across train!() calls
    xla::Dict{Any, Any} = Dict{Any, Any}() # compiled XLA programs, built on first use (`compiled!`)
end

# Build a Learner around a Lux model. Every keyword except capacity is a field name (defaults
# above), so Learner(model; n=3, polyak=true) sets exactly those fields.
#   capacity replay buffer size
# envname and η are fields too, but the constructor needs them up front (buffer size d, Adam).
function Learner(model; η=1f-2, capacity=50_000, envname=LUNAR_LANDER, kw...)
    p, st = Lux.setup(Xoshiro(0), model) |> rdev
    ts = Training.TrainState(model, p, st, Optimisers.Adam(Float32(η)))
    # explicit type parameters → Julia converts each keyword to its field type (e.g. τ=0.01 → Float32)
    l = Learner{typeof(model), typeof(p)}(;
            envname, model, η, ts, p⁻=fmap(copy, p), buffer=Replay(capacity, spec(envname).obs), kw...)
    l.ε = l.ε₀                                            # annealing starts at ε₀
    0 < l.γ ≤ 1 || throw(ArgumentError("γ must be in (0, 1]"))
    0 ≤ l.ε_min ≤ l.ε₀ ≤ 1 || throw(ArgumentError("need 0 ≤ ε_min ≤ ε₀ ≤ 1"))
    l.n ≥ 1 || throw(ArgumentError("n must be ≥ 1"))
    0 < l.τ ≤ 1 || throw(ArgumentError("τ must be in (0, 1]"))
    return l
end

# Set ε for episode k (1-based): linear ε₀ → ε_min over ε_episodes, then flat at ε_min.
anneal!(l::Learner, k) = (l.ε = max(l.ε_min, l.ε₀ - (l.ε₀ - l.ε_min) * (k - 1) / l.ε_episodes))

# Display: the REPL shows nothing for a Learner (train!() returns one); println/errors get one line
# instead of the full parametric type plus every weight array.
Base.show(io::IO, ::MIME"text/plain", ::Learner) = nothing
Base.show(io::IO, l::Learner) =
    print(io, "Learner(γ=$(l.γ), ε=$(round(l.ε; digits=3)), $(Lux.parameterlength(online(l))) params)")

online(l::Learner) = l.ts.parameters      # θ, on the XLA device

# The compiled program stored under `key`, building it with `build()` the first time. A program is
# fixed to its argument shapes and to the constants baked in when it was traced; `key` names the
# constants that may change (e.g. τ), so changing one compiles a new program instead of reusing a stale one.
compiled!(build, l::Learner, key) = get!(build, l.xla, key)

# ε-greedy on Q_θ(s, ·). Compiled forward pass on a d × 1 batch: nothing here is differentiated.
function action(learner::Learner, obs)
    rand() < learner.ε && return rand(1:spec(learner.envname).actions)     # uniform over K
    x = rdev(reshape(obs, :, 1))
    q = compiled!(learner, :q) do
        @compile learner.model(x, online(learner), learner.ts.states)
    end
    Q, _ = q(x, online(learner), learner.ts.states)                # Lux returns (y, st′)
    return argmax(vec(Array(Q)))
end

# Pop the oldest queued (s₀, a₀) and store it as an m-step transition, m = length(queue):
#   R = Σₖ₌₀ᵐ⁻¹ γᵏ rₖ,   bootstrap from s′ with g = γᵐ   (d = terminated: no bootstrap at all)
function emit!(learner::Learner, s′, terminated)
    q = learner.queue
    m = length(q)
    R = sum(learner.γ^(k - 1) * q[k][3] for k ∈ 1:m)
    s, a, _ = popfirst!(q)
    push!(learner.buffer, s, a, R, s′, terminated, learner.γ^m)
end

# Queue (s, a, r). Once n are queued the oldest is complete (n real rewards + the state n steps
# later) and goes to the buffer. At episode end every queued one is flushed with its shorter m:
#   terminated: d = 1, no bootstrap      truncated (time limit): d = 0, bootstrap from obs′ with γᵐ
# Then every `train_every` steps (after warmup) take one batch step.
# Returns the batch loss ℒ if a step was taken, else NaN.
function on_step!(learner::Learner, obs, a, r, obs′, terminated, truncated)
    push!(learner.queue, (obs, a, Float32(r)))
    if terminated || truncated
        while !isempty(learner.queue)
            emit!(learner, obs′, terminated)
        end
    elseif length(learner.queue) == learner.n
        emit!(learner, obs′, false)
    end
    learner.steps += 1
    update_target!(learner)
    length(learner.buffer) ≥ learner.warmup && learner.steps % learner.train_every == 0 || return NaN
    return train_batch!(learner)
end

# Target network θ⁻, one of two ways:
#   hard:   every C = target_every env steps, θ⁻ ← θ               (sample-and-hold: flat, then a jump)
#   polyak: every env step, θ⁻ ← τθ + (1 − τ)θ⁻                     (EMA: θ⁻ = τ Σₖ (1 − τ)ᵏ θₜ₋ₖ, creeps)
# The blend is one compiled program with τ baked in; fmap walks p⁻ and θ in parallel, leaf = one
# weight array. Both ways p⁻ gets fresh device buffers, never aliases of θ's.
struct Polyak
    τ::Float32
end
(b::Polyak)(p⁻, p) = fmap((w⁻, w) -> b.τ .* w .+ (1 - b.τ) .* w⁻, p⁻, p)

function update_target!(learner::Learner)
    if learner.polyak
        blend = compiled!(learner, (:polyak, learner.τ)) do
            @compile Polyak(learner.τ)(learner.p⁻, online(learner))
        end
        learner.p⁻ = blend(learner.p⁻, online(learner))
    elseif learner.steps % learner.target_every == 0
        learner.p⁻ = fmap(copy, online(learner))
    end
    return learner
end

# Huber loss of a residual δ, with q = min(|δ|, κ):
#   h(δ) = ½q² + κ(|δ| − q)  =  ½δ² if |δ| ≤ κ,  κ(|δ| − ½κ) otherwise
#   h′(δ) = clamp(δ, −κ, κ): the error signal is capped, so one big δ can't dominate the step.
function huber(δ, κ)
    a = abs(δ)
    q = min(a, κ)
    return q^2 / 2 + κ * (a - q)
end

# One semi-gradient Q-learning step on a random minibatch, Double DQN target:
#   a*ᵢ = argmaxₐ′ Q_θ(s′ᵢ, a′)                  (online θ PICKS the next action)
#   tᵢ  = rᵢ + gᵢ (1 − dᵢ) Q_θ⁻(s′ᵢ, a*ᵢ)         (frozen θ⁻ SCORES it; constant; gᵢ = γᵐ, γ for 1-step)
#   ℒ  = (1/B) Σᵢ h(tᵢ − Q_θ(sᵢ, aᵢ))             (Huber; differentiated w.r.t. θ)
#
# As a Lux objective (model, θ, st, data) → (ℒ, st′, stats), so the whole step (both target forward
# passes, the loss, Enzyme's reverse pass and Adam's update) is ONE compiled XLA program.
#   PICK  without argmax/gather: Mᵢ = 1[Q_θ(s′ᵢ, ·) = maxₐ′ Q_θ(s′ᵢ, a′)] ∈ {0,1}ᴷ, then
#         Vnᵢ = ⟨Mᵢ, Q_θ⁻(s′ᵢ, ·)⟩ / ‖Mᵢ‖₁, which is Q_θ⁻(s′ᵢ, a*ᵢ) unless the max is tied (then the
#         mean over the tied actions; ties between float outputs don't happen in practice).
#   t is a constant: ignore_derivatives stops the gradient (it would be zero through the argmax
#   anyway, but this keeps Enzyme from differentiating the two extra forward passes).
# κ is baked in when the step first compiles.
struct DDQNLoss
    κ::Float32
end

function (o::DDQNLoss)(model, p, st, (s, mask, r, g, d, s′, p⁻))
    Qn⁻, _ = model(s′, p⁻, st)                                     # Q_θ⁻(s′, ·), K × B: SCORE
    Qn, _ = model(s′, p, st)                                       # Q_θ(s′, ·),  K × B: PICK
    M = Float32.(Qn .== maximum(Qn; dims=1))                       # one-hot of a*ᵢ per column
    Vn = vec(sum(M .* Qn⁻; dims=1) ./ sum(M; dims=1))              # θ⁻'s value of θ's choice
    t = Reactant.ignore_derivatives(r .+ g .* (1 .- d) .* Vn)      # tᵢ = Rᵢ + gᵢ(1 − dᵢ)Vnᵢ, gᵢ = γᵐ
    Q, st′ = model(s, p, st)                                       # K × B
    Q_sa = vec(sum(Q .* mask; dims=1))                             # Q(sᵢ, aᵢ): pick row aᵢ in column i
    return sum(huber.(t .- Q_sa, o.κ)) / size(Q, 2), st′, (;)
end

# Sampling stays on the host (the buffer is plain arrays); one host → device copy of the minibatch,
# one compiled step, one device → host copy of the scalar ℒ.
function train_batch!(learner::Learner)
    b = sample(learner.buffer, learner.batch)
    B = learner.batch
    K = spec(learner.envname).actions
    onehot = zeros(Float32, K, B)                                  # column i has a 1 in row aᵢ
    onehot[CartesianIndex.(b.a, 1:B)] .= 1
    data = (rdev((b.s, onehot, b.r, b.g, b.d, b.s′))..., learner.p⁻)
    _, ℒ, _, learner.ts = Training.single_train_step!(AutoEnzyme(), DDQNLoss(learner.κ), data, learner.ts)
    return Float32(ℒ)
end

# plots: a PlotStyle (NoPlot, UnicodePlot, MakiePlot, BothPlots) or Bool (true = UnicodePlot).
# MakiePlot / BothPlots write a PNG to `plot_path` (default resources/<env>.png).
# success = landed (LunarLander) or balanced to the 500-step cap (CartPole), see envs.jl.
function train!(learner, env; episodes=1000, seed=0, plots=NoPlot,
                plot_path=plot_path(learner.envname))
    successes = 0
    returns = Float64[]
    prog = Progress(episodes; desc="training ", showspeed=true)   # redraws in place
    try
        for k ∈ 1:episodes
            anneal!(learner, k)
            res = run_episode(env, learner; seed=seed + k)
            successes += success(res.outcome)
            push!(returns, res.G)
            push!(learner.returns, res.G)                         # lifetime history (saved with the learner)
            last100 = returns[max(1, end - 99):end]
            next!(prog; showvalues = [
                ("episode",          k),
                ("return",           round(res.G; digits=1)),
                ("steps",            res.steps),
                ("avg return (100)", round(sum(last100) / length(last100); digits=1)),
                ("mean ℓ",           round(res.loss; digits=3)),
                ("ε",                round(learner.ε; digits=3)),
                ("success",          successes),
            ])
        end
    finally
        finish!(prog)
    end
    println("success $successes/$(length(returns)),  avg return of last 100: ",
            round(sum(returns[max(1, end - 99):end]) / min(100, length(returns)); digits=1))
    sp = spec(learner.envname)
    show_curve(returns, plots; path=plot_path, solved=sp.solved, ylim=sp.ylim,
               title="$(sp.id) DQN learning curve")              # utils.jl
    return learner
end

# Run a trained learner greedily (ε = 0, no learning) on fresh seeds; ε is restored afterwards.
function infer(learner, env; episodes=5, seed=100_000)
    ε = learner.ε
    learner.ε = 0f0
    returns = Float64[]
    successes = 0
    try
        for k ∈ 1:episodes
            res = run_episode(env, learner; seed=seed + k, learn=false)
            successes += success(res.outcome)
            push!(returns, res.G)
            println("episode $k: return $(round(res.G; digits=1)), steps $(res.steps), $(res.outcome)")
        end
    finally
        learner.ε = ε
    end
    println("greedy: success $successes/$episodes,  avg return ", round(sum(returns) / episodes; digits=1))
    return returns
end


include("checkpoint.jl")        # save_learner / load_learner (needs Learner defined above)
