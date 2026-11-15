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
using Optimisers, Zygote                 # Adam, and ∇ of the loss w.r.t. the parameters
using Functors                           # fmap over the (nested NamedTuple) parameter trees
using ProgressMeter                      # live-updating progress bar for train!()
using CUDA, cuDNN                        # loaded so gpu_device() finds the GPU

const cdev = cpu_device()
const gdev = gpu_device()        # switch here; pays off only with batches / bigger nets
const tdev = cdev                # training device: where p, st and the forward/backward live

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
        obs′, r, terminated, truncated, _ = env.step(a - 1)        # Python: outside Zygote entirely
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
Base.@kwdef mutable struct Learner{M, P<:NamedTuple, S<:NamedTuple, O}
    envname::EnvName = LUNAR_LANDER   # which problem: sets d, K (envs.jl), outcome labels, plot
    model::M                      # Q_θ : ℝᵈ → ℝᴷ, must match spec(envname)
    γ::Float32 = 0.99f0
    p::P                          # online parameters θ: trained, used to act
    st::S

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
    opt_state::O                  # Adam's per-parameter moments (m, v), same tree shape as p
end

# Build a Learner around a Lux model. Every keyword except η and capacity is a field name (defaults
# above), so Learner(model; n=3, polyak=true) sets exactly those fields.
#   η        Adam step size (lives inside opt_state, not a field)
#   capacity replay buffer size
# envname is a field too, but the constructor needs it up front for the buffer's state size d.
function Learner(model; η=1f-2, capacity=50_000, envname=LUNAR_LANDER, kw...)
    p, st = Lux.setup(Xoshiro(0), model) .|> tdev
    opt_state = Optimisers.setup(Optimisers.Adam(η), p)
    # explicit type parameters → Julia converts each keyword to its field type (e.g. τ=0.01 → Float32)
    l = Learner{typeof(model), typeof(p), typeof(st), typeof(opt_state)}(;
            envname, model, p, st, p⁻=deepcopy(p), buffer=Replay(capacity, spec(envname).obs),
            opt_state, kw...)
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
    print(io, "Learner(γ=$(l.γ), ε=$(round(l.ε; digits=3)), $(Lux.parameterlength(l.p)) params)")

# ε-greedy on Q_p(s, ·). Plain forward pass with learner.p: nothing here is differentiated.
function action(learner::Learner, obs)
    rand() < learner.ε && return rand(1:spec(learner.envname).actions)     # uniform over K
    Q, _ = learner.model(obs |> tdev, learner.p, learner.st)       # Lux returns (y, st′)
    return argmax(Q |> cdev)
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
# fmap walks p⁻ and p in parallel, leaf = one weight array; the blend is in place (p⁻ never aliases p).
function update_target!(learner::Learner)
    if learner.polyak
        τ = learner.τ
        fmap((w⁻, w) -> (w⁻ .= τ .* w .+ (1 - τ) .* w⁻), learner.p⁻, learner.p)
    elseif learner.steps % learner.target_every == 0
        learner.p⁻ = deepcopy(learner.p)
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
#   a*ᵢ = argmaxₐ′ Q_p(s′ᵢ, a′)                  (online θ PICKS the next action)
#   tᵢ  = rᵢ + gᵢ (1 − dᵢ) Q_p⁻(s′ᵢ, a*ᵢ)         (frozen θ⁻ SCORES it; constant; gᵢ = γᵐ, γ for 1-step)
#   ℒ  = (1/B) Σᵢ h(tᵢ − Q_p(sᵢ, aᵢ))             (Huber; differentiated w.r.t. p)
function train_batch!(learner::Learner)
    b = sample(learner.buffer, learner.batch)
    B = learner.batch

    s′ = b.s′ |> tdev
    Qn⁻, _ = learner.model(s′, learner.p⁻, learner.st)             # Q_θ⁻(s′, ·), K × B: SCORE
    Qn, _ = learner.model(s′, learner.p, learner.st)               # Q_θ(s′, ·),  K × B: PICK (no gradient)
    K = size(Qn, 1)
    a′ = argmax(Qn; dims=1)                                        # 1 × B CartesianIndex (a*ᵢ, i)
    Vn = vec(Qn⁻[a′])                                              # θ⁻'s value of θ's choice
    t = (b.r |> tdev) .+ (b.g |> tdev) .* (1 .- (b.d |> tdev)) .* Vn   # tᵢ = Rᵢ + gᵢ(1 − dᵢ)Vnᵢ, gᵢ = γᵐ

    onehot = zeros(Float32, K, B)                                  # column i has a 1 in row aᵢ
    onehot[CartesianIndex.(b.a, 1:B)] .= 1
    x, mask = b.s |> tdev, onehot |> tdev

    ℒ, (∇,) = Zygote.withgradient(learner.p) do p
        Q, _ = learner.model(x, p, learner.st)                     # K × B
        Q_sa = vec(sum(Q .* mask; dims=1))                         # Q(sᵢ, aᵢ): pick row aᵢ in column i
        sum(huber.(t .- Q_sa, learner.κ)) / B
    end
    learner.opt_state, learner.p = Optimisers.update!(learner.opt_state, learner.p, ∇)
    return ℒ
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
