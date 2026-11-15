# DDQN CPU benchmark; any version of this project with the same API (this code, or the Reactant
# version from commit b75a0ca checked out elsewhere):
#   julia -t 1 --project=<ddqn dir> experiments/bench.jl <label>
# Times (1) one action, (2) one training step (B = 256), (3) one Gymnasium env.step, (4) env steps/s
# while learning, and (5) the split of an env step inside the real loop. ~3 min.
const LABEL = ARGS[1]
include(joinpath(Base.active_project() |> dirname, "src", "setup.jl"))
using Statistics, Random

env, l = create_learner(LUNAR_LANDER)
Random.seed!(0)
# Fill the buffer past warmup and compile everything (action and training step).
k = 0
while length(l.buffer) < 5_000
    global k += 1
    run_episode(env, l; seed = k)
end

function per_call(f, n; reps = 5)
    f()                                      # make sure it is compiled
    ts = map(1:reps) do _
        t = time_ns(); for _ ∈ 1:n; f(); end; (time_ns() - t) / n / 1e3
    end
    return minimum(ts)                       # μs per call, best of reps
end

obs, _ = env.reset(seed = 1); obs = Float32.(obs)
l.ε = 0.0                                    # always the network's action, never a random one
μ_act   = per_call(() -> action(l, obs), 20_000)
μ_train = per_call(() -> train_batch!(l), 2_000)
env.reset(seed = 2)
function env_step()
    _, _, te, tr, _ = env.step(rand(0:3))
    (te || tr) && env.reset()
end
μ_env   = per_call(env_step, 20_000)

l.ε = l.ε_min                                # the post-annealing regime: 95 % greedy
# End-to-end: learning episodes for ≥ 60 s, env steps per second.
steps = 0; t0 = time()
while time() - t0 < 60
    global k += 1
    global steps += run_episode(env, l; seed = k).steps
end
sps = steps / (time() - t0)

println("RESULT $LABEL  action $(round(μ_act; digits=1)) μs | train step $(round(μ_train; digits=1)) μs | env.step $(round(μ_env; digits=1)) μs | end-to-end $(round(Int, sps)) env steps/s ($(steps) steps)")

# Where an env step's time goes inside the real loop (same logic as run_episode), 60 s.
tA = tE = tO = 0.0; n = 0; nT = 0; t0 = time()
while time() - t0 < 60
    global k += 1
    obs, _ = env.reset(seed = k); obs = Float32.(obs)
    for t ∈ 1:1000
        a1 = time_ns(); a = action(l, obs); a2 = time_ns()
        obs′, r, te, tr, _ = env.step(a - 1); a3 = time_ns()
        obs′ = Float32.(obs′)
        ℓ = on_step!(l, obs, a, r, obs′, te, tr || t == 1000); a4 = time_ns()
        global tA += a2 - a1; global tE += a3 - a2; global tO += a4 - a3; global n += 1
        isnan(ℓ) || (global nT += 1)
        obs = obs′
        (te || tr) && break
    end
end
T = time() - t0
f(x) = round(x / n / 1e3; digits = 1)
println("SPLIT $LABEL  per env step: action $(f(tA)) μs | env.step $(f(tE)) μs | on_step! $(f(tO)) μs (train steps $(round(nT/n; digits=3))/env step) | other $(round(T*1e6/n - (tA+tE+tO)/n/1e3; digits=1)) μs | $(round(Int, n/T)) steps/s")
