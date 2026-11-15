# Save / load a trained Learner with Serialization.
#
# We don't serialize the Learner struct itself: if its fields change (they do, upgrade by upgrade),
# an old file would no longer match the type. Instead we store plain data and rebuild through the
# Learner(model; ...) constructor, which re-applies defaults for any field added later:
#
#   model                       Lux Chain (Dense + tanh: plain data, no closures, so it serializes)
#   p, p⁻, st, opt_state        nested NamedTuples of arrays, moved to the CPU
#   hyper                       envname + every scalar hyperparameter field (γ, ε₀, n, τ, polyak, …)
#   ε, steps, returns, capacity training progress
#
# The replay buffer is NOT saved (≈ 3 MB of stale transitions): a loaded learner starts with an
# empty buffer and refills it, past `warmup`, if you keep training. Inference doesn't need it.
# Serialization's format is tied to the Julia version; fine on this box (JLD2 for archiving).

using Serialization

# envname rides along with the hyperparameters: a checkpoint saved before it existed has none, and
# the constructor's default (LUNAR_LANDER) is then right.
const HYPER_FIELDS = (:envname, :γ, :ε₀, :ε_min, :ε_episodes, :batch, :warmup, :train_every, :target_every,
                      :polyak, :τ, :n, :κ)

function save_learner(path, l::Learner)
    mkpath(dirname(path))
    hyper = NamedTuple{HYPER_FIELDS}(map(f -> getfield(l, f), HYPER_FIELDS))
    serialize(path, (; l.model, p=l.p |> cdev, p⁻=l.p⁻ |> cdev, st=l.st |> cdev,
                       opt_state=l.opt_state |> cdev, hyper, l.ε, l.steps, l.returns,
                       capacity=length(l.buffer.a)))
    return path
end

function load_learner(path)
    s = deserialize(path)
    l = Learner(s.model; capacity=s.capacity, s.hyper...)
    l.p, l.p⁻, l.st = s.p |> tdev, s.p⁻ |> tdev, s.st |> tdev
    l.opt_state = s.opt_state |> tdev
    l.ε, l.steps = s.ε, s.steps
    append!(l.returns, s.returns)
    return l
end
