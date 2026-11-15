# Save / load a trained Learner with Serialization.
#
# We don't serialize the Learner struct itself: if its fields change (they do, upgrade by upgrade),
# an old file would no longer match the type. Nor its device arrays or compiled programs, which only
# exist in the process that made them. Instead we store plain host data and rebuild through the
# Learner(model; ...) constructor, which re-applies defaults for any field added later:
#
#   model                       Lux Chain (Dense + tanh: plain data, no closures, so it serializes)
#   p, p⁻, st                   nested NamedTuples of host Arrays
#   adam                        Adam's state per weight array, (m, v, (β₁ᵗ, β₂ᵗ)), in p's tree shape
#   hyper                       envname + every scalar hyperparameter field (γ, η, ε₀, n, τ, polyak, …)
#   ε, steps, returns, capacity training progress
#
# The replay buffer is NOT saved (≈ 3 MB of stale transitions): a loaded learner starts with an
# empty buffer and refills it, past `warmup`, if you keep training. Inference doesn't need it.
# Serialization's format is tied to the Julia version; fine on this box (JLD2 for archiving).
#
# Files from the Zygote version (CPU arrays, `opt_state` = an Optimisers tree, no η in hyper) load too.

using Serialization

# envname rides along with the hyperparameters: a checkpoint saved before it existed has none, and
# the constructor's default (LUNAR_LANDER) is then right. Same for η, read from the old `opt_state`.
const HYPER_FIELDS = (:envname, :γ, :η, :ε₀, :ε_min, :ε_episodes, :batch, :warmup, :train_every, :target_every,
                      :polyak, :τ, :n, :κ)

isoptleaf(x) = x isa Optimisers.Leaf

# Device scalars (Adam's β powers after the first step) → Float32; arrays → host Arrays.
tohost(x::AbstractArray) = Array(x)
tohost(x::Tuple) = map(tohost, x)
tohost(x::Number) = Float32(x)
tohost(x::Reactant.ConcreteRNumber) = Float32(x)

# Adam's state per weight array as host data, (m, v, (β₁ᵗ, β₂ᵗ)), in p's tree shape. Works on both the
# Reactant TrainState's tree and the old Zygote checkpoints' plain Optimisers tree.
adam_host(os) = fmap(leaf -> tohost(leaf.state), os; exclude=isoptleaf)

# Write saved moments into a fresh TrainState's leaves. They keep the types a fresh setup has (device
# arrays, Float32 β powers), so the first training step compiles exactly as it would from scratch.
function adam_restore!(os, adam)
    fmap(os, adam; exclude=isoptleaf) do leaf, (m, v, β)
        leaf.state = (rdev(m), rdev(v), β)
        leaf
    end
    return os
end

# Adam's η from an old checkpoint's Optimisers tree: the rule on any leaf.
first_leaf(os) = isoptleaf(os) ? os : first_leaf(first(values(os)))

function save_learner(path, l::Learner)
    mkpath(dirname(path))
    hyper = NamedTuple{HYPER_FIELDS}(map(f -> getfield(l, f), HYPER_FIELDS))
    serialize(path, (; l.model, p=online(l) |> cdev, p⁻=l.p⁻ |> cdev, st=l.ts.states |> cdev,
                       adam=adam_host(l.ts.optimizer_state), hyper, l.ε, l.steps, l.returns,
                       capacity=length(l.buffer.a)))
    return path
end

function load_learner(path)
    s = deserialize(path)
    hyper = haskey(s.hyper, :η) ? s.hyper : (; s.hyper..., η=first_leaf(s.opt_state).rule.eta)
    adam = haskey(s, :adam) ? s.adam : adam_host(s.opt_state)
    l = Learner(s.model; capacity=s.capacity, hyper...)
    p, st = (s.p, s.st) |> rdev
    l.ts = Training.TrainState(s.model, p, st, Optimisers.Adam(l.η))
    adam_restore!(l.ts.optimizer_state, adam)
    l.p⁻ = s.p⁻ |> rdev
    l.ε, l.steps = s.ε, s.steps
    append!(l.returns, s.returns)
    return l
end
