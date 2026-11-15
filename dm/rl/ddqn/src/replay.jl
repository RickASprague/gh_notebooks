mutable struct Replay
    s::Matrix{Float32}       # d × N   state
    a::Vector{Int}           # N       action, 1-based
    r::Vector{Float32}       # N       reward
    s′::Matrix{Float32}      # d × N   next state
    d::Vector{Float32}       # N       1 if terminated (no bootstrap), else 0
    g::Vector{Float32}       # N       discount on the bootstrap, γᵐ for an m-step transition (γ for 1-step)
    i::Int                   # index of the last write
    n::Int                   # number of valid entries, ≤ N
end

# N transitions of a d-dimensional state.
Replay(N::Int, d::Int) = Replay(zeros(Float32, d, N), zeros(Int, N), zeros(Float32, N),
                                zeros(Float32, d, N), zeros(Float32, N), zeros(Float32, N), 0, 0)

Base.length(b::Replay) = b.n

function Base.push!(b::Replay, s, a, r, s′, terminated, g)
    N = length(b.a)
    b.i = mod1(b.i + 1, N)                  # wrap: N+1 → 1
    b.s[:, b.i] = s
    b.a[b.i] = a
    b.r[b.i] = r
    b.s′[:, b.i] = s′
    b.d[b.i] = terminated
    b.g[b.i] = g
    b.n = min(b.n + 1, N)
    return b
end

function sample(b::Replay, B::Int)
    idx = rand(1:b.n, B)
    return (; s=b.s[:, idx], a=b.a[idx], r=b.r[idx], s′=b.s′[:, idx], d=b.d[idx], g=b.g[idx])
end
