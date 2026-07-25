# Shared by sindy.jl and dl.jl: ground truth, the 20-orbit training set, acceleration-field plots.

ENV["GKSwstype"] = "100"   # headless GR
using OrdinaryDiffEq, LinearAlgebra, Random, Printf, Plots

const RESOURCES = joinpath(@__DIR__, "..", "resources")   # figures
const RESULTS = joinpath(@__DIR__, "..", "results")       # fitted models (Ξ, network weights)
const α = -1.0
coulomb!(du, u, p, t) = (x = u[1:2]; du[1:2] = u[3:4]; du[3:4] = α * x / norm(x)^3)
a_gt(X) = α .* X ./ (sum(abs2, X; dims = 1) .^ 1.5)          # true acceleration, 2 × M

const dt = 0.01
const T = 4.0

integrate(u0) = reduce(hcat, solve(ODEProblem(coulomb!, u0, (0.0, T)), Tsit5();
    abstol = 1e-10, reltol = 1e-10, saveat = dt).u)          # 4 × N

# Positions and v̇ (central differences of the sampled velocity), both 2 × (N−2).
function simulate(u0)
    Z = integrate(u0)
    V̇ = (Z[3:4, 3:end] - Z[3:4, 1:end-2]) / (2dt)
    Z[1:2, 2:end-1], V̇
end

# n orbits: random start on r ∈ [2, 4], random velocity, reject passes closer than r = 0.3.
function training_orbits(n = 20; seed = 7)
    rng = Xoshiro(seed)
    Xs, V̇s, U0s = Matrix{Float64}[], Matrix{Float64}[], Vector{Float64}[]
    while length(Xs) < n
        θ0, r0 = 2π * rand(rng), 2 + 2rand(rng)
        u0 = [r0 * cos(θ0), r0 * sin(θ0), 0.8randn(rng), 0.8randn(rng)]
        X, V̇ = simulate(u0)
        minimum(hypot.(X[1, :], X[2, :])) > 0.3 || continue
        push!(Xs, X); push!(V̇s, V̇); push!(U0s, u0)
    end
    Xs, V̇s, U0s
end

# ── Library in position only: monomials of degree ≤ 2 in (x, y), plus rational terms in r ──
const vars = ["x", "y"]
function library(Z)
    x, y = Z[1, :], Z[2, :]
    r = hypot.(x, y)
    cols, names = Vector{Vector{Float64}}(), String[]
    add(c, n) = (push!(cols, c); push!(names, n))
    add(ones(size(Z, 2)), "1")
    for i in 1:2
        add(Z[i, :], vars[i])
    end
    for i in 1:2, j in i:2
        add(Z[i, :] .* Z[j, :], i == j ? "$(vars[i])²" : "$(vars[i])·$(vars[j])")
    end
    for n in 1:3
        sup = ("", "²", "³")[n]
        add(r .^ -n, "1/r$sup")
        add(x ./ r .^ n, "x/r$sup")
        add(y ./ r .^ n, "y/r$sup")
    end
    reduce(hcat, cols), names
end

# Evaluation grid [-4,4]², r > 0.3, for the scalar error printouts in sindy.jl / dl.jl.
const Xg = reduce(hcat, [[x, y] for x in range(-4, 4; length = 17) for y in range(-4, 4; length = 17)
                         if hypot(x, y) > 0.3])
rel_err(afun, X) = map(norm, eachcol(afun(X) - a_gt(X))) ./ map(norm, eachcol(a_gt(X)))

# STLSQ (Brunton et al. 2016): least squares, zero |ξ| < λ, refit on the survivors.
function stlsq(Θ, Ż, λ; iters = 10)
    Ξ = Θ \ Ż'
    for _ in 1:iters, k in axes(Ξ, 2)
        small = abs.(Ξ[:, k]) .< λ
        Ξ[small, k] .= 0
        Ξ[.!small, k] = Θ[:, .!small] \ Ż[k, :]
    end
    Ξ
end

# Arrows: direction of a, length ∝ log|a| (0.25 … 0.8 grid spacings), drawn as line segments.
function arrow_xy(X, A)
    n = map(norm, eachcol(A))
    ℓ = 0.25 .+ 0.17 .* clamp.(log10.(n) .+ 2.2, 0, 3.3)
    xs, ys = Float64[], Float64[]
    for i in axes(X, 2)
        x0, y0 = X[:, i]
        θ = atan(A[2, i], A[1, i])
        x1, y1 = x0 + ℓ[i] * cos(θ), y0 + ℓ[i] * sin(θ)
        h = 0.35ℓ[i]
        append!(xs, [x0, x1, NaN, x1, x1 - h * cos(θ - 0.45), NaN, x1, x1 - h * cos(θ + 0.45), NaN])
        append!(ys, [y0, y1, NaN, y1, y1 - h * sin(θ - 0.45), NaN, y1, y1 - h * sin(θ + 0.45), NaN])
    end
    xs, ys
end
