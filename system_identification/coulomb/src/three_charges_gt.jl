# Ground truth for three fixed charges: field magnitude, field lines, null points (a = 0),
# and the 20 orbits from the same recipe as three_charges.jl.
#
#   julia --project src/three_charges_gt.jl

ENV["GKSwstype"] = "100"
using OrdinaryDiffEq, LinearAlgebra, Random, Plots, Printf
const C = [1.5 .* [cos(φ), sin(φ)] for φ in (π/2, π/2 + 2π/3, π/2 + 4π/3)]
const Q = [-1.0, 1.0, 2.0]
accel3(x) = sum(q .* (x .- c) ./ norm(x .- c)^3 for (c, q) in zip(C, Q))
three!(du, u, p, t) = (du[1:2] = u[3:4]; du[3:4] = accel3(u[1:2]))
dmin(x) = minimum(norm(x .- c) for c in C)
# null points a = 0
nulls = Vector{Vector{Float64}}()
for x0 in -6:0.5:6, y0 in -6:0.5:6
    x = [x0, y0]; dmin(x) > 0.2 || continue
    for _ in 1:60   # Newton, finite-difference Jacobian
        a = accel3(x); h = 1e-6
        J = reduce(hcat, [(accel3(x .+ h .* e) .- a) ./ h for e in ([1.0, 0], [0, 1.0])])
        x = x .- J \ a
        (norm(x) > 50 || dmin(x) < 0.05) && break
    end
    (norm(accel3(x)) < 1e-10 && dmin(x) > 0.1 && norm(x) < 20 && all(norm(x - n) > 1e-4 for n in nulls)) && push!(nulls, x)
end
println("charges: ", [(round.(c, digits = 3), q) for (c, q) in zip(C, Q)])
println("null points (a = 0): ", [round.(n, digits = 3) for n in nulls])
println("net charge: ", sum(Q), "  dipole moment Σ Qᵢcᵢ: ", round.(sum(q .* c for (c, q) in zip(C, Q)), digits = 3))
L = 6.0; hx = range(-L, L; length = 301)
Z = [dmin([x, y]) > 0.3 ? log10(norm(accel3([x, y]))) : NaN for y in hx, x in hx]
p = heatmap(hx, hx, Z; c = :viridis, clims = (-2, 1.5), colorbar_title = "log₁₀|a|", aspect_ratio = :equal,
    xlims = (-L, L), ylims = (-L, L), size = (900, 820), title = "Ground truth: Q = −1, +1, +2 on a triangle",
    xlabel = "x", ylabel = "y")
# field lines-ish: streamlines of a via short integrations of dx/ds = a/|a|
for (c, q) in zip(C, Q), k in 0:15
    φ = 2π * k / 16
    x = c .+ 0.32 .* [cos(φ), sin(φ)]
    pts = [copy(x)]
    s = q > 0 ? 1.0 : -1.0
    for _ in 1:800
        a = accel3(x); x = x .+ s * 0.03 .* a ./ norm(a); push!(pts, copy(x))
        (dmin(x) < 0.25 || maximum(abs.(x)) > L) && break
    end
    plot!(p, first.(pts), last.(pts); color = :white, alpha = 0.45, lw = 0.8, label = false)
end
# the 20 training orbits (same recipe)
rng = Xoshiro(7); n = 0
while n < 20
    θ0, r0 = 2π * rand(rng), 2 + 2rand(rng)
    u0 = [r0 * cos(θ0), r0 * sin(θ0), 0.8randn(rng), 0.8randn(rng)]
    dmin(u0[1:2]) > 0.3 || continue
    sol = solve(ODEProblem(three!, u0, (0.0, 4.0)), Tsit5(); abstol = 1e-10, reltol = 1e-10, saveat = 0.01)
    X = reduce(hcat, sol.u)
    minimum(dmin.(eachcol(X[1:2, :]))) > 0.3 || continue
    global n += 1
    plot!(p, X[1, :], X[2, :]; color = :orange, lw = 1.6, label = n == 1 ? "training orbits" : false)
    scatter!(p, [u0[1]], [u0[2]]; color = :orange, ms = 3, label = false)
end
scatter!(p, first.(nulls), last.(nulls); marker = :x, ms = 8, color = :magenta, markerstrokewidth = 3, label = "a = 0")
scatter!(p, first.(C), last.(C); ms = 9, color = [q < 0 ? :red : :dodgerblue for q in Q], markerstrokecolor = :white, label = false)
for (c, q) in zip(C, Q); annotate!(p, c[1] + 0.5, c[2] + 0.45, text(q < 0 ? "−1" : "+$(Int(q))", 13, :white)); end
savefig(p, joinpath(@__DIR__, "..", "resources", "three_charges_gt.png"))
