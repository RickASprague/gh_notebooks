# SINDy for the dynamics only:  v̇ = Θ(x) Ξ,   truth: v̇x = α x/r³, v̇y = α y/r³.
# (ẋ = v is kinematics — not regressed.) v̇ from central differences of the sampled
# velocity; library depends on position only; Ξ by STLSQ.
# Run on one trajectory, then on many (different E, L).
#
#   julia --project src/sindy.jl

include("shared.jl")

function report(title, Θ, Ż, names; λ = 0.05)
    println("\n══ $title ══  (N = $(size(Θ, 1)) samples, $(size(Θ, 2)) library terms)")
    Θn = Θ ./ map(norm, eachcol(Θ))'            # column-normalized, for the conditioning check
    σ = svdvals(Θn)
    @printf "  smallest singular values of normalized Θ: %s\n" join((@sprintf("%.1e", s) for s in σ[end-4:end]), "  ")
    Ξ = stlsq(Θ, Ż, λ)
    for (k, lhs) in enumerate(["v̇x", "v̇y"])
        terms = [@sprintf("%+.4f·%s", Ξ[i, k], names[i]) for i in eachindex(names) if Ξ[i, k] != 0]
        println("  $lhs = ", isempty(terms) ? "0" : join(terms, "  "))
    end
    @printf "  residual RMS: %.2e\n" norm(Θ * Ξ - Ż') / sqrt(length(Ż))
    Ξ, σ
end

# ── 1. Single trajectory ──
Z1, Ż1 = simulate([3.0, 1.0, -1.0, 0.0])
Θ1, names = library(Z1)
report("Single trajectory", Θ1, Ż1, names)

# ── 2. Many trajectories (shared training set) ──
Zs, Żs, U0s = training_orbits(20)
ZM, ŻM = reduce(hcat, Zs), reduce(hcat, Żs)
ΘM, _ = library(ZM)
ΞM, _ = report("20 trajectories", ΘM, ŻM, names)

# ── Per-orbit invariants and figure ──
println("\n  orbit   E        L        e      type")
plt = plot(; aspect_ratio = :equal, xlabel = "x", ylabel = "y", legend = false,
    title = "SINDy training data: 20 orbits", xlims = (-5, 5), ylims = (-5, 5))
for (i, u0) in enumerate(U0s)
    x0, v0 = u0[1:2], u0[3:4]
    E = 0.5 * dot(v0, v0) + α / norm(x0)
    L = x0[1] * v0[2] - x0[2] * v0[1]
    e = sqrt(1 + 2E * L^2 / α^2)
    @printf "  %2d   %+.3f   %+.3f   %.3f   %s\n" i E L e (E < 0 ? "ellipse" : "hyperbola")
    X = integrate(u0)
    plot!(plt, X[1, :], X[2, :]; lw = 1.5, ls = E < 0 ? :solid : :dash)
    scatter!(plt, [x0[1]], [x0[2]]; ms = 3, color = :black)
end
scatter!(plt, [0], [0]; ms = 8, color = :red)
savefig(plt, joinpath(RESOURCES, "sindy_orbits.png"))

# ── Save Ξ for compare.jl ──
a_sindy(X) = (library(X)[1] * ΞM)'
@printf "\n  field error on grid: max |a_sindy − a_gt| / |a_gt| = %.2e\n" maximum(rel_err(a_sindy, Xg))
mkpath(RESULTS)
open(joinpath(RESULTS, "sindy_xi.txt"), "w") do io
    for i in axes(ΞM, 1)
        println(io, join(ΞM[i, :], " "))
    end
end
