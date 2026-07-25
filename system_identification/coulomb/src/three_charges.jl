# Three fixed charges: where the SINDy library is wrong and DL is the better tool.
#   d/dt v = a(x) = Σᵢ Qᵢ (x − cᵢ)/‖x − cᵢ‖³      (unit test charge, k/m = 1: αᵢ = Qᵢ)
# Charges Q = −1, +1, +2 on an equilateral triangle (circumradius 1.5) centred on the origin.
# Same data recipe, same 15-term library (centred on the origin), same MLP and training as the
# single-charge experiment.
#
#   julia --project src/three_charges.jl

include("shared.jl")      # library(), RESOURCES, RESULTS, packages
using Lux, Optimisers, Zygote, ComponentArrays, Statistics, Serialization

# ── Ground truth ──
const Rtri = 1.5
const C = [Rtri .* [cos(φ), sin(φ)] for φ in (π / 2, π / 2 + 2π / 3, π / 2 + 4π / 3)]   # 2-vectors
const Q = [-1.0, 1.0, 2.0]

function accel3(x)
    a = zeros(2)
    for (c, q) in zip(C, Q)
        d = x .- c
        a .+= q .* d ./ norm(d)^3
    end
    a
end
a3(X) = reduce(hcat, [accel3(x) for x in eachcol(X)])                 # 2 × M
three!(du, u, p, t) = (du[1:2] = u[3:4]; du[3:4] = accel3(u[1:2]))
dmin(x) = minimum(norm(x .- c) for c in C)                           # distance to nearest charge

# ── Data: 20 orbits, same recipe as the single-charge set ──
function simulate3(u0)
    Z = reduce(hcat, solve(ODEProblem(three!, u0, (0.0, T)), Tsit5();
        abstol = 1e-10, reltol = 1e-10, saveat = dt).u)
    Z[1:2, 2:end-1], (Z[3:4, 3:end] - Z[3:4, 1:end-2]) / (2dt)
end

rng = Xoshiro(7)
Xs, V̇s = Matrix{Float64}[], Matrix{Float64}[]
while length(Xs) < 20
    θ0, r0 = 2π * rand(rng), 2 + 2rand(rng)
    u0 = [r0 * cos(θ0), r0 * sin(θ0), 0.8randn(rng), 0.8randn(rng)]
    dmin(u0[1:2]) > 0.3 || continue
    X, V̇ = simulate3(u0)
    minimum(dmin.(eachcol(X))) > 0.3 || continue
    push!(Xs, X); push!(V̇s, V̇)
end
X, V̇ = reduce(hcat, Xs), reduce(hcat, V̇s)
@printf "Training set: N = %d, ‖x‖ ∈ [%.2f, %.2f], closest approach to a charge %.2f\n" size(X, 2) minimum(norm.(eachcol(X))) maximum(norm.(eachcol(X))) minimum(dmin.(eachcol(X)))

# ── SINDy: same 15-term library, same STLSQ ──
function stlsq(Θ, Ż, λ; iters = 10)
    Ξ = Θ \ Ż'
    for _ in 1:iters, k in axes(Ξ, 2)
        small = abs.(Ξ[:, k]) .< λ
        Ξ[small, k] .= 0
        Ξ[.!small, k] = Θ[:, .!small] \ Ż[k, :]
    end
    Ξ
end

Θ, names = library(X)
Ξ = stlsq(Θ, V̇, 0.05)
println("\n══ SINDy ══")
for (k, lhs) in enumerate(["v̇x", "v̇y"])
    terms = [@sprintf("%+.3f·%s", Ξ[i, k], names[i]) for i in eachindex(names) if Ξ[i, k] != 0]
    println("  $lhs = ", join(terms, "  "), "   ($(length(terms)) of 15 terms)")
end
@printf "  train RMS residual: %.3e   (RMS |v̇| = %.3e)\n" norm(Θ * Ξ - V̇') / sqrt(length(V̇)) norm(V̇) / sqrt(length(V̇))
a_sindy(Xq) = (library(Xq)[1] * Ξ)'

# ── DL: same MLP, same schedule ──
nn = Chain(Dense(2 => 64, tanh), Dense(64 => 64, tanh), Dense(64 => 2))
ps, st = Lux.setup(Xoshiro(0), nn)
θ = ComponentArray{Float64}(ps)
mse(θ) = mean(abs2, first(nn(X, θ, st)) .- V̇)
const DL_FILE = joinpath(RESULTS, "three_charges_dl.jls")
if isfile(DL_FILE)                                   # cached: skip the ~7 min training
    θ, losses = deserialize(DL_FILE)
    println("\n══ DL ══  (loaded $(DL_FILE))")
else
    opt = Optimisers.setup(Optimisers.Adam(1e-3), θ)
    losses = Float64[]
    println("\n══ DL ══")
    @time for epoch in 1:30_000
        epoch == 15_000 && Optimisers.adjust!(opt, 1e-4)
        epoch == 25_000 && Optimisers.adjust!(opt, 1e-5)
        l, g = Zygote.withgradient(mse, θ)
        global opt, θ = Optimisers.update(opt, θ, g[1])
        push!(losses, l)
        epoch % 5000 == 0 && @printf "  Epoch %6d | MSE %.3e\n" epoch l
    end
    mkpath(RESULTS)
    serialize(DL_FILE, (θ, losses))
end
a_dl(Xq) = first(nn(Xq, θ, st))
@printf "  train MSE: SINDy %.3e   DL %.3e\n" mean(abs2, Θ * Ξ - V̇') losses[end]

# ── Field comparison on [-L, L]², masking 0.3 around each charge ──
const L = 8.0
hx = range(-L, L; length = 321)
H = reduce(hcat, [[x, y] for y in hx for x in hx])
keep = vec(dmin.(eachcol(H)) .> 0.3)
on_grid(v) = (out = fill(NaN, size(H, 2)); out[keep] = v; reshape(out, length(hx), length(hx))')
Hk = H[:, keep]
A_gt = a3(Hk)
# Non-finite predictions (SINDy's 1/rⁿ terms at the origin) count as infinite error.
relerr(A) = (e = map(norm, eachcol(A - A_gt)) ./ map(norm, eachcol(A_gt)); e[.!isfinite.(e)] .= Inf; e)
E_sindy, E_dl = relerr(a_sindy(Hk)), relerr(a_dl(Hk))

# distance from each grid point to the nearest training sample
dnear = [minimum(sum(abs2, X .- h; dims = 1)) |> sqrt for h in eachcol(Hk)]
println("\n══ Field error (relative), median by distance to nearest training sample ══")
for (lab, m) in [("d < 0.25 (on the data)", dnear .< 0.25), ("0.25 ≤ d < 1", 0.25 .<= dnear .< 1),
                 ("d ≥ 1 (off the data)", dnear .>= 1), ("all of [-8,8]²", trues(length(dnear)))]
    @printf "  %-24s  %5.1f%% of grid   SINDy %.2e   DL %.2e\n" lab 100mean(m) median(E_sindy[m]) median(E_dl[m])
end

# Arrows: direction of a, length ∝ log|a|, drawn as line segments (backend-independent heads).
ga = range(-L, L; length = 17)
XA = reduce(hcat, [[x, y] for x in ga for y in ga if dmin([x, y]) > 0.45])
function arrow_xy(Xp, A)
    n = map(norm, eachcol(A))
    ℓ = 0.25 .+ 0.17 .* clamp.(log10.(n) .+ 2.2, 0, 3.3)
    xs, ys = Float64[], Float64[]
    for i in axes(Xp, 2)
        x0, y0 = Xp[:, i]
        φ = atan(A[2, i], A[1, i])
        x1, y1 = x0 + ℓ[i] * cos(φ), y0 + ℓ[i] * sin(φ)
        h = 0.35ℓ[i]
        append!(xs, [x0, x1, NaN, x1, x1 - h * cos(φ - 0.45), NaN, x1, x1 - h * cos(φ + 0.45), NaN])
        append!(ys, [y0, y1, NaN, y1, y1 - h * sin(φ - 0.45), NaN, y1, y1 - h * sin(φ + 0.45), NaN])
    end
    xs, ys
end

function charges!(p; tc = :white)
    scatter!(p, first.(C), last.(C); ms = 7, label = false,
        color = [q < 0 ? :red : :dodgerblue for q in Q], markerstrokecolor = :white)
    for (c, q) in zip(C, Q)
        annotate!(p, c[1] + 0.55, c[2] + 0.45, text(q < 0 ? "−1" : "+$(Int(q))", 11, tc))
    end
    p
end
orbits!(p; color = :white) = (for Xo in Xs; plot!(p, Xo[1, :], Xo[2, :]; color = color, lw = 1, alpha = 0.8, label = false); end; p)
panel(z, title; c, clims, cb) = heatmap(hx, hx, z; c = c, clims = clims, colorbar_title = cb, title = title,
    aspect_ratio = :equal, xlims = (-L, L), ylims = (-L, L), xlabel = "x", ylabel = "y")

function field_panel(afun, title, color)
    p = panel(on_grid(log10.(map(norm, eachcol(afun(Hk))))), title; c = :viridis, clims = (-2.2, 1.1), cb = "log₁₀|a|")
    plot!(p, arrow_xy(XA, afun(XA))...; color = color, lw = 1.1, label = false)
    charges!(p)
end
error_panel(E, title) = (p = panel(on_grid(log10.(E)), title; c = :inferno, clims = (-6, 1), cb = "log₁₀ rel. error"); orbits!(p); charges!(p))

f1 = field_panel(a3, "Ground truth  a = Σ Qᵢ(x−cᵢ)/‖x−cᵢ‖³", :white)
f2 = field_panel(a_sindy, "SINDy  a = Θ(x)Ξ", :orange)
f3 = field_panel(a_dl, "DL  a = NN(x)", :orange)
e1 = error_panel(E_sindy, "SINDy error  (white: training orbits)")
e2 = error_panel(E_dl, "DL error  (white: training orbits)")

# Same layout as compare.jl: fields on top; overlay and errors below.
c1 = plot(; aspect_ratio = :equal, xlims = (-L, L), ylims = (-L, L), legend = :topright,
    title = "Overlay: GT (α=0.5) + SINDy + DL", xlabel = "x", ylabel = "y")
plot!(c1, arrow_xy(XA, a3(XA))...; color = :blue, lw = 4, alpha = 0.5, label = "ground truth")
plot!(c1, arrow_xy(XA, a_sindy(XA))...; color = :darkorange, lw = 1.5, label = "SINDy")
plot!(c1, arrow_xy(XA, a_dl(XA))...; color = :green, lw = 1.5, label = "DL")
charges!(c1; tc = :black)
savefig(plot(f1, f2, f3, c1, e1, e2; layout = (2, 3), size = (1800, 1150), margin = 5Plots.mm),
    joinpath(RESOURCES, "three_charges_compare.png"))
