# DL for three fixed charges: v̇ = NN_θ(x), best fit within a wall-clock budget.
# 20 training orbits (same as three_charges.jl), 10 held-out orbits select the best weights.
#
#   julia --project src/three_charges_dl.jl

include("shared.jl")
using Lux, Optimisers, Zygote, ComponentArrays, Statistics, Serialization
BLAS.set_num_threads(16)

const BUDGET = 420.0                                  # training seconds
const C = [1.5 .* [cos(φ), sin(φ)] for φ in (π / 2, π / 2 + 2π / 3, π / 2 + 4π / 3)]
const Q = [-1.0, 1.0, 2.0]
accel3(x) = sum(q .* (x .- c) ./ norm(x .- c)^3 for (c, q) in zip(C, Q))
a3(X) = reduce(hcat, [accel3(x) for x in eachcol(X)])
three!(du, u, p, t) = (du[1:2] = u[3:4]; du[3:4] = accel3(u[1:2]))
dmin(x) = minimum(norm(x .- c) for c in C)

function orbits(n, seed)
    rng = Xoshiro(seed)
    Xs, V̇s = Matrix{Float64}[], Matrix{Float64}[]
    while length(Xs) < n
        θ0, r0 = 2π * rand(rng), 2 + 2rand(rng)
        u0 = [r0 * cos(θ0), r0 * sin(θ0), 0.8randn(rng), 0.8randn(rng)]
        dmin(u0[1:2]) > 0.3 || continue
        Z = reduce(hcat, solve(ODEProblem(three!, u0, (0.0, T)), Tsit5();
            abstol = 1e-10, reltol = 1e-10, saveat = dt).u)
        X, V̇ = Z[1:2, 2:end-1], (Z[3:4, 3:end] - Z[3:4, 1:end-2]) / (2dt)
        minimum(dmin.(eachcol(X))) > 0.3 || continue
        push!(Xs, X); push!(V̇s, V̇)
    end
    Xs, V̇s
end

Xs, V̇s = orbits(20, 7)
Xh, V̇h = (reduce(hcat, v) for v in orbits(10, 8))
X, V̇ = reduce(hcat, Xs), reduce(hcat, V̇s)
Xf, V̇f, Xhf, V̇hf = Float32.(X ./ 4), Float32.(V̇), Float32.(Xh ./ 4), Float32.(V̇h)

nn = Chain(Dense(2 => 128, tanh), Dense(128 => 128, tanh), Dense(128 => 128, tanh), Dense(128 => 2))
ps, st = Lux.setup(Xoshiro(0), nn)
θ = ComponentArray(ps)
println("Parameters: $(length(θ)), N = $(size(X, 2)), held out = $(size(Xh, 2))")
loss(θ, A, B) = mean(abs2, first(nn(A, θ, st)) .- B)

opt = Optimisers.setup(Optimisers.Adam(3f-3), θ)
Zygote.withgradient(t -> loss(t, Xf, V̇f), θ)         # compile outside the budget
best, bestθ = Inf, copy(θ)
hist = NTuple{4,Float64}[]
t0, epoch = time(), 0
while (el = time() - t0) < BUDGET
    global epoch += 1
    Optimisers.adjust!(opt, Float32(1e-5 + 0.5 * (3e-3 - 1e-5) * (1 + cos(π * el / BUDGET))))
    l, g = Zygote.withgradient(t -> loss(t, Xf, V̇f), θ)
    global opt, θ = Optimisers.update(opt, θ, g[1])
    if epoch % 100 == 0
        h = loss(θ, Xhf, V̇hf)
        push!(hist, (epoch, el, l, h))
        h < best && (global best = h; global bestθ = copy(θ))
        epoch % 2000 == 0 && @printf "epoch %6d  %5.0fs  train %.3e  held-out %.3e\n" epoch el l h
    end
end
@printf "Done: %d epochs, best held-out MSE %.3e, train MSE at best %.3e\n" epoch best loss(bestθ, Xf, V̇f)
mkpath(RESULTS)
serialize(joinpath(RESULTS, "three_charges_dl_best.jls"), (; nn, θ = bestθ, st, hist))

a_new(Q_) = Float64.(first(nn(Float32.(Q_ ./ 4), bestθ, st)))
old_nn = Chain(Dense(2 => 64, tanh), Dense(64 => 64, tanh), Dense(64 => 2))
_, st_old = Lux.setup(Xoshiro(0), old_nn)
θ_old, _ = deserialize(joinpath(RESULTS, "three_charges_dl.jls"))
a_old(Q_) = first(old_nn(Q_, θ_old, st_old))
@printf "Held-out MSE: first model %.3e   this model %.3e\n" mean(abs2, a_old(Xh) .- V̇h) mean(abs2, a_new(Xh) .- V̇h)

# ── Field error on [-8,8]² by distance to the nearest training sample ──
g = range(-8, 8; length = 161)
G = reduce(hcat, [[x, y] for y in g for x in g if dmin([x, y]) > 0.3])
A = a3(G)
rel(f) = map(norm, eachcol(f(G) - A)) ./ map(norm, eachcol(A))
d = [sqrt(minimum(sum(abs2, X .- h; dims = 1))) for h in eachcol(G)]
Eo, En = rel(a_old), rel(a_new)
for (lab, m) in [("d < 0.25", d .< 0.25), ("0.25 ≤ d < 1", 0.25 .<= d .< 1), ("d ≥ 1", d .>= 1), ("all", trues(length(d)))]
    @printf "%-14s %5.1f%% of grid   first model %.3f   this model %.3f\n" lab 100mean(m) median(Eo[m]) median(En[m])
end

# ── Figure: ground truth, DL, overlay, error ──
L = 8.0
hx = range(-L, L; length = 321)
H = reduce(hcat, [[x, y] for y in hx for x in hx])
keep = vec(dmin.(eachcol(H)) .> 0.3)
Hk = H[:, keep]
grid(v) = (o = fill(NaN, size(H, 2)); o[keep] = v; reshape(o, length(hx), length(hx))')
ga = range(-L, L; length = 17)
XA = reduce(hcat, [[x, y] for x in ga for y in ga if dmin([x, y]) > 0.45])
function arrows(Xp, Ap)
    n = map(norm, eachcol(Ap)); ℓ = 0.25 .+ 0.17 .* clamp.(log10.(n) .+ 2.2, 0, 3.3)
    xs, ys = Float64[], Float64[]
    for i in axes(Xp, 2)
        x0, y0 = Xp[:, i]; φ = atan(Ap[2, i], Ap[1, i])
        x1, y1 = x0 + ℓ[i] * cos(φ), y0 + ℓ[i] * sin(φ); h = 0.35ℓ[i]
        append!(xs, [x0, x1, NaN, x1, x1 - h * cos(φ - 0.45), NaN, x1, x1 - h * cos(φ + 0.45), NaN])
        append!(ys, [y0, y1, NaN, y1, y1 - h * sin(φ - 0.45), NaN, y1, y1 - h * sin(φ + 0.45), NaN])
    end
    xs, ys
end
function charges!(p; tc = :white)
    scatter!(p, first.(C), last.(C); ms = 7, label = false, color = [q < 0 ? :red : :dodgerblue for q in Q], markerstrokecolor = :white)
    for (c, q) in zip(C, Q); annotate!(p, c[1] + 0.55, c[2] + 0.45, text(q < 0 ? "−1" : "+$(Int(q))", 11, tc)); end
    p
end
hm(z, t; c, clims, cb) = heatmap(hx, hx, z; c, clims, colorbar_title = cb, title = t, aspect_ratio = :equal,
    xlims = (-L, L), ylims = (-L, L), xlabel = "x", ylabel = "y")
function fieldp(f, t, col)
    p = hm(grid(log10.(map(norm, eachcol(f(Hk))))), t; c = :viridis, clims = (-2.2, 1.1), cb = "log₁₀|a|")
    plot!(p, arrows(XA, f(XA))...; color = col, lw = 1.2, label = false); charges!(p)
end
f1 = fieldp(a3, "Ground truth  a = Σ Qᵢ(x−cᵢ)/‖x−cᵢ‖³", :white)
f2 = fieldp(a_new, "DL  a = NN(x)", :orange)
c1 = plot(; aspect_ratio = :equal, xlims = (-L, L), ylims = (-L, L), legend = :topright,
    title = "Overlay: GT (α=0.5) + DL", xlabel = "x", ylabel = "y")
plot!(c1, arrows(XA, a3(XA))...; color = :blue, lw = 4, alpha = 0.5, label = "ground truth")
plot!(c1, arrows(XA, a_new(XA))...; color = :green, lw = 1.5, label = "DL")
charges!(c1; tc = :black)
e1 = hm(grid(log10.(map(norm, eachcol(a_new(Hk) - a3(Hk))) ./ map(norm, eachcol(a3(Hk))))),
    "DL error  (white: training orbits)"; c = :inferno, clims = (-6, 1), cb = "log₁₀ rel. error")
for Xo in Xs; plot!(e1, Xo[1, :], Xo[2, :]; color = :white, lw = 1, alpha = 0.8, label = false); end
charges!(e1)
savefig(plot(f1, f2, c1, e1; layout = (2, 2), size = (1250, 1150), margin = 5Plots.mm),
    joinpath(RESOURCES, "three_charges_dl.png"))
