# Compare the identified acceleration fields a(x): ground truth, SINDy, DL — out to ±8,
# well beyond the training orbits, to see where each model fails.
# Needs results/ from sindy.jl and dl.jl.
#
#   julia --project src/compare.jl

include("shared.jl")
using Lux, ComponentArrays, Serialization, Statistics

# ── Load models ──
Ξ = reduce(vcat, [parse.(Float64, split(l))' for l in eachline(joinpath(RESULTS, "sindy_xi.txt"))])
a_sindy(X) = (library(X)[1] * Ξ)'
m = deserialize(joinpath(RESULTS, "dl_model.jls"))
a_dl(X) = first(m.nn(X, m.θ, m.st))

Xs, _, _ = training_orbits(20)                       # positions only, for the coverage overlay
Xtr = reduce(hcat, Xs)
rtr = hypot.(Xtr[1, :], Xtr[2, :])

# ── Grids ──
const L = 8.0                                        # plot extent ±L
const rmin = 0.3
hx = range(-L, L; length = 321)
H = reduce(hcat, [[x, y] for y in hx for x in hx])
keep = vec(hypot.(H[1, :], H[2, :]) .> rmin)
on_grid(v) = (out = fill(NaN, size(H, 2)); out[keep] = v; reshape(out, length(hx), length(hx))')

ga = range(-L, L; length = 17)                       # arrow sites, spacing 1
XA = reduce(hcat, [[x, y] for x in ga for y in ga if hypot(x, y) > 0.5])

logmag(afun) = on_grid(log10.(map(norm, eachcol(afun(H[:, keep])))))
logerr(afun) = on_grid(log10.(rel_err(afun, H[:, keep])))

base(title; kw...) = plot(; aspect_ratio = :equal, xlims = (-L, L), ylims = (-L, L), title = title,
    xlabel = "x", ylabel = "y", kw...)
charge!(p) = scatter!(p, [0], [0]; ms = 5, color = :red, label = false)
orbits!(p) = (for X in Xs; plot!(p, X[1, :], X[2, :]; color = :white, lw = 1, alpha = 0.8, label = false); end; p)

function field_panel(afun, title, color)
    p = heatmap(hx, hx, logmag(afun); c = :viridis, clims = (-2.2, 1.1), colorbar_title = "log₁₀|a|",
        aspect_ratio = :equal, xlims = (-L, L), ylims = (-L, L), title = title, xlabel = "x", ylabel = "y")
    plot!(p, arrow_xy(XA, afun(XA))...; color = color, lw = 1.2, label = false)
    charge!(p)
end

function error_panel(afun, title)
    p = heatmap(hx, hx, logerr(afun); c = :inferno, clims = (-6, 1), colorbar_title = "log₁₀ rel. error",
        aspect_ratio = :equal, xlims = (-L, L), ylims = (-L, L), title = title, xlabel = "x", ylabel = "y")
    orbits!(p)
    charge!(p)
end

# ── Figure 1: fields (top), training coverage + errors (bottom) ──
f1 = field_panel(a_gt, "Ground truth  a = αx/r³", :white)
f2 = field_panel(a_sindy, "SINDy  a = Θ(x)Ξ", :orange)
f3 = field_panel(a_dl, "DL  a = NN(x)", :orange)

c1 = base("Overlay: GT (α=0.5) + SINDy + DL"; legend = :topright)
plot!(c1, arrow_xy(XA, a_gt(XA))...; color = :blue, lw = 4, alpha = 0.5, label = "ground truth")
plot!(c1, arrow_xy(XA, a_sindy(XA))...; color = :darkorange, lw = 1.5, label = "SINDy")
plot!(c1, arrow_xy(XA, a_dl(XA))...; color = :green, lw = 1.5, label = "DL")
charge!(c1)
e1 = error_panel(a_sindy, "SINDy error  (white: training orbits)")
e2 = error_panel(a_dl, "DL error  (white: training orbits)")

savefig(plot(f1, f2, f3, c1, e1, e2; layout = (2, 3), size = (1800, 1150), margin = 5Plots.mm),
    joinpath(RESOURCES, "compare_field.png"))

# ── Figure 2: radial profile |a|(r), averaged over 64 directions, out to r = 100 ──
rs = exp10.(range(-1, 2; length = 300))
φs = range(0, 2π; length = 65)[1:end-1]
function radial(afun)
    map(rs) do r
        Xr = reduce(hcat, [[r * cos(φ), r * sin(φ)] for φ in φs])
        A = afun(Xr)
        # radial component a·r̂ (negative = attractive) and its magnitude
        mean(-(A[1, :] .* cos.(φs) .+ A[2, :] .* sin.(φs)))
    end
end
p = plot(; xscale = :log10, yscale = :log10, xlabel = "r", ylabel = "−a·r̂  (inward acceleration)",
    title = "Radial profile, mean over 64 directions", legend = :bottomleft, size = (800, 550))
vspan!(p, [minimum(rtr), maximum(rtr)]; color = :gray, alpha = 0.2, label = "training r-range")
plot!(p, rs, abs(α) ./ rs .^ 2; lw = 4, color = :blue, alpha = 0.5, label = "ground truth  |α|/r²")
plot!(p, rs, max.(radial(a_sindy), 1e-12); lw = 2, color = :darkorange, label = "SINDy")
dl = radial(a_dl)
plot!(p, rs[dl .> 0], dl[dl .> 0]; lw = 2, color = :green, label = "DL (where > 0)")
savefig(p, joinpath(RESOURCES, "compare_radial.png"))

# ── Numbers ──
rq = hypot.(H[1, keep], H[2, keep])
for (name, f) in [("SINDy", a_sindy), ("DL", a_dl)]
    E = rel_err(f, H[:, keep])
    inside = (rq .>= minimum(rtr)) .& (rq .<= 4)
    @printf "%-5s  median rel err: r<0.84 %.2e | 0.84≤r≤4 %.2e | r>4 %.2e | r>8 %.2e\n" name median(E[rq .< minimum(rtr)]) median(E[inside]) median(E[rq .> 4]) median(E[rq .> 8])
end
for r in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    i = argmin(abs.(rs .- r))
    @printf "r = %6.1f   GT %.3e   SINDy %.3e   DL %+.3e\n" rs[i] abs(α) / rs[i]^2 radial(a_sindy)[i] dl[i]
end
