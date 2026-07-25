# Deep learning for the dynamics:  v̇ = NN_θ(x),  an MLP regressed on the same (x, v̇) pairs as SINDy.
# No library, no ODE solver: plain supervised regression, MSE, Adam.
#
#   julia --project src/dl.jl

include("shared.jl")
using Lux, Optimisers, Zygote, ComponentArrays, Serialization, Statistics

Xs, V̇s, _ = training_orbits(20)
X, V̇ = reduce(hcat, Xs), reduce(hcat, V̇s)            # 2 × 7980 each
r = hypot.(X[1, :], X[2, :])
@printf "Training set: N = %d, r ∈ [%.2f, %.2f]\n" size(X, 2) minimum(r) maximum(r)

nn = Chain(Dense(2 => 64, tanh), Dense(64 => 64, tanh), Dense(64 => 2))
ps, st = Lux.setup(Xoshiro(0), nn)
θ = ComponentArray{Float64}(ps)
println("Parameters: $(length(θ))")

mse(θ) = mean(abs2, first(nn(X, θ, st)) .- V̇)

# Adam, lr 1e-3 → 1e-4 → 1e-5 (step decay), full batch.
opt = Optimisers.setup(Optimisers.Adam(1e-3), θ)
losses = Float64[]
epochs = 30_000
@time for epoch in 1:epochs
    epoch == 15_000 && Optimisers.adjust!(opt, 1e-4)
    epoch == 25_000 && Optimisers.adjust!(opt, 1e-5)
    l, g = Zygote.withgradient(mse, θ)
    global opt, θ = Optimisers.update(opt, θ, g[1])
    push!(losses, l)
    epoch % 5000 == 0 && @printf "Epoch %6d | MSE %.3e\n" epoch l
end

plt = plot(losses; yscale = :log10, lw = 2, label = false, xlabel = "Epoch", ylabel = "MSE (log)",
    title = "DL: v̇ = NN(x) training")
vline!(plt, [15_000, 25_000]; ls = :dash, color = :gray, label = "lr ÷10")
savefig(plt, joinpath(RESOURCES, "dl_loss.png"))

a_dl(Xq) = first(nn(Xq, θ, st))
@printf "Final train MSE: %.3e\n" losses[end]
@printf "Field error on grid [-4,4]², r>0.3: median %.2e, max %.2e\n" median(rel_err(a_dl, Xg)) maximum(rel_err(a_dl, Xg))

mkpath(RESULTS)
serialize(joinpath(RESULTS, "dl_model.jls"), (; nn, θ, st))
