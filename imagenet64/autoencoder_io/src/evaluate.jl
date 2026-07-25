# Validation evaluation and figures, shared by normal/reconstruct.jl and data_parallel/reconstruct.jl.
#
# Loads CairoMakie, so it is only included by the reconstruction scripts and never alongside a
# training loop (CairoMakie + Zygote in one process stalls the first gradient's compile).

include(joinpath(@__DIR__, "shared.jl"))
using JLD2, CairoMakie, DelimitedFiles

# Final checkpoint → (parameters, BatchNorm state in test mode), on the GPU.
function load_checkpoint(dir; epoch = 10)
    ck  = joinpath(dir, "model_$epoch.jld2")
    dev = gpu_device()
    ps  = load(ck, "parameters") |> dev
    st  = Lux.testmode(load(ck, "state")) |> dev        # BN running statistics
    return ps, st
end

# Per-image MSE and PSNR (peak 1) over the whole validation set.
function val_metrics(model, ps, st, Xv; B = 1000)
    mse = Float32[]
    for i ∈ 1:B:size(Xv, 4)
        j = min(i + B - 1, size(Xv, 4))
        x = CuArray(Xv[:, :, :, i:j])
        ŷ, _ = Lux.apply(model, x, ps, st)
        append!(mse, vec(Array(mean((ŷ .- x) .^ 2, dims = (1, 2, 3)))))
    end
    psnr = -10 .* log10.(mse)
    @printf("val images     : %d\n", length(mse))
    @printf("val MSE        : mean %.5f  median %.5f\n", mean(mse), median(mse))
    @printf("val PSNR (dB)  : mean %.2f  p10 %.2f  p90 %.2f\n",
            mean(psnr), quantile(psnr, 0.1), quantile(psnr, 0.9))
    return mse, psnr
end

# 8 validation images spanning the PSNR range, worst → best; inputs over reconstructions.
function plot_grid(model, ps, st, Xv, psnr, path)
    order = sortperm(psnr)
    pick  = order[round.(Int, range(1, length(psnr), length = 8))]
    ŷ = Array(first(Lux.apply(model, CuArray(Xv[:, :, :, pick]), ps, st)))
    fig = Figure(size = (8 * 130, 2 * 140 + 30))
    for (c, k) ∈ enumerate(pick)
        a1 = Axis(fig[1, c], title = @sprintf("%.1f dB", psnr[k]), aspect = DataAspect())
        a2 = Axis(fig[2, c], aspect = DataAspect())
        image!(a1, Utils.makie_image(Xv[:, :, :, k]))
        image!(a2, Utils.makie_image(ŷ[:, :, :, c]))
        hidedecorations!.((a1, a2)); hidespines!.((a1, a2))
    end
    Label(fig[1, 0], "input",   rotation = π/2, tellheight = false)
    Label(fig[2, 0], "4×4×256", rotation = π/2, tellheight = false)
    save(path, fig)
end

function plot_psnr(psnr, path; title = "Validation PSNR, 50k images")
    fig = Figure(size = (600, 300))
    ax  = Axis(fig[1, 1], xlabel = "PSNR (dB)", ylabel = "count", title = title)
    hist!(ax, psnr, bins = 80)
    save(path, fig)
end

# Two training-loss curves (50-step moving averages) against epochs and against optimizer
# updates: the same images per epoch, different numbers of updates.
function plot_loss_compare(runs, path)
    fig = Figure(size = (1000, 340))
    ax1 = Axis(fig[1, 1], xlabel = "epoch", ylabel = "MSE", yscale = log10, title = "per epoch")
    ax2 = Axis(fig[1, 2], xlabel = "optimizer update", ylabel = "MSE", yscale = log10,
               xscale = log10, title = "per update")
    for (label, loss_csv, epochs_csv, color) in runs
        loss    = vec(readdlm(loss_csv, ',', Float32))
        nepochs = size(readdlm(epochs_csv, ',', header = true)[1], 1)
        w = max(1, min(50, length(loss) ÷ (2nepochs)))
        smooth = [mean(loss[max(1, i - w + 1):i]) for i in eachindex(loss)]
        steps  = collect(1:length(loss))
        lines!(ax1, steps .* (nepochs / length(loss)), smooth, color = color, label = label)
        lines!(ax2, steps, smooth, color = color, label = label)
    end
    axislegend(ax1, position = :rt); axislegend(ax2, position = :rt)
    save(path, fig)
end

# Per-step training loss with a 50-step moving average, epoch boundaries dashed.
function plot_loss(loss_csv, epochs_csv, path; title)
    loss    = vec(readdlm(loss_csv, ',', Float32))
    nepochs = size(readdlm(epochs_csv, ',', header = true)[1], 1)
    fig = Figure(size = (700, 320))
    ax  = Axis(fig[1, 1], xlabel = "optimizer step", ylabel = "MSE", yscale = log10, title = title)
    lines!(ax, eachindex(loss), loss, linewidth = 0.5, color = (:steelblue, 0.4))
    w = max(1, min(50, length(loss) ÷ (2nepochs)))
    smooth = [mean(loss[max(1, i - w + 1):i]) for i in eachindex(loss)]
    lines!(ax, eachindex(smooth), smooth, color = :black, linewidth = 1.5)
    vlines!(ax, cumsum(fill(length(loss) ÷ nepochs, nepochs - 1)), color = :gray, linestyle = :dash)
    save(path, fig)
end
