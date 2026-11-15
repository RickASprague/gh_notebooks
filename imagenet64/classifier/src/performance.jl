# Per-class performance on the 50 000 validation images: the 1000 × 1000 confusion matrix, each
# class's recall and precision, and a precision–recall scatter with one dot per class.
#
#   julia --project src/performance.jl                       # BF16 Reactant ResNet-18, epoch 9 (best)
#   julia --project src/performance.jl <checkpoint.jld2> [model] [tag]
#
# Writes results/performance-<tag>.csv (per class) and results/confusion-<tag>.csv (the nonzero
# cells), and the plot to resources/pr_scatter-<tag>.png (tracked, so the README can show it).
# Plotting lives here, never in a training script.
#
# Class names come from the ILSVRC2012 devkit: label k of the ImageNet-64 shards is the devkit's
# ILSVRC2012_ID k. Checked by matching 2 000 original val JPEGs, downsampled to 64 × 64, to their
# twins in the val shard (cosine similarity ≥ 0.987, every one distinct): 2 000 / 2 000 agree. The
# shard's image order differs from the devkit's, so its ground-truth file can't be used directly.

using Lux, LuxCUDA, JLD2, MAT, CairoMakie, Printf, DelimitedFiles

const ROOT   = dirname(@__DIR__)
const DEVKIT = "/data/datasets/imagenet/2012/ILSVRC2012_devkit_t12/data"
const CHKPT  = length(ARGS) ≥ 1 ? ARGS[1] :
               joinpath(ROOT, "build", "checkpoint-resnet18-reactant-bf16", "model_9.jld2")
const MODEL  = length(ARGS) ≥ 2 ? ARGS[2] : "resnet18"
const TAG    = length(ARGS) ≥ 3 ? ARGS[3] : "resnet18-reactant-bf16"
const NCLASS = 1000
const BATCH  = 1000

include(joinpath(dirname(ROOT), "utils", "data.jl"))        # load_val
include(joinpath(@__DIR__, "model.jl"))                      # build_model

# (wnid, short name) per label 1…1000, from the devkit's synsets (ILSVRC2012_ID ≤ 1000 are the leaves).
function class_names()
    m = matread(joinpath(DEVKIT, "meta.mat"))["synsets"]
    ids = vec(Int.(m["ILSVRC2012_ID"]))
    wnid, words = vec(m["WNID"]), vec(m["words"])
    names = Vector{Tuple{String, String}}(undef, NCLASS)
    full = Vector{String}(undef, NCLASS)
    for i ∈ eachindex(ids)
        ids[i] ≤ NCLASS || continue
        names[ids[i]] = (wnid[i], first(split(words[i], ", ")))
        full[ids[i]] = words[i]
    end
    # A few short names are shared by two classes (crane the bird and the machine; maillot twice):
    # those get their full WordNet words instead.
    short = last.(names)
    for k ∈ 1:NCLASS
        count(==(short[k]), short) > 1 && (names[k] = (names[k][1], full[k]))
    end
    return names
end

# Top-1 prediction for every val image: an FP32 forward pass in test mode, batches of BATCH.
function predict(model, ps, st, X)
    dev = gpu_device()
    ps, st = dev(ps), dev(Lux.testmode(st))
    N = size(X, 4)
    ŷ = Vector{Int}(undef, N)
    for i ∈ 1:BATCH:N
        j = min(i + BATCH - 1, N)
        z, _ = model(dev(Array(view(X, :, :, :, i:j))), ps, st)
        ŷ[i:j] = vec(getindex.(argmax(Array(z); dims=1), 1))
    end
    return ŷ
end

# Wilson score interval for k successes in n trials (z = 1.96, 95 %): unlike k/n ± 1.96·se it stays
# inside [0, 1] and is sensible at k = 0 or n, which matters with 50 images per class.
function wilson(k, n; z = 1.96)
    p = k / n
    c = (p + z^2 / 2n) / (1 + z^2 / n)
    h = z / (1 + z^2 / n) * sqrt(p * (1 - p) / n + z^2 / 4n^2)
    return c - h, c + h
end

function main()
    yv, Xv = load_val()
    names = class_names()
    model = build_model(MODEL; nclass = NCLASS)
    ck = JLD2.load(CHKPT)
    ŷ = predict(model, ck["parameters"], ck["state"], Xv)

    # C[y, ŷ]: rows are the true class, columns the prediction.
    C = zeros(Int, NCLASS, NCLASS)
    for (y, p) ∈ zip(yv, ŷ)
        C[y, p] += 1
    end
    n_true = vec(sum(C; dims=2))                 # 50 per class
    n_pred = vec(sum(C; dims=1))                 # how often each class is predicted
    hits   = [C[i, i] for i ∈ 1:NCLASS]
    recall    = hits ./ n_true
    precision = [n_pred[i] == 0 ? NaN : hits[i] / n_pred[i] for i ∈ 1:NCLASS]
    f1 = [isnan(p) || p + r == 0 ? 0.0 : 2p * r / (p + r) for (p, r) ∈ zip(precision, recall)]
    top1 = sum(hits) / length(yv)
    @printf("%s: top-1 %.2f %% on %d val images; %d classes never predicted\n",
            basename(CHKPT), 100top1, length(yv), count(==(0), n_pred))

    # The most frequent wrong predictions for class i: its row without the diagonal.
    function confusions(i, k = 3)
        row = copy(C[i, :]); row[i] = 0
        js = partialsortperm(row, 1:k; rev = true)
        return join(["$(names[j][2]) ($(row[j]))" for j ∈ js if row[j] > 0], "; ")
    end

    mkpath(joinpath(ROOT, "results"))
    open(joinpath(ROOT, "results", "performance-$TAG.csv"), "w") do io
        println(io, "label,wnid,name,recall,recall_lo95,recall_hi95,precision,f1,n_predicted,top_confusions")
        for i ∈ 1:NCLASS
            lo, hi = wilson(hits[i], n_true[i])
            @printf(io, "%d,%s,\"%s\",%.2f,%.3f,%.3f,%.3f,%.3f,%d,\"%s\"\n", i, names[i]..., recall[i],
                    lo, hi, precision[i], f1[i], n_pred[i], confusions(i))
        end
    end
    ij = findall(!iszero, C)
    writedlm(joinpath(ROOT, "results", "confusion-$TAG.csv"),
             vcat(["true" "predicted" "count"], hcat(first.(Tuple.(ij)), last.(Tuple.(ij)), C[ij])), ',')

    # Best and worst by F1, which needs both: found when present (recall) and right when claimed
    # (precision). A high-recall, low-precision class is a sink that absorbs other classes' images.
    order = sortperm(f1)
    show(i) = @printf("  %-28s recall %4.0f %%  precision %4.0f %%  F1 %.2f   ← %s\n",
                      names[i][2], 100recall[i], 100precision[i], f1[i], confusions(i))
    println("\nworst 15 by F1:"); foreach(show, order[1:15])
    println("\nbest 15 by F1:");  foreach(show, reverse(order[end-14:end]))

    # The scatter. Recall takes only the 51 values k/50, so the dots form vertical stripes; a small
    # horizontal jitter (±0.004, a fifth of a stripe gap) shows how many sit on each.
    ok = .!isnan.(precision)
    jit = 0.008 .* (rand(NCLASS) .- 0.5)
    fig = Figure(size = (1100, 1000))
    ax = Axis(fig[1, 1]; xlabel = "recall (of the 50 val images of the class, how many are found)",
              ylabel = "precision (of the images predicted as the class, how many are right)",
              title = @sprintf("%s — per-class precision vs recall, %d classes, top-1 %.1f %%", TAG, NCLASS, 100top1),
              limits = (-0.02, 1.02, -0.02, 1.02), aspect = 1)
    sc = scatter!(ax, recall[ok] .+ jit[ok], precision[ok]; color = f1[ok], colormap = :viridis,
                  colorrange = (0, 1), markersize = 6, alpha = 0.8)
    lines!(ax, [0, 1], [0, 1]; color = (:gray, 0.4), linestyle = :dash)
    Colorbar(fig[1, 2], sc; label = "F1")
    # Name the 8 best and 8 worst by F1 and the 6 biggest sinks (most often predicted).
    sinks = partialsortperm(n_pred, 1:6; rev = true)
    for (i, c) ∈ [(i, :darkgreen) for i ∈ order[end-7:end]] ∪ [(i, :firebrick) for i ∈ order[1:8] if ok[i]] ∪
                 [(i, :darkorange) for i ∈ sinks]
        right = recall[i] > 0.85                              # near the right edge: label on the left
        text!(ax, recall[i] + jit[i], precision[i]; text = right ? names[i][2] * " " : " " * names[i][2],
              fontsize = 10, color = c, align = (right ? :right : :left, :center))
    end
    mkpath(joinpath(ROOT, "resources"))
    path = joinpath(ROOT, "resources", "pr_scatter-$TAG.png")
    save(path, fig)
    println("\nwrote results/performance-$TAG.csv, results/confusion-$TAG.csv, resources/pr_scatter-$TAG.png")
end

main()
