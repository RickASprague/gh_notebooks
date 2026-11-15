# Replay a finished run's CSVs into TensorBoard, as a curve to compare live runs against.
#
#   julia --project src/tb_import.jl <results dir> <run name> [csv suffix]
#   julia --project src/tb_import.jl results/nas-titanx-20261115 vgg-b512-nas-titanx
#
# Reads <dir>/train_loss<suffix>.csv (per step) and <dir>/train_epochs<suffix>.csv (per epoch) and
# writes build/tensorboard/<run name>/ with the same tags train.jl logs live: train/loss per step,
# val/top1, val/top5, train/epoch_loss, time/epoch_s and train/img_per_s at each epoch's last step.

using TensorBoardLogger, DelimitedFiles

const ROOT = dirname(@__DIR__)

dir, name = ARGS[1], ARGS[2]
suffix    = length(ARGS) ≥ 3 ? ARGS[3] : ""

loss = vec(readdlm(joinpath(dir, "train_loss$suffix.csv"), ',', Float64))
ep, hdr = readdlm(joinpath(dir, "train_epochs$suffix.csv"), ',', Float64; header = true)
col = Dict(h => i for (i, h) in enumerate(vec(hdr)))
nep = size(ep, 1)
spe = length(loss) ÷ nep                         # steps per epoch

tb = TBLogger(joinpath(ROOT, "build", "tensorboard", name), tb_overwrite)
for (s, ℓ) in enumerate(loss)
    log_value(tb, "train/loss", ℓ; step = s)
end
for e in 1:nep
    step = e * spe
    log_value(tb, "val/top1",         100ep[e, col["val_top1"]]; step)
    log_value(tb, "val/top5",         100ep[e, col["val_top5"]]; step)
    log_value(tb, "train/epoch_loss", ep[e, col["loss"]];        step)
    log_value(tb, "time/epoch_s",     ep[e, col["seconds"]];     step)
    log_value(tb, "train/img_per_s",  ep[e, col["img_per_s"]];   step)
end
println("wrote build/tensorboard/$name: $(length(loss)) steps, $nep epochs, $spe steps/epoch")
