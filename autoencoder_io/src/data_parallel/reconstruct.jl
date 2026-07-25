# Held-out reconstruction quality of the data-parallel model (data_parallel/train.jl, data-parallel run).
#
#   julia --project src/data_parallel/reconstruct.jl
#
# Same evaluation as normal/reconstruct.jl, on build/checkpoint_dp. Figures used by
# Auto Encoder - Data Parallel.md: resources/dp_{reconstruction,val_psnr,train_loss,loss_compare}.png.
# The comparison needs results/train_loss.csv from normal/train.jl.

include(joinpath(@__DIR__, "..", "evaluate.jl"))

model  = build_model()
ps, st = load_checkpoint(CHKPT_DIR_DP)
yv, Xv = load_val()

mse, psnr = val_metrics(model, ps, st, Xv)

plot_grid(model, ps, st, Xv, psnr, joinpath(RESOURCES, "dp_reconstruction.png"))
plot_psnr(psnr, joinpath(RESOURCES, "dp_val_psnr.png"); title = "Validation PSNR, 50k images (data parallel)")
plot_loss(joinpath(RESULTS, "dp_train_loss.csv"), joinpath(RESULTS, "dp_train_epochs.csv"),
          joinpath(RESOURCES, "dp_train_loss.png");
          title = "Training loss (5 × 1024 per step, epoch boundaries dashed)")

# Data parallel vs single GPU (normal/train.jl), per epoch and per update.
plot_loss_compare([
    ("single GPU, 1024", joinpath(RESULTS, "train_loss.csv"),    joinpath(RESULTS, "train_epochs.csv"),    :steelblue),
    ("data parallel, 5 × 1024", joinpath(RESULTS, "dp_train_loss.csv"), joinpath(RESULTS, "dp_train_epochs.csv"), :firebrick),
], joinpath(RESOURCES, "dp_loss_compare.png"))
