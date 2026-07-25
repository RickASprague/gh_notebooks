# Held-out reconstruction quality of the final checkpoint of normal/train.jl.
#
#   julia --project src/normal/reconstruct.jl
#
# Validation MSE / PSNR over all 50k val images, and the figures used by Auto Encoder.md:
# resources/{reconstruction,val_psnr,train_loss}.png.

include(joinpath(@__DIR__, "..", "evaluate.jl"))

model  = build_model()
ps, st = load_checkpoint(CHKPT_DIR)
yv, Xv = load_val()

mse, psnr = val_metrics(model, ps, st, Xv)
@printf("latent / input : %d / %d = %.3f\n", 4*4*256, 64*64*3, 4*4*256 / (64*64*3))

plot_grid(model, ps, st, Xv, psnr, joinpath(RESOURCES, "reconstruction.png"))
plot_psnr(psnr, joinpath(RESOURCES, "val_psnr.png"))
plot_loss(joinpath(RESULTS, "train_loss.csv"), joinpath(RESULTS, "train_epochs.csv"),
          joinpath(RESOURCES, "train_loss.png");
          title = "Training loss (batch 1024, epoch boundaries dashed)")
