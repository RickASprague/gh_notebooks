# ImageNet Autoencoder: Making the GPU the Bottleneck

A convolutional autoencoder trained on all of ImageNet-64 (1.28M images), with data loading kept
off the critical path. The dataset is converted once from Python pickles to flat binary shards and
memory-mapped at training time, so there is no per-pass deserialization and no upfront load.

## Documents

| document | covers |
|---|---|
| [Image Preprocess.md](Image%20Preprocess.md) | pickle → mmap shard conversion and the binary format |
| [Auto Encoder.md](Auto%20Encoder.md) | data loading, model, training, IO accounting, Nsight Systems / Compute analysis, reconstruction quality |
| [Auto Encoder - Data Parallel.md](Auto%20Encoder%20-%20Data%20Parallel.md) | simulated 5-way synchronous data parallelism on one GPU, and its reassembled multi-GPU cost |

## Code

```
src/
  shared.jl            shard loader, model, MSE gradient, the normal training loop (used by everything)
  evaluate.jl          validation metrics and figures (used by both reconstruct scripts)
  preprocess.jl        one-time pickle → mmap shard conversion
  normal/
    train.jl           10-epoch training run
    reconstruct.jl     validation quality and figures
    profile.jl         training loop with NVTX ranges, for Nsight Systems / Compute
  data_parallel/
    common.jl          data-parallel primitives (all-reduce, synchronized timing)
    verify.jl          arithmetic checks
    train.jl           simulated 5-GPU run (10 epochs, saves the model), baseline, reassembly
    reconstruct.jl     validation quality and figures for the data-parallel model
lib/
  utils.jl             binary serialization and MMapReader (zero-copy mmap views)
  imagenet.jl          pickle loading via PythonCall (preprocessing only)
```

Run from this folder, e.g. `julia --project src/normal/train.jl`. Outputs go to `build/`
(checkpoints, shards) and `results/` (logs, CSVs, profiler reports); figures go to `resources/`.

## Results

- **Loading:** mapping the 63 GB of shards takes 0.17 s; pages are faulted in on first use.
- **Training:** 24.8 s per epoch in steady state, ~52k img/s, batch 1024. The GPU has work 91% of
  each step.
- **GPU efficiency:** the kernels are latency-bound. SM issue is 12% and tensor cores are active 4%,
  because the model's small feature maps and per-channel BatchNorm cannot fill 188 SMs.
- **Reconstruction:** 4×4×256 latent (1/3 of the input's values), validation PSNR 29.5 dB.
- **Data parallel:** 5 GPUs would bring an epoch to ~5–7.1 s; the all-reduce of 0.78 M parameters
  costs 0.12 ms per step. With the learning rate unscaled, 10 epochs reach 26.8 dB: per update it
  learns at almost the same rate as the single-GPU model, with 5× fewer updates per epoch.

## Setup

| component | spec |
|---|---|
| CPU | AMD Threadripper PRO 7965WX, 256 GB RAM |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB, CC 12.0, 188 SMs) |
| storage | ZFS, striped 2×4 TB NVMe Gen5 |
| OS | Ubuntu 26.04, NVIDIA driver 615.71 (open), CUDA 13.4 |
| stack | Julia 1.12, Lux.jl, CUDA.jl, Zygote, CairoMakie; Nsight Systems / Compute for profiling |
