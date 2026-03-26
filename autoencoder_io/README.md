# ImageNet Autoencoder: Maximizing IO Throughput

Can we make the GPU the bottleneck instead of data loading? This project trains a convolutional autoencoder on the full ImageNet-64 dataset (1.28M images, 10 epochs) while minimizing IO overhead using Julia's memory-mapped files.

## The Problem

ImageNet-64 ships as Python pickle files containing NumPy arrays. Deserializing these from Julia is brutally slow — unpickling + converting 1.28M images takes ~3 minutes per pass. With 10 epochs, that's 30 minutes of pure IO before any GPU work happens.

## The Solution

A one-time preprocessing step converts the Python pickles to flat binary files (just shape headers + raw Float32 payloads). At training time, Julia's `Mmap` maps these files directly into virtual memory — no deserialization, no parsing, no upfront load step. The OS page cache streams data from NVMe on demand. **The ~3 min/pass deserialization cost is completely eliminated.**

## Hardware

| Component | Spec |
|-----------|------|
| CPU | AMD Threadripper Pro 7965WX |
| RAM | 256 GB |
| GPU | NVIDIA RTX 6000 Blackwell Pro (Workstation Edition) |
| Storage | ZFS striped 2x4TB NVMe Gen5 |
| OS | Ubuntu 24 |

## Stack

- **Julia** / **Lux.jl** (deep learning framework)
- **CUDA.jl** / **LuxCUDA** (GPU acceleration)
- **TensorBoard** (loss monitoring)
- **CairoMakie** (visualization)
- Custom `MMapReader` for zero-copy data loading (see `lib/utils.jl`)

## Notebooks

| Notebook | Description |
|----------|-------------|
| `Image Preprocess.ipynb` | One-time conversion: ImageNet-64 Python pickles → memory-mapped Julia binary files |
| `Auto Encoder.ipynb` | Trains the autoencoder, profiles IO throughput, compares reconstruction quality at different bottleneck sizes |

## Key Files

- `lib/utils.jl` — Custom binary serialization and `MMapReader` (mmap-backed zero-copy deserialization)
- `lib/imagenet.jl` — Python pickle loading via PythonCall (used only in preprocessing)

## Results

- **Data loading:** No upfront load — mmap eliminates the ~3 min/pass pickle deserialization entirely. Data streams from disk on demand.
- **Bottleneck comparison:** 8x8x128 latent (good reconstruction) vs 4x4x256 latent (~1/3 compression, visible degradation)
- **GPU utilization:** ~50%. Mmap solved the IO problem; remaining gap needs Nsight profiling to diagnose.
