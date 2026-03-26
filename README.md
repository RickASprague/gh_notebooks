Various Jupyter notebooks — Julia/CUDA experiments on a Threadripper Pro + RTX 6000 Blackwell workstation.

* **registration_se2** — Lie group registration in 2D.
* **autoencoder_io** — Can we make the GPU the bottleneck instead of data loading? Trains a convolutional autoencoder on full ImageNet-64 (1.28M images) using memory-mapped IO. Python pickle deserialization was too slow (~3 min/pass), so a custom `MMapReader` eliminates the upfront load entirely — data streams from NVMe on demand via the OS page cache. Measures reconstruction quality vs latent dimension size (8x8x128 vs 4x4x256).
