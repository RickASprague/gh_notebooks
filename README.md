Various Jupyter notebooks:

* **registration_se2** — Lie group registration in 2D.
* **autoencoder_io** — Can we make the GPU the bottleneck instead of data loading? Trains a convolutional autoencoder on full ImageNet-64 (1.28M images) using memory-mapped IO. Python pickle deserialization was too slow (~3 min/pass), so a custom `MMapReader` eliminates the upfront load entirely — data streams from NVMe on demand via the OS page cache. Measures reconstruction quality vs latent dimension size (8x8x128 vs 4x4x256).
* **system_identification** 
    — `coulomb/` identifies Coulomb's law $\dot{\mathbf v} = \alpha\mathbf x/r^3$ with SINDy vs a neural network, and shows where the network fails out of sample (Julia, markdown + scripts).
* **other** — standalone notes.
  * `android_imu.ipynb` — the Android device sensor coordinate frame and what an accelerometer's "specific force" reading actually reports.

