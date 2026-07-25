Various Jupyter notebooks:

* **registration_se2** — Lie group registration in 2D.
* **autoencoder_io** — Can we make the GPU the bottleneck instead of data loading? Trains a convolutional autoencoder on full ImageNet-64 (1.28M images) from memory-mapped shards, so there is no per-pass deserialization and no upfront load. Profiles the training step with Nsight Systems and Nsight Compute, measures reconstruction quality, and simulates 5-way data-parallel training on one GPU (Julia, markdown + scripts).
* **system_identification** 
    — `coulomb/` identifies Coulomb's law $\dot{\mathbf v} = \alpha\mathbf x/r^3$ with SINDy vs a neural network, and shows where the network fails out of sample (Julia, markdown + scripts).
* **other** — standalone notes.
  * `android_imu.ipynb` — the Android device sensor coordinate frame and what an accelerometer's "specific force" reading actually reports.

