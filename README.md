Various Jupyter notebooks:

* **registration_se2** — Lie group registration in 2D.
* **autoencoder_io** — Can we make the GPU the bottleneck instead of data loading? Trains a convolutional autoencoder on full ImageNet-64 (1.28M images) using memory-mapped IO. Python pickle deserialization was too slow (~3 min/pass), so a custom `MMapReader` eliminates the upfront load entirely — data streams from NVMe on demand via the OS page cache. Measures reconstruction quality vs latent dimension size (8x8x128 vs 4x4x256).
* **other** — standalone notes.
  * `rotating_frame.ipynb` — rotating-reference-frame kinematics from first principles: the transport theorem and the four-term acceleration (relative / Coriolis / centripetal / Euler — and why the same terms read as real inward accelerations in the inertial frame but fictitious centrifugal/Coriolis forces in the body frame), then nested transforms C→B→A in three cases: a static mount (a fixed extrinsic R folds into the composition, no new terms), a dynamic inner spin (cross-Coriolis / gyroscopic couplings plus centripetal & Euler terms on the mounting offset), and coincident origins (rotations multiply, angular velocities add, the stack collapses to one rotating frame). Worked carousel + turning-car examples close it out — a steady turn *is* a carousel, differing only in where you put the origin.
  * `android_imu.ipynb` — the Android device sensor coordinate frame and what an accelerometer's "specific force" reading actually reports.

