# System Identification

Learning the governing equations of a dynamical system from observed trajectories.

* **[coulomb](coulomb/Coulomb.md)**: identify $\frac{d}{dt}\mathbf v = \mathbf a(\mathbf x)$ for a test particle near a fixed point charge, from 20 simulated orbits. Sparse regression (SINDy) recovers $\alpha\,\mathbf x/r^3$ exactly. An MLP fits the same data equally well but is wrong by orders of magnitude outside the training range. The writeup also shows why a single orbit leaves SINDy rank-deficient: the conic's focus–directrix identity puts four null vectors in the library. Julia; run the scripts in `coulomb/src/`.
