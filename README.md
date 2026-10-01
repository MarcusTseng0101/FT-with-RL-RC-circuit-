# Fourier analysis of RC / RL circuits driven by a square wave

V(t) for series RC (V_C) and RL (V_L) circuits driven by a 0/10 V square wave, solved as
general solution = particular (periodic steady state, via Fourier) + homogeneous (A·e^(−t/τ)),
computed three ways and checked against the piecewise analytic solution:

1. `m1_fourier_series.py` — Fourier series: split into harmonics, multiply by H(nω0) = 1/(1 + jnω0τ), sum.
2. `m2_linear_algebra.py` — DFT matrix: x_p = F⁻¹ · diag(H) · F · v (F diagonalizes the periodic derivative).
3. `m3_qft.py` — Quantum Fourier transform (Qiskit statevector): QFT†, block-encode diag(H) with one ancilla, QFT, postselect.

Parameters (`common.py`): R = 2 Ω, C = 1 F (τ = 2 s), L = 1 H (τ = 0.5 s), T = 20 s.

```
pip install numpy scipy matplotlib qiskit
python make_figures.py
```

Outputs `fig1_response`, `fig2_convergence`, `fig3_decomposition` (PDF + PNG).
