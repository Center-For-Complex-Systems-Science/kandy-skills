---
name: kandy
description: >
  Data-driven discovery of dynamical systems with KANDy (Kolmogorov-Arnold
  Networks for Dynamics). Use when fitting governing equations to trajectory
  data, doing system identification / equation discovery (SINDy-style), Koopman
  lifting, symbolic regression of ODEs, PDEs, or discrete maps, or generating
  finite-volume PDE training data. Triggers: "KANDy", "KAN", "discover
  equations", "system identification", "Koopman lift", "symbolic extraction",
  "learn dynamics from data".
---

# KANDy — Kolmogorov-Arnold Networks for Dynamics

KANDy identifies governing equations from trajectory data by learning

```
x_dot = A · Ψ(φ(x))
```

where **φ** is a user-chosen Koopman lift (encodes all cross-terms), **Ψ** is a
separable single-layer KAN spline map, and **A** is a linear mixing matrix.
It replaces SINDy's sparse regression with a KAN and returns closed-form SymPy
formulas.

Install: `pip install kandy` (Python 3.11–3.13, PyTorch ≥ 2.0, PyKAN ≥ 0.2.0).

## The one rule that matters

**The KAN is separable — each spline sees exactly one lifted coordinate.
Cross-interaction terms (x·y, x·z, sin(θⱼ−θᵢ), …) can NEVER be learned from raw
inputs; they must be encoded explicitly in the lift φ.** A missing cross-term
makes the model structurally wrong, not just inaccurate. This is a theorem
(bilinear obstruction), not a tuning issue. When results are bad, check the
lift first.

## Standard workflow

```python
from kandy import KANDy, PolynomialLift

lift  = PolynomialLift(degree=2)                 # Lorenz: R^3 → R^9, has xy, xz
model = KANDy(lift=lift, grid=5, k=3, steps=500)

model.fit(X, X_dot)          # X: (N, n) states, X_dot: (N, n) derivatives
# or, with only a trajectory:  model.fit(X, dt=0.01)   ← central-difference

traj     = model.rollout(x0, T=5000, dt=0.005)   # autoregressive validation
formulas = model.get_formula(var_names=["x", "y", "z"])   # SymPy expressions
r2       = model.score_formula(formulas, X_test, Xdot_test)
A        = model.get_A()                          # (n × m) mixing matrix
```

Validate in this order: (1) rollout reproduces the attractor/trajectory,
(2) symbolic formulas score R² ≈ 1 on held-out data, (3) formulas match known
structure if the system is a benchmark.

## Choosing the lift (most critical decision)

| Situation | Lift |
|---|---|
| Polynomial RHS (Lorenz, Hénon, predator-prey) | `PolynomialLift(degree=2 or 3)` |
| Periodic PDE field u ∈ ℝᴺ (Burgers, KS) | `FourierLift(n_modes)` or local features via `CustomLift` |
| Unknown structure, smooth | `RadialBasisLift(n_centers=50, center_method="kmeans")` |
| Data-driven Koopman eigenfunctions | `DMDLift(n_modes, dictionary=PolynomialLift(2))` |
| Scalar time series | `DelayEmbedding(delays)` |
| Known physics terms (trig products, gates, coupling) | `CustomLift(fn, output_dim)` |
| Learn the lift itself (experimental) | `KANELift(latent_dim)` |

Details and code for every lift: `references/lifts.md`.

## Training choices

- **LBFGS** (default, `lr=1.0`) for most systems. **Adam** (`lr≈1e-3–2e-3`,
  `batch=4096`) for large datasets or discrete maps with many parameters.
- **Discrete maps**: pass current state as `X` and next state as `X_dot` —
  KANDy learns the one-step map. For long-horizon rollout use the increment
  trick (`f(s) = map(s) − s`, Euler with dt=1).
- **Periodic phases** (Kuramoto): `model.fit(..., rollout_weight=0.3,
  rollout_loss_fn=angle_mse)` so phase errors wrap to (−π, π].
- **Rollout loss** for multi-step accuracy: `fit_kan(..., rollout_weight=0.6,
  rollout_horizon=15, dynamics_fn=..., integrator="rk4")`.

Full options: `references/training.md`.

## Symbolic extraction

`model.get_formula()` for the basic path. For physics-informed extraction
(cheap costs on known-physics edges), custom symbol libraries, R² scoring, and
LaTeX export, read `references/symbolic.md`.

## PDE training data

KANDy ships a finite-volume solver suite (MUSCL + Rusanov/Roe/HLLC fluxes,
SSP-RK2/3) for 1D periodic conservation laws: `solve_burgers`,
`solve_viscous_burgers`, `solve_scalar`, `cfl_dt`. See `references/numerics.md`.

## Examples

Complete, runnable scripts in `examples/`, grouped by fit recipe — pick the
category matching the data type, then the script closest to the target system:

| Category | System | Script | Notable technique |
|---|---|---|---|
| `odes/` | Lorenz-63 (chaotic ODE) | `lorenz_example.py` | Polynomial lift with cross-terms |
| `odes/` | x + sin(x) recovery | `sinx_recover_sine.py` | Restricted symbol library for clean `auto_symbolic` |
| `maps/` | Hénon map | `henon_example.py` | Discrete map, [x, y, x²] lift |
| `maps/` | Ikeda map | `ikeda_example.py` | Custom trig physics lift + rollout loss |
| `oscillators/` | Kuramoto | `kuramoto_example.py` | Periodic phases, `angle_mse` |
| `oscillators/` | Adaptive Kuramoto | `adaptive_kuramoto_example.py` | Coupled 20D lift |
| `pdes/` | Inviscid Burgers | `burgers_example.py`, `burgers_fourier_example.py` | FV data, flux-form lift |
| `pdes/` | Kuramoto–Sivashinsky | `kuramoto_sivashinsky_example.py` | 12D local PDE features |
| `fluids/` | Navier–Stokes (ABC flow) | `navier_stokes_example.py` | 3D vorticity/velocity lift |
| `mathbio/` | Holling Type II | `holling_type_ii_example.py` | 21D physics library, Adam |
| `geometry/` | Hopf fibration | `hopf_example.py` | Map S³ → S², engineered lift |
| `geometry/` | Trefoil knot | `trefoil_knot_example.py`, `trefoil_knot_hero.py` | Hopf fibers, figures |

## Full API

Complete signatures for every public function: `references/api.md`.
