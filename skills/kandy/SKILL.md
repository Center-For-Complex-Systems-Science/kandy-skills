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

The corollary is just as important and easy to get backwards:

**Put products of DIFFERENT variables in the lift; leave powers of a SINGLE
variable to the edges.** A spline edge already represents any univariate
function, so `x²`, `x³`, `sin x` need not be in φ — the edge learns them
(`odes/pendulum_dictionary_completion_example.py` recovers `sin θ` with no
trig dictionary at all). Adding them anyway is actively harmful for symbolic
extraction: if φ contains both `x` and `x²`, the two edges are functionally
dependent, the decomposition becomes non-unique, and the recovered
coefficients are meaningless even at R² = 1
(`mathbio/lotka_volterra_competition_example.py`). **Lift coordinates must be
functionally independent of one another.**

So a missing cross-term breaks *learning* and cannot be repaired later; a
missing dictionary entry only breaks *naming* and costs you nothing but a
readable formula.

## Standard workflow

```python
from kandy import KANDy, PolynomialLift

lift  = PolynomialLift(degree=2)                 # Lorenz: R^3 → R^9, has xy, xz
model = KANDy(lift=lift, grid=5, k=3, steps=500)

model.fit(X, X_dot)          # X: (N, n) states, X_dot: (N, n) derivatives
# or, with only a trajectory:  model.fit(X, dt=0.01)   ← central-difference

traj     = model.rollout(x0, T=5000, dt=0.005)   # autoregressive validation
# ↓ get_formula MUTATES the model — do rollout/predict/plots BEFORE this line
formulas = model.get_formula(var_names=FEATURE_NAMES)     # SymPy expressions
r2       = model.score_formula(formulas, X_test, Xdot_test, var_names=FEATURE_NAMES)
# Do NOT use model.get_A() for coefficients — it returns near-init spline
# scales that correlate with nothing. Least-squares the network output onto
# the lifted features instead (see references/api.md):
A, *_    = np.linalg.lstsq(model.lift(X), model.predict(X), rcond=None); A = A.T
```

Two things that bite immediately:

- **`get_formula()` rewrites the fitted model in place** — `auto_symbolic`
  replaces every spline edge with its snapped surrogate. Anything afterwards
  (`rollout`, `predict`, edge plots) sees the surrogate, and if snapping went
  badly the model is now wrong (R² 1.00 → 0.50 is easy). Validate first, or
  `copy.deepcopy` the model per extraction attempt.
- **`var_names` names the LIFTED features, not the state variables** — pass
  the same list to `score_formula`, or it silently returns `NaN`.

Validate in this order: (1) rollout reproduces the attractor/trajectory,
(2) symbolic formulas score R² ≈ 1 on held-out data, (3) formulas match known
structure if the system is a benchmark.

Sampling matters as much as the lift: if the system has a **conserved
quantity** (total population, energy), trajectory data lies on a lower-dimensional
manifold where the lifted features are linearly dependent and the coefficients
are non-identifiable. Sample states independently over a box instead — check
`np.linalg.cond` of the lifted design matrix (`mathbio/sir_example.py`). Note
`cond` catches only *linear* dependence: a constraint manifold can leave it
looking healthy while still making coefficients non-unique
(`geometry/spherical_pendulum_example.py`).

## Choosing the lift (most critical decision)

| Situation | Lift |
|---|---|
| Polynomial RHS (Lorenz, Hénon, predator-prey) | `CustomLift` with states + cross-products only — see the corollary above. `PolynomialLift(degree=2)` is fine for fit quality but adds `x²`-type coordinates that corrupt symbolic extraction |
| No cross-terms at all (FitzHugh–Nagumo, pendulum) | Identity `CustomLift(fn=lambda X: X, ...)` — the edges do the work |
| Periodic PDE field u ∈ ℝᴺ (Burgers, KS) | `FourierLift(n_modes)` or local features via `CustomLift` |
| Unknown structure, smooth | `RadialBasisLift(n_centers=50, center_method="kmeans")` |
| Data-driven Koopman eigenfunctions | `DMDLift(n_modes, dictionary=PolynomialLift(2))` |
| Scalar time series | `DelayEmbedding(delays)` |
| Known physics terms (trig products, gates, coupling) | `CustomLift(fn, output_dim)` |

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
- **Stiff systems**: the loss is an unweighted MSE across equations, so a slow
  variable whose derivative is orders of magnitude smaller is simply not
  fitted. Fit on `X_dot / X_dot.std(0)` and rescale the formulas afterwards.

Full options: `references/training.md`.

## Symbolic extraction

`model.get_formula()` for the basic path, but its defaults are tuned for
sparsity and often return a formula that is far too short:

- **`lib=`** must match the shape of each EDGE, not the term structure you
  expect. RHS linear in the lifted features → `lib=["x", "0"]`. Quadratic
  edges → add `"x^2"`. A sine on an edge → add `"sin"`.
- **`weight_simple=0.8`** (the default) is a strong simplicity pressure that
  snaps genuine curved edges to `0`. Use `weight_simple=0.0` whenever real
  nonlinear edges are being dropped.
- Some edges **cannot** be snapped at all: PyKAN fits one primitive under an
  affine composition `c·f(a·x+b)+d`, which cannot represent e.g. `v − v³/3`.
  Fit the edge as a polynomial and sum the edges instead — a single-layer KAN
  output is exactly `Σᵢ edge_ij(θᵢ)`
  (`mathbio/fitzhugh_nagumo_example.py`).

For physics-informed extraction (cheap costs on known-physics edges), custom
symbol libraries, R² scoring, and LaTeX export, read `references/symbolic.md`.

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
| `odes/` | Damped pendulum | `pendulum_dictionary_completion_example.py` | Dictionary completion — spline learns `sin θ` with no trig dictionary |
| `odes/` | Van der Pol | `van_der_pol_rbf_example.py` | `RadialBasisLift` for unknown structure; validate by rollout |
| `odes/` | x + sin(x) recovery | `sinx_recover_sine.py` | Restricted symbol library for clean `auto_symbolic` |
| `odes/` | Mackey–Glass (delay DE) | `mackey_glass_delay_example.py` | `DelayEmbedding` on a scalar series; recovering the delay; splines learn the Hill term |
| `odes/` | Koopman slow manifold | `koopman_slow_manifold_example.py` | `DMDLift`: data-derived Koopman coordinates; exact spectrum recovery; EDMD traps |
| `maps/` | Hénon map | `henon_example.py` | Discrete map, [x, y, x²] lift |
| `maps/` | Ikeda map | `ikeda_example.py` | Custom trig physics lift + rollout loss |
| `oscillators/` | Kuramoto | `kuramoto_example.py` | Periodic phases, `angle_mse` |
| `oscillators/` | Adaptive Kuramoto | `adaptive_kuramoto_example.py` | Coupled 20D lift |
| `oscillators/` | Network reconstruction | `network_reconstruction_example.py` | Recover the adjacency matrix; per-node regression; unknown coupling function |
| `pdes/` | Inviscid Burgers | `burgers_example.py`, `burgers_fourier_example.py` | FV data, flux-form lift |
| `pdes/` | Kuramoto–Sivashinsky | `kuramoto_sivashinsky_example.py` | 12D local PDE features |
| `fluids/` | Navier–Stokes (ABC flow) | `navier_stokes_example.py` | 3D vorticity/velocity lift |
| `fluids/` | 2D decaying turbulence | `navier_stokes_2d_vorticity_example.py` | Exact vorticity equation + viscosity; spectral data, PDE rollout |
| `fluids/` | Vortex shedding (cylinder wake) | `vortex_shedding_example.py` | Reduced-order model: DNS → POD → modal ODE; slaved shift mode, Stuart–Landau |
| `fluids/` | LES subgrid closure | `turbulence_closure_example.py` | Fitting a residual; recovers Δ²/12; measures what is *not* closable |
| `mathbio/` | Holling Type II | `holling_type_ii_example.py` | 21D physics library, Adam |
| `mathbio/` | SIR epidemic | `sir_example.py` | `S·I` cross-term; conservation laws break identifiability |
| `mathbio/` | Lotka–Volterra competition | `lotka_volterra_competition_example.py` | Cross-terms in the lift, powers on the edges |
| `mathbio/` | FitzHugh–Nagumo | `fitzhugh_nagumo_example.py` | Identity lift; edge-wise polynomial reconstruction |
| `mathbio/` | Epileptor (seizures) | `epileptor_example.py` | Gated features for switching nonlinearities; per-equation target scaling for stiff systems |
| `mathbio/` | Hodgkin–Huxley | `hodgkin_huxley_example.py` | Recovers biophysical constants; `m³h·V` triple product; grid-plateau diagnostic |
| `geometry/` | Hopf fibration | `hopf_example.py` | Map S³ → S², engineered lift |
| `geometry/` | Trefoil knot | `trefoil_knot_example.py`, `trefoil_knot_hero.py` | Hopf fibers, figures |
| `geometry/` | Euler top (rigid body) | `euler_top_example.py` | Pure cross-product RHS — the minimal lift, stated exactly |
| `geometry/` | Heisenberg group H¹ | `heisenberg_example.py` | Minimal counterexample to separability: `x·v − y·u` |
| `geometry/` | Frenet–Serret frame | `frenet_serret_example.py` | Bilinear moving frame; κ, τ carried as states |
| `geometry/` | Unicycle / Dubins car | `unicycle_se2_example.py` | `v·cos θ`: mixed trig×linear, no stock lift works |
| `geometry/` | 2-torus winding | `torus_winding_example.py` | Trig embedding lift; map mode + flow |
| `geometry/` | Möbius transformations | `mobius_riemann_sphere_example.py` | Rational lift `1/|cz+d|²`; pole exclusion, rank diagnostics |
| `geometry/` | Spherical pendulum | `spherical_pendulum_example.py` | Constraint-force lift; `cond()` misses separable degeneracy |

## Full API

Complete signatures for every public function: `references/api.md`.
