# KANDy Examples

All scripts are self-contained — they simulate their own training data
(via `scipy.integrate.solve_ivp`, map iteration, or KANDy's finite-volume
solvers), fit a KANDy model, validate by rollout, and extract symbolic
formulas.

Run any script directly:

```bash
python odes/lorenz_example.py
```

Figures are written to the working directory (scripts use the `Agg` backend;
no display needed).

Examples are grouped by **fit recipe** — pick the category matching your data
type, then the script closest to your system.

## `odes/` — continuous ODEs

Fit `(X, X_dot)` pairs (or a trajectory + `dt` for central differences).

| Script | System | Highlights |
|---|---|---|
| `lorenz_example.py` | Lorenz-63 chaotic ODE | Polynomial lift with xy, xz cross-terms; attractor overlay |
| `sinx_recover_sine.py` | 1D `x_dot = x + sin(x)` | Making `auto_symbolic` name `sin` explicitly: heterogeneous base functions + restricted symbol library |

## `maps/` — discrete maps

Pass current state as `X` and next state as `X_dot`; for long-horizon rollout
use the increment trick (`f(s) = map(s) − s`, Euler with dt=1).

| Script | System | Highlights |
|---|---|---|
| `henon_example.py` | Hénon map | Minimal [x, y, x²] lift |
| `ikeda_example.py` | Ikeda optical cavity | CustomLift with trig physics; rollout loss; physics-informed symbolic costs |

## `oscillators/` — coupled phase oscillators

Periodic phases need `angle_mse` rollout loss so errors wrap to (−π, π].

| Script | System | Highlights |
|---|---|---|
| `kuramoto_example.py` | Kuramoto oscillators | Periodic phases; `angle_mse` rollout loss |
| `adaptive_kuramoto_example.py` | Adaptive Kuramoto–Sakaguchi | 20D coupled lift; adaptive coupling weights |

## `pdes/` — partial differential equations

Field u ∈ ℝᴺ on a periodic grid; lifts built from local features
(u, derivatives, flux terms) or Fourier modes. Training data from KANDy's
finite-volume solvers.

| Script | System | Highlights |
|---|---|---|
| `burgers_example.py` | Inviscid Burgers | FV training data; flux-form lift [u, u_x, ∂(u²/2)/∂x] |
| `burgers_fourier_example.py` | Burgers, random Fourier ICs | Generalisation across initial conditions |
| `kuramoto_sivashinsky_example.py` | Kuramoto–Sivashinsky | 12D local features; chaotic PDE |

## `fluids/` — fluid dynamics

| Script | System | Highlights |
|---|---|---|
| `navier_stokes_example.py` | 3D Navier–Stokes (ABC Beltrami flow) | 12D vorticity/velocity lift; pseudo-spectral data |

## `mathbio/` — mathematical biology

| Script | System | Highlights |
|---|---|---|
| `holling_type_ii_example.py` | Rosenzweig–MacArthur predator-prey | 21D physics library; Adam optimizer |

## `geometry/` — maps between manifolds, engineered lifts

| Script | System | Highlights |
|---|---|---|
| `hopf_example.py` | Hopf fibration S³ → S² | Engineered lift for a map between manifolds |
| `trefoil_knot_example.py` | Trefoil knot via Hopf fibration | Knotted trajectory recovery |
| `trefoil_knot_hero.py` | Trefoil hero figure | Publication figure generation |

## `baselines/` — comparison methods

| Script | System | Highlights |
|---|---|---|
| `sindy_baselines.py` | SINDy comparison | Baseline for benchmark tables |
| `pdefind_baseline.py` | PDE-FIND comparison | Baseline on Burgers with Fourier ICs |
