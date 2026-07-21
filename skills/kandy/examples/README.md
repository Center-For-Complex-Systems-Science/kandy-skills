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
| `pendulum_dictionary_completion_example.py` | Damped nonlinear pendulum | **Dictionary completion**: the spline learns `sin θ` with no trig dictionary; the library only *names* it. Polynomial-vs-trig library contrast |
| `van_der_pol_rbf_example.py` | Van der Pol oscillator | `RadialBasisLift` when the RHS is unknown; identity-lift failure baseline; validate by rollout, not formula |
| `sinx_recover_sine.py` | 1D `x_dot = x + sin(x)` | Making `auto_symbolic` name `sin` explicitly: heterogeneous base functions + restricted symbol library |
| `koopman_slow_manifold_example.py` | Brunton–Proctor–Kutz slow manifold | **`DMDLift`** — the data picks the lift. Recovers the exact Koopman spectrum {μ, 2μ, λ} to ~1e-14; three traps: `n_modes` drops the fastest mode, gluing trajectories corrupts EDMD, and the lift needs a trajectory while the regression needs a region |
| `mackey_glass_delay_example.py` | Mackey–Glass DDE | **`DelayEmbedding` from a scalar series**: recover the delay τ by lag scan; why a high R² in delay space is cheap (Takens); splines recover the Hill term with no dictionary |

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
| `network_reconstruction_example.py` | Kuramoto on a sparse graph | **Recovering the adjacency matrix**: one regression per node, edge amplitude = coupling strength. Assuming the wrong coupling function invents *false edges*; a phase-locked network is unidentifiable |

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
| `navier_stokes_2d_vorticity_example.py` | 2D decaying turbulence | Recovers `∂ω/∂t = −u·ω_x − v·ω_y + ν∇²ω` exactly, viscosity included; spectral solver, one sample per grid point, full PDE rollout |
| `turbulence_closure_example.py` | LES subgrid closure, 2D turbulence | **Fitting a residual, not an RHS**: recovers the Clark coefficient Δ²/12 to ~3% under a smooth filter; under a sharp filter ~85% of the subgrid term is *not* locally closable — a measured negative result |
| `vortex_shedding_example.py` | Cylinder wake at Re=100 | **Reduced-order model**: DNS → POD → ODE in modal coordinates. Shift mode vs shedding pair; the shift mode is *slaved*, so λ is unidentifiable; Stuart–Landau normal form on the slow manifold. `lamb>0` is what fixes the limit-cycle radius |

## `mathbio/` — mathematical biology

Population and physiological models. Most are polynomial with a bilinear
interaction term; see each script's docstring for which lift the structure
demands.

| Script | System | Highlights |
|---|---|---|
| `holling_type_ii_example.py` | Rosenzweig–MacArthur predator-prey | 21D physics library; Adam optimizer; *rational* functional response needs a hand-built lift |
| `sir_example.py` | SIR epidemic | The `S·I` cross-term lesson; why conservation laws (`S+I+R` const) make trajectory-only data non-identifiable |
| `lotka_volterra_competition_example.py` | Two-species competition | Put cross-terms in the lift, NOT powers — why `PolynomialLift` gives the wrong symbolic answer here |
| `fitzhugh_nagumo_example.py` | FitzHugh–Nagumo neuron | Identity lift (no cross-terms); edge-wise polynomial reconstruction when `auto_symbolic` structurally cannot snap the cubic |
| `hodgkin_huxley_example.py` | Hodgkin–Huxley neuron | Recovers **real biophysical constants** (gNa, gK, ENa, EK to ~0.01%); triple-product cross-term `m³h·V`; LBFGS *required* — Adam fails outright; the grid-plateau diagnostic for a missing lift term |
| `epileptor_example.py` | Epileptor (seizure dynamics) | **Switching nonlinearities as gated features** (`m*x1^3`, `p*x1*x2`); **per-equation target scaling** for a 6000× stiff system; validate by rollout — gated features are zero-inflated and degrade symbolic snapping |

## `geometry/` — maps between manifolds, engineered lifts

| Script | System | Highlights |
|---|---|---|
| `hopf_example.py` | Hopf fibration S³ → S² | Engineered lift for a map between manifolds |
| `trefoil_knot_example.py` | Trefoil knot via Hopf fibration | Knotted trajectory recovery |
| `trefoil_knot_hero.py` | Trefoil hero figure | Publication figure generation |
| `euler_top_example.py` | Free rigid body on SO(3) | The cross-term rule at its purest — the RHS is *nothing but* cross-products, so the minimal lift is exactly `[W2W3, W3W1, W1W2]`; conserves energy and angular momentum |
| `heisenberg_example.py` | Sub-Riemannian geodesics on H¹ | The minimal counterexample to separability: `dz/dt = x·v − y·u` is a bare antisymmetric bilinear form |
| `frenet_serret_example.py` | Frenet–Serret moving frame | 11D autonomous system carrying κ and τ as states; every RHS term is invariant × frame component |
| `unicycle_se2_example.py` | Kinematic unicycle on SE(2) | Mixed trig×linear cross-terms (`v·cos θ`) that **no stock lift** provides — neither `PolynomialLift` nor `FourierLift` |
| `torus_winding_example.py` | Flat 2-torus and winding flows | Map mode + winding flow; 5-feature trig lift with a known 3×5 mixing matrix |
| `mobius_riemann_sphere_example.py` | Möbius maps on the Riemann sphere | Rational lift `[u, xu, yu, r²u]` with `u = 1/|cz+d|²`; the linear identity is verified to machine precision before training |
