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

Verification convention:

- Print one `[CHECK] PASS/FAIL` line with concrete thresholds.
- Write `results/<Example>/metrics.json` with the same metrics and thresholds.
- Keep plotting code in named functions near the bottom of the script.
- Share only boring mechanics through `examples/_utils.py`; keep equations,
  lifts, simulators, pruning rules, and symbolic readouts inside each example.

Examples are grouped by **fit recipe** — pick the category matching your data
type, then the script closest to your system.

## `odes/` — continuous ODEs

Fit `(X, X_dot)` pairs (or a trajectory + `dt` for central differences).

| Script | System | Highlights |
|---|---|---|
| `lorenz_example.py` | Lorenz-63 chaotic ODE | Polynomial lift with xy, xz cross-terms; attractor overlay |
| `quadratic_obstruction_example.py` | `ż = z²` — complex squaring map | **Why hidden width 2 is not enough.** Two inputs, two outputs, and the required space `V = span{x²−y², 2xy}` contains no perfect square — so no span of two squares can hold it. Proved for quadratic activations (R² ≤ 0.486, deterministic), measured for real splines (~0.87–0.91 across seeds and runs), both jumping to ~1.0 at width 3. Includes the `(xy, x)` control that **succeeds** at width 2, so the failure can't be blamed on capacity. KANDy on `φ=[x²−y², xy]` recovers the equation at depth 1. PyKAN `model.plot()` for all three |
| `hopf_cross_term_example.py` | Hopf (Stuart–Landau) normal form | **The cross-term obstruction, quantified in 2D.** Two inputs, two outputs, two *different* mixed monomials — the Lorenz obstruction small enough to plot. Without the lift there is no equation to read and the fit is wrong too (best possible additive model capped at R²=0.962; limit cycle collapses to r=0.55). With `φ=[x,y,xy²,x²y]` the RHS comes back exact to 2e-5 by summing the edges. Also: why "just add a layer" returns no equation, and why training on the attractor hides the obstruction entirely |
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
| `kuramoto_example.py` | Kuramoto oscillators | Independent pair-sine lift; effective linear readout |
| `adaptive_kuramoto_example.py` | Adaptive Kuramoto–Sakaguchi | Kappa-trig product lift; separate phase/kappa models |
| `network_reconstruction_example.py` | Kuramoto on a sparse graph | **Recovering the adjacency matrix**: one regression per node, edge amplitude = coupling strength, post-training pruning. Assuming the wrong coupling function invents *false edges*; a phase-locked network is unidentifiable |

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
| `vortex_shedding_example.py` | Cylinder wake at Re=100 | **Reduced-order model**: DNS → POD → ODE in modal coordinates. Shift mode vs shedding pair; the shift mode is *slaved*, so λ is unidentifiable; Stuart–Landau normal form on the slow manifold. Record length, not `lamb`, is what fixes the limit-cycle radius — the record needs ~10+ *saturated* periods or the cubic damping is unidentifiable (~45 min) |

## `plasma/` — magnetohydrodynamics

Ideal and resistive MHD, where every equation couples a velocity field to a
magnetic field. The recurring theme is that the *state variables* you choose
decide which physics the fit can get wrong: evolve the vector potential and
`div(B) = 0` holds identically, fit a flux function instead of a right-hand
side and the finite-volume discretisation cancels out of the identification.

| Script | System | Highlights |
|---|---|---|
| `orszag_tang_example.py` | Orszag–Tang vortex, incompressible 2D MHD | Evolving `A` rather than `B` makes **`div(B) = 0` structural, not learned**; three *decoy* features that correlate with `dw/dt` at r ≈ 0.4 must go to zero; ν and η recovered separately (Pm = 0.4). Also the cleanest instance of the independence rule: `j` cannot be in the lift because `j = -lap(A)` exactly, and including it splits η across two columns at R² = 1 |
| `brio_wu_shock_tube_example.py` | Brio–Wu MHD Riemann problem (compound wave) | Discovering the **flux function**, not the RHS: which of twelve candidate fluxes enters which conservation law, magnetic tension included. **A single Riemann problem is self-similar and therefore unidentifiable** (23× the coefficient error, decoys at 1.0) — fit an ensemble of tubes with varying `Bx`. Shock cells carry ~40× the residual and bias an MSE fit; validation re-runs the held-out tube with the discovered flux |
| `mhd_2d_turbulence_example.py` | Decaying 2D MHD turbulence, subgrid EMF | Closure, not identification: the turbulent EMF is **not** a function of the resolved fields, so the deliverable is how much is locally closable (R² ≈ 0.96–0.98) and with what coefficient. Two rival closures share one lift — the gradient model wins, eddy resistivity alone reaches R² ≈ 0.11–0.20. Validated by a **scaling law**: the coefficient tracks Δ²/12 across a factor of 7 in Δ². Viscosity is set by a gated resolution check, since a subgrid term on an under-resolved grid measures the discretisation |

| `heat_flux_closure_example.py` | 1D kinetic heat-flux closure | **A lift coordinate that is NONLOCAL** — a Hilbert transform of the whole profile, which no local derivative library can express at any order; the local-only ablation is ~370× worse. Divide out the known physics (`q_fs = T^{3/2}`, the Fourier floor) and fit the dimensionless closure, so the two KAN edges *are* the flux limiter and the nonlocal term: `a`, `z0`, `b` recovered to four digits, then exported as two lookup tables that drop back into the solver with no torch in the loop. Three traps, each measured: saturation is rare (stratify in `z`), `z` and `h` are correlated (broad-spectrum initial conditions separate them), and the knee needs knots finer than `z0`. Also the clearest instance of the chronological-split trap — stratified rows must be shuffled before `fit` |

## `mathbio/` — mathematical biology

Population and physiological models. Most are polynomial with a bilinear
interaction term; see each script's docstring for which lift the structure
demands.

| Script | System | Highlights |
|---|---|---|
| `holling_type_ii_example.py` | Rosenzweig–MacArthur predator-prey | 21D physics library; Adam optimizer; *rational* functional response needs a hand-built lift |
| `holling_mult_node_example.py` | Rosenzweig–MacArthur, nothing supplied | A separable lift is capped at R²=0.9376 on a product of two variables; one multiplication node breaks it, and alternating least squares reads the product term off the trained model — recovering *a* and *h*, which the oracle-lift version above is told |
| `sir_example.py` | SIR epidemic | The `S·I` cross-term lesson; why conservation laws (`S+I+R` const) make trajectory-only data non-identifiable |
| `lotka_volterra_competition_example.py` | Two-species competition | Put cross-terms in the lift, NOT powers — why `PolynomialLift` gives the wrong symbolic answer here |
| `fitzhugh_nagumo_example.py` | FitzHugh–Nagumo neuron | Identity lift (no cross-terms); edge-wise polynomial reconstruction when `auto_symbolic` structurally cannot snap the cubic |
| `hodgkin_huxley_example.py` | Hodgkin–Huxley neuron | Recovers **real biophysical constants** (gNa, gK, ENa, EK to ~0.01%); triple-product cross-term `m³h·V`; LBFGS *required* — Adam fails outright; the grid-plateau diagnostic for a missing lift term |
| `epileptor_example.py` | Epileptor (seizure dynamics) | **Switching nonlinearities as gated features** (`m*x1^3`, `p*x1*x2`); **per-equation target scaling** for a 6000× stiff system; validate by rollout — gated features are zero-inflated and degrade symbolic snapping |
| `fmri_dcm_example.py` | fMRI: DCM neural layer + Balloon–Windkessel → BOLD | The only example where **the state is not observed**. Balloon constants recovered with nothing supplied — `α` and `E0` come from the *shapes* of two edges, `τ` three independent ways. Then the negative result: regional HRF lag makes a fit on BOLD report the connection that does not exist as the strongest, and *deconvolving first does not fix it either*. Keeping the observation operator inside the loss does |

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
| `mobius_riemann_sphere_example.py` | Möbius maps on the Riemann sphere | The only *rational* lift — `u = 1/|cz+d|²`; pole exclusion to keep features bounded, and dropping `u` itself because `[u, xu, yu, r²u]` is exactly rank-deficient against the constant (`cond` 1.3e16 → 9.7) |
| `spherical_pendulum_example.py` | Spherical pendulum on S² | Constraint-force lift (triple products `qᵢ·|p|²`); why a healthy `cond()` can still hide a *separable* degeneracy, and how off-manifold sampling fixes it |
