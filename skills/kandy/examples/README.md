# KANDy Examples

All scripts are self-contained — they simulate their own training data
(via `scipy.integrate.solve_ivp`, map iteration, or KANDy's finite-volume
solvers), fit a KANDy model, validate by rollout, and extract symbolic
formulas — **except** the three iEEG scripts (`ieeg_example.py`,
`ieeg_relu_gated.py`, `ieeg_kuramoto_virtual_surgery.py`), which require a
local `data/E3Data.mat` intracranial-EEG recording that is not distributed
with this repository. Use them as structural templates for fitting KANDy to
real experimental time series.

Run any script directly:

```bash
python lorenz_example.py
```

Figures are written to the working directory (scripts use the `Agg` backend;
no display needed).

| Script | System | Highlights |
|---|---|---|
| `lorenz_example.py` | Lorenz-63 chaotic ODE | Polynomial lift with xy, xz cross-terms; attractor overlay |
| `henon_example.py` | Hénon map | Discrete map via (X, X_next); minimal [x, y, x²] lift |
| `ikeda_example.py` | Ikeda optical cavity | CustomLift with trig physics; rollout loss; physics-informed symbolic costs |
| `holling_type_ii_example.py` | Rosenzweig–MacArthur predator-prey | 21D physics library; Adam optimizer |
| `kuramoto_example.py` | Kuramoto oscillators | Periodic phases; `angle_mse` rollout loss |
| `adaptive_kuramoto_example.py` | Adaptive Kuramoto–Sakaguchi | 20D coupled lift; adaptive coupling weights |
| `hopf_example.py` | Hopf fibration S³ → S² | Engineered lift for a map between manifolds |
| `trefoil_knot_example.py` | Trefoil knot via Hopf fibration | Knotted trajectory recovery |
| `trefoil_knot_hero.py` | Trefoil hero figure | Publication figure generation |
| `burgers_example.py` | Inviscid Burgers PDE | FV training data; flux-form lift [u, u_x, ∂(u²/2)/∂x] |
| `burgers_fourier_example.py` | Burgers, random Fourier ICs | Generalisation across initial conditions |
| `kuramoto_sivashinsky_example.py` | Kuramoto–Sivashinsky PDE | 12D local features; chaotic PDE |
| `navier_stokes_example.py` | 3D Navier–Stokes (ABC flow) | Vorticity/velocity lift |
| `ieeg_example.py` | iEEG mode dynamics | Duffing-ReLU oscillator on real data (needs E3Data.mat) |
| `ieeg_relu_gated.py` | iEEG, ReLU-gated Duffing | Gate functions in the lift; time forcing (needs E3Data.mat) |
| `ieeg_kuramoto_virtual_surgery.py` | iEEG virtual surgery | SOZ/NSOZ Kuramoto-ReLU communities (needs E3Data.mat) |
| `sindy_baselines.py` | SINDy comparison | Baseline for benchmark tables |
| `pdefind_baseline.py` | PDE-FIND comparison | Baseline on Burgers with Fourier ICs |
