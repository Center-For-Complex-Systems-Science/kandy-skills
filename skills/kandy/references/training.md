# Training

## KANDy.fit

```python
model.fit(
    X,                    # (N, n) state trajectory
    X_dot=None,           # (N, n) time derivatives; omit if passing dt
    dt=None,              # time step → central-difference derivative estimation
    opt="LBFGS",          # "LBFGS" (default) or "Adam"
    lr=1.0,               # learning rate (use ~1e-3 for Adam)
    batch=-1,             # mini-batch size (-1 = full batch)
    lamb=0.0,             # sparsity regularisation (L1 + entropy)
    rollout_weight=0.0,   # weight on trajectory rollout loss
    rollout_loss_fn=None, # separate loss for rollout; defaults to MSE
    fit_steps=None,       # override self.steps for this call
    val_frac=0.15,
    test_frac=0.15,
)
```

Optimizer choice:

- **LBFGS** (default, `lr=1.0`, full batch) — most ODE/PDE systems.
- **Adam** (`lr=1e-3` to `2e-3`, `batch=4096`) — large datasets, or discrete
  maps with many parameters (Holling Type II, Ikeda with rollout).

Constructor knobs: `KANDy(lift, grid=5, k=3, steps=500, seed=42, device=None,
base_fun=None)`. `grid` = spline grid points, `k` = spline order. Increase
`grid` (e.g. 10–20) for sharper nonlinearities; increase `steps` if train loss
is still falling.

## Derivatives

Prefer true derivatives (`X_dot`) when the simulator provides them. Otherwise
pass `dt` for central differences — but denoise first if the data is
experimental; central differences amplify noise.

## Is the fit capacity-limited or structurally limited?

When a fit stalls below R²≈1, refine `grid` (5 → 10 → 20 → 40) and watch:

- **R² improves** → a resolution problem. Keep more knots.
- **R² is flat** → the lift is missing a term. More capacity will not help; add
  the feature instead.

The plateau happens because a separable KAN computes `sum_j psi_j(theta_j)`,
so any genuine product of two states is unreachable no matter how fine the
splines. `mathbio/hodgkin_huxley_example.py` measures this: the gating equation
`dm/dt = alpha(V) - (alpha+beta)(V)*m` sits at R²≈0.990 across an 8x grid
refinement (spread ~0.001), and jumps to 1.000000 the moment the products are
supplied in the lift.

Note the trap: 0.990 looks like success. Always run the refinement check before
concluding a lift is adequate.

## Stiff systems: scale the targets per equation

The loss is an unweighted MSE over all output equations, so an equation whose
derivative is orders of magnitude smaller than the others contributes nothing
and is effectively not fitted. This bites exactly where it hurts: the slow
variable is usually the one carrying the interesting physics (a permittivity
variable, a slow gate, an adiabatic invariant).

Check first, then fix by fitting on unit-variance targets and rescaling the
recovered formulas afterwards:

```python
scale = X_dot.std(axis=0)
print(scale.max() / scale.min())          # > ~100 means you must scale

model.fit(X=states, X_dot=X_dot / scale, ...)

# predictions and formulas are in scaled units — undo it
physical = model.predict(states) * scale
formula_i = sympy.expand(formulas[i] * float(scale[i]))
```

`mathbio/epileptor_example.py` is the worked case: `tau0 = 2857` makes `dz/dt`
~6000x smaller than the fast equations, and without scaling its R² drops to
0.86 with coefficients off by 45–75%. Scaling also conditions the problem, so
LBFGS converges in far fewer steps.

## Discrete maps

Pass current state as `X` and next state as `X_dot`; KANDy learns the one-step
map directly:

```python
model.fit(X_current, X_next, opt="Adam", lr=2e-3, batch=4096)
```

For long-horizon discrete rollout, use the **increment trick** — define the
dynamics as the increment so Euler with dt=1 reproduces exact map iteration:

```python
def discrete_rhs(state):
    return map_fn(state) - state

fit_kan(model.model_, dataset, integrator="euler", dynamics_fn=discrete_rhs, ...)
```

## Periodic phases (Kuramoto-type)

Use `angle_mse` so phase differences wrap to (−π, π] before squaring:

```python
from kandy import angle_mse
model.fit(X, X_dot, rollout_weight=0.3, rollout_loss_fn=angle_mse)
```

Helpers: `wrap_pi_torch(x)` wraps angles; `order_param_torch(theta)` computes
the Kuramoto order parameter |⟨e^{iθ}⟩|.

## Multi-step rollout loss (advanced)

`fit_kan` accepts a trajectory dataset for differentiable multi-step loss —
this is what stabilises long-horizon prediction on chaotic/stiff systems:

```python
from kandy import fit_kan, make_windows

train_windows = make_windows(train_traj, window=16)   # (Nw, 16, state_dim)

dataset = {
    "train_input": Theta_train,  "train_label": Y_train,
    "test_input":  Theta_test,   "test_label":  Y_test,
    "train_traj":  train_windows, "train_t": t_window,
    "test_traj":   test_windows,  "test_t":  t_window,
}

fit_kan(
    model.model_, dataset,
    opt="LBFGS", steps=100,
    rollout_weight=0.6, rollout_horizon=15,
    dynamics_fn=my_dynamics_fn,   # state → derivative (must apply lift internally)
    integrator="rk4",             # "rk4" or "euler"
    rollout_loss_fn=angle_mse,    # optional; defaults to MSE
    update_grid=True, stop_grid_update_step=50,
)
```

`dynamics_fn` receives the raw state and must apply the lift internally (torch
ops, differentiable). `rk4_step` / `euler_step` / `integrate_trajectory` are
the differentiable torch integrators used inside the rollout loss;
`rk4_integrate_numpy(f, x0, T, dt)` is for post-fit numpy rollouts and figures.

## Validation and diagnostics

- `model.rollout(x0, T, dt, integrator="rk4")` — does the autoregressive
  trajectory stay on the attractor?
- `model.predict(X)` — one-step derivative/next-state predictions.
- `plot_loss_curves`, `plot_attractor_overlay`, `plot_trajectory_error`,
  `plot_all_edges` (edge activations reveal which lifted features matter),
  `plot_kan_architecture`. Call `use_pub_style()` for publication figures.
- Rollout diverges → suspect a missing cross-term in the lift before touching
  optimizer settings (see references/lifts.md).

## Stochastic data (SDEs)

For `dX = f(X)dt + g(X)dW` the regression target does not exist: the increment
rate `ΔX/Δt = f + g·ΔW/Δt` has standard deviation `~g/√Δt`, which **diverges**
as `Δt → 0`. The conditional mean is still `f`, so least squares is consistent —
but every habit from the deterministic examples inverts. Five rules, all
measured in `mathbio/stochastic_logistic_sde_example.py`:

- **Never use `fit(X, dt=...)`.** It fills `X_dot` with *central* differences,
  and on an SDE that is catastrophic rather than merely noisy: the centred
  target contains `ΔW_{n−1}`, which the regressor `x_n` also depends on. The
  endogeneity biases the slope by `≈ σ²E[x]/(2Var x)`, which in that example is
  `+0.99` against a true slope of `−1.0` — the fit reports **zero** interaction,
  at every `Δt`, with no warning. Build forward differences and pass `X_dot`.
- **Sampling faster makes it worse**, not better. SNR of the target went 0.035 →
  0.366 as `Δt` went 0.002 → 0.2.
- **What buys drift accuracy is total observed time `T`, not sample count.**
  Error is `~1/√T` and nearly flat across a 25× change in `Δt`. Independent
  replicate paths count toward `T`, so many short runs are fine.
- **Diffusion has the opposite preference**: `σ²` comes from the quadratic
  variation `E[(ΔX)²]/Δt`, consistent as `Δt → 0` at fixed `T`. Fit it with a
  second model; a flat edge means state-independent diffusion.
- **R² is near zero for the correct model** — 0.012 against the target there,
  because the drift is ~1% of its variance and the rest is Brownian by
  construction. Score against the recovered coefficients, and set
  `r2_threshold=0.0` in `get_formula` or the default 0.80 rejects every edge.

Validate by **distribution, not trajectory**: two realisations from the same
initial condition diverge immediately. Simulate the fitted SDE and compare
invariant laws against a calibrated null, as `_utils.attractor_tv` does for
chaotic orbits.
