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
