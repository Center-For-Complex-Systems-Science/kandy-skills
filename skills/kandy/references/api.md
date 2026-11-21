# KANDy API Reference

Everything below is importable from the top-level `kandy` package.

## Model

```python
KANDy(lift, grid=5, k=3, steps=500, seed=42, device=None, base_fun=None)
```

| Method | Description |
|---|---|
| `fit(X, X_dot=None, *, dt=None, t=None, val_frac=0.15, test_frac=0.15, lamb=0.0, rollout_weight=0.0, rollout_horizon=None, rollout_loss_fn=None, dynamics_fn=None, dataset_extras=None, opt="LBFGS", lr=1.0, batch=-1, stop_grid_update_step=50, fit_steps=None, patience=10, verbose=True)` | Fit the model |
| `predict(X)` | Predict derivatives / next state |
| `rollout(x0, T, dt, integrator="rk4")` | Autoregressive trajectory integration. **`T` is a step count (int), not an end time** — output has shape `(T, n)` |
| `get_formula(var_names=None, round_places=3, simplify=False, lib=None, r2_threshold=0.80, weight_simple=0.8)` | Symbolic extraction → list of SymPy expressions. **Mutates the model in place** (see warning below) |
| `score_formula(formulas, X, y_true, var_names=None)` | R² of formulas on raw-state data (applies lift) |
| `get_A()` | Per-edge spline **scale factors**, shape ℝⁿˣᵐ — *not* the fitted coefficients (see below) |

Attributes after fit: `model_` (the underlying PyKAN model), `lift`,
`train_results_` (loss history for `plot_loss_curves`).

> **`get_formula` is destructive.** It runs `robust_auto_symbolic`, which
> replaces each spline edge with its snapped symbolic surrogate inside
> `model_`. Every later `predict`, `rollout` or edge plot uses the surrogate,
> so a bad snap silently corrupts the model. Do all numeric validation before
> extraction, or `copy.deepcopy(model)` per attempt.
>
> **`var_names` refers to the LIFTED feature names**, not the state variables,
> and `score_formula` returns `NaN` (not an error) if they don't match the
> symbols in the formulas — pass the same list to both.

> **`get_A()` does not return the coefficients — do not use it.** It returns
> PyKAN's per-edge `scale_sp`, and an edge computes
> `scale_base * base(θ) + scale_sp * spline(θ)` with `spline(θ) = Σ coef·B(θ)`.
> Since `scale_sp` and `coef` are both trainable and multiply each other, the
> optimizer puts the fit into `coef` and leaves `scale_sp` near its random
> initialization — measured drift over a full fit is often < 0.01. `get_A()`
> is therefore reporting initialization noise.
>
> It correlates with nothing: not magnitude, not sign, not even the sparsity
> pattern. On a linear system fitted to RMSE 1e-5 with true coefficients
> `[[3, -1, 0], [0, 0.5, 2]]`, `get_A()` returned
> `[[0.923, -0.001, 0.854], [0.882, 0.000, 0.932]]` — ~0 for the two true
> nonzeros, ~0.9 for two true zeros.
>
> To recover actual coefficients, either read the slope of each edge activation
> (`get_edge_activation`), or least-squares the network output against the
> lifted features:
> ```python
> Theta = model.lift(X)
> A_eff, *_ = np.linalg.lstsq(Theta, model.predict(X), rcond=None)
> A_eff = A_eff.T                      # (n, m), matches the true matrix
> ```
> `geometry/euler_top_example.py` and `geometry/torus_winding_example.py` both
> do this and recover their coefficients to ~1e-5.

## Lifts

```python
Lift                       # ABC: __call__(X), output_dim, feature_names (attribute), fit(X)
PolynomialLift(degree=2, include_bias=False)
FourierLift(n_modes)
RadialBasisLift(n_centers, sigma=None, center_method="random")
DMDLift(n_modes, dictionary=None, sort_by="magnitude")
DelayEmbedding(delays=3)   # (T, n) -> (T - delays + 1, n*delays); TRIM TARGETS: X_dot[delays-1:]
CustomLift(fn, output_dim, torch_fn=None, name="custom")   # torch_fn required for rollout-loss training
```

## Training

```python
fit_kan(model, dataset, opt="LBFGS", steps=500, lr=1.0, batch=-1,
        lamb=0.0, rollout_weight=0.0, rollout_horizon=10,
        dynamics_fn=None, integrator="rk4", rollout_loss_fn=None,
        update_grid=True, stop_grid_update_step=50)
make_windows(traj, window)          # (T, n) → (T-w+1, w, n) overlapping windows
angle_mse(pred, true)               # MSE with wrapped angle differences
wrap_pi_torch(x)                    # wrap angles to (−π, π]
order_param_torch(theta)            # Kuramoto order parameter |⟨e^{iθ}⟩|
rk4_step(f, x, dt)                  # differentiable RK4 step (torch)
euler_step(f, x, dt)                # differentiable Euler step (torch)
integrate_trajectory(f, x0, t)      # integrate over time array t (torch)
rk4_integrate_numpy(f, x0, T, dt)   # post-fit numpy rollout for figures
```

## Symbolic

```python
make_symbolic_lib({name: (torch_fn, sympy_fn, cost)})   # → PyKAN lib dict
POLY_LIB, POLY_LIB_CHEAP, TRIG_LIB, TRIG_LIB_CHEAP      # pre-built libraries
auto_symbolic_with_costs(model, preferred_idx, preferred_lib, other_lib,
                         weight_simple=0.8, r2_threshold=0.90, verbose=1)
robust_auto_symbolic(...)                               # retrying fallback
score_formula(formulas, theta, y_true, var_names)       # → list[float] R²
formulas_to_latex(formulas, lhs_names, environment="align*")  # → str
substitute_params(formulas, params)                     # → list[sympy.Expr]
```

## Numerics (finite volume, periodic 1D)

```python
cfl_dt(u, dx, cfl=0.8)
solve_burgers(u0, n_steps, dt, domain_length=None, scheme="rusanov",
              limiter="minmod", time_stepper="tvdrk2", save_every=1)
solve_viscous_burgers(u0, n_steps, dt, nu, domain_length=None, scheme="rusanov",
                      limiter="minmod", time_stepper="tvdrk2", save_every=1)
solve_scalar(u0, dx, n_steps, dt, flux_fn, speed_fn, roe_speed_fn=None,
             scheme="rusanov", limiter="minmod", time_stepper="tvdrk2",
             save_every=1)
fv_rhs(u, dx, flux_fn, speed_fn, roe_speed_fn=None, scheme="rusanov",
       limiter="minmod")
muscl_reconstruct(u, dx, limiter="minmod")   # → (u_L, u_R)
spectral_derivative(u, dx, order=1)
rusanov_flux(u_L, u_R, flux_fn, speed_fn)
roe_flux(u_L, u_R, flux_fn, roe_speed_fn, entropy_fix=True)
hllc_flux(u_L, u_R, flux_fn, speed_fn)
tvdrk2_step(u, rhs, dt)
tvdrk3_step(u, rhs, dt)
burgers_flux(u); burgers_speed(u); burgers_roe_speed(u_L, u_R)
minmod(a, b); van_leer(a, b); superbee(a, b)
```

## Plotting

```python
use_pub_style()                                  # publication matplotlib style
get_edge_activation(model, l, i, j, X=None)      # -> (x, y) sampled spline of one edge
                                                 # pass X (a tensor) or it raises: the forward
                                                 # pass with save_act=True must have been run
get_all_edge_activations(model, X=None)          # -> {(l, i, j): (x, y)}
plot_kan_architecture(...)                       # network diagram
fit_linear, fit_polynomial, fit_sine, fit_sech2, fit_sech2_tanh   # curve fits for edge annotation

plot_all_edges(model, X=None, *, fits=(), in_var_names=None, out_var_names=None,
               figsize_per_panel=(3.0, 2.5), poly_degree=2, save=None)
plot_loss_curves(results, *, ax=None, log_scale=True, show_rollout=True, save=None)
plot_attractor_overlay(true_traj, *other_trajs, dim_x=0, dim_y=2, labels=None,
                       colors=None, ax=None, xlabel=None, ylabel=None, xlim=None,
                       ylim=None, drop=0, lw=1.0, alpha_true=0.35, save=None)
plot_trajectory_error(true_traj, pred_traj, t=None, *, lyapunov_time=None, ax=None,
                      log_scale=True, label="KANDy", color="#1f77b4", save=None)
```

Note the exact keyword names — these functions take **no `title=` argument**
(set titles on the returned axis), and `plot_all_edges` uses
`in_var_names` / `out_var_names`, not `input_names` / `output_names`. Each
returns `(fig, ax)` — `plot_all_edges` returns `(fig, axes_array)`. Passing
`save="path/stem"` writes both `.png` and `.pdf`.
