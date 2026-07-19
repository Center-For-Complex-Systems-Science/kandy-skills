# KANDy API Reference

Everything below is importable from the top-level `kandy` package.

## Model

```python
KANDy(lift, grid=5, k=3, steps=500, seed=42, device=None, base_fun=None)
```

| Method | Description |
|---|---|
| `fit(X, X_dot=None, dt=None, opt="LBFGS", lr=1.0, batch=-1, lamb=0.0, rollout_weight=0.0, rollout_loss_fn=None, fit_steps=None, val_frac=0.15, test_frac=0.15)` | Fit the model |
| `predict(X)` | Predict derivatives / next state |
| `rollout(x0, T, dt, integrator="rk4")` | Autoregressive trajectory integration |
| `get_formula(var_names=None, round_places=3, simplify=False)` | Symbolic extraction → list of SymPy expressions |
| `score_formula(formulas, X, y_true, var_names=None)` | R² of formulas on raw-state data (applies lift) |
| `get_A()` | Extract linear mixing matrix A ∈ ℝⁿˣᵐ |

Attributes after fit: `model_` (the underlying PyKAN model), `lift`.

## Lifts

```python
Lift                       # ABC: __call__(X), output_dim, feature_names(), fit(X)
PolynomialLift(degree=2, include_bias=False)
FourierLift(n_modes)
RadialBasisLift(n_centers, sigma=None, center_method="random")
DMDLift(n_modes, dictionary=None, sort_by="magnitude")
DelayEmbedding(delays=3)
CustomLift(fn, output_dim, name="custom")
KANELift(latent_dim, hidden_dim=None, grid=5, k=3)   # EXPERIMENTAL
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
get_edge_activation(model, l, i, j)              # sampled spline of one edge
get_all_edge_activations(model)
plot_edge(...); plot_all_edges(...)              # per-edge spline plots
plot_kan_architecture(...)                       # network diagram
plot_loss_curves(...)                            # train/val/test losses
plot_attractor_overlay(...)                      # true vs rollout attractor
plot_trajectory_error(...)                       # per-dimension error over time
fit_linear, fit_polynomial, fit_sine, fit_sech2, fit_sech2_tanh   # curve fits for edge annotation
```
