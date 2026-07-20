# Finite-Volume Numerics

Self-contained finite-volume module for generating PDE training data on
periodic 1D domains, for scalar conservation laws `u_t + ∂_x F(u) = 0`.
All schemes use MUSCL reconstruction (2nd order) with a slope limiter, and
TVD/SSP Runge–Kutta time stepping.

## Convenience solvers

```python
from kandy import solve_burgers, solve_viscous_burgers, cfl_dt
import numpy as np

N  = 256
x  = np.linspace(0, 2*np.pi, N, endpoint=False)
u0 = np.sin(x)
dx = x[1] - x[0]

dt = cfl_dt(u0, dx, cfl=0.4)          # stable step from CFL condition

# Inviscid Burgers  u_t + (u²/2)_x = 0
U = solve_burgers(u0, n_steps=500, dt=dt, scheme="roe", limiter="van_leer")

# Viscous Burgers  u_t + (u²/2)_x = ν u_xx
# IMEX: explicit TVD-RK convection + exact spectral implicit diffusion
U = solve_viscous_burgers(u0, n_steps=500, dt=dt, nu=0.01)

U.shape   # (n_steps, N)
```

Custom conservation laws via `solve_scalar`:

```python
from kandy import solve_scalar

U = solve_scalar(
    u0, dx, n_steps=1000, dt=dt,
    flux_fn=my_flux, speed_fn=my_speed,
    roe_speed_fn=my_roe_speed,   # required for scheme="roe"
    scheme="roe", limiter="superbee", time_stepper="tvdrk3",
)
```

## Options

| Flux scheme | Function | Notes |
|---|---|---|
| `"rusanov"` (LLF) | `rusanov_flux` | Most diffusive; always stable — safe default |
| `"roe"` | `roe_flux` | Less diffusive; Harten–Hyman entropy fix at sonic points |
| `"hllc"` | `hllc_flux` | Two-wave solver; HLLC = HLL for scalar laws |

| Limiter | Character |
|---|---|
| `"minmod"` | Most dissipative, most robust |
| `"van_leer"` | Middle ground |
| `"superbee"` | Sharpest, can steepen smooth extrema |

| Time stepper | Order |
|---|---|
| `"tvdrk2"` (Heun) | 2nd, TVD |
| `"tvdrk3"` (Shu–Osher) | 3rd, TVD |

## Building blocks

For custom RHS assembly (e.g. inside a `dynamics_fn` for rollout training):

```python
fv_rhs(u, dx, flux_fn, speed_fn, roe_speed_fn=None,
       scheme="rusanov", limiter="minmod")       # semi-discrete RHS
muscl_reconstruct(u, dx, limiter="minmod")       # → (u_L, u_R) interface states
spectral_derivative(u, dx, order=1)              # FFT derivative (periodic)
tvdrk2_step(u, rhs, dt);  tvdrk3_step(u, rhs, dt)
burgers_flux(u); burgers_speed(u); burgers_roe_speed(u_L, u_R)
minmod(a, b); van_leer(a, b); superbee(a, b)
```

## Typical KANDy-PDE pipeline

1. Generate `U` with `solve_burgers` / `solve_viscous_burgers` / `solve_scalar`.
2. Build lifted features per grid point — e.g. `[u, u_x, ∂(u²/2)/∂x]` using
   `spectral_derivative` or the FV flux divergence.
3. Fit KANDy on (features, u_t) pairs; validate by rolling the discovered RHS
   forward with the same FV machinery and comparing space-time fields.

See `examples/pdes/burgers_example.py`, `examples/pdes/burgers_fourier_example.py`,
and `examples/pdes/kuramoto_sivashinsky_example.py`.
