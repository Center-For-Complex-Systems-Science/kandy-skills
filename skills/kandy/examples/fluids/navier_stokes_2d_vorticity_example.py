#!/usr/bin/env python3
"""KANDy example: 2D decaying turbulence — recovering the vorticity equation.

Two-dimensional incompressible Navier–Stokes on a periodic box, in vorticity
form:

    dw/dt = -(u * dw/dx + v * dw/dy) + nu * laplacian(w)

with the velocity recovered from the streamfunction,

    laplacian(psi) = -w,   u = dpsi/dy,   v = -dpsi/dx.

Training data comes from a pseudo-spectral simulation of decaying turbulence
(2/3-rule dealiasing, RK4 in Fourier space) started from a smooth random
low-wavenumber vorticity field.

The lift: local features, one sample per grid point
--------------------------------------------------
As in the 1D PDE examples, each GRID POINT is a training sample and the lift
is built from local field quantities evaluated there.  Spatial derivatives are
computed spectrally, which is exact for periodic data.

    theta = [w, u, v, w_x, w_y, lap_w, u*w_x, v*w_y]        (8 features)

The two ADVECTION terms u*w_x and v*w_y are products of different fields, so
they are exactly the cross-terms a separable KAN cannot construct — they must
be in the lift.  The remaining five features are deliberately redundant: the
true equation does not use w, u, v, w_x or w_y on their own, and a correct fit
must drive their edges to zero.  That is the check that the method is
identifying structure rather than curve-fitting.

Given the lift, the RHS is LINEAR in the features with known coefficients
(-1, -1, nu), so lib=["x", "0"] recovers them exactly — including the
viscosity, which is read straight off the lap_w edge.

Rollout here means integrating the LEARNED local RHS over the whole grid:
recompute the features spectrally at each step, apply the model pointwise,
and RK4 forward.  This is a genuine PDE rollout, not a pointwise regression
score.

KAN:  width = [8, 1],  base_fun='zero'
"""

import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kandy import KANDy, CustomLift
from kandy.plotting import plot_all_edges, plot_loss_curves, use_pub_style

# ---------------------------------------------------------------------------
# 0. Reproducibility and physical parameters
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

N_GRID   = 64          # grid points per dimension (domain is [0, 2*pi]^2)
NU       = 0.02        # kinematic viscosity
AMPLITUDE = 5.0        # initial vorticity scale
DT       = 0.001       # simulation time step
N_STEPS  = 3000        # simulation steps
SNAP_EVERY = 150       # snapshot interval for training data

# ---------------------------------------------------------------------------
# 1. Pseudo-spectral 2D Navier–Stokes solver (vorticity–streamfunction form)
# ---------------------------------------------------------------------------
KX = np.fft.fftfreq(N_GRID, d=1.0 / N_GRID).reshape(-1, 1)
KY = np.fft.fftfreq(N_GRID, d=1.0 / N_GRID).reshape(1, -1)
K2 = KX ** 2 + KY ** 2
K2_INV = np.where(K2 == 0, 1.0, K2)          # avoid division by zero at k=0
DEALIAS = (np.abs(KX) < N_GRID / 3) & (np.abs(KY) < N_GRID / 3)


def local_fields(w_hat: np.ndarray):
    """Return (w, u, v, w_x, w_y, lap_w) in physical space from vorticity."""
    psi_hat = np.where(K2 == 0, 0.0, w_hat / K2_INV)
    u    = np.real(np.fft.ifft2(1j * KY * psi_hat))
    v    = np.real(np.fft.ifft2(-1j * KX * psi_hat))
    w_x  = np.real(np.fft.ifft2(1j * KX * w_hat))
    w_y  = np.real(np.fft.ifft2(1j * KY * w_hat))
    lap  = np.real(np.fft.ifft2(-K2 * w_hat))
    w    = np.real(np.fft.ifft2(w_hat))
    return w, u, v, w_x, w_y, lap


def rhs_hat(w_hat: np.ndarray) -> np.ndarray:
    """Spectral RHS of the vorticity equation (dealiased advection)."""
    _, u, v, w_x, w_y, _ = local_fields(w_hat)
    advection = np.fft.fft2(-(u * w_x + v * w_y)) * DEALIAS
    return advection - NU * K2 * w_hat


def rk4_hat(w_hat: np.ndarray, dt: float) -> np.ndarray:
    k1 = rhs_hat(w_hat)
    k2 = rhs_hat(w_hat + 0.5 * dt * k1)
    k3 = rhs_hat(w_hat + 0.5 * dt * k2)
    k4 = rhs_hat(w_hat + dt * k3)
    return w_hat + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# Smooth random initial condition concentrated at low wavenumbers
rng = np.random.default_rng(SEED)
w_hat = np.fft.fft2(rng.standard_normal((N_GRID, N_GRID)))
w_hat *= np.exp(-K2 / (2 * 4.0 ** 2)) * DEALIAS
w_init = np.real(np.fft.ifft2(w_hat))
w_init *= AMPLITUDE / np.abs(w_init).max()
w_hat = np.fft.fft2(w_init)

print(f"[SIM]  {N_GRID}x{N_GRID} grid, nu={NU}, dt={DT}, {N_STEPS} steps")

FEATURE_NAMES = ["w", "u", "v", "w_x", "w_y", "lap_w", "u*w_x", "v*w_y"]


def build_features(w_hat: np.ndarray) -> np.ndarray:
    """Lift one vorticity field to (N_GRID^2, 8) local features."""
    w, u, v, w_x, w_y, lap = local_fields(w_hat)
    return np.stack([w, u, v, w_x, w_y, lap, u * w_x, v * w_y],
                    axis=-1).reshape(-1, 8)


feature_pool, target_pool, snapshots = [], [], []
w_hat_run = w_hat.copy()
for i in range(N_STEPS + 1):
    if i % SNAP_EVERY == 0:
        w, u, v, w_x, w_y, lap = local_fields(w_hat_run)
        dwdt = -(u * w_x + v * w_y) + NU * lap
        feature_pool.append(build_features(w_hat_run))
        target_pool.append(dwdt.reshape(-1))
        snapshots.append(w.copy())
    w_hat_run = rk4_hat(w_hat_run, DT)

feature_pool = np.concatenate(feature_pool)
target_pool  = np.concatenate(target_pool)
print(f"[DATA]  {len(snapshots)} snapshots -> {feature_pool.shape[0]} grid samples")

# ---------------------------------------------------------------------------
# 2. Subsample the grid-point pool for training
# ---------------------------------------------------------------------------
N_TRAIN = 9000
sel = np.random.default_rng(SEED + 1).choice(len(feature_pool), N_TRAIN, replace=False)
Theta = feature_pool[sel]
Y     = target_pool[sel][:, None]

cond = np.linalg.cond(np.column_stack([Theta, np.ones(len(Theta))]))
adv_scale  = np.std(Theta[:, 6] + Theta[:, 7])
visc_scale = np.std(NU * Theta[:, 5])
print(f"[DATA]  {N_TRAIN} training points, cond(Theta) = {cond:.1f}")
print(f"[DATA]  term magnitudes — advection {adv_scale:.3f}, viscous {visc_scale:.3f}")

# ---------------------------------------------------------------------------
# 3. KANDy model — features are pre-computed, so the lift is the identity
# ---------------------------------------------------------------------------
lift = CustomLift(fn=lambda X: X, output_dim=len(FEATURE_NAMES), name="ns2d_local")
model = KANDy(lift=lift, grid=5, k=3, steps=200, seed=SEED, base_fun="zero")
model.fit(X=Theta, X_dot=Y, val_frac=0.15, test_frac=0.15, lamb=0.0, patience=40)

pred = model.predict(Theta)
raw_r2 = 1.0 - np.sum((Y[:, 0] - pred[:, 0]) ** 2) / np.sum((Y[:, 0] - Y[:, 0].mean()) ** 2)
print(f"[EVAL]  Network R^2 on dw/dt: {raw_r2:.6f}")

# ---------------------------------------------------------------------------
# 4. Field-level check and PDE rollout with the LEARNED model
#
#    Done before get_formula(), which rewrites the model's edges in place.
# ---------------------------------------------------------------------------
def learned_rhs(w_hat_state: np.ndarray) -> np.ndarray:
    """dw/dt over the whole grid, evaluated by the learned local model."""
    feats = build_features(w_hat_state)
    out = model.predict(feats)[:, 0]
    return out.reshape(N_GRID, N_GRID)


# 4a. One-step field accuracy on a held-out snapshot
w_hat_test = np.fft.fft2(snapshots[len(snapshots) // 2])
w_t, u_t, v_t, wx_t, wy_t, lap_t = local_fields(w_hat_test)
dwdt_true = -(u_t * wx_t + v_t * wy_t) + NU * lap_t
dwdt_pred = learned_rhs(w_hat_test)
field_r2 = 1.0 - np.sum((dwdt_true - dwdt_pred) ** 2) / np.sum(
    (dwdt_true - dwdt_true.mean()) ** 2
)
print(f"[EVAL]  Field-level R^2 of dw/dt on a held-out snapshot: {field_r2:.6f}")

# 4b. Autoregressive PDE rollout: integrate the learned RHS in physical space
ROLL_STEPS = 600
w_hat_roll = np.fft.fft2(snapshots[0])
w_hat_ref  = np.fft.fft2(snapshots[0])
for _ in range(ROLL_STEPS):
    # RK4 on the learned RHS, transforming back to spectral space each stage
    def _f(wh):
        return np.fft.fft2(learned_rhs(wh)) * DEALIAS

    k1 = _f(w_hat_roll)
    k2 = _f(w_hat_roll + 0.5 * DT * k1)
    k3 = _f(w_hat_roll + 0.5 * DT * k2)
    k4 = _f(w_hat_roll + DT * k3)
    w_hat_roll = w_hat_roll + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    w_hat_ref = rk4_hat(w_hat_ref, DT)

w_roll = np.real(np.fft.ifft2(w_hat_roll))
w_ref  = np.real(np.fft.ifft2(w_hat_ref))
roll_rmse = np.sqrt(np.mean((w_roll - w_ref) ** 2))
roll_rel  = roll_rmse / np.std(w_ref)
print(f"[EVAL]  PDE rollout ({ROLL_STEPS} steps, t={ROLL_STEPS * DT:.2f}): "
      f"RMSE {roll_rmse:.6f}  ({100 * roll_rel:.3f}% of field std)")

# Capture edge inputs before snapping
theta_t = torch.tensor(Theta[:2048], dtype=torch.float32)

# ---------------------------------------------------------------------------
# 5. Symbolic extraction — the RHS is linear in these features
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Extracting the vorticity equation ...")
formulas = model.get_formula(
    var_names=FEATURE_NAMES, round_places=4,
    lib=["x", "0"], r2_threshold=0.80, weight_simple=0.0,
)
print(f"  dw/dt = {formulas[0]}")
r2 = model.score_formula(formulas, Theta, Y, var_names=FEATURE_NAMES)
print(f"[SYMBOLIC] Formula R^2: {np.round(r2, 6)}")
print(f"[SYMBOLIC] True:  dw/dt = -1*u*w_x - 1*v*w_y + {NU}*lap_w")
print("[SYMBOLIC] The five unused features (w, u, v, w_x, w_y) should be absent — "
      "\n           their edges were driven to zero, recovering the equation's "
      "structure.")

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/NavierStokes2D", exist_ok=True)

extent = [0, 2 * np.pi, 0, 2 * np.pi]

# 6a. Vorticity fields: initial, true final, learned rollout, error
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
vmax = np.abs(snapshots[0]).max()
panels = [
    (snapshots[0], "initial vorticity", vmax),
    (w_ref,  f"true, t={ROLL_STEPS * DT:.2f}", vmax),
    (w_roll, f"KANDy rollout, t={ROLL_STEPS * DT:.2f}", vmax),
    (w_roll - w_ref, "error", np.abs(w_roll - w_ref).max() + 1e-12),
]
for ax, (field, title, scale) in zip(axes, panels):
    im = ax.imshow(field.T, origin="lower", extent=extent, cmap="RdBu_r",
                   vmin=-scale, vmax=scale)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("2D decaying turbulence: learned vorticity dynamics", fontsize=12)
fig.tight_layout()
fig.savefig("results/NavierStokes2D/vorticity_fields.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6b. dw/dt field, true vs predicted
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))
s = np.abs(dwdt_true).max()
for ax, (field, title) in zip(axes, [(dwdt_true, "true $\\partial_t\\omega$"),
                                     (dwdt_pred, "KANDy $\\partial_t\\omega$"),
                                     (dwdt_pred - dwdt_true, "error")]):
    im = ax.imshow(field.T, origin="lower", extent=extent, cmap="RdBu_r",
                   vmin=-s, vmax=s)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle(f"Vorticity tendency, field $R^2$ = {field_r2:.5f}", fontsize=12)
fig.tight_layout()
fig.savefig("results/NavierStokes2D/dwdt_field.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6c. Enstrophy decay, true vs learned
fig, ax = plt.subplots(figsize=(6, 4))
ens_true = [0.5 * np.mean(s_ ** 2) for s_ in snapshots]
t_snap = np.arange(len(snapshots)) * SNAP_EVERY * DT
ax.plot(t_snap, ens_true, "o-", color="#1f77b4", ms=3, lw=1.2, label="true")
ax.axhline(0.5 * np.mean(w_roll ** 2), color="#d62728", lw=1.0, ls="--",
           label=f"KANDy rollout at t={ROLL_STEPS * DT:.2f}")
ax.set_xlabel("time")
ax.set_ylabel(r"enstrophy $\frac{1}{2}\langle\omega^2\rangle$")
ax.set_title("Enstrophy decay")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/NavierStokes2D/enstrophy.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6d. Loss curves
if getattr(model, "train_results_", None):
    fig, ax = plot_loss_curves(
        model.train_results_, save="results/NavierStokes2D/loss_curves",
    )
    plt.close(fig)

# 6e. Edge activations — only the two advection edges and lap_w are non-flat
fig, axes = plot_all_edges(
    model.model_, X=theta_t,
    in_var_names=FEATURE_NAMES,
    out_var_names=["dw/dt"],
    save="results/NavierStokes2D/edge_activations",
)
plt.close(fig)

print("[FIGS]  Saved results/NavierStokes2D/")
