#!/usr/bin/env python3
"""KANDy example: kinematic unicycle (Dubins car) — a flow on SE(2).

The unicycle rolls without slipping in the plane.  Carrying the two controls
as constant extra coordinates makes the system autonomous, so KANDy.rollout
can integrate it directly:

    dx/dt     = v * cos(theta)
    dy/dt     = v * sin(theta)
    dtheta/dt = omega
    dv/dt     = 0
    domega/dt = 0

State = (x, y, theta, v, omega) in R^5.  Trajectories are circular arcs of
radius v/omega, degenerating to straight lines as omega -> 0.

THE LESSON: mixed trig x linear cross-terms need a hand-built lift
------------------------------------------------------------------
1. NO STOCK LIFT CAN DO THIS.  The RHS needs the product v*cos(theta), a
   trigonometric function of one state multiplied by another state.

     * ``PolynomialLift`` supplies v, theta, v*theta, theta^2, ... — every
       monomial, and not one trig function.  A spline edge on v*theta cannot
       become v*cos(theta): the map (v, theta) -> v*theta is not invertible
       into the target.
     * ``FourierLift`` supplies cos(theta), sin(theta), cos(2*theta), ... —
       trig, but no way to multiply the amplitude v back in.

   The bilinear obstruction (v*cos(theta) != h(u(v) + w(theta)) for any
   continuous h, u, w) says a separable KAN cannot recover the missing
   product, so it has to be built by hand.  ``CustomLift`` composing BOTH
   families is the only option:

       theta_lift = [v*cos(theta),  v*sin(theta),  omega]

   Three features for a five-dimensional state, and every output is then a
   SINGLE linear feature (dv/dt and domega/dt are identically zero).

2. x, y AND theta NEVER APPEAR IN THE LIFT.  This is the corrective to "just
   throw the states in".  The lift encodes the functional dependence of the
   RIGHT-HAND SIDE, not the state vector.  The unicycle is translation
   invariant — the RHS does not depend on where you are — so position needs no
   feature at all, and theta enters only through the two trig products that
   already carry it.  Adding x, y, theta as extra coordinates would only add
   near-dependent columns for the extraction to trip over.

3. THE FAILURE MODE IS STRUCTURAL, NOT A TUNING PROBLEM.  Model B below uses
   the identity lift (KAN [5, 5]) — the naive "feed it the states" choice.  It
   trains to convergence and still has an order-of-magnitude worse derivative
   RMSE, and its rollouts drift off the true arcs.  No amount of grid
   refinement, steps, or width fixes it: the function it needs is not in the
   span of a sum of univariate functions.

Lift  phi: R^5 -> R^3   theta = [v*cos(theta), v*sin(theta), omega]
KAN:  width = [3, 5],  base_fun='zero'
"""

import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from kandy import KANDy, CustomLift
from kandy.plotting import (
    plot_all_edges,
    plot_loss_curves,
    use_pub_style,
)

# ---------------------------------------------------------------------------
# 0. Reproducibility and parameters
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# These KANs are tiny ([3, 5] and [5, 5]); letting torch spread them over every
# core costs far more in thread synchronisation than it saves.  Capping the
# thread count turns a ~20 minute run into a ~1 minute one on a many-core box.
torch.set_num_threads(min(4, torch.get_num_threads()))

XY_BOX = 3.0            # positions sampled over [-3, 3]^2 (irrelevant to the RHS)
V_RANGE = (0.2, 2.0)    # forward speed
W_RANGE = (-1.5, 1.5)   # turn rate

print("[MODEL] Kinematic unicycle on SE(2): state (x, y, theta, v, omega)")
print("[MODEL] dx/dt = v*cos(theta), dy/dt = v*sin(theta), dtheta/dt = omega, "
      "dv/dt = domega/dt = 0")
print("[MODEL] Trajectories are arcs of radius v/omega "
      "(straight lines as omega -> 0)")


def unicycle_rhs_np(S: np.ndarray) -> np.ndarray:
    """Right-hand side for an (N, 5) block of states."""
    theta, v, omega = S[:, 2], S[:, 3], S[:, 4]
    z = np.zeros_like(v)
    return np.column_stack([v * np.cos(theta), v * np.sin(theta), omega, z, z])


def unicycle(t, state):
    return unicycle_rhs_np(np.asarray(state, dtype=float)[None, :])[0]


# ---------------------------------------------------------------------------
# 1. Training data — independent uniform samples, NOT a trajectory
#
#    Sampling theta, v and omega independently keeps v*cos(theta),
#    v*sin(theta) and omega linearly independent.  Along a single arc v and
#    omega are constant, so the feature matrix would be rank deficient and the
#    coefficients unidentifiable.  x and y are sampled too — they are inputs
#    the model must learn to IGNORE — but they never reach the KAN, because the
#    lift drops them.
# ---------------------------------------------------------------------------
N_SAMPLES = 6000
rng = np.random.default_rng(SEED)

X = np.column_stack([
    rng.uniform(-XY_BOX, XY_BOX, N_SAMPLES),                    # x
    rng.uniform(-XY_BOX, XY_BOX, N_SAMPLES),                    # y
    rng.uniform(-np.pi, np.pi, N_SAMPLES),                      # theta
    rng.uniform(V_RANGE[0], V_RANGE[1], N_SAMPLES),             # v
    rng.uniform(W_RANGE[0], W_RANGE[1], N_SAMPLES),             # omega
])
X_dot = unicycle_rhs_np(X)

print(f"\n[DATA]  {N_SAMPLES} independent state samples")
print(f"[DATA]  theta ~ U(-pi, pi],  v ~ U{V_RANGE},  omega ~ U{W_RANGE},  "
      f"x, y ~ U(-{XY_BOX}, {XY_BOX})")

# ---------------------------------------------------------------------------
# 2. The lift — three mixed trig x linear features, no position, no theta
#
#    Symbol names are plain identifiers so SymPy/lambdify can handle them in
#    score_formula:  v_cos_theta = v*cos(theta), v_sin_theta = v*sin(theta).
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["v_cos_theta", "v_sin_theta", "omega"]
STATE_NAMES = ["x", "y", "theta", "v", "omega"]
OUT_NAMES = ["dx/dt", "dy/dt", "dtheta/dt", "dv/dt", "domega/dt"]


def se2_lift(S: np.ndarray) -> np.ndarray:
    theta, v, omega = S[:, 2], S[:, 3], S[:, 4]
    return np.column_stack([v * np.cos(theta), v * np.sin(theta), omega])


lift = CustomLift(fn=se2_lift, output_dim=3, name="se2_lift")

# Identifiability diagnostic: a well-conditioned lifted feature matrix means
# the linear coefficients are uniquely determined by the data.
Theta = se2_lift(X)
cond_lift = np.linalg.cond(np.column_stack([Theta, np.ones(len(Theta))]))
cond_raw = np.linalg.cond(np.column_stack([X, np.ones(len(X))]))
print(f"[DATA]  cond(lifted features + bias) = {cond_lift:.2f}   "
      f"(identity lift: {cond_raw:.2f})")

# ---------------------------------------------------------------------------
# 3. Models — A: physics lift [3, 5];  B: identity lift [5, 5]
# ---------------------------------------------------------------------------
print("\n--- Model A: SE(2) lift  phi: R^5 -> R^3  (KAN = [3, 5]) ---")
model = KANDy(lift=lift, grid=5, k=3, steps=200, seed=SEED, base_fun="zero")
model.fit(X=X, X_dot=X_dot, val_frac=0.15, test_frac=0.15, lamb=0.0, patience=50)

print("\n--- Model B: identity lift  phi: R^5 -> R^5  (KAN = [5, 5]) ---")
id_lift = CustomLift(fn=lambda S: S.copy(), output_dim=5, name="identity")
model_id = KANDy(lift=id_lift, grid=5, k=3, steps=200, seed=SEED, base_fun="zero")
model_id.fit(X=X, X_dot=X_dot, val_frac=0.15, test_frac=0.15, lamb=0.0, patience=50)

pred = model.predict(X)
pred_id = model_id.predict(X)
rmse_deriv = np.sqrt(np.mean((pred - X_dot) ** 2))
rmse_deriv_id = np.sqrt(np.mean((pred_id - X_dot) ** 2))


def per_eq_r2(y, yhat):
    out = []
    for i in range(y.shape[1]):
        var = np.sum((y[:, i] - y[:, i].mean()) ** 2)
        out.append(1.0 if var == 0.0 else
                   1.0 - np.sum((y[:, i] - yhat[:, i]) ** 2) / var)
    return np.array(out)


print(f"\n[EVAL]  Derivative RMSE  — SE(2) lift: {rmse_deriv:.6e}")
print(f"[EVAL]  Derivative RMSE  — identity lift: {rmse_deriv_id:.6e}   "
      f"({rmse_deriv_id / rmse_deriv:.0f}x worse)")
print(f"[EVAL]  R^2 per equation — SE(2) lift:    "
      f"{np.round(per_eq_r2(X_dot, pred), 6)}")
print(f"[EVAL]  R^2 per equation — identity lift: "
      f"{np.round(per_eq_r2(X_dot, pred_id), 6)}")
print("[EVAL]  The identity-lift model is beaten by the bilinear obstruction: "
      "a sum of\n        univariate functions of (x, y, theta, v, omega) cannot "
      "represent v*cos(theta).")

# ---------------------------------------------------------------------------
# 4. Rollout validation — a tight arc and a near-straight run
#
#    Done BEFORE get_formula(): symbolic extraction replaces the spline edges
#    in place, so every later predict/rollout would use the snapped surrogate.
# ---------------------------------------------------------------------------
DT = 0.02
T_MAX = 6.0
t_eval = np.arange(0.0, T_MAX, DT)

# (x0, y0, theta0, v, omega)
INITIAL_CONDITIONS = [
    ("tight arc  (r = v/omega = 0.83)", [0.0, 0.0, 0.0, 1.0, 1.2]),
    ("near-straight (r = 30.0)", [-1.5, 1.0, 0.6, 1.5, 0.05]),
    ("wide arc, reversed turn (r = 2.67)", [1.0, -1.0, -2.0, 0.8, -0.3]),
]

rollout_rows = []
for label, x0 in INITIAL_CONDITIONS:
    sol = solve_ivp(unicycle, [t_eval[0], t_eval[-1]], x0, t_eval=t_eval,
                    method="RK45", rtol=1e-11, atol=1e-13)
    true_traj = sol.y.T
    pred_traj = model.rollout(np.array(x0, dtype=float), T=len(true_traj),
                              dt=DT, integrator="rk4")
    pred_id_traj = model_id.rollout(np.array(x0, dtype=float), T=len(true_traj),
                                    dt=DT, integrator="rk4")
    rollout_rows.append((label, true_traj, pred_traj, pred_id_traj))

print(f"\n[EVAL]  Rollout: {len(t_eval)} RK4 steps at dt={DT} (T = {T_MAX})")
for label, s, p, pid in rollout_rows:
    rmse = np.sqrt(np.mean((p - s) ** 2))
    rmse_id = np.sqrt(np.mean((pid - s) ** 2))
    pose_err = np.linalg.norm(p[-1, :2] - s[-1, :2])
    pose_err_id = np.linalg.norm(pid[-1, :2] - s[-1, :2])
    head_err = abs(p[-1, 2] - s[-1, 2])
    print(f"  {label}")
    print(f"    RMSE          SE(2) {rmse:.3e}   identity {rmse_id:.3e}")
    print(f"    final xy err  SE(2) {pose_err:.3e}   identity {pose_err_id:.3e}")
    print(f"    final heading err SE(2) {head_err:.3e} rad")

# Heading note: theta is a genuine angle, but neither the true nor the learned
# trajectory is wrapped here — both integrate theta0 + omega*t on the real
# line, so a plain difference is the right error metric.  Wrapping would be
# required only if the states were stored modulo 2*pi; kandy.training provides
# ``wrap_pi_torch`` and ``angle_mse`` for that case (see the Kuramoto example).
# Note also that theta being unbounded is harmless for the lift: v*cos(theta)
# and v*sin(theta) stay inside the training range no matter how far theta winds.
th_span = max(abs(s[:, 2]).max() for _, s, _, _ in rollout_rows)
print(f"[EVAL]  max |theta| along rollouts = {th_span:.2f} rad "
      f"(> pi, but the lift is bounded, so no wrapping is needed)")

# Recovered linear mixing, captured before snapping.
A_hat = model.get_A()
A_eff = np.linalg.lstsq(Theta, pred, rcond=None)[0].T   # effective gains, (5, 3)
A_true = np.array([
    [1.0, 0.0, 0.0],    # dx/dt     = 1 * v*cos(theta)
    [0.0, 1.0, 0.0],    # dy/dt     = 1 * v*sin(theta)
    [0.0, 0.0, 1.0],    # dtheta/dt = 1 * omega
    [0.0, 0.0, 0.0],    # dv/dt     = 0
    [0.0, 0.0, 0.0],    # domega/dt = 0
])
print("\n[EVAL]  True A (5 x 3):")
print(np.array2string(A_true, precision=3, suppress_small=True))
print("[EVAL]  Effective A recovered by least squares on the network output:")
print(np.array2string(A_eff, precision=4, suppress_small=True))
print(f"[EVAL]  max |A_eff - A_true| = {np.abs(A_eff - A_true).max():.3e}")
print("[EVAL]  model.get_A() (PyKAN's raw per-edge spline SCALES — an internal\n"
      "        parameterisation, not the coefficients of the fit; a large\n"
      "        entry on a dead edge just means its spline shape is ~0.  Read\n"
      "        the coefficients off A_eff above or the formulas below):")
print(np.array2string(A_hat, precision=4, suppress_small=True))

# Edge activations, also captured before snapping.
train_theta = torch.tensor(Theta[:2048], dtype=torch.float32)

# ---------------------------------------------------------------------------
# 5. Symbolic extraction  (LAST — get_formula mutates the model)
#
#    Every edge is a pure identity here, so the library only needs 'x' and
#    '0'.  The zero rows dv/dt and domega/dt should snap to 0 on all edges.
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Extracting formulas ...")
formulas = model.get_formula(
    var_names=FEATURE_NAMES,
    round_places=4,
    lib=["x", "0"],
    r2_threshold=0.80,
    weight_simple=0.8,
)
for lab, f in zip(OUT_NAMES, formulas):
    print(f"  {lab} = {f}")

r2 = model.score_formula(formulas, X, X_dot, var_names=FEATURE_NAMES)
print(f"[SYMBOLIC] Formula R^2 per equation: {np.round(r2, 6)}")
print("[SYMBOLIC] True:  dx/dt = 1.0*v_cos_theta,  dy/dt = 1.0*v_sin_theta,")
print("[SYMBOLIC]        dtheta/dt = 1.0*omega,    dv/dt = domega/dt = 0")

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
use_pub_style()
OUT = "results/UnicycleSE2"
os.makedirs(OUT, exist_ok=True)

# 6a. xy-plane paths — true vs SE(2) lift vs identity lift
fig, axes = plt.subplots(1, len(rollout_rows), figsize=(4.2 * len(rollout_rows), 4.0))
for ax, (label, s, p, pid) in zip(np.atleast_1d(axes), rollout_rows):
    ax.plot(s[:, 0], s[:, 1], color="#1f77b4", lw=2.0, alpha=0.6, label="true")
    ax.plot(p[:, 0], p[:, 1], color="#d62728", lw=1.2, ls="--", label="KANDy SE(2) lift")
    ax.plot(pid[:, 0], pid[:, 1], color="#7f7f7f", lw=1.2, ls=":", label="identity lift")
    ax.plot(s[0, 0], s[0, 1], "ko", ms=4)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(label, fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
np.atleast_1d(axes)[0].legend(loc="best", fontsize=7)
fig.suptitle("Unicycle on SE(2): planar paths", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/xy_paths.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/xy_paths.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6b. Heading time series
fig, ax = plt.subplots(figsize=(8, 4))
for ci, (label, s, p, pid) in enumerate(rollout_rows):
    c = ["#1f77b4", "#2ca02c", "#9467bd"][ci % 3]
    ax.plot(t_eval, s[:, 2], color=c, lw=1.6, alpha=0.6)
    ax.plot(t_eval, p[:, 2], color=c, lw=1.0, ls="--")
    ax.plot(t_eval, pid[:, 2], color=c, lw=1.0, ls=":")
ax.plot([], [], color="k", lw=1.6, alpha=0.6, label="true")
ax.plot([], [], color="k", lw=1.0, ls="--", label="KANDy SE(2) lift")
ax.plot([], [], color="k", lw=1.0, ls=":", label="identity lift")
ax.set_xlabel("time")
ax.set_ylabel(r"$\theta$  (rad, unwrapped)")
ax.set_title(r"Heading $\theta(t) = \theta_0 + \omega t$")
ax.legend(loc="best", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/theta_timeseries.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/theta_timeseries.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6c. Loss curves for both models
for tag, mdl in [("se2", model), ("identity", model_id)]:
    if getattr(mdl, "train_results_", None):
        fig, ax = plot_loss_curves(mdl.train_results_, save=f"{OUT}/loss_curves_{tag}")
        plt.close(fig)

# 6d. Edge activations — three straight lines, which IS the result
fig, _axes = plot_all_edges(
    model.model_, X=train_theta,
    in_var_names=FEATURE_NAMES,
    out_var_names=OUT_NAMES,
    save=f"{OUT}/edge_activations",
)
plt.close(fig)

print(f"[FIGS]  Saved {OUT}/")
