#!/usr/bin/env python3
"""KANDy example: free rigid body (Euler top) on SO(3).

Angular velocity in the body frame, with distinct principal moments of
inertia I1 < I2 < I3 and no applied torque:

    dW1/dt = (I2 - I3)/I1 * W2*W3
    dW2/dt = (I3 - I1)/I2 * W3*W1
    dW3/dt = (I1 - I2)/I3 * W1*W2

The flow is Hamiltonian on the dual of so(3) and conserves two quantities:

    E      = 1/2 * (I1*W1^2 + I2*W2^2 + I3*W3^2)        (kinetic energy)
    ||L||^2 = (I1*W1)^2 + (I2*W2)^2 + (I3*W3)^2         (angular momentum)

so every orbit lies on the intersection of an energy ellipsoid with a
momentum sphere — a closed curve (the classic polhode).

THE LESSON: the purest possible statement of the cross-term rule
-----------------------------------------------------------------
The Euler equations are the cleanest test case the lift rule has, because the
RHS is *nothing but* cross-products.  There are no linear terms and no powers
of a single variable anywhere.  So the minimal lift is exactly three features

    theta = [W2*W3, W3*W1, W1*W2]

and NOT [W1, W2, W3, W1^2, ..., W2*W3, ...].  Three consequences:

  1. Each output is a SINGLE lifted feature times a constant.  The KAN has
     nothing left to learn but three straight lines: every edge activation is
     linear, and the two off-diagonal edges per output are zero.  A straight
     edge is not a boring plot — it *is* the result, the visual statement that
     the lift already contains the whole nonlinearity.

  2. The recovered coefficients are read straight off the network and match
     (I2-I3)/I1, (I3-I1)/I2, (I1-I2)/I3 to several decimals.

  3. Adding W1, W2, W3 or their squares would break rule 2 of lifts.md: a
     spline edge on W1 can already produce any function of W1, so the
     decomposition would stop being unique and the constants would come out
     meaningless even at R^2 = 1.

The contrast model below makes the point structurally rather than
rhetorically: with the identity lift (KAN [3,3]) the network is a sum of three
univariate functions, f1(W1) + f2(W2) + f3(W3), which by the bilinear
obstruction *cannot* represent W2*W3 no matter how long it trains.  Its RMSE
is orders of magnitude worse, and no amount of tuning fixes it.

FREE CORRECTNESS CHECK: because E and ||L||^2 are conserved by the true flow,
their drift along the *learned* rollout is a label-free error measure.  A model
that has the coefficient ratios slightly wrong drifts off the polhode even when
its one-step RMSE looks fine, so this catches errors the pointwise metrics hide.

Lift  phi: R^3 -> R^3   theta = [W2*W3, W3*W1, W1*W2]
KAN:  width = [3, 3],  base_fun='zero'
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
    fit_linear,
    get_edge_activation,
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

I1, I2, I3 = 1.0, 2.0, 3.0          # distinct principal moments of inertia
C1 = (I2 - I3) / I1                 # -1
C2 = (I3 - I1) / I2                 # +1
C3 = (I1 - I2) / I3                 # -1/3
TRUE_COEFFS = np.array([C1, C2, C3])

print(f"[MODEL] Euler top with I = ({I1}, {I2}, {I3})")
print(f"[MODEL] True coefficients: (I2-I3)/I1 = {C1:+.6f}, "
      f"(I3-I1)/I2 = {C2:+.6f}, (I1-I2)/I3 = {C3:+.6f}")


def euler_rhs_np(W1, W2, W3):
    return np.column_stack([C1 * W2 * W3, C2 * W3 * W1, C3 * W1 * W2])


def euler_top(t, state):
    W1, W2, W3 = state
    return [C1 * W2 * W3, C2 * W3 * W1, C3 * W1 * W2]


def energy(W):
    """E = 1/2 sum_i I_i W_i^2."""
    W = np.atleast_2d(W)
    return 0.5 * (I1 * W[:, 0] ** 2 + I2 * W[:, 1] ** 2 + I3 * W[:, 2] ** 2)


def momentum_sq(W):
    """||L||^2 = sum_i (I_i W_i)^2."""
    W = np.atleast_2d(W)
    return (I1 * W[:, 0]) ** 2 + (I2 * W[:, 1]) ** 2 + (I3 * W[:, 2]) ** 2


# ---------------------------------------------------------------------------
# 1. Training data — INDEPENDENT samples over a box, not one trajectory
#
#    A single Euler-top trajectory is a closed curve confined to the
#    intersection of two level sets, so along it the lifted features satisfy
#    exact algebraic relations and the coefficients stop being identifiable.
#    Sampling W1, W2, W3 independently over [-2, 2]^3 breaks those relations.
#    The condition number of [Theta | 1] is the diagnostic — see the same
#    argument spelled out in mathbio/sir_example.py.
# ---------------------------------------------------------------------------
N_SAMPLES = 8000
BOX = 2.0
rng = np.random.default_rng(SEED)

X = rng.uniform(-BOX, BOX, size=(N_SAMPLES, 3))
X_dot = euler_rhs_np(X[:, 0], X[:, 1], X[:, 2])
print(f"[DATA]  {N_SAMPLES} independent state samples over [-{BOX}, {BOX}]^3")

# ---------------------------------------------------------------------------
# 2. Minimal lift — the three cross-products and nothing else
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["W2*W3", "W3*W1", "W1*W2"]


def top_lift(X: np.ndarray) -> np.ndarray:
    W1, W2, W3 = X[:, 0], X[:, 1], X[:, 2]
    return np.column_stack([W2 * W3, W3 * W1, W1 * W2])


lift = CustomLift(fn=top_lift, output_dim=3, name="euler_top_lift")

Theta = top_lift(X)
cond = np.linalg.cond(np.column_stack([Theta, np.ones(len(Theta))]))
print(f"[DATA]  cond(Theta) = {cond:.1f}   (small => identifiable; "
      f"a single trajectory instead of box samples pushes this into the 1e3+ range)")

# ---------------------------------------------------------------------------
# 3. KANDy model  (+ the identity-lift contrast model)
# ---------------------------------------------------------------------------
print("\n--- Model A: minimal cross-product lift (KAN=[3,3]) ---")
model = KANDy(lift=lift, grid=5, k=3, steps=100, seed=SEED, base_fun="zero")
model.fit(X=X, X_dot=X_dot, val_frac=0.15, test_frac=0.15, lamb=0.0, patience=20)

print("\n--- Model B: identity lift (KAN=[3,3]) — structurally unable to fit ---")
id_lift = CustomLift(fn=lambda X: X, output_dim=3, name="identity")
model_id = KANDy(lift=id_lift, grid=5, k=3, steps=100, seed=SEED, base_fun="zero")
model_id.fit(X=X, X_dot=X_dot, val_frac=0.15, test_frac=0.15, lamb=0.0, patience=20)

pred = model.predict(X)
pred_id = model_id.predict(X)
rmse_lift = np.sqrt(np.mean((pred - X_dot) ** 2))
rmse_id = np.sqrt(np.mean((pred_id - X_dot) ** 2))
raw_r2 = [
    1.0 - np.sum((X_dot[:, i] - pred[:, i]) ** 2)
        / np.sum((X_dot[:, i] - X_dot[:, i].mean()) ** 2)
    for i in range(3)
]
print(f"\n[EVAL]  Network R^2 per equation (cross-product lift): {np.round(raw_r2, 6)}")
print(f"[EVAL]  Derivative RMSE — cross-product lift: {rmse_lift:.6e}")
print(f"[EVAL]  Derivative RMSE — identity lift:      {rmse_id:.6e}"
      f"   ({rmse_id / rmse_lift:.0f}x worse)")
print("[EVAL]  The identity-lift model is a sum of univariate functions of W1,"
      " W2, W3;\n        no separable sum equals W2*W3, so this gap is structural,"
      " not a tuning issue.")

# ---------------------------------------------------------------------------
# 3b. Recovered coefficients — straight off the trained network
#
#     get_A() returns the raw per-edge spline scales, which absorb the spline's
#     own normalisation, so the physical coefficient is the SLOPE of each edge
#     activation.  Both are printed; the slopes are the ones to compare with
#     the truth.  Edge (0, j, i) carries lifted feature j into output i.
# ---------------------------------------------------------------------------
train_theta = torch.tensor(Theta[:2048], dtype=torch.float32)
A_slope = np.zeros((3, 3))
for i in range(3):                                  # output index
    for j in range(3):                              # lifted feature index
        xe, ye = get_edge_activation(model.model_, 0, j, i, X=train_theta)
        A_slope[i, j] = fit_linear(xe, ye)["params"][0]

np.set_printoptions(precision=6, suppress=True)
print(f"\n[EVAL]  model.get_A()  (raw spline scales):\n{model.get_A()}")
print(f"[EVAL]  Edge slopes = recovered A (rows dW_i/dt, cols {FEATURE_NAMES}):\n{A_slope}")
recovered = np.array([A_slope[0, 0], A_slope[1, 1], A_slope[2, 2]])
for lab, tru, rec in zip(["(I2-I3)/I1", "(I3-I1)/I2", "(I1-I2)/I3"],
                         TRUE_COEFFS, recovered):
    print(f"[EVAL]    {lab}:  true {tru:+.6f}   recovered {rec:+.6f}   "
          f"abs err {abs(rec - tru):.2e}")
off_diag = np.max(np.abs(A_slope - np.diag(np.diag(A_slope))))
print(f"[EVAL]  Largest off-diagonal edge slope: {off_diag:.2e}  (should be ~0)")

# ---------------------------------------------------------------------------
# 4. Rollout validation — the polhode, plus conserved-quantity drift
#
#    All of this runs BEFORE get_formula(): symbolic extraction replaces the
#    spline edges in place, so any rollout or edge query afterwards would be
#    measuring the snapped surrogate instead of the trained network.
# ---------------------------------------------------------------------------
DT = 0.02
T_MAX = 40.0
t_eval = np.arange(0.0, T_MAX, DT)
X0 = np.array([1.0, 0.35, 0.55])

sol = solve_ivp(euler_top, [t_eval[0], t_eval[-1]], X0, t_eval=t_eval,
                method="RK45", rtol=1e-12, atol=1e-14)
true_traj = sol.y.T
pred_traj = model.rollout(X0, T=len(true_traj), dt=DT, integrator="rk4")

rmse = np.sqrt(np.mean((pred_traj - true_traj) ** 2))
print(f"\n[EVAL]  Rollout RMSE (T={len(true_traj)} steps, t<={T_MAX}): {rmse:.6e}")

E_true, E_pred = energy(true_traj), energy(pred_traj)
L_true, L_pred = momentum_sq(true_traj), momentum_sq(pred_traj)


def drift(q):
    """Max relative departure of a conserved quantity from its initial value."""
    return np.max(np.abs(q - q[0])) / abs(q[0])


print(f"[EVAL]  Energy  E(0) = {E_true[0]:.6f}   relative drift: "
      f"true {drift(E_true):.2e}, KANDy {drift(E_pred):.2e}")
print(f"[EVAL]  ||L||^2 (0) = {L_true[0]:.6f}   relative drift: "
      f"true {drift(L_true):.2e}, KANDy {drift(L_pred):.2e}")
print("[EVAL]  Both are label-free checks: nothing in training told the model"
      " about E or ||L||^2.")

# A few extra orbits, rescaled onto the SAME momentum ellipsoid as X0 so the
# phase portrait shows the classical polhode family on one surface.
L0 = momentum_sq(X0)[0]
POLHODES = [
    np.array([1.0, 0.35, 0.55]),
    np.array([0.2, 0.95, 0.30]),
    np.array([0.6, 0.75, 0.45]),
    np.array([1.1, 0.10, 0.30]),
    np.array([0.1, 0.20, 0.68]),
]
POLHODES = [w * np.sqrt(L0 / momentum_sq(w)[0]) for w in POLHODES]
orbit_pairs = []
for x0 in POLHODES:
    s = solve_ivp(euler_top, [0, T_MAX], x0, t_eval=t_eval,
                  method="RK45", rtol=1e-12, atol=1e-14).y.T
    p = model.rollout(x0, T=len(s), dt=DT, integrator="rk4")
    orbit_pairs.append((s, p))

# ---------------------------------------------------------------------------
# 5. Symbolic extraction  (LAST — get_formula mutates model in place)
#
#    Every edge here is exactly linear, so the library is just {'x', '0'}:
#    the three diagonal edges must snap to 'x' and the six off-diagonal edges
#    to '0'.  weight_simple=0.0 removes the simplicity pressure that would
#    otherwise let '0' win against a genuine but small-slope edge — the
#    (I1-I2)/I3 = -1/3 edge is the one at risk.
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Extracting formulas ...")
formulas = model.get_formula(
    var_names=FEATURE_NAMES,
    round_places=6,
    lib=["x", "0"],
    r2_threshold=0.80,
    weight_simple=0.0,
)
for lab, f in zip(["dW1/dt", "dW2/dt", "dW3/dt"], formulas):
    print(f"  {lab} = {f}")

r2 = model.score_formula(formulas, X, X_dot, var_names=FEATURE_NAMES)
print(f"[SYMBOLIC] Formula R^2 per equation: {np.round(r2, 6)}")
print(f"[SYMBOLIC] True:  dW1/dt = {C1:+.6f}*W2*W3")
print(f"[SYMBOLIC]        dW2/dt = {C2:+.6f}*W3*W1")
print(f"[SYMBOLIC]        dW3/dt = {C3:+.6f}*W1*W2")

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
use_pub_style()
OUT = "results/EulerTop"
os.makedirs(OUT, exist_ok=True)

# 6a. 3D phase space — polhodes on the momentum sphere, true vs KANDy
fig = plt.figure(figsize=(6, 5.5))
ax = fig.add_subplot(111, projection="3d")
r = np.sqrt(L_true[0])
u = np.linspace(0, 2 * np.pi, 60)
v = np.linspace(0, np.pi, 30)
ax.plot_surface(
    (r / I1) * np.outer(np.cos(u), np.sin(v)),
    (r / I2) * np.outer(np.sin(u), np.sin(v)),
    (r / I3) * np.outer(np.ones_like(u), np.cos(v)),
    color="0.7", alpha=0.12, linewidth=0, rasterized=True,
)
for s, p in orbit_pairs:
    ax.plot(s[:, 0], s[:, 1], s[:, 2], color="#1f77b4", lw=1.4, alpha=0.8)
    ax.plot(p[:, 0], p[:, 1], p[:, 2], color="#d62728", lw=1.0, ls="--")
ax.plot([], [], color="#1f77b4", lw=1.4, label="true")
ax.plot([], [], color="#d62728", lw=1.0, ls="--", label="KANDy")
ax.set_xlabel("$\\omega_1$"); ax.set_ylabel("$\\omega_2$"); ax.set_zlabel("$\\omega_3$")
ax.set_title("Euler top: polhodes on the momentum ellipsoid")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/phase_space_3d.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/phase_space_3d.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6b. Time series of the angular-velocity components
fig, ax = plt.subplots(figsize=(9, 4))
for ci, (lab, c) in enumerate(zip(["$\\omega_1$", "$\\omega_2$", "$\\omega_3$"],
                                  ["#1f77b4", "#2ca02c", "#d62728"])):
    ax.plot(t_eval, true_traj[:, ci], color=c, lw=1.4, label=f"{lab} true")
    ax.plot(t_eval, pred_traj[:, ci], color=c, lw=1.0, ls="--", label=f"{lab} KANDy")
ax.set_xlabel("time")
ax.set_ylabel("body-frame angular velocity")
ax.set_title("Euler top: free precession")
ax.legend(loc="best", fontsize=8, ncol=3)
fig.tight_layout()
fig.savefig(f"{OUT}/timeseries.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6c. Conserved-quantity drift — the free correctness check
fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
axes[0].plot(t_eval, E_true / E_true[0], color="#1f77b4", lw=1.4, label="true")
axes[0].plot(t_eval, E_pred / E_true[0], color="#d62728", lw=1.0, ls="--", label="KANDy")
axes[0].set_ylabel("$E(t) / E(0)$")
axes[0].set_title("Conserved quantities along the rollout")
axes[0].legend(loc="best", fontsize=8)
axes[1].plot(t_eval, L_true / L_true[0], color="#1f77b4", lw=1.4, label="true")
axes[1].plot(t_eval, L_pred / L_true[0], color="#d62728", lw=1.0, ls="--", label="KANDy")
axes[1].set_ylabel("$\\|L\\|^2(t) / \\|L\\|^2(0)$")
axes[1].set_xlabel("time")
axes[1].legend(loc="best", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/conserved_drift.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/conserved_drift.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6d. Loss curves
if getattr(model, "train_results_", None):
    fig, ax = plot_loss_curves(model.train_results_, save=f"{OUT}/loss_curves")
    plt.close(fig)

# 6e. Edge activations — every panel should be a straight line (or flat zero)
fig, _axes = plot_all_edges(
    model.model_, X=train_theta,
    in_var_names=FEATURE_NAMES,
    out_var_names=["dW1/dt", "dW2/dt", "dW3/dt"],
    fits=("linear",),
    save=f"{OUT}/edge_activations",
)
plt.close(fig)

print(f"[FIGS]  Saved {OUT}/")
