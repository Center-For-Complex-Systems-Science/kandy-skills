#!/usr/bin/env python3
"""KANDy example: FitzHugh–Nagumo neuron — recovering an edge that
``auto_symbolic`` cannot snap.

The FitzHugh–Nagumo model, a two-variable reduction of Hodgkin–Huxley:

    dv/dt = v - v^3/3 - w + I_ext        (fast voltage-like variable)
    dw/dt = eps * (v + a - b*w)          (slow recovery variable)

With I_ext above threshold the system settles onto a relaxation-oscillation
limit cycle: slow recovery punctuated by fast spikes.

Lesson 1 — no cross-terms, so the lift is the identity
------------------------------------------------------
Unlike SIR (S*I) or Lotka–Volterra (N1*N2), this RHS contains no product of
two DIFFERENT state variables; v and w appear only additively.  The single
nonlinearity -v^3/3 is a power of one variable, which a spline edge learns
directly.  So phi(v, w) = [v, w] suffices and the KAN does genuine univariate
function discovery rather than regression on hand-built features.

Do NOT be tempted to add v^3 as a third lift coordinate.  v -> v^3 is
invertible, so an edge acting on v^3 can represent any function of v: the two
edges become functionally dependent, the decomposition is non-unique, and the
recovered coefficients are meaningless (try it — the cubic coefficient comes
out ~10x too small).  Lift coordinates must be functionally INDEPENDENT.

Lesson 2 — when per-edge snapping structurally cannot work
----------------------------------------------------------
PyKAN's ``auto_symbolic`` replaces each edge with ONE library primitive under
an affine composition,  c * f(a*x + b) + d.  The v edge here must represent
v - v^3/3, and that is not expressible in this form for any f in the library:

    c*(a*v + b)^3 + d  =  c*a^3 v^3 + 3c*a^2*b v^2 + 3c*a*b^2 v + (c*b^3 + d)

Matching the missing v^2 term forces b = 0, which also kills the v term.  So
``get_formula`` silently drops the cubic edge and returns  dv/dt = 0.49 - w.
(Contrast Lotka–Volterra, where the edges are quadratics: c*(a*x+b)^2 + d has
enough freedom for a linear + quadratic edge, so snapping works there.)

The fix is to fit each edge as a POLYNOMIAL instead of one primitive, then sum
the edges.  A single-layer KAN output is exactly

    output_j = sum_i  edge_ij(theta_i)

so summing per-edge polynomial fits reconstructs the governing equation.  Here
that recovers  dv/dt = -v^3/3 + v - w + 1/2  exactly.

Reach for this whenever an edge plot looks like a clean polynomial but
``get_formula`` returns a suspiciously short expression.

Lift  phi: R^2 -> R^2   theta = [v, w]   (identity)
KAN:  width = [2, 2],  base_fun='zero'
"""

import os
import numpy as np
import sympy as sp
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from kandy import KANDy, CustomLift
from kandy.plotting import (
    get_edge_activation,
    plot_all_edges,
    plot_trajectory_error,
    plot_loss_curves,
    use_pub_style,
)

# ---------------------------------------------------------------------------
# 0. Reproducibility and parameters
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

A_FHN = 0.7
B_FHN = 0.8
EPS   = 0.08
I_EXT = 0.5      # above threshold -> sustained spiking


def fhn_rhs_np(v, w):
    return np.column_stack([
        v - v ** 3 / 3.0 - w + I_EXT,
        EPS * (v + A_FHN - B_FHN * w),
    ])


def fhn(t, state):
    return fhn_rhs_np(np.array([state[0]]), np.array([state[1]]))[0]


# ---------------------------------------------------------------------------
# 1. Training data — independent samples covering the limit cycle and beyond
# ---------------------------------------------------------------------------
N_SAMPLES = 6000
rng = np.random.default_rng(SEED)

V_RANGE = (-2.5, 2.5)
W_RANGE = (-1.0, 2.0)
X = np.column_stack([
    rng.uniform(*V_RANGE, size=N_SAMPLES),
    rng.uniform(*W_RANGE, size=N_SAMPLES),
])
X_dot = fhn_rhs_np(X[:, 0], X[:, 1])
print(f"[DATA]  {N_SAMPLES} independent state samples, v in {V_RANGE}, w in {W_RANGE}")

# ---------------------------------------------------------------------------
# 2. Lift — identity: this system has no cross-terms
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["v", "w"]
lift = CustomLift(fn=lambda X: X, output_dim=2, name="identity")

# ---------------------------------------------------------------------------
# 3. KANDy model
# ---------------------------------------------------------------------------
model = KANDy(lift=lift, grid=5, k=3, steps=300, seed=SEED, base_fun="zero")
model.fit(X=X, X_dot=X_dot, val_frac=0.15, test_frac=0.15, lamb=0.0, patience=50)

pred = model.predict(X)
raw_r2 = [
    1.0 - np.sum((X_dot[:, i] - pred[:, i]) ** 2)
        / np.sum((X_dot[:, i] - X_dot[:, i].mean()) ** 2)
    for i in range(2)
]
print(f"[EVAL]  Network R^2 per equation: {np.round(raw_r2, 6)}")

# ---------------------------------------------------------------------------
# 4. Edge-wise polynomial reconstruction — the method that works
#
#    NOTE ON ORDERING: get_formula() MUTATES the fitted model — auto_symbolic
#    replaces each spline edge with its snapped symbolic surrogate in place.
#    Anything done afterwards (rollout, predict, edge plots) sees the snapped
#    model, not the trained one.  Here that would be fatal, since snapping
#    destroys the cubic edge (network R^2 drops 1.00 -> 0.50).  So we do the
#    reconstruction, the rollout and the figures FIRST, and demonstrate the
#    snapping failure at the very end of the script.
# ---------------------------------------------------------------------------
POLY_DEGREE = 3
SYMS = [sp.Symbol(n) for n in FEATURE_NAMES]

model.model_.save_act = True
theta_t = torch.tensor(X[:4000], dtype=torch.float32)
with torch.no_grad():
    model_out = model.model_(theta_t).cpu().numpy()

print(f"\n[SYMBOLIC] Edge-wise polynomial reconstruction (degree {POLY_DEGREE}):")
recovered = []
for j in range(len(FEATURE_NAMES)):
    expr = sp.Integer(0)
    recon = np.zeros(len(theta_t))
    for i in range(len(FEATURE_NAMES)):
        x_e, y_e = get_edge_activation(model.model_, 0, i, j)
        x_e = np.asarray(x_e).ravel()
        y_e = np.asarray(y_e).ravel()

        coeffs = np.polyfit(x_e, y_e, POLY_DEGREE)
        edge_r2 = 1.0 - np.sum((y_e - np.polyval(coeffs, x_e)) ** 2) / (
            np.sum((y_e - y_e.mean()) ** 2) + 1e-15
        )
        print(f"    edge  d{FEATURE_NAMES[j]}/dt <- {FEATURE_NAMES[i]}:  "
              f"poly R^2 = {edge_r2:.6f}")

        recon += np.polyval(coeffs, theta_t[:, i].numpy())
        expr += sum(float(coeffs[k]) * SYMS[i] ** (POLY_DEGREE - k)
                    for k in range(POLY_DEGREE + 1))

    # The single-layer KAN output is exactly the sum of its edges — verify it.
    max_err = np.abs(recon - model_out[:, j]).max()
    print(f"    sum of edges vs network output: max |diff| = {max_err:.2e}")
    recovered.append(sp.nsimplify(sp.expand(expr), rational=False, tolerance=1e-3))

for lab, e in zip(["dv/dt", "dw/dt"], recovered):
    print(f"  {lab} = {e}")

r2 = model.score_formula(recovered, X, X_dot, var_names=FEATURE_NAMES)
print(f"[SYMBOLIC] Formula R^2 per equation: {np.round(r2, 6)}")
print(f"[SYMBOLIC] True:  dv/dt = v - v**3/3 - w + {I_EXT}")
print(f"[SYMBOLIC]        dw/dt = {EPS}*v - {EPS * B_FHN:.3f}*w + {EPS * A_FHN:.3f}")

# ---------------------------------------------------------------------------
# 5. Rollout validation — the relaxation-oscillation limit cycle
# ---------------------------------------------------------------------------
DT     = 0.05
T_MAX  = 200.0
t_eval = np.arange(0.0, T_MAX, DT)
X0     = [-1.0, 1.0]

sol = solve_ivp(fhn, [t_eval[0], t_eval[-1]], X0, t_eval=t_eval,
                method="RK45", rtol=1e-10, atol=1e-12)
true_traj = sol.y.T
pred_traj = model.rollout(np.array(X0), T=len(true_traj), dt=DT, integrator="rk4")

rmse = np.sqrt(np.mean((pred_traj - true_traj) ** 2))
print(f"\n[EVAL]  Rollout RMSE (T={len(true_traj)} steps): {rmse:.6f}")


def spike_count(v, thresh=1.0):
    """Count upward threshold crossings — a phase-insensitive limit-cycle check."""
    above = v > thresh
    return int(np.sum(above[1:] & ~above[:-1]))


print(f"[EVAL]  Spikes over {T_MAX:g} time units — "
      f"true {spike_count(true_traj[:, 0])}, KANDy {spike_count(pred_traj[:, 0])}")

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/FitzHughNagumo", exist_ok=True)

# 6a. Voltage and recovery traces
fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
for ax, ci, lab in zip(axes, [0, 1], ["v (voltage)", "w (recovery)"]):
    ax.plot(t_eval, true_traj[:, ci], color="#1f77b4", lw=1.2, label="true")
    ax.plot(t_eval, pred_traj[:, ci], color="#d62728", lw=1.0, ls="--", label="KANDy")
    ax.set_ylabel(lab)
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("time")
fig.suptitle("FitzHugh–Nagumo: spiking rollout", fontsize=12)
fig.tight_layout()
fig.savefig("results/FitzHughNagumo/timeseries.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6b. Phase plane with nullclines
fig, ax = plt.subplots(figsize=(5.5, 5))
vv = np.linspace(*V_RANGE, 400)
ax.plot(vv, vv - vv ** 3 / 3.0 + I_EXT, color="#7f7f7f", lw=1.0, ls=":",
        label=r"$\dot v = 0$ nullcline")
ax.plot(vv, (vv + A_FHN) / B_FHN, color="#2ca02c", lw=1.0, ls=":",
        label=r"$\dot w = 0$ nullcline")
ax.plot(true_traj[:, 0], true_traj[:, 1], color="#1f77b4", lw=1.4, label="true")
ax.plot(pred_traj[:, 0], pred_traj[:, 1], color="#d62728", lw=1.0, ls="--", label="KANDy")
ax.set_xlim(*V_RANGE)
ax.set_ylim(*W_RANGE)
ax.set_xlabel("$v$")
ax.set_ylabel("$w$")
ax.set_title("Phase plane and limit cycle")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig("results/FitzHughNagumo/phase_plane.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6c. Trajectory error
fig, ax = plot_trajectory_error(
    true_traj, pred_traj, t=t_eval, save="results/FitzHughNagumo/trajectory_error",
)
plt.close(fig)

# 6d. Loss curves
if getattr(model, "train_results_", None):
    fig, ax = plot_loss_curves(
        model.train_results_, save="results/FitzHughNagumo/loss_curves",
    )
    plt.close(fig)

# 6e. Edge activations — the v -> dv/dt edge is the cubic that would not snap
fig, axes = plot_all_edges(
    model.model_, X=theta_t[:2048],
    in_var_names=FEATURE_NAMES,
    out_var_names=["dv/dt", "dw/dt"],
    fits=("poly",),
    poly_degree=POLY_DEGREE,
    save="results/FitzHughNagumo/edge_activations",
)
plt.close(fig)

print("[FIGS]  Saved results/FitzHughNagumo/")

# ---------------------------------------------------------------------------
# 7. Appendix: the auto_symbolic failure, demonstrated
#
#    RUN THIS LAST — get_formula() rewrites the model's edges in place, so the
#    model is no longer usable for rollout or plotting afterwards.
# ---------------------------------------------------------------------------
print("\n[APPENDIX] Standard get_formula on the same model "
      "(expected to drop the cubic edge):")
snapped = model.get_formula(
    var_names=FEATURE_NAMES, round_places=3,
    lib=["x", "x^2", "x^3", "0"], r2_threshold=0.80, weight_simple=0.0,
)
for lab, f in zip(["dv/dt", "dw/dt"], snapped):
    print(f"  {lab} = {f}")
snapped_r2 = model.score_formula(snapped, X, X_dot, var_names=FEATURE_NAMES)
print(f"  R^2 = {np.round(snapped_r2, 4)}")
print("  dv/dt lost the cubic: c*f(a*v + b) + d cannot represent v - v^3/3.")

post_snap = model.predict(X)
post_r2 = [
    1.0 - np.sum((X_dot[:, i] - post_snap[:, i]) ** 2)
        / np.sum((X_dot[:, i] - X_dot[:, i].mean()) ** 2)
    for i in range(2)
]
print(f"  Network R^2 after snapping: {np.round(post_r2, 4)}  "
      f"(was {np.round(raw_r2, 4)}) — get_formula mutated the model.")
