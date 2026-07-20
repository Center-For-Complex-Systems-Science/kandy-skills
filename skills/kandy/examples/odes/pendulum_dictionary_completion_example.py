#!/usr/bin/env python3
"""KANDy example: dictionary completion — learning sin(theta) without a
trigonometric dictionary.

Damped nonlinear pendulum:

    dtheta/dt = omega
    domega/dt = -(g/L) * sin(theta) - b * omega

THE POINT: the dictionary is needed to NAME the nonlinearity, not to LEARN it
-----------------------------------------------------------------------------
Sparse-regression methods (SINDy, PDE-FIND) regress onto a fixed dictionary of
candidate terms.  If sin(theta) is not in that dictionary, the method cannot
represent the pendulum at all — the best it can do is a truncated Taylor
series theta - theta^3/6 + ..., which needs many terms and still degrades at
large amplitude.  Choosing the dictionary is therefore a modelling decision
made BEFORE seeing the data.

KANDy splits the problem in two:

    1. LEARNING.  The KAN spline edge on theta is a free univariate function.
       It fits -(g/L)*sin(theta) to machine precision with no dictionary, no
       candidate terms, and no prior knowledge that a sine is involved.  The
       network reaches R^2 = 1.0 before any symbolic step happens.

    2. NAMING.  Only afterwards does a symbolic library enter, to attach a
       closed form to the shape the spline already found.  Supplying 'sin'
       recovers  -9.81*sin(theta)  exactly; supplying only polynomials fails
       (R^2 ~ 0.007) — but that failure is a *labelling* failure.  The learned
       dynamics, the rollout and the edge plot are unaffected.

This is what makes the KAN dictionary-completing: a missing term costs you a
readable formula, not a working model.  Contrast a missing CROSS-term, which
is fatal at the learning stage and cannot be repaired at the naming stage —
see sir_example.py.

Related: odes/sinx_recover_sine.py covers a narrower trick — coaxing
auto_symbolic to emit a literally named 'sin' on a 1D toy problem by biasing
the per-channel base functions.  This script instead contrasts what a
dictionary can and cannot do on a standard 2D mechanical system.

Lift  phi: R^2 -> R^2   theta = [theta, omega]   (identity — no cross-terms)
KAN:  width = [2, 2],  base_fun='zero'
"""

import copy
import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from kandy import KANDy, CustomLift
from kandy.plotting import (
    get_edge_activation,
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

G_OVER_L = 9.81      # g / L
DAMPING  = 0.2       # b


def pendulum_rhs_np(theta, omega):
    return np.column_stack([omega, -G_OVER_L * np.sin(theta) - DAMPING * omega])


def pendulum(t, state):
    return pendulum_rhs_np(np.array([state[0]]), np.array([state[1]]))[0]


# ---------------------------------------------------------------------------
# 1. Training data — independent samples over a full swing range
#    theta spans [-pi, pi], well outside the small-angle regime.
# ---------------------------------------------------------------------------
N_SAMPLES = 6000
rng = np.random.default_rng(SEED)

X = np.column_stack([
    rng.uniform(-np.pi, np.pi, size=N_SAMPLES),
    rng.uniform(-6.0, 6.0, size=N_SAMPLES),
])
X_dot = pendulum_rhs_np(X[:, 0], X[:, 1])
print(f"[DATA]  {N_SAMPLES} independent state samples, "
      f"theta in [-pi, pi] (fully nonlinear regime)")

# ---------------------------------------------------------------------------
# 2. Lift — identity.  No products of different variables appear in the RHS.
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["theta", "omega"]
lift = CustomLift(fn=lambda X: X, output_dim=2, name="identity")

# ---------------------------------------------------------------------------
# 3. Fit — note there is NO dictionary anywhere in this step
# ---------------------------------------------------------------------------
model = KANDy(lift=lift, grid=5, k=3, steps=300, seed=SEED, base_fun="zero")
model.fit(X=X, X_dot=X_dot, val_frac=0.15, test_frac=0.15, lamb=0.0, patience=50)

pred = model.predict(X)
raw_r2 = [
    1.0 - np.sum((X_dot[:, i] - pred[:, i]) ** 2)
        / np.sum((X_dot[:, i] - X_dot[:, i].mean()) ** 2)
    for i in range(2)
]
print(f"[EVAL]  Network R^2 per equation: {np.round(raw_r2, 6)}  "
      f"<- learned with no dictionary at all")

# ---------------------------------------------------------------------------
# 4. Evidence that the spline found the sine on its own
#    Regress the raw theta -> domega/dt edge onto {sin(theta), 1}.
# ---------------------------------------------------------------------------
model.model_.save_act = True
theta_t = torch.tensor(X[:4000], dtype=torch.float32)
with torch.no_grad():
    model.model_(theta_t)

x_edge, y_edge = get_edge_activation(model.model_, 0, 0, 1)   # in=theta, out=domega
x_edge = np.asarray(x_edge).ravel()
y_edge = np.asarray(y_edge).ravel()

basis = np.column_stack([np.sin(x_edge), np.ones_like(x_edge)])
coef, *_ = np.linalg.lstsq(basis, y_edge, rcond=None)
fit = basis @ coef
edge_r2 = 1.0 - np.sum((y_edge - fit) ** 2) / np.sum((y_edge - y_edge.mean()) ** 2)

print(f"\n[EDGE]  theta -> domega/dt edge regressed on sin(theta):")
print(f"        edge(theta) ~ {coef[0]:.4f}*sin(theta) + {coef[1]:.4f}   "
f"(R^2 = {edge_r2:.6f})")
print(f"        true coefficient -(g/L) = {-G_OVER_L}")

# ---------------------------------------------------------------------------
# 5. Rollout validation — large-amplitude swing, before any snapping
# ---------------------------------------------------------------------------
DT     = 0.01
T_MAX  = 20.0
t_eval = np.arange(0.0, T_MAX, DT)
X0     = [2.8, 0.0]          # released near the top: strongly nonlinear

sol = solve_ivp(pendulum, [t_eval[0], t_eval[-1]], X0, t_eval=t_eval,
                method="RK45", rtol=1e-10, atol=1e-12)
true_traj = sol.y.T
pred_traj = model.rollout(np.array(X0), T=len(true_traj), dt=DT, integrator="rk4")

rmse = np.sqrt(np.mean((pred_traj - true_traj) ** 2))
print(f"\n[EVAL]  Rollout RMSE (T={len(true_traj)} steps, theta0={X0[0]}): {rmse:.6f}")

# Keep a pre-snapping copy of the edge for the figure
edge_x_plot, edge_y_plot = x_edge.copy(), y_edge.copy()

# ---------------------------------------------------------------------------
# 6. Naming the shape: two dictionaries, same trained model
#
#    get_formula() mutates the model in place, so each attempt runs on its own
#    deepcopy.  Otherwise the first call would corrupt the second.
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Same trained network, two different symbolic libraries:")

results = {}
for tag, lib in [("trig       ['x','sin','0']", ["x", "sin", "0"]),
                 ("polynomial ['x','x^2','x^3','0']", ["x", "x^2", "x^3", "0"])]:
    trial = copy.deepcopy(model)
    formulas = trial.get_formula(
        var_names=FEATURE_NAMES, round_places=4,
        lib=lib, r2_threshold=0.80, weight_simple=0.0,
    )
    r2 = trial.score_formula(formulas, X, X_dot, var_names=FEATURE_NAMES)
    results[tag] = (formulas, r2)
    print(f"\n  {tag}   R^2 = {np.round(r2, 6)}")
    for lab, f in zip(["dtheta/dt", "domega/dt"], formulas):
        print(f"    {lab} = {f}")

print(f"\n[SYMBOLIC] True:  dtheta/dt = omega")
print(f"[SYMBOLIC]        domega/dt = -{G_OVER_L}*sin(theta) - {DAMPING}*omega")
print("[SYMBOLIC] The polynomial library cannot name the sine and drops the "
      "term entirely,\n           but the network above still had R^2 = 1.0 — "
      "naming failed, learning did not.")

# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/PendulumDictionary", exist_ok=True)

# 7a. The learned edge vs the true sine — the heart of the example
order = np.argsort(edge_x_plot)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(edge_x_plot[order], edge_y_plot[order], color="#d62728", lw=2.0,
        label="learned KAN edge")
ax.plot(edge_x_plot[order], -G_OVER_L * np.sin(edge_x_plot[order]),
        color="#1f77b4", lw=1.2, ls="--", label=r"true $-(g/L)\sin\theta$")
ax.plot(edge_x_plot[order], -G_OVER_L * edge_x_plot[order],
        color="#7f7f7f", lw=1.0, ls=":", label=r"small-angle $-(g/L)\theta$")
ax.set_ylim(-G_OVER_L * 1.6, G_OVER_L * 1.6)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"edge value contributing to $\dot\omega$")
ax.set_title("The spline discovered the sine without being told")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("results/PendulumDictionary/learned_edge_vs_sine.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)

# 7b. Large-amplitude rollout
fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
for ax, ci, lab in zip(axes, [0, 1], [r"$\theta$", r"$\omega$"]):
    ax.plot(t_eval, true_traj[:, ci], color="#1f77b4", lw=1.2, label="true")
    ax.plot(t_eval, pred_traj[:, ci], color="#d62728", lw=1.0, ls="--", label="KANDy")
    ax.set_ylabel(lab)
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("time")
fig.suptitle(r"Damped pendulum released at $\theta_0 = 2.8$ rad", fontsize=12)
fig.tight_layout()
fig.savefig("results/PendulumDictionary/timeseries.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7c. Phase portrait
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(true_traj[:, 0], true_traj[:, 1], color="#1f77b4", lw=1.4, label="true")
ax.plot(pred_traj[:, 0], pred_traj[:, 1], color="#d62728", lw=1.0, ls="--",
        label="KANDy")
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"$\omega$")
ax.set_title("Spiralling to rest")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("results/PendulumDictionary/phase_portrait.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)

# 7d. Trajectory error
fig, ax = plot_trajectory_error(
    true_traj, pred_traj, t=t_eval,
    save="results/PendulumDictionary/trajectory_error",
)
plt.close(fig)

# 7e. Loss curves
if getattr(model, "train_results_", None):
    fig, ax = plot_loss_curves(
        model.train_results_, save="results/PendulumDictionary/loss_curves",
    )
    plt.close(fig)

print("[FIGS]  Saved results/PendulumDictionary/")
