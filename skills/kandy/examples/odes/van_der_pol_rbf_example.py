#!/usr/bin/env python3
"""KANDy example: fitting an unknown vector field with RadialBasisLift.

Van der Pol oscillator:

    dx/dt = y
    dy/dt = mu * (1 - x^2) * y - x

The nonlinear damping term mu*(1-x^2)*y contains x^2*y — a product of two
DIFFERENT state variables, so by the separability rule a lift that does not
encode it cannot represent the dynamics.  This script uses Van der Pol as a
stand-in for the common real-world case: *you do not know the RHS*, so you
cannot hand-build the right cross-terms.

When you cannot specify the lift
--------------------------------
``RadialBasisLift`` places Gaussian bumps  exp(-||x - c_k||^2 / 2*sigma^2)
throughout the visited region of state space.  Each feature is a function of
the FULL state vector, not of one coordinate, so cross-interactions are
carried by the features themselves — no prior knowledge of x^2*y required.
With centres from k-means the bumps follow the data density.

This script makes the contrast explicit by fitting twice:

    identity lift  phi(x, y) = [x, y]     -> R^2 ~ 0.37 on dy/dt   (structurally
                                             incapable: no cross-term)
    RBF lift       80 k-means centres     -> R^2 ~ 0.97 on dy/dt

WHAT YOU GIVE UP: interpretability.  RBF features are anonymous bumps, so a
symbolic formula in terms of them ("2.1*phi_37 - 0.8*phi_12 + ...") carries no
physical meaning, and this script deliberately does not extract one.  Validate
an RBF model by ROLLOUT — does it reproduce the attractor? — not by reading
its formula.  RBF also only interpolates where it has data; outside the
sampled region the Gaussians decay and predictions are meaningless.

Use RBF as the "I don't know the physics yet" workhorse: confirm the dynamics
are learnable, inspect the trajectory, then, once you have a hypothesis about
the structure, switch to a CustomLift and re-fit for an interpretable model.

Lift  phi: R^2 -> R^80  (Gaussian RBFs, k-means centres)
KAN:  width = [80, 2],  base_fun='zero'
"""

import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from kandy import KANDy, CustomLift, RadialBasisLift
from kandy.plotting import (
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

MU = 1.5          # nonlinearity strength; larger -> more relaxation-like


def vdp_rhs_np(x, y):
    return np.column_stack([y, MU * (1.0 - x ** 2) * y - x])


def vdp(t, state):
    return vdp_rhs_np(np.array([state[0]]), np.array([state[1]]))[0]


# ---------------------------------------------------------------------------
# 1. Training data — independent samples covering the limit cycle region
#
#    RBF only interpolates where it has data, so the sampling box must cover
#    the region the rollout will visit.  The Van der Pol limit cycle for
#    mu = 1.5 stays within roughly |x| < 2.2, |y| < 3.5.
# ---------------------------------------------------------------------------
N_SAMPLES = 4000
rng = np.random.default_rng(SEED)

X = np.column_stack([
    rng.uniform(-3.0, 3.0, size=N_SAMPLES),
    rng.uniform(-5.0, 5.0, size=N_SAMPLES),
])
X_dot = vdp_rhs_np(X[:, 0], X[:, 1])
print(f"[DATA]  {N_SAMPLES} independent state samples over [-3,3]x[-5,5]")


def equation_r2(model, X, X_dot):
    pred = model.predict(X)
    return [
        1.0 - np.sum((X_dot[:, i] - pred[:, i]) ** 2)
            / np.sum((X_dot[:, i] - X_dot[:, i].mean()) ** 2)
        for i in range(X_dot.shape[1])
    ]


# ---------------------------------------------------------------------------
# 2. Baseline — identity lift, which CANNOT work
#
#    dy/dt needs x^2*y.  A separable KAN over [x, y] can only produce
#    f(x) + g(y), never a product, so this fit must fail on the second
#    equation no matter how long it trains.
# ---------------------------------------------------------------------------
print("\n[BASELINE] Identity lift phi(x, y) = [x, y] ...")
torch.manual_seed(SEED)
np.random.seed(SEED)
baseline = KANDy(
    lift=CustomLift(fn=lambda X: X, output_dim=2, name="identity"),
    grid=5, k=3, steps=150, seed=SEED, base_fun="zero",
)
baseline.fit(X=X, X_dot=X_dot, lamb=0.0, patience=30, verbose=False)
base_r2 = equation_r2(baseline, X, X_dot)
print(f"[BASELINE] R^2 per equation: {np.round(base_r2, 5)}")
print("[BASELINE] dy/dt is unlearnable without a cross-term — as the theory says.")

# ---------------------------------------------------------------------------
# 3. RBF lift — no structural knowledge required
# ---------------------------------------------------------------------------
N_CENTERS = 80
print(f"\n[RBF]  Fitting RadialBasisLift(n_centers={N_CENTERS}, "
      f"center_method='kmeans') ...")

torch.manual_seed(SEED)
np.random.seed(SEED)
lift = RadialBasisLift(n_centers=N_CENTERS, center_method="kmeans")
model = KANDy(lift=lift, grid=5, k=3, steps=120, seed=SEED, base_fun="zero")
model.fit(X=X, X_dot=X_dot, val_frac=0.15, test_frac=0.15, lamb=0.0, patience=40)

rbf_r2 = equation_r2(model, X, X_dot)
print(f"[RBF]  R^2 per equation: {np.round(rbf_r2, 5)}")
print(f"[RBF]  dy/dt: {base_r2[1]:.3f} (identity) -> {rbf_r2[1]:.3f} (RBF)")

# ---------------------------------------------------------------------------
# 4. Validation is by ROLLOUT, not by formula
#
#    Start off the limit cycle and check the model is drawn onto the same
#    closed orbit, with the right amplitude and period.
# ---------------------------------------------------------------------------
DT     = 0.01
T_MAX  = 60.0
t_eval = np.arange(0.0, T_MAX, DT)
X0     = [0.5, 0.5]           # inside the limit cycle

sol = solve_ivp(vdp, [t_eval[0], t_eval[-1]], X0, t_eval=t_eval,
                method="RK45", rtol=1e-10, atol=1e-12)
true_traj = sol.y.T
pred_traj = model.rollout(np.array(X0), T=len(true_traj), dt=DT, integrator="rk4")

rmse = np.sqrt(np.mean((pred_traj - true_traj) ** 2))
print(f"\n[EVAL]  Rollout RMSE (T={len(true_traj)} steps): {rmse:.6f}")


def cycle_stats(traj, tail_frac=0.5):
    """Amplitude and period measured on the settled part of the trajectory."""
    tail = traj[int(len(traj) * tail_frac):]
    amp = tail[:, 0].max() - tail[:, 0].min()
    x = tail[:, 0] - tail[:, 0].mean()
    crossings = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
    period = float(np.mean(np.diff(crossings)) * DT) if len(crossings) >= 2 else np.nan
    return amp, period


amp_t, per_t = cycle_stats(true_traj)
amp_p, per_p = cycle_stats(pred_traj)
print(f"[EVAL]  Limit-cycle amplitude — true {amp_t:.4f}, KANDy {amp_p:.4f}  "
      f"({100 * abs(amp_p - amp_t) / amp_t:.2f}% error)")
print(f"[EVAL]  Limit-cycle period    — true {per_t:.4f}, KANDy {per_p:.4f}  "
      f"({100 * abs(per_p - per_t) / per_t:.2f}% error)")
print("[EVAL]  Pointwise RMSE is the WRONG metric for a limit cycle: a small "
      "period error\n        accumulates into phase drift, so RMSE grows with "
      "time even when the orbit\n        is right.  Judge amplitude, period "
      "and the closed shape in phase space.")
print("[EVAL]  No symbolic extraction: RBF features are anonymous bumps, so a "
      "formula\n        in terms of them would not be physically meaningful.")

# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/VanDerPolRBF", exist_ok=True)

# 5a. Phase portrait with the RBF centres overlaid
fig, ax = plt.subplots(figsize=(5.5, 5))
centers = getattr(lift, "centers_", getattr(lift, "centers", None))
if centers is not None:
    centers = np.asarray(centers)
    ax.scatter(centers[:, 0], centers[:, 1], s=12, color="#7f7f7f", alpha=0.5,
               label=f"{N_CENTERS} RBF centres")
ax.plot(true_traj[:, 0], true_traj[:, 1], color="#1f77b4", lw=1.6, label="true")
ax.plot(pred_traj[:, 0], pred_traj[:, 1], color="#d62728", lw=1.1, ls="--",
        label="KANDy (RBF)")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_title("Van der Pol limit cycle recovered without knowing the RHS")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("results/VanDerPolRBF/phase_portrait.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 5b. Time series
fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
for ax, ci, lab in zip(axes, [0, 1], ["$x$", "$y$"]):
    ax.plot(t_eval, true_traj[:, ci], color="#1f77b4", lw=1.2, label="true")
    ax.plot(t_eval, pred_traj[:, ci], color="#d62728", lw=1.0, ls="--", label="KANDy")
    ax.set_ylabel(lab)
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("time")
fig.suptitle("Van der Pol: relaxation oscillations", fontsize=12)
fig.tight_layout()
fig.savefig("results/VanDerPolRBF/timeseries.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 5c. Identity vs RBF, per equation
fig, ax = plt.subplots(figsize=(6, 4))
idx = np.arange(2)
ax.bar(idx - 0.18, base_r2, width=0.36, color="#7f7f7f", label="identity lift")
ax.bar(idx + 0.18, rbf_r2, width=0.36, color="#d62728", label="RBF lift")
ax.set_xticks(idx)
ax.set_xticklabels(["$dx/dt$", "$dy/dt$"])
ax.set_ylabel("$R^2$")
ax.set_ylim(0, 1.05)
ax.set_title("The cross-term is what the identity lift cannot reach")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig("results/VanDerPolRBF/lift_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 5d. Trajectory error
fig, ax = plot_trajectory_error(
    true_traj, pred_traj, t=t_eval, save="results/VanDerPolRBF/trajectory_error",
)
plt.close(fig)

# 5e. Loss curves
if getattr(model, "train_results_", None):
    fig, ax = plot_loss_curves(
        model.train_results_, save="results/VanDerPolRBF/loss_curves",
    )
    plt.close(fig)

print("[FIGS]  Saved results/VanDerPolRBF/")
