#!/usr/bin/env python3
"""KANDy example: SIR epidemic model.

The Kermack–McKendrick compartmental model:

    dS/dt = -beta * S * I
    dI/dt =  beta * S * I - gamma * I
    dR/dt =  gamma * I

with transmission rate beta and recovery rate gamma.

Two lessons, both of which generalise to most math-bio models
-------------------------------------------------------------

1. THE CROSS-TERM MUST BE IN THE LIFT.
   The only nonlinearity is the mass-action incidence  beta*S*I — a bilinear
   product of two *different* state variables.  The KAN is separable: each
   spline sees exactly one lifted coordinate, so it can never manufacture S*I
   from raw S and I.  Supply it explicitly:  theta = [S, I, R, S*I].

2. SAMPLE THE VECTOR FIELD OFF THE CONSERVED MANIFOLD.
   SIR conserves total population: S + I + R = const.  Training only on
   trajectories therefore gives data on a 2D plane inside R^3, where the lifted
   features are *linearly dependent* (R is an exact affine function of S and I).
   The regression is then non-identifiable — many different coefficient sets
   reproduce the data equally well, and the recovered constants come out wrong
   even though the fit looks perfect.  Sampling states independently over a box
   makes the features independent and the coefficients identifiable.

   This is the general rule: any conservation law (total population, total
   probability, energy) creates a degeneracy the lift inherits.

Symbolic settings
-----------------
Every RHS term is *linear* in the lifted features, so restrict the symbolic
library to ``lib=["x", "0"]``.  With the full default library the snapper fits
spurious quadratics to near-zero edges and R^2 collapses.

Lift  phi: R^3 -> R^4     theta = [S, I, R, S*I]
KAN:  width = [4, 3],  base_fun='zero' (pure spline, no SiLU bias)
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
    plot_trajectory_error,
    plot_loss_curves,
    use_pub_style,
)

# ---------------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BETA  = 0.9      # transmission rate
GAMMA = 0.2      # recovery rate


def sir_rhs_np(S, I, R):
    """Vectorised SIR right-hand side."""
    infection = BETA * S * I
    return np.column_stack([-infection, infection - GAMMA * I, GAMMA * I])


def sir(t, state):
    S, I, R = state
    return sir_rhs_np(np.array([S]), np.array([I]), np.array([R]))[0]


# ---------------------------------------------------------------------------
# 1. Training data — independent samples of the state space
#
#    NOT trajectory data: see lesson 2 in the docstring.  We sample S, I, R
#    independently so the lifted features are linearly independent.
# ---------------------------------------------------------------------------
N_SAMPLES = 6000
rng = np.random.default_rng(SEED)

X = rng.uniform(0.0, 1.0, size=(N_SAMPLES, 3))
X_dot = sir_rhs_np(X[:, 0], X[:, 1], X[:, 2])

print(f"[DATA]  {N_SAMPLES} independent state samples")

# ---------------------------------------------------------------------------
# 2. Koopman lift  phi: R^3 -> R^4    theta = [S, I, R, S*I]
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["S", "I", "R", "SI"]


def sir_lift(X: np.ndarray) -> np.ndarray:
    S, I, R = X[:, 0], X[:, 1], X[:, 2]
    return np.column_stack([S, I, R, S * I])


# Condition number of the lifted design matrix — the identifiability diagnostic.
# On trajectory data (S+I+R const) this blows up; on box samples it stays small.
Theta = sir_lift(X)
cond = np.linalg.cond(np.column_stack([Theta, np.ones(len(Theta))]))
print(f"[DATA]  cond(Theta) = {cond:.1f}   (small => identifiable; "
      f"trajectory-only data pushes this into the thousands)")

lift = CustomLift(fn=sir_lift, output_dim=4, name="sir_lift")

# ---------------------------------------------------------------------------
# 3. KANDy model  (single-layer KAN: width=[4, 3])
# ---------------------------------------------------------------------------
model = KANDy(lift=lift, grid=5, k=3, steps=300, seed=SEED, base_fun="zero")
model.fit(X=X, X_dot=X_dot, val_frac=0.15, test_frac=0.15, lamb=0.0, patience=50)

pred = model.predict(X)
raw_r2 = [
    1.0 - np.sum((X_dot[:, i] - pred[:, i]) ** 2)
        / np.sum((X_dot[:, i] - X_dot[:, i].mean()) ** 2)
    for i in range(3)
]
print(f"[EVAL]  Network R^2 per equation: {np.round(raw_r2, 6)}")

# ---------------------------------------------------------------------------
# 4. Rollout validation — reproduce a real outbreak curve
#
#    The model was never trained on trajectory data; this checks that the
#    learned vector field integrates correctly on the physical manifold.
#
#    Do this BEFORE get_formula(): auto_symbolic rewrites the model's spline
#    edges in place, so everything after it sees the snapped surrogate rather
#    than the trained network.  (See fitzhugh_nagumo_example.py, where the
#    difference is fatal.)
# ---------------------------------------------------------------------------
DT     = 0.05
T_MAX  = 60.0
t_eval = np.arange(0.0, T_MAX, DT)
sol = solve_ivp(sir, [t_eval[0], t_eval[-1]], [0.99, 0.01, 0.0],
                t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-12)
true_traj = sol.y.T

pred_traj = model.rollout(true_traj[0], T=len(true_traj), dt=DT, integrator="rk4")

rmse = np.sqrt(np.mean((pred_traj - true_traj) ** 2))
peak_true = true_traj[:, 1].max()
peak_pred = pred_traj[:, 1].max()
print(f"\n[EVAL]  Rollout RMSE (T={len(true_traj)} steps): {rmse:.6f}")
print(f"[EVAL]  Peak infected — true {peak_true:.4f}, KANDy {peak_pred:.4f}")

# ---------------------------------------------------------------------------
# 5. Symbolic extraction — linear library only
#
#    Every RHS term is linear in the lifted features, so lib=["x", "0"].  With
#    the default library the snapper fits spurious quadratics to near-zero
#    edges and R^2 collapses.
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Extracting formulas ...")
formulas = model.get_formula(
    var_names=FEATURE_NAMES,
    round_places=3,
    lib=["x", "0"],
    r2_threshold=0.80,
)
for lab, f in zip(["dS/dt", "dI/dt", "dR/dt"], formulas):
    print(f"  {lab} = {f}")

r2 = model.score_formula(formulas, X, X_dot, var_names=FEATURE_NAMES)
print(f"[SYMBOLIC] Formula R^2 per equation: {np.round(r2, 6)}")
print(f"[SYMBOLIC] True:  dS/dt = -{BETA}*SI      "
      f"dI/dt = {BETA}*SI - {GAMMA}*I      dR/dt = {GAMMA}*I")

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/SIR", exist_ok=True)

# 6a. Outbreak curves: true vs rollout
fig, ax = plt.subplots(figsize=(9, 4))
for ci, (lab, c) in enumerate(zip(["S", "I", "R"], ["#1f77b4", "#d62728", "#2ca02c"])):
    ax.plot(t_eval, true_traj[:, ci], color=c, lw=1.4, label=f"{lab} true")
    ax.plot(t_eval, pred_traj[:, ci], color=c, lw=1.0, ls="--", label=f"{lab} KANDy")
ax.set_xlabel("time")
ax.set_ylabel("population fraction")
ax.set_title("SIR epidemic: rollout vs truth")
ax.legend(loc="center right", fontsize=8, ncol=3)
fig.tight_layout()
fig.savefig("results/SIR/outbreak.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6b. Trajectory error
fig, ax = plot_trajectory_error(
    true_traj, pred_traj, t=t_eval, save="results/SIR/trajectory_error",
)
plt.close(fig)

# 6c. Loss curves
if getattr(model, "train_results_", None):
    fig, ax = plot_loss_curves(model.train_results_, save="results/SIR/loss_curves")
    plt.close(fig)

# 6d. Edge activations — the S*I edge should be a clean straight line
train_theta = torch.tensor(Theta[:2048], dtype=torch.float32)
fig, axes = plot_all_edges(
    model.model_, X=train_theta,
    in_var_names=FEATURE_NAMES,
    out_var_names=["dS/dt", "dI/dt", "dR/dt"],
    save="results/SIR/edge_activations",
)
plt.close(fig)

print("[FIGS]  Saved results/SIR/")
