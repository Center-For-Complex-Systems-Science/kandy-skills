#!/usr/bin/env python3
"""KANDy example: Hénon map.

The Hénon map is a 2D discrete dynamical system:
    x_{n+1} = 1 - a*x_n^2 + y_n
    y_{n+1} = b*x_n
with a=1.4, b=0.3.

For discrete maps there is no derivative to predict; the KAN directly learns
the one-step-ahead map  (x_n, y_n) -> (x_{n+1}, y_{n+1}).  The Koopman lift
is the identity (raw states), and the KAN learns the full nonlinear map.

Note: the Hénon map uses the KANDy model in "map mode" (not ODE mode).
The API is identical — fit() takes X (current state) and X_dot (next state).
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from kandy import KANDy, CustomLift
from _utils import attractor_tv, ensure_outdir, print_check, save_metrics

# ---------------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# 1. Data generation
# ---------------------------------------------------------------------------
A_PARAM = 1.4
B_PARAM = 0.3
N_TOTAL = 10_000
BURN_IN = 1_000


def generate_henon(n_total=N_TOTAL, burn_in=BURN_IN, x0=0.1, y0=0.0):
    xy = np.zeros((n_total + burn_in, 2), dtype=np.float32)
    xy[0] = [x0, y0]
    for i in range(n_total + burn_in - 1):
        x, y = xy[i]
        xy[i + 1] = [1 - A_PARAM * x**2 + y, B_PARAM * x]
    return xy[burn_in:]


series = generate_henon()
X = series[:-1]   # current state   (N, 2)
Y = series[1:]    # next state       (N, 2)  — this is our "X_dot" in map mode

# Chronological 60 / 20 / 20 split
N = len(X)
n_test = int(N * 0.20)
n_val  = int(N * 0.20)
n_train = N - n_test - n_val

print(f"[DATA]  train={n_train}, val={n_val}, test={n_test}")

# ---------------------------------------------------------------------------
# 2. Lift definition
# ---------------------------------------------------------------------------
# For the Hénon map the lift is the identity — the KAN learns the full map.
# We use a CustomLift that wraps the identity function.
identity_lift = CustomLift(
    fn=lambda X: X,
    output_dim=2,
    name="identity",
)

# ---------------------------------------------------------------------------
# 3. KANDy model
# ---------------------------------------------------------------------------
model = KANDy(
    lift=identity_lift,
    grid=5,
    k=3,
    steps=2000,
    seed=SEED,
    base_fun=lambda x: torch.exp(-(x**2)),   # RBF base (from research code)
)

# In map mode: X = current state, X_dot = next state
# val_frac / test_frac are computed from the combined data here;
# we pass the full dataset and let KANDy split chronologically.
model.fit(
    X=X,
    X_dot=Y,
    val_frac=0.20,
    test_frac=0.20,
    lamb=0.0,
)

# ---------------------------------------------------------------------------
# 5. Autoregressive rollout (map iteration)
# ---------------------------------------------------------------------------
def rollout_map(model, x0, T):
    """Iterate the learned map: x_{n+1} = model(x_n)."""
    traj = [x0.copy()]
    x = x0.copy()
    for _ in range(T - 1):
        x = model.predict(x)
    traj.append(x)
    # full rollout
    traj = [x0.copy()]
    x = x0.copy()
    for _ in range(T - 1):
        x = model.predict(x)
        traj.append(x.copy())
    return np.array(traj)


test_start = series[N - n_test]
T_test = n_test + 1
learned_traj = rollout_map(model, test_start, T_test)
true_traj = series[N - n_test : N - n_test + T_test]

rmse = np.sqrt(np.mean((learned_traj - true_traj) ** 2))
print(f"\n[EVAL]  Rollout RMSE over {T_test} steps: {rmse:.6f}")

# ---------------------------------------------------------------------------
# 5b. Verification
#
#     The Henon map is chaotic, so an autoregressive rollout decorrelates from
#     the true orbit no matter how good the model is — rollout RMSE is recorded
#     but NOT used as a pass criterion.  What must hold is the ONE-STEP map on
#     held-out data, plus the learned orbit landing on the same attractor
#     (compared through its invariant measure, not pointwise).
# ---------------------------------------------------------------------------
X_test, Y_test = X[N - n_test:], Y[N - n_test:]
pred_test = model.predict(X_test)
onestep_r2 = [
    float(1.0 - np.sum((Y_test[:, i] - pred_test[:, i]) ** 2)
          / np.sum((Y_test[:, i] - Y_test[:, i].mean()) ** 2))
    for i in range(2)
]
print(f"[EVAL]  One-step held-out R^2 per component: {np.round(onestep_r2, 6)}")

# Invariant-measure agreement, with a calibrated null: two finite samples of
# the same attractor never give TV = 0, so the null (true orbit vs itself) is
# what the measured distance has to be judged against.
tv_distance, tv_null = attractor_tv(true_traj, learned_traj)
print(f"[EVAL]  Attractor occupancy TV distance: {tv_distance:.4f}  "
      f"(null, true-vs-true: {tv_null:.4f})")

THRESHOLDS = {
    "onestep_r2_min": 0.999,     # the learned map itself must be right
    "tv_distance_vs_null_max": 2.5,   # relative to the true-vs-true null
}
passed = print_check(
    min(onestep_r2) >= THRESHOLDS["onestep_r2_min"]
    and tv_distance <= THRESHOLDS["tv_distance_vs_null_max"] * tv_null,
    "Henon one-step map and attractor occupancy",
)
RESULT_DIR = ensure_outdir("results/Henon")
save_metrics(RESULT_DIR / "metrics.json", {
    "task": "Henon",
    "category": "maps",
    "seed": SEED,
    "n_train": int(n_train),
    "n_test": int(n_test),
    "a": A_PARAM,
    "b": B_PARAM,
    "onestep_r2": onestep_r2,
    "tv_distance": tv_distance,
    "tv_null": tv_null,
    "rollout_rmse": float(rmse),
    "thresholds": THRESHOLDS,
    "passed": passed,
})

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------

DROP = 200   # drop transients for attractor plot
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(true_traj[DROP:, 0], true_traj[DROP:, 1], s=1.5, alpha=0.6,
           c="#1f77b4", label="True Hénon", rasterized=True)
ax.scatter(learned_traj[DROP:, 0], learned_traj[DROP:, 1], s=1.5, alpha=0.6,
           c="#ff7f0e", label="KANDy", rasterized=True)
ax.set_xlabel(r"$x_n$")
ax.set_ylabel(r"$y_n$")
ax.legend(framealpha=0.9, edgecolor="black")
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
fig.savefig("results/Henon/attractor_overlay.png", dpi=300, bbox_inches="tight")
fig.savefig("results/Henon/attractor_overlay.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print("[FIGS]  Saved results/Henon/attractor_overlay.{png,pdf}")

# ---------------------------------------------------------------------------
# 7. Symbolic extraction  (LAST — get_formula mutates the model in place)
#
#     This used to run immediately after fit(), which meant the rollout and the
#     one-step scores above were measuring the snapped surrogate rather than the
#     trained network.  Nothing may be evaluated after this line.
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Extracting formulas...")
try:
    formulas = model.get_formula(var_names=["x_n", "y_n"])
    print("  x_{n+1} =", formulas[0])
    print("  y_{n+1} =", formulas[1])
    print(f"  [TRUE] x_(n+1) = 1 - {A_PARAM}*x_n**2 + y_n,  y_(n+1) = {B_PARAM}*x_n")
except Exception as e:
    print(f"  Symbolic extraction failed: {e}")
