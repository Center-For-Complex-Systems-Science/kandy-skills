#!/usr/bin/env python3
"""KANDy example: the Epileptor — seizure dynamics with switching nonlinearities.

The Epileptor (Jirsa et al., Brain 2014) is the standard phenomenological model
of focal seizures: a fast subsystem (x1, y1) generating spike-wave discharges,
an intermediate subsystem (x2, y2) generating slow-wave events, and an ultra-slow
"permittivity" variable z that drives seizure onset and, crucially, terminates
it.  The 0.002*g(x1) memory coupling is dropped here (see the module note at the
bottom); everything else is the published model.

    dx1/dt = y1 - f1(x1, x2, z) - z + I1
    dy1/dt = 1 - 5*x1^2 - y1
    dz/dt  = (1/tau0) * (4*(x1 - x0) - z)                       tau0 = 2857
    dx2/dt = -y2 + x2 - x2^3 + I2 - 0.3*(z - 3.5)
    dy2/dt = (1/tau2) * (-y2 + f2(x2))                          tau2 = 10

    f1 = x1^3 - 3*x1^2               if x1 <  0
         (x2 - 0.6*(z-4)^2) * x1     if x1 >= 0
    f2 = 0                           if x2 <  -0.25
         6*(x2 + 0.25)               if x2 >= -0.25

Two techniques this example exists to teach
-------------------------------------------

1. **SWITCHING NONLINEARITIES GO IN THE LIFT AS GATED FEATURES.**
   f1 and f2 are piecewise.  No polynomial dictionary can represent them, and
   the KAN cannot manufacture them either — not only because of the switch, but
   because the active branch of f1 is `x1*x2` and `x1*(z-4)^2`, products of
   *different* states.  The fix is to multiply each branch by its own indicator
   and hand the result over as a precomputed feature:

       m = 1{x1 <  0}      p = 1{x1 >= 0}      n = 1{x2 >= -0.25}

       theta = [x1, y1, z, x2, y2,
                m*x1^2, p*x1^2, m*x1^3,             <- gated fast subsystem
                p*x1*x2, p*x1*(z-4)^2,              <- gated CROSS-terms
                x2^3, n*(x2+0.25)]                  (12 features)

   Note `x1^2` is split into `m*x1^2` and `p*x1^2` rather than passed whole.
   Passing both `x1^2` and `m*x1^2` makes them ~70% correlated (they agree
   wherever x1 < 0), which measurably degrades the symbolic extraction.  When a
   gate is in play, gate *everything* that the gate touches.

   Given this lift every one of the five equations is exactly LINEAR in the
   features, so lib=["x", "0"] is the right symbolic setting.

2. **STIFF SYSTEMS NEED PER-EQUATION TARGET SCALING.**
   tau0 = 2857 makes dz/dt some three orders of magnitude smaller than the fast
   equations (this script prints the measured ratio — around 6000x over the
   sampling box).  An unweighted MSE over all five outputs therefore *ignores*
   the z equation: its entire contribution to the loss sits below the residual
   of the fast equations.  This script fits twice to show it:

       naive  (raw targets)      dz/dt R^2 ~ 0.86, and the recovered
                                 coefficients are off by 45-75%
       scaled (X_dot / std)      dz/dt R^2 ~ 1.00, and the -z/tau0 feedback
                                 comes out to four significant figures

   That accuracy is not cosmetic: -z/tau0 is the negative feedback that ends
   the seizure, and it is the smallest coefficient in the whole system.
   Whichever equation carries your slow variable is exactly the one an
   unweighted loss will neglect.

   The recipe is to fit on `X_dot / s` with `s = X_dot.std(0)`, then multiply
   the formula for row i by `s[i]` to get physical coefficients.

Sampling
--------
States are drawn from a BOX covering the attractor rather than from the
trajectory.  On the trajectory the lifted features are strongly correlated and
the coefficients come out wrong even at R^2 = 1 — the same non-identifiability
as the conserved population in `mathbio/sir_example.py`.  Derivatives are the
exact RHS evaluated at the sampled states.

An honest limit: gated features are zero-inflated
-------------------------------------------------
Roughly half of every gated feature's samples are exactly 0, which is a point
mass the spline grid adapts poorly to.  The consequence is visible below: the
NETWORK is excellent (R^2 ~ 1.0 on all five equations, and the rollout below
reproduces the seizure), and `dz/dt` — the one equation built purely from
ungated features — extracts exactly.  But `get_formula` recovers the gated
coefficients only approximately, and drops some outright.  Mitigations used
here: a coarse grid (grid=3) and validating by ROLLOUT rather than by formula,
as in `odes/van_der_pol_rbf_example.py`.  Do not read the gated coefficients as
the result of this example; read the rollout.  Concretely, `dy1/dt` should be
`1 - 5*m*x1^2 - 5*p*x1^2 - y1` and the snap loses both quadratic terms, while
the same network integrates for 1500 time units with the correct spike count
and a maximum z error of ~0.006.

Runtime: ~4 minutes.
KAN:  width = [12, 5],  base_fun='zero'
"""

import os
import time
import copy
import numpy as np
import torch
import sympy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from kandy import KANDy, CustomLift
from kandy.plotting import plot_all_edges, plot_loss_curves, use_pub_style

# ---------------------------------------------------------------------------
# 0. Reproducibility and Epileptor parameters (Jirsa et al. 2014)
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

X0, Y0 = -1.6, 1.0          # excitability (x0 > -2.05 => seizures) and offset
TAU0, TAU2 = 2857.0, 10.0   # permittivity and intermediate time constants
I1, I2 = 3.1, 0.45          # external inputs

STATE_NAMES = ["x1", "y1", "z", "x2", "y2"]
EQ_NAMES = ["dx1/dt", "dy1/dt", "dz/dt", "dx2/dt", "dy2/dt"]


def f1(x1, x2, z):
    """Fast-subsystem switching nonlinearity."""
    return np.where(x1 < 0, x1 ** 3 - 3 * x1 ** 2, (x2 - 0.6 * (z - 4) ** 2) * x1)


def f2(x2):
    """Intermediate-subsystem switching nonlinearity."""
    return np.where(x2 < -0.25, 0.0, 6 * (x2 + 0.25))


def epileptor_rhs(state):
    """Exact RHS for an (N, 5) array of states -> (N, 5)."""
    x1, y1, z, x2, y2 = state.T
    return np.stack([
        y1 - f1(x1, x2, z) - z + I1,
        Y0 - 5 * x1 ** 2 - y1,
        (1.0 / TAU0) * (4 * (x1 - X0) - z),
        -y2 + x2 - x2 ** 3 + I2 - 0.3 * (z - 3.5),
        (1.0 / TAU2) * (-y2 + f2(x2)),
    ], axis=1)


# ---------------------------------------------------------------------------
# 1. Simulate a seizure train
# ---------------------------------------------------------------------------
T_END, DT_SIM = 6000.0, 0.05
t_grid = np.arange(0, T_END, DT_SIM)

print(f"[SIM]  Epileptor, x0={X0}, tau0={TAU0}, T={T_END:.0f}")
sol = solve_ivp(lambda t, s: epileptor_rhs(s[None, :])[0], [0, T_END],
                [0.0, -5.0, 3.0, 0.0, 0.0], t_eval=t_grid,
                method="LSODA", rtol=1e-8, atol=1e-10, max_step=1.0)
if not sol.success:
    raise RuntimeError(f"integration failed: {sol.message}")
traj_true = sol.y.T
lfp_true = -traj_true[:, 0] + traj_true[:, 3]     # the usual LFP proxy
n_spikes = int((np.diff((traj_true[:, 0] > 0).astype(int)) == 1).sum())
print(f"[SIM]  {len(traj_true)} samples, {n_spikes} spikes, "
      f"z sweeps {traj_true[:, 2].min():.3f} -> {traj_true[:, 2].max():.3f}")

# ---------------------------------------------------------------------------
# 2. Box-sampled training data (see docstring: NOT the trajectory)
# ---------------------------------------------------------------------------
N_TRAIN = 4000
span = np.ptp(traj_true, axis=0)
lo, hi = traj_true.min(axis=0) - 0.1 * span, traj_true.max(axis=0) + 0.1 * span
rng = np.random.default_rng(SEED)
states = rng.uniform(lo, hi, size=(N_TRAIN, 5))
targets = epileptor_rhs(states)

scale = targets.std(axis=0)
print(f"[DATA]  target std per equation: {np.round(scale, 6)}")
print(f"[DATA]  timescale separation: fastest/slowest = "
      f"{scale.max() / scale.min():.0f}x  <- why scaling is required")

# ---------------------------------------------------------------------------
# 3. The gated lift
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["x1", "y1", "z", "x2", "y2", "m*x1^2", "p*x1^2", "m*x1^3",
                 "p*x1*x2", "p*x1*(z-4)^2", "x2^3", "n*(x2+0.25)"]


def phi_np(Z):
    x1, y1, z, x2, y2 = Z.T
    m = (x1 < 0).astype(float)
    p = 1.0 - m
    n = (x2 >= -0.25).astype(float)
    return np.stack([x1, y1, z, x2, y2, m * x1 ** 2, p * x1 ** 2, m * x1 ** 3,
                     p * x1 * x2, p * x1 * (z - 4) ** 2, x2 ** 3,
                     n * (x2 + 0.25)], axis=1)


def phi_torch(Z):
    x1, y1, z, x2, y2 = Z[:, 0], Z[:, 1], Z[:, 2], Z[:, 3], Z[:, 4]
    m = (x1 < 0).float()
    p = 1.0 - m
    n = (x2 >= -0.25).float()
    return torch.stack([x1, y1, z, x2, y2, m * x1 ** 2, p * x1 ** 2, m * x1 ** 3,
                        p * x1 * x2, p * x1 * (z - 4) ** 2, x2 ** 3,
                        n * (x2 + 0.25)], dim=1)


Theta = phi_np(states)
print(f"[DATA]  {N_TRAIN} box samples, cond(Theta) = "
      f"{np.linalg.cond(np.column_stack([Theta, np.ones(N_TRAIN)])):.1f}")
zero_frac = np.mean(Theta[:, 5:] == 0.0)
print(f"[DATA]  {100 * zero_frac:.0f}% of the gated features are exactly zero "
      "(the zero-inflation noted in the docstring)")


def build_model():
    lift = CustomLift(fn=phi_np, output_dim=len(FEATURE_NAMES),
                      torch_fn=phi_torch, name="epileptor")
    # grid=3: the RHS is linear in the features, and a coarse grid copes far
    # better with the zero-inflated gated coordinates than the default grid=5.
    return KANDy(lift=lift, grid=3, k=3, steps=80, seed=SEED, base_fun="zero")


def eval_r2(m, s):
    pred = m.predict(states) * s
    return np.array([
        1.0 - np.sum((targets[:, i] - pred[:, i]) ** 2)
        / np.sum((targets[:, i] - targets[:, i].mean()) ** 2)
        for i in range(5)
    ])


# ---------------------------------------------------------------------------
# 4. The naive fit — raw targets, and the z equation is lost
# ---------------------------------------------------------------------------
print("\n[NAIVE]  Fitting on RAW targets (no per-equation scaling) ...")
naive = build_model()
naive.fit(X=states, X_dot=targets, val_frac=0.15, test_frac=0.15, lamb=0.0,
          opt="LBFGS", lr=1.0, batch=-1, stop_grid_update_step=5, verbose=False)
r2_naive = eval_r2(naive, np.ones(5))
print(f"[NAIVE]  R^2 per equation: {np.round(r2_naive, 5)}")
naive_dz = sympy.expand(naive.get_formula(
    var_names=FEATURE_NAMES, round_places=6, lib=["x", "0"],
    r2_threshold=0.80, weight_simple=0.0)[2])
print(f"[NAIVE]  dz/dt = {naive_dz}")
print(f"[NAIVE]  true  = {4/TAU0:.6f}*x1 - {1/TAU0:.6f}*z + {-4*X0/TAU0:.6f}")
print("[NAIVE]  ^ same features, but the coefficients on the seizure clock are\n"
      "         badly distorted — the slow equation is invisible to the loss.")

# ---------------------------------------------------------------------------
# 5. The scaled fit — divide each target by its own std
# ---------------------------------------------------------------------------
print("\n[SCALED] Fitting on X_dot / std ...")
t0 = time.time()
model = build_model()
model.fit(X=states, X_dot=targets / scale, val_frac=0.15, test_frac=0.15,
          lamb=0.0, opt="LBFGS", lr=1.0, batch=-1, stop_grid_update_step=5,
          verbose=False)
r2_scaled = eval_r2(model, scale)
print(f"[SCALED] R^2 per equation: {np.round(r2_scaled, 6)}  ({time.time()-t0:.0f}s)")
print(f"[SCALED] dz/dt R^2: {r2_naive[2]:.4f} (naive) -> {r2_scaled[2]:.6f} (scaled)")

# ---------------------------------------------------------------------------
# 6. Validation by ROLLOUT — the real test (before get_formula mutates edges)
# ---------------------------------------------------------------------------
scale_t = torch.tensor(scale, dtype=torch.float32)


def learned_rhs(v):
    """Physical RHS from the scaled network, evaluated on a single state."""
    with torch.no_grad():
        z_in = torch.tensor(v[None, :], dtype=torch.float32)
        return (model.model_(phi_torch(z_in))[0] * scale_t).numpy().astype(float)


T_ROLL = 1500.0
n_roll = int(T_ROLL / DT_SIM)
print(f"\n[EVAL]  Autonomous rollout for t={T_ROLL:.0f} ({n_roll} RK4 steps) ...")
t0 = time.time()
x = traj_true[0].copy()
roll = np.empty((n_roll + 1, 5))
roll[0] = x
for i in range(n_roll):
    k1 = learned_rhs(x)
    k2 = learned_rhs(x + 0.5 * DT_SIM * k1)
    k3 = learned_rhs(x + 0.5 * DT_SIM * k2)
    k4 = learned_rhs(x + DT_SIM * k3)
    x = x + (DT_SIM / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    if not np.isfinite(x).all():
        raise RuntimeError(f"rollout diverged at t={i * DT_SIM:.1f}")
    roll[i + 1] = x

ref = traj_true[:n_roll + 1]
spikes_roll = int((np.diff((roll[:, 0] > 0).astype(int)) == 1).sum())
spikes_ref = int((np.diff((ref[:, 0] > 0).astype(int)) == 1).sum())
z_err = np.abs(roll[:, 2] - ref[:, 2]).max()
print(f"[EVAL]  rollout {time.time() - t0:.0f}s")
print(f"[EVAL]  spike count: KANDy {spikes_roll} vs true {spikes_ref}")
print(f"[EVAL]  x1 range: KANDy [{roll[:,0].min():.3f}, {roll[:,0].max():.3f}]  "
      f"true [{ref[:,0].min():.3f}, {ref[:,0].max():.3f}]")
print(f"[EVAL]  max |z_KANDy - z_true| over the run: {z_err:.5f}  "
      "(the slow seizure clock is tracked)")

theta_t = torch.tensor(Theta[:2048], dtype=torch.float32)
model_for_formula = copy.deepcopy(model)

# ---------------------------------------------------------------------------
# 7. Symbolic extraction — exact for dz/dt, approximate on the gated edges
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Extracting (coefficients rescaled by the target std) ...")
formulas = model_for_formula.get_formula(
    var_names=FEATURE_NAMES, round_places=6,
    lib=["x", "0"], r2_threshold=0.80, weight_simple=0.0,
)
for i, expr in enumerate(formulas):
    print(f"  {EQ_NAMES[i]} = {sympy.expand(expr * float(scale[i]))}")
print("\n[SYMBOLIC] Ground truth:")
print(f"  dz/dt  = {4/TAU0:.6f}*x1 - {1/TAU0:.6f}*z + {-4*X0/TAU0:.6f}   <- z coefficient to 4 s.f.")
print("  dx1/dt = y1 - z + 3.1 - m*x1^3 + 3*m*x1^2 - p*x1*x2 + 0.6*p*x1*(z-4)^2")
print("  dy1/dt = 1 - 5*m*x1^2 - 5*p*x1^2 - y1")
print("  dx2/dt = x2 - x2^3 - y2 - 0.3*z + 1.5")
print("  dy2/dt = -0.1*y2 + 0.6*n*(x2+0.25)")
print("[SYMBOLIC] The gated equations are recovered only approximately — see the\n"
      "           zero-inflation note in the docstring.  The rollout above, not\n"
      "           these coefficients, is the validation of this model.")

# ---------------------------------------------------------------------------
# 8. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/Epileptor", exist_ok=True)

# 8a. The seizure train: LFP with the slow permittivity variable underneath
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
axes[0].plot(t_grid, lfp_true, lw=0.4, color="#1f77b4")
axes[0].set_ylabel("LFP  $(-x_1 + x_2)$")
axes[0].set_title("Epileptor seizure train")
axes[1].plot(t_grid, traj_true[:, 2], lw=1.2, color="#d62728")
axes[1].set_ylabel("$z$ (permittivity)")
axes[1].set_xlabel("time")
fig.tight_layout()
fig.savefig("results/Epileptor/seizure_train.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 8b. Rollout vs truth
t_roll = np.arange(n_roll + 1) * DT_SIM
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
axes[0].plot(t_roll, ref[:, 0], lw=0.5, color="#1f77b4", label="true")
axes[0].plot(t_roll, roll[:, 0], lw=0.5, color="#d62728", alpha=0.7, label="KANDy")
axes[0].set_ylabel("$x_1$"); axes[0].legend(fontsize=8, loc="upper right")
axes[0].set_title(f"Autonomous rollout — {spikes_roll} spikes vs {spikes_ref} true")
axes[1].plot(t_roll, ref[:, 2], lw=1.4, color="#1f77b4", label="true")
axes[1].plot(t_roll, roll[:, 2], lw=1.4, color="#d62728", ls="--", label="KANDy")
axes[1].set_ylabel("$z$"); axes[1].set_xlabel("time")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/Epileptor/rollout.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 8c. The scaling lesson: predicted vs true dz/dt, naive against scaled
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
preds = [naive.predict(states)[:, 2], model.predict(states)[:, 2] * scale[2]]
for ax, p, ttl, r2v in zip(axes, preds,
                           ["naive (raw targets)", "scaled (X_dot / std)"],
                           [r2_naive[2], r2_scaled[2]]):
    ax.scatter(targets[:, 2], p, s=4, alpha=0.3, color="#1f77b4")
    lim = [targets[:, 2].min(), targets[:, 2].max()]
    ax.plot(lim, lim, "k--", lw=1.0)
    ax.set_xlabel("true $dz/dt$")
    ax.set_title(f"{ttl}\n$R^2$ = {r2v:.4f}", fontsize=10)
axes[0].set_ylabel("predicted $dz/dt$")
fig.suptitle("Why stiff systems need per-equation target scaling", fontsize=12)
fig.tight_layout()
fig.savefig("results/Epileptor/scaling_lesson.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 8d. Fast-subsystem phase portrait, true vs rollout
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.plot(ref[:, 0], ref[:, 1], lw=0.4, color="#1f77b4", alpha=0.6, label="true")
ax.plot(roll[:, 0], roll[:, 1], lw=0.4, color="#d62728", alpha=0.6, label="KANDy")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$y_1$")
ax.set_title("Fast subsystem attractor")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/Epileptor/phase_portrait.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 8e. Loss curves and edge activations
if getattr(model, "train_results_", None):
    fig, ax = plot_loss_curves(model.train_results_,
                               save="results/Epileptor/loss_curves")
    plt.close(fig)

fig, axes = plot_all_edges(
    model.model_, X=theta_t,
    in_var_names=FEATURE_NAMES,
    out_var_names=EQ_NAMES,
    save="results/Epileptor/edge_activations",
)
plt.close(fig)

print("[FIGS]  Saved results/Epileptor/")

# ---------------------------------------------------------------------------
# Note on the dropped g(x1) coupling
# ---------------------------------------------------------------------------
# The published Epileptor adds 0.002*g(x1) to dx2/dt, where g is a low-pass
# filter of x1 obeying dg/dt = -0.01*(g - x1).  It is omitted here because the
# coefficient 0.002 sits at the level of the extraction noise on the gated
# edges, so including it would add a sixth state whose contribution could not be
# identified anyway.  To restore it: append g to the state, add the g column to
# the lift, and add the filter as a sixth output equation.
