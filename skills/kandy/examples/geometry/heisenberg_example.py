#!/usr/bin/env python3
"""KANDy example: sub-Riemannian geodesic flow on the Heisenberg group H^1.

The Heisenberg group is the simplest non-abelian nilpotent Lie group.  Its
horizontal distribution is spanned by the two left-invariant fields
X = d/dx - (y/2) d/dz and Y = d/dy + (x/2) d/dz, and [X, Y] = d/dz: the
bracket generates the missing direction.  Carrying the two horizontal
controls (u, v) as constant extra state coordinates makes the geodesic flow
autonomous on R^5:

    dx/dt = u
    dy/dt = v
    dz/dt = x*v - y*u        <-- contact form / Lie-bracket term
    du/dt = 0
    dv/dt = 0

With (u, v) constant the (x, y) motion is a straight line, and z accumulates
twice the signed area swept by the position vector — the trajectory is that
line lifted onto a ruled surface in z.

THE LESSON: the minimal counterexample to separability
-------------------------------------------------------
A single-layer KAN computes a SEPARABLE sum, dz/dt ~ sum_i a_i psi_i(s_i),
one univariate spline per input coordinate.  The Heisenberg term

    dz/dt = x*v - y*u

is exactly an antisymmetric bilinear form on (x, y, u, v) — nothing else, no
higher-order clutter — so it is the smallest honest counterexample to
separability that a dynamical system offers.

1. Why no separable sum works, at ANY width or grid size (zero-set
   corollary).  The zero set of a separable sum sum_i psi_i(s_i) = 0 is,
   locally, a union of graphs foliated by coordinate directions; for the
   affine case it is a union of HYPERPLANES.  The zero set of x*v - y*u is
   the quadric cone {x*v = y*u} in R^4, an irreducible quadric that is NOT a
   union of hyperplanes.  A separable model therefore cannot even get the
   SIGN pattern of dz/dt right, let alone its magnitude.  This is a
   structural obstruction: refining the grid, widening the layer or training
   longer moves the error by a few percent and then stops.  The grid sweep
   in section 4b demonstrates exactly that.

2. The fix is four features:  theta = [u, v, x*v, y*u].  Each output is then
   ONE or TWO linear features, and the KAN only has to learn a linear map:

       dx/dt = theta_1                dz/dt = theta_3 - theta_4
       dy/dt = theta_2                du/dt = dv/dt = 0

   Note what the lift does NOT contain: no x, no y, no x^2, no powers of a
   single variable.  x and y enter the RHS only through the bilinear pairing,
   so only the pairing belongs in phi.  Adding x and y alongside x*v would
   violate rule 2 of references/lifts.md (functional independence).

3. The failure is LOCALISED, and the per-component RMSE proves it.  The
   identity-lift model learns dx/dt = u and dy/dt = v essentially exactly
   (those ARE separable — each is a single coordinate) and fails only on
   dz/dt.  That is the clean signature of a missing cross-term: the linear
   rows are fine, the bilinear row is not.  When you see this pattern in your
   own fit, the diagnosis is the lift, not the optimiser.

Lift phi: R^5 -> R^4   theta = [u, v, x*v, y*u]
KAN: width = [4, 5],  base_fun='zero'     (compare: identity lift, [5, 5])
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

STATE_NAMES = ["x", "y", "z", "u", "v"]
OUT_NAMES = ["dx/dt", "dy/dt", "dz/dt", "du/dt", "dv/dt"]
RESULTS_DIR = "results/Heisenberg"

print("[MODEL] Heisenberg H^1 geodesic flow, state (x, y, z, u, v)")
print("[MODEL]   dx/dt = u,  dy/dt = v,  dz/dt = x*v - y*u,  du/dt = dv/dt = 0")
print("[MODEL]   zero set of dz/dt is the quadric cone {x*v = y*u} -> not separable")


def heisenberg_rhs(S: np.ndarray) -> np.ndarray:
    """Vectorised RHS.  S is (N, 5) = (x, y, z, u, v)."""
    x, y, u, v = S[:, 0], S[:, 1], S[:, 3], S[:, 4]
    return np.column_stack([u, v, x * v - y * u, np.zeros_like(u), np.zeros_like(u)])


def heisenberg_ivp(t, s):
    return heisenberg_rhs(np.asarray(s, dtype=float)[None, :])[0]


# ---------------------------------------------------------------------------
# 1. Training data — independent uniform samples over a box
#
#    Sampling the five coordinates independently (not along trajectories)
#    keeps the lifted features linearly independent; along a single geodesic
#    u and v are constant and x*v, y*u would be perfectly collinear with time.
#    z never appears in the RHS, so it is sampled but carries no information —
#    the identity-lift model has to discover that its z edge is dead.
# ---------------------------------------------------------------------------
N_SAMPLES = 6000
rng = np.random.default_rng(SEED)

X = np.column_stack([
    rng.uniform(-2.0, 2.0, N_SAMPLES),   # x
    rng.uniform(-2.0, 2.0, N_SAMPLES),   # y
    rng.uniform(-2.0, 2.0, N_SAMPLES),   # z  (does not enter the RHS)
    rng.uniform(-1.0, 1.0, N_SAMPLES),   # u
    rng.uniform(-1.0, 1.0, N_SAMPLES),   # v
])
X_dot = heisenberg_rhs(X)

print(f"\n[DATA]  {N_SAMPLES} independent samples: "
      f"x,y,z in [-2,2], u,v in [-1,1]")
print(f"[DATA]  |dz/dt| range: [{np.abs(X_dot[:, 2]).min():.4f}, "
      f"{np.abs(X_dot[:, 2]).max():.4f}], std = {X_dot[:, 2].std():.4f}")

# ---------------------------------------------------------------------------
# 2. Lifts
#
#    (A) identity  phi(s) = s                    -> R^5, the naive choice
#    (B) engineered  theta = [u, v, x*v, y*u]    -> R^4, the bracket features
# ---------------------------------------------------------------------------
RAW_FEATURE_NAMES = STATE_NAMES
raw_lift = CustomLift(fn=lambda S: S, output_dim=5, name="identity_h1")

ENG_FEATURE_NAMES = ["u", "v", "x*v", "y*u"]


def heisenberg_lift(S: np.ndarray) -> np.ndarray:
    x, y, u, v = S[:, 0], S[:, 1], S[:, 3], S[:, 4]
    return np.column_stack([u, v, x * v, y * u])


eng_lift = CustomLift(fn=heisenberg_lift, output_dim=4, name="heisenberg_bracket")

# Identifiability diagnostic — see references/lifts.md.  A large condition
# number means the lifted features are near-dependent and the recovered
# coefficients would be meaningless even at R^2 = 1.
Theta = heisenberg_lift(X)
cond = np.linalg.cond(np.column_stack([Theta, np.ones(len(Theta))]))
print(f"[DATA]  cond([Theta, 1]) = {cond:.3f}  (near 1 -> fully identifiable)")

# ---------------------------------------------------------------------------
# 3. Models — train both, identical hyperparameters
# ---------------------------------------------------------------------------
STEPS = 200

print("\n--- Model A: identity lift (KAN = [5, 5]) ---")
model_raw = KANDy(lift=raw_lift, grid=5, k=3, steps=STEPS, seed=SEED,
                  base_fun="zero")
model_raw.fit(X=X, X_dot=X_dot, val_frac=0.15, test_frac=0.15, lamb=0.0,
              patience=50)

print("\n--- Model B: engineered bracket lift (KAN = [4, 5]) ---")
model_eng = KANDy(lift=eng_lift, grid=5, k=3, steps=STEPS, seed=SEED,
                  base_fun="zero")
model_eng.fit(X=X, X_dot=X_dot, val_frac=0.15, test_frac=0.15, lamb=0.0,
              patience=50)

# ---------------------------------------------------------------------------
# 4a. One-step accuracy, broken out per output component
#
#    This is the sharp part of the demonstration: dx/dt and dy/dt are single
#    coordinates and therefore separable, so BOTH models nail them.  Only
#    dz/dt distinguishes the two — that is where the missing cross-term lives.
# ---------------------------------------------------------------------------
pred_raw = model_raw.predict(X)
pred_eng = model_eng.predict(X)


def per_component_rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2, axis=0))


rmse_comp_raw = per_component_rmse(pred_raw, X_dot)
rmse_comp_eng = per_component_rmse(pred_eng, X_dot)

print("\n[EVAL]  One-step RMSE per output component")
print(f"[EVAL]  {'component':>8}  {'identity lift':>14}  {'bracket lift':>13}")
for name, a, b in zip(OUT_NAMES, rmse_comp_raw, rmse_comp_eng):
    print(f"[EVAL]  {name:>8}  {a:>14.6f}  {b:>13.6f}")
print(f"[EVAL]  {'overall':>8}  {np.sqrt(np.mean((pred_raw - X_dot) ** 2)):>14.6f}"
      f"  {np.sqrt(np.mean((pred_eng - X_dot) ** 2)):>13.6f}")
print(f"[EVAL]  dz/dt RMSE ratio (identity / bracket): "
      f"{rmse_comp_raw[2] / max(rmse_comp_eng[2], 1e-12):.1f}x")
sst = np.sum((X_dot[:, 2] - X_dot[:, 2].mean()) ** 2)
r2_raw_z = 1.0 - np.sum((pred_raw[:, 2] - X_dot[:, 2]) ** 2) / sst
r2_eng_z = 1.0 - np.sum((pred_eng[:, 2] - X_dot[:, 2]) ** 2) / sst
print(f"[EVAL]  dz/dt target std = {X_dot[:, 2].std():.4f}; the identity model's "
      f"dz/dt RMSE is {100 * rmse_comp_raw[2] / X_dot[:, 2].std():.1f}% of it")
print(f"[EVAL]  dz/dt R^2 — identity lift: {r2_raw_z:.6f}  "
      f"(i.e. it explains none of the bracket term), "
      f"bracket lift: {r2_eng_z:.6f}")

# ---------------------------------------------------------------------------
# 4b. The obstruction is structural, not a capacity problem
#
#    Refining the spline grid gives the identity-lift model strictly more
#    univariate resolution.  It does not help, because no separable sum has
#    the right zero set.
# ---------------------------------------------------------------------------
print("\n[EVAL]  Grid sweep for the identity lift (dz/dt only):")
for g in (5, 10, 20):
    m = KANDy(lift=CustomLift(fn=lambda S: S, output_dim=5, name="identity_h1"),
              grid=g, k=3, steps=100, seed=SEED, base_fun="zero")
    m.fit(X=X, X_dot=X_dot, val_frac=0.15, test_frac=0.15, lamb=0.0,
          patience=30, verbose=False)
    e = per_component_rmse(m.predict(X), X_dot)[2]
    print(f"[EVAL]    grid={g:<3d}  dz/dt RMSE = {e:.6f}")
print("[EVAL]  -> flat in the grid size: this is the zero-set obstruction, "
      "not under-fitting")

# ---------------------------------------------------------------------------
# 4c. Rollout validation — several geodesics with different constant (u, v)
#
#    Do this BEFORE get_formula(): symbolic extraction rewrites the spline
#    edges in place, so anything after it sees the snapped surrogate.
# ---------------------------------------------------------------------------
#    The horizon is chosen so that |x|, |y| <= 2 for the whole rollout: the
#    splines were only trained on x, y in [-2, 2], and a KAN extrapolates
#    poorly outside its grid.  Since |u|, |v| <= 1, a horizon of T = 2 moves
#    each coordinate by at most 2.  Running longer degrades BOTH models for a
#    reason that has nothing to do with the lift, which would muddy the point.
DT = 0.02
T_MAX = 2.0
t_eval = np.arange(0.0, T_MAX, DT)

INITIAL_CONDITIONS = [
    #  x     y     z     u      v
    [0.0, 0.0, 0.0, 1.0, 0.6],
    [-1.0, 0.5, 0.0, 0.8, -0.9],
    [1.2, -1.0, 0.5, -0.7, 0.5],
    [0.5, -1.5, -0.5, 0.7, 0.9],
]

roll_true, roll_raw, roll_eng = [], [], []
for s0 in INITIAL_CONDITIONS:
    sol = solve_ivp(heisenberg_ivp, [t_eval[0], t_eval[-1]], s0, t_eval=t_eval,
                    method="RK45", rtol=1e-11, atol=1e-13)
    roll_true.append(sol.y.T)
    roll_raw.append(model_raw.rollout(np.array(s0), T=len(t_eval), dt=DT,
                                      integrator="rk4"))
    roll_eng.append(model_eng.rollout(np.array(s0), T=len(t_eval), dt=DT,
                                      integrator="rk4"))

T_all = np.concatenate(roll_true, axis=0)
R_all = np.concatenate(roll_raw, axis=0)
E_all = np.concatenate(roll_eng, axis=0)

roll_rmse_raw = np.sqrt(np.mean((R_all - T_all) ** 2))
roll_rmse_eng = np.sqrt(np.mean((E_all - T_all) ** 2))
roll_comp_raw = per_component_rmse(R_all, T_all)
roll_comp_eng = per_component_rmse(E_all, T_all)

print(f"\n[EVAL]  Rollout over {len(INITIAL_CONDITIONS)} geodesics, "
      f"{len(t_eval)} steps each (dt={DT})")
print(f"[EVAL]  Overall rollout RMSE — identity lift: {roll_rmse_raw:.6f}, "
      f"bracket lift: {roll_rmse_eng:.6f}")
print(f"[EVAL]  {'component':>8}  {'identity lift':>14}  {'bracket lift':>13}")
for name, a, b in zip(STATE_NAMES, roll_comp_raw, roll_comp_eng):
    print(f"[EVAL]  {name:>8}  {a:>14.6f}  {b:>13.6f}")

# ---------------------------------------------------------------------------
# 4d. The linear mixing matrix A
#
#    get_A() returns the KAN's per-edge spline SCALE, which is defined only
#    up to the internal normalisation of each spline — so it shows the
#    sparsity pattern, not the physical coefficients.  The identifiable
#    quantity is the effective slope d(output)/d(theta), recovered here by
#    least squares against the lifted features.  For the bracket lift every
#    edge is linear, so those slopes ARE the coefficients of the true system.
# ---------------------------------------------------------------------------
A_true = np.array([
    [1.0, 0.0, 0.0, 0.0],    # dx/dt = u
    [0.0, 1.0, 0.0, 0.0],    # dy/dt = v
    [0.0, 0.0, 1.0, -1.0],   # dz/dt = x*v - y*u
    [0.0, 0.0, 0.0, 0.0],    # du/dt = 0
    [0.0, 0.0, 0.0, 0.0],    # dv/dt = 0
])

Theta_1 = np.column_stack([Theta, np.ones(len(Theta))])
A_eff = np.linalg.lstsq(Theta_1, model_eng.predict(X), rcond=None)[0][:-1].T

np.set_printoptions(precision=4, suppress=True)
print(f"\n[MODEL] Bracket-lift columns: {ENG_FEATURE_NAMES}")
print("[MODEL] True A:")
print(A_true)
print("[MODEL] Effective A (least squares on the lifted features):")
print(A_eff)
print(f"[MODEL] max |A_eff - A_true| = {np.abs(A_eff - A_true).max():.6f}")
print("[MODEL] model.get_A()  (raw PyKAN spline scales — sparsity pattern only):")
print(model_eng.get_A())

# Edge activations captured before the snap
train_theta = torch.tensor(heisenberg_lift(X[:2048]), dtype=torch.float32)

# ---------------------------------------------------------------------------
# 5. Symbolic extraction  (LAST — get_formula mutates the model in place)
#
#    Every edge of the bracket model is linear, so the library is just
#    ['x', '0'] and the coefficients should come out as +1 / -1.
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Bracket-lift model:")
formulas = model_eng.get_formula(var_names=ENG_FEATURE_NAMES, round_places=4,
                                 lib=["x", "0"], r2_threshold=0.80,
                                 weight_simple=0.0)
for name, f in zip(OUT_NAMES, formulas):
    print(f"  {name} = {f}")
r2 = model_eng.score_formula(formulas, X, X_dot, var_names=ENG_FEATURE_NAMES)
print(f"[SYMBOLIC] Formula R^2 per output: {np.round(r2, 6)}")
print("[SYMBOLIC] True:  dx/dt = u,  dy/dt = v,  dz/dt = (x*v) - (y*u),  "
      "du/dt = dv/dt = 0")

print("\n[SYMBOLIC] Identity-lift model (for contrast):")
formulas_raw = model_raw.get_formula(var_names=RAW_FEATURE_NAMES,
                                     round_places=3, r2_threshold=0.80)
for name, f in zip(OUT_NAMES, formulas_raw):
    print(f"  {name} = {f}")
print("[SYMBOLIC] dz/dt is a sum of univariate terms — no product of two "
      "state variables can appear, whatever the library.")

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs(RESULTS_DIR, exist_ok=True)

C_TRUE, C_ENG, C_RAW = "#1f77b4", "#2ca02c", "#d62728"

# 6a. Lifted geodesics in (x, y, z) — the identity model fails to climb in z
fig = plt.figure(figsize=(12, 4.6))
for panel, (trajs, title, color) in enumerate([
    (roll_true, "True geodesics", C_TRUE),
    (roll_eng, f"KANDy, bracket lift\nRMSE {roll_rmse_eng:.2e}", C_ENG),
    (roll_raw, f"KANDy, identity lift\nRMSE {roll_rmse_raw:.2e}", C_RAW),
]):
    ax = fig.add_subplot(1, 3, panel + 1, projection="3d")
    for tr, ref in zip(trajs, roll_true):
        ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], color="gray", lw=0.7, alpha=0.45)
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=color, lw=1.8)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_zlabel("$z$")
    ax.set_title(title, fontsize=10, y=0.98)
    ax.view_init(elev=20, azim=-60)
fig.suptitle("Heisenberg $H^1$ sub-Riemannian geodesics (grey = truth)",
             fontsize=12, y=1.03)
fig.tight_layout()
fig.savefig(f"{RESULTS_DIR}/geodesics_3d.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{RESULTS_DIR}/geodesics_3d.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6b. z(t) — the bracket term, isolated
fig, axes = plt.subplots(1, len(INITIAL_CONDITIONS), figsize=(13, 3.2),
                         sharex=True)
for ax, tr, re_, rr, s0 in zip(axes, roll_true, roll_eng, roll_raw,
                               INITIAL_CONDITIONS):
    ax.plot(t_eval, tr[:, 2], color=C_TRUE, lw=1.6, label="true")
    ax.plot(t_eval, re_[:, 2], color=C_ENG, lw=1.2, ls="--", label="bracket lift")
    ax.plot(t_eval, rr[:, 2], color=C_RAW, lw=1.2, ls=":", label="identity lift")
    ax.set_title(f"$u={s0[3]}$, $v={s0[4]}$", fontsize=9)
    ax.set_xlabel("time")
axes[0].set_ylabel("$z(t)$")
axes[0].legend(fontsize=7, loc="best")
fig.suptitle("Vertical (bracket) coordinate $z(t)$", fontsize=12)
fig.tight_layout()
fig.savefig(f"{RESULTS_DIR}/z_timeseries.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6c. Per-component RMSE — localises the failure to dz/dt
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
idx = np.arange(5)
w = 0.38
for ax, (a, b, title, labels) in zip(axs, [
    (rmse_comp_raw, rmse_comp_eng, "One-step RMSE per component", OUT_NAMES),
    (roll_comp_raw, roll_comp_eng, "Rollout RMSE per component", STATE_NAMES),
]):
    ax.bar(idx - w / 2, np.maximum(a, 1e-12), w, color=C_RAW, label="identity lift")
    ax.bar(idx + w / 2, np.maximum(b, 1e-12), w, color=C_ENG, label="bracket lift")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("RMSE (log)")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3, ls="--")
axs[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{RESULTS_DIR}/rmse_per_component.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{RESULTS_DIR}/rmse_per_component.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6d. Loss curves
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, mdl, title in [(axs[0], model_raw, "identity lift"),
                       (axs[1], model_eng, "bracket lift")]:
    if getattr(mdl, "train_results_", None):
        plot_loss_curves(mdl.train_results_, ax=ax)
        ax.set_title(title, fontsize=10)
fig.tight_layout()
fig.savefig(f"{RESULTS_DIR}/loss_curves.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6e. Edge activations for the bracket model — all four edges linear
fig, _axes = plot_all_edges(
    model_eng.model_, X=train_theta,
    in_var_names=ENG_FEATURE_NAMES,
    out_var_names=OUT_NAMES,
    save=f"{RESULTS_DIR}/edge_activations_bracket",
)
plt.close(fig)

print(f"\n[FIGS]  Saved {RESULTS_DIR}/")
