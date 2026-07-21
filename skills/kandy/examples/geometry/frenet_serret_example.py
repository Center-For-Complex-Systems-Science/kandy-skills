#!/usr/bin/env python3
"""KANDy example: the Frenet–Serret moving frame along a space curve.

A unit-speed curve r(s) in R^3 carries an orthonormal frame (T, N, B) whose
evolution in arclength s is governed by the Frenet–Serret equations.  Carrying
the curvature kappa and the torsion tau as constant extra state coordinates
turns the system into an autonomous 11-dimensional ODE:

    dT/ds     =  kappa * N
    dN/ds     = -kappa * T + tau * B
    dB/ds     = -tau  * N
    dkappa/ds =  0
    dtau/ds   =  0

State = (T1,T2,T3, N1,N2,N3, B1,B2,B3, kappa, tau).  With kappa and tau
constant the curve is a circular helix of radius kappa/(kappa^2+tau^2) and
pitch tau/(kappa^2+tau^2), so there is an exact closed-form ground truth: the
frame rotates rigidly about the constant Darboux vector

    omega = tau*T + kappa*B,     d(omega)/ds = 0,

at angular rate |omega| = sqrt(kappa^2 + tau^2).

THE LESSON: a bilinear moving frame, and the invariant the lift does not know
------------------------------------------------------------------------------
1. EVERY term on the right-hand side is a product of a scalar geometric
   invariant (kappa or tau) with a frame-vector component.  Because kappa and
   tau are *state variables* here, none of these products degenerate into a
   constant that a linear model could absorb — they are all genuine
   cross-terms between DIFFERENT state coordinates, and by the bilinear
   obstruction a separable KAN cannot manufacture any of them.  So all twelve
   go into the lift:

       theta = [kappa*T1..3, kappa*N1..3, tau*N1..3, tau*B1..3]

   Then each of the nine frame equations is one or two *linear* features
   (dT_i = +1*(kappa*N_i);  dN_i = -1*(kappa*T_i) + 1*(tau*B_i);
   dB_i = -1*(tau*N_i)), and dkappa/ds = dtau/ds = 0.  Every edge of the KAN
   is a straight line and the whole system collapses to one sparse 11x12
   matrix A with entries in {0, +1, -1}.  Note what is NOT in the lift: no
   kappa^2, no T_i^2, no kappa*tau.  Powers of a single variable are exactly
   what a spline edge already represents for free.

2. A HIDDEN INVARIANT the lift does not enforce.  The true flow preserves
   orthonormality: (T, N, B) stays in SO(3) for all s, because the
   Frenet–Serret generator is skew-symmetric.  The learned vector field knows
   nothing about that — it only ever saw pointwise derivative samples.  Any
   residual error in A perturbs the generator away from skew-symmetry, and the
   symmetric part acts as a slow exponential growth or decay of the frame.  So
   this example measures ||T||-1, ||N||-1, ||B||-1, T.N, T.B, N.B and
   det[T N B]-1 along the LEARNED rollout and compares them to the same
   quantities along a tight RK45 solution of the true system.  It is a case
   where an accurate one-step fit still degrades *geometrically* over a long
   rollout: the pointwise R^2 says nothing about whether the frame is still a
   frame 2000 steps later.

   What a reader could do about it, without building a whole framework:
     * train with ``rollout_weight > 0`` and a long ``rollout_horizon`` so the
       loss actually penalises multi-step drift rather than one-step error;
     * re-orthonormalise (Gram–Schmidt / polar projection of [T N B]) after
       each integration step, i.e. project the learned flow back onto SO(3);
     * or add the six orthonormality residuals to the loss as a soft penalty.
   The structural fix — parameterising the learned generator as skew-symmetric
   by construction — is out of scope here, but it is the right answer.

3. The drift is made VISIBLE by reconstructing the curve itself:
   integrating dr/ds = T for the true and the learned frame and overlaying the
   two space curves turns an abstract 1e-k number into a geometric error.

Lift  phi: R^11 -> R^12
      theta = [kappa*T1,kappa*T2,kappa*T3, kappa*N1,kappa*N2,kappa*N3,
               tau*N1,tau*N2,tau*N3,       tau*B1,tau*B2,tau*B3]
KAN:  width = [12, 11],  base_fun='zero'   (all 132 edges linear)
"""

import os
import time

import numpy as np
import sympy as sp
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from scipy.integrate import solve_ivp

from kandy import KANDy, CustomLift
from kandy.plotting import plot_all_edges, plot_loss_curves, use_pub_style

T_START = time.time()

# ---------------------------------------------------------------------------
# 0. Reproducibility and parameters
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

KAPPA_RANGE = (0.3, 2.0)     # training range for curvature
TAU_RANGE = (-1.5, 1.5)      # training range for torsion

STATE_NAMES = ["T1", "T2", "T3", "N1", "N2", "N3", "B1", "B2", "B3",
               "kappa", "tau"]
DERIV_NAMES = ["dT1", "dT2", "dT3", "dN1", "dN2", "dN3", "dB1", "dB2", "dB3",
               "dkappa", "dtau"]

print("[MODEL] Frenet-Serret frame with kappa, tau carried as state "
      "(11-dim autonomous system)")
print(f"[MODEL] training ranges: kappa in {KAPPA_RANGE}, tau in {TAU_RANGE}")


def frenet_rhs_np(X: np.ndarray) -> np.ndarray:
    """Vectorised Frenet-Serret right-hand side for a batch of states."""
    T, N, B = X[:, 0:3], X[:, 3:6], X[:, 6:9]
    kappa, tau = X[:, 9:10], X[:, 10:11]
    dT = kappa * N
    dN = -kappa * T + tau * B
    dB = -tau * N
    return np.column_stack([dT, dN, dB, np.zeros((len(X), 2))])


def frenet(s, state):
    return frenet_rhs_np(np.asarray(state)[None, :])[0]


# ---------------------------------------------------------------------------
# 1. Training data — independent samples: random frames x random (kappa, tau)
#
#    Sampling frames uniformly over SO(3) (QR of a Gaussian matrix, sign-fixed
#    so det = +1) and kappa, tau independently keeps the twelve lifted features
#    linearly independent.  A single helix trajectory would NOT: along one
#    solution kappa and tau are frozen and the frame stays on a 1-parameter
#    subgroup, so the feature matrix would be badly conditioned and the
#    coefficients unidentifiable.
# ---------------------------------------------------------------------------
N_SAMPLES = 4000
rng = np.random.default_rng(SEED)


def random_frames(n: int, rng) -> np.ndarray:
    """n uniformly random right-handed orthonormal frames, as (n, 9) rows."""
    A = rng.standard_normal((n, 3, 3))
    Q, R = np.linalg.qr(A)
    # Fix the QR sign ambiguity, then force det(Q) = +1 (right-handed frame).
    Q = Q * np.sign(np.einsum("nii->ni", R))[:, None, :]
    flip = np.linalg.det(Q) < 0
    Q[flip, :, 2] *= -1.0
    # Columns of Q are T, N, B; flatten row-wise as (T1..3, N1..3, B1..3).
    return np.concatenate([Q[:, :, 0], Q[:, :, 1], Q[:, :, 2]], axis=1)


frames = random_frames(N_SAMPLES, rng)
kappa_s = rng.uniform(*KAPPA_RANGE, size=(N_SAMPLES, 1))
tau_s = rng.uniform(*TAU_RANGE, size=(N_SAMPLES, 1))
X = np.column_stack([frames, kappa_s, tau_s])
X_dot = frenet_rhs_np(X)

print(f"[DATA]  {N_SAMPLES} independent samples "
      f"(uniform SO(3) frames x uniform kappa, tau)")
_F = frames.reshape(-1, 3, 3)          # rows are T, N, B
print(f"[DATA]  sample frames: max |F F^T - I| = "
      f"{np.abs(_F @ _F.transpose(0, 2, 1) - np.eye(3)).max():.2e}, "
      f"min det = {np.linalg.det(_F).min():.6f}")

# ---------------------------------------------------------------------------
# 2. Lift — the twelve genuine cross-terms, nothing else
# ---------------------------------------------------------------------------
FEATURE_NAMES = (
    [f"k*T{i}" for i in (1, 2, 3)]
    + [f"k*N{i}" for i in (1, 2, 3)]
    + [f"t*N{i}" for i in (1, 2, 3)]
    + [f"t*B{i}" for i in (1, 2, 3)]
)


def frenet_lift(X: np.ndarray) -> np.ndarray:
    T, N, B = X[:, 0:3], X[:, 3:6], X[:, 6:9]
    kappa, tau = X[:, 9:10], X[:, 10:11]
    return np.column_stack([kappa * T, kappa * N, tau * N, tau * B])


lift = CustomLift(fn=frenet_lift, output_dim=12, name="frenet_lift")

Theta = frenet_lift(X)
cond = np.linalg.cond(np.column_stack([Theta, np.ones(len(Theta))]))
print(f"[DATA]  cond(lifted feature matrix + ones column) = {cond:.2f}"
      "   (small -> coefficients identifiable)")

# True mixing matrix: 11 states x 12 features, entries in {0, +1, -1}.
A_TRUE = np.zeros((11, 12))
for i in range(3):
    A_TRUE[0 + i, 3 + i] = +1.0      # dT_i = + kappa*N_i
    A_TRUE[3 + i, 0 + i] = -1.0      # dN_i = - kappa*T_i
    A_TRUE[3 + i, 9 + i] = +1.0      #        + tau*B_i
    A_TRUE[6 + i, 6 + i] = -1.0      # dB_i = - tau*N_i

# ---------------------------------------------------------------------------
# 3. KANDy model
#
#    12 -> 11 is wide (132 edges), but every edge is a straight line, so LBFGS
#    converges fast: the loss is already ~1e-8 by step 40.  Past that the
#    gradient is at machine precision and the line search thrashes, costing
#    minutes for no accuracy — hence steps=60 rather than the usual few hundred.
# ---------------------------------------------------------------------------
model = KANDy(lift=lift, grid=5, k=3, steps=60, seed=SEED, base_fun="zero")
model.fit(X=X, X_dot=X_dot, val_frac=0.15, test_frac=0.15, lamb=0.0,
          patience=50)

pred = model.predict(X)
var = X_dot.var(axis=0)
raw_r2 = np.array([
    1.0 - np.sum((X_dot[:, i] - pred[:, i]) ** 2)
    / np.sum((X_dot[:, i] - X_dot[:, i].mean()) ** 2)
    if var[i] > 1e-12 else np.nan
    for i in range(11)
])
print(f"\n[EVAL]  Network R^2, frame rows: {np.round(raw_r2[:9], 6)}")
print(f"[EVAL]  dkappa/ds, dtau/ds are identically zero (R^2 undefined); "
       f"max |pred| = {np.abs(pred[:, 9:]).max():.2e}")

# --- recovered mixing matrix vs truth -------------------------------------
A_hat = model.get_A()
# get_A returns PyKAN's per-edge spline SCALE, which carries the sign and
# magnitude of each edge but is not itself the fitted slope.  The effective
# linear coefficients are read off directly by least squares on the network's
# own output — an honest apples-to-apples comparison with A_TRUE.
A_eff, *_ = np.linalg.lstsq(Theta, pred, rcond=None)
A_eff = A_eff.T

np.set_printoptions(precision=3, suppress=True, linewidth=200)
print("\n[EVAL]  Effective mixing matrix (nine frame rows, 12 features)")
print("        features:", " ".join(f"{n:>6s}" for n in FEATURE_NAMES))
for i in range(9):
    print(f"   {DERIV_NAMES[i]:>6s} learned " +
          " ".join(f"{v:6.3f}" for v in A_eff[i]))
    print(f"   {'':>6s} true    " +
          " ".join(f"{v:6.3f}" for v in A_TRUE[i]))
print(f"[EVAL]  max |A_eff - A_true| over frame rows: "
      f"{np.abs(A_eff[:9] - A_TRUE[:9]).max():.3e}")
active = A_TRUE != 0
print(f"[EVAL]  get_A() edge scales: median |A| = "
      f"{np.median(np.abs(A_hat[active])):.4f} on the {active.sum()} "
      f"structurally active edges vs "
      f"{np.median(np.abs(A_hat[~active])):.4f} on the {(~active).sum()} "
      "edges that should be zero -- i.e. scale_sp does NOT separate them, so "
      "read structure off A_eff above, not off get_A().")

# ---------------------------------------------------------------------------
# 4. Rollout validation — long arclength, plus the SO(3) invariants
#
#    All of this happens BEFORE get_formula(): symbolic extraction rewrites the
#    spline edges in place, so any predict/rollout after it uses the snapped
#    surrogate instead of the trained network.
# ---------------------------------------------------------------------------
KAPPA0, TAU0 = 1.2, 0.5
DS = 0.02
S_MAX = 40.0
s_eval = np.arange(0.0, S_MAX, DS)
X0 = np.array([1.0, 0.0, 0.0,
               0.0, 1.0, 0.0,
               0.0, 0.0, 1.0,
               KAPPA0, TAU0])

sol = solve_ivp(frenet, [s_eval[0], s_eval[-1]], X0, t_eval=s_eval,
                method="RK45", rtol=1e-12, atol=1e-14)
true_traj = sol.y.T
pred_traj = model.rollout(X0, T=len(true_traj), dt=DS, integrator="rk4")


def helix_closed_form(s: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Exact frame: rigid rotation about the constant Darboux vector."""
    T0, N0, B0 = x0[0:3], x0[3:6], x0[6:9]
    kappa, tau = x0[9], x0[10]
    omega = tau * T0 + kappa * B0
    w = np.linalg.norm(omega)
    e = omega / w
    c, sn = np.cos(w * s)[:, None], np.sin(w * s)[:, None]
    out = []
    for v in (T0, N0, B0):                      # Rodrigues rotation formula
        out.append(c * v + sn * np.cross(e, v) + (1 - c) * np.dot(e, v) * e)
    return np.column_stack(out)


exact_frame = helix_closed_form(s_eval, X0)
print(f"\n[EVAL]  solve_ivp vs closed-form helix frame: max err "
      f"{np.abs(true_traj[:, :9] - exact_frame).max():.2e}  "
      "(ground truth confirmed)")

frame_rmse = np.sqrt(np.mean((pred_traj[:, :9] - true_traj[:, :9]) ** 2))
print(f"[EVAL]  Rollout frame RMSE over {len(s_eval)} steps (s up to "
      f"{S_MAX}): {frame_rmse:.3e}")
print(f"[EVAL]  Invariant drift of the carried parameters: "
      f"|kappa(S)-kappa0| = {abs(pred_traj[-1, 9] - KAPPA0):.2e}, "
      f"|tau(S)-tau0| = {abs(pred_traj[-1, 10] - TAU0):.2e}")


def so3_invariants(traj: np.ndarray) -> dict:
    """Orthonormality + right-handedness residuals along a trajectory."""
    T, N, B = traj[:, 0:3], traj[:, 3:6], traj[:, 6:9]
    det = np.einsum("ni,ni->n", np.cross(T, N), B)
    return {
        "|T|-1": np.linalg.norm(T, axis=1) - 1.0,
        "|N|-1": np.linalg.norm(N, axis=1) - 1.0,
        "|B|-1": np.linalg.norm(B, axis=1) - 1.0,
        "T.N": np.einsum("ni,ni->n", T, N),
        "T.B": np.einsum("ni,ni->n", T, B),
        "N.B": np.einsum("ni,ni->n", N, B),
        "det-1": det - 1.0,
    }


inv_pred = so3_invariants(pred_traj)
inv_true = so3_invariants(true_traj)
print("\n[EVAL]  SO(3) drift at s = %.1f   (learned vs true RK45)" % S_MAX)
for key in inv_pred:
    print(f"          {key:>6s}:  learned {inv_pred[key][-1]:+.3e}   "
          f"true {inv_true[key][-1]:+.3e}")
worst = max(np.abs(v).max() for v in inv_pred.values())
print(f"[EVAL]  worst learned SO(3) residual over the whole rollout: "
      f"{worst:.3e}")
print("[EVAL]  The lift never encoded orthonormality; the drift is the "
      "learned generator's symmetric part leaking in.")

# --- reconstruct the curve r(s) by integrating dr/ds = T -------------------
def integrate_curve(traj: np.ndarray, ds: float) -> np.ndarray:
    Tv = traj[:, 0:3]
    r = np.zeros_like(Tv)
    r[1:] = np.cumsum(0.5 * (Tv[1:] + Tv[:-1]) * ds, axis=0)
    return r


curve_true = integrate_curve(true_traj, DS)
curve_pred = integrate_curve(pred_traj, DS)
curve_rmse = np.sqrt(np.mean((curve_pred - curve_true) ** 2))
print(f"[EVAL]  Reconstructed curve RMSE: {curve_rmse:.3e}   "
      f"(final-point error {np.linalg.norm(curve_pred[-1] - curve_true[-1]):.3e}, "
      f"curve length {S_MAX})")

# Edge activations captured before the snap
train_theta = torch.tensor(Theta[:2048], dtype=torch.float32)

# ---------------------------------------------------------------------------
# 5. Symbolic extraction — every edge should snap to a straight line
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Extracting formulas (132 edges, lib = ['x', '0']) ...")
formulas = model.get_formula(
    var_names=FEATURE_NAMES,
    round_places=6,
    lib=["x", "0"],
    r2_threshold=0.80,
    weight_simple=0.8,
)


def polish(expr, tol: float = 5e-3, places: int = 3):
    """Expand and drop numerically negligible terms, then round."""
    e = sp.expand(expr)
    kept = [t for t in sp.Add.make_args(e)
            if abs(float(t.as_coeff_Mul()[0])) >= tol]
    e = sp.Add(*kept) if kept else sp.Integer(0)
    return e.xreplace({n: sp.Float(round(float(n), places))
                       for n in e.atoms(sp.Float)})


TRUE_STR = ([f"1.0*k*N{i}" for i in (1, 2, 3)]
            + [f"-1.0*k*T{i} + 1.0*t*B{i}" for i in (1, 2, 3)]
            + [f"-1.0*t*N{i}" for i in (1, 2, 3)]
            + ["0", "0"])
for lab, f, truth in zip(DERIV_NAMES, formulas, TRUE_STR):
    print(f"  {lab:>7s}/ds = {polish(f)}")
    print(f"  {'true':>7s}     = {truth}")

r2 = model.score_formula(formulas, X, X_dot, var_names=FEATURE_NAMES)
print(f"[SYMBOLIC] Formula R^2, frame rows: {np.round(np.asarray(r2)[:9], 6)}")

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
use_pub_style()
OUT = "results/FrenetSerret"
os.makedirs(OUT, exist_ok=True)

# 6a. 3D reconstructed curve — true helix vs KANDy
fig = plt.figure(figsize=(6.5, 5.5))
ax = fig.add_subplot(111, projection="3d")
ax.plot(curve_true[:, 0], curve_true[:, 1], curve_true[:, 2],
        color="#1f77b4", lw=1.6, label="true helix")
ax.plot(curve_pred[:, 0], curve_pred[:, 1], curve_pred[:, 2],
        color="#d62728", lw=1.0, ls="--", label="KANDy rollout")
ax.scatter(*curve_true[0], color="k", s=20)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title(f"Curve from dr/ds = T  (kappa={KAPPA0}, tau={TAU0}, s<={S_MAX})")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/curve_3d.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/curve_3d.pdf", bbox_inches="tight")
plt.close(fig)

# 6b. Orthonormality / determinant drift along the learned rollout
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, inv, title in zip(axes, [inv_pred, inv_true],
                          ["learned rollout", "true RK45 solution"]):
    for key, v in inv.items():
        ax.semilogy(s_eval, np.abs(v) + 1e-18, lw=1.0, label=key)
    ax.set_xlabel("arclength $s$")
    ax.set_title(f"SO(3) residuals — {title}")
    ax.grid(alpha=0.3, ls="--")
axes[0].set_ylabel("|residual|")
axes[0].legend(loc="lower right", fontsize=7, ncol=2)
fig.suptitle("The frame is only orthonormal because the true flow says so — "
             "the lift never encoded it", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/so3_drift.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/so3_drift.pdf", bbox_inches="tight")
plt.close(fig)

# 6c. Frame component series
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
for blk, (ax, name, cols) in enumerate(zip(
        axes, ["T", "N", "B"], [(0, 1, 2), (3, 4, 5), (6, 7, 8)])):
    for ci, c in zip(cols, ["#1f77b4", "#2ca02c", "#d62728"]):
        ax.plot(s_eval, true_traj[:, ci], color=c, lw=1.2)
        ax.plot(s_eval, pred_traj[:, ci], color=c, lw=0.9, ls="--")
    ax.set_ylabel(f"${name}$ components")
    ax.grid(alpha=0.3, ls="--")
axes[0].plot([], [], color="k", lw=1.2, label="true")
axes[0].plot([], [], color="k", lw=0.9, ls="--", label="KANDy")
axes[0].legend(loc="upper right", fontsize=8, ncol=2)
axes[-1].set_xlabel("arclength $s$")
axes[0].set_title("Frenet frame components along the helix")
fig.tight_layout()
fig.savefig(f"{OUT}/frame_components.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/frame_components.pdf", bbox_inches="tight")
plt.close(fig)

# 6d. Loss curves
if getattr(model, "train_results_", None):
    fig, ax = plot_loss_curves(model.train_results_, save=f"{OUT}/loss_curves")
    plt.close(fig)

# 6e. Edge activations — all 132 edges should be straight lines
fig, _axes = plot_all_edges(
    model.model_, X=train_theta,
    in_var_names=FEATURE_NAMES,
    out_var_names=DERIV_NAMES,
    figsize_per_panel=(1.5, 1.2),
)
fig.suptitle("Frenet-Serret KAN edges — every edge is linear", fontsize=12)
fig.savefig(f"{OUT}/edge_activations.png", dpi=110, bbox_inches="tight")
plt.close(fig)

print(f"[FIGS]  Saved {OUT}/")
print(f"[FIGS]  total runtime {time.time() - T_START:.1f} s")
