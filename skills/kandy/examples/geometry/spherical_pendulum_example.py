#!/usr/bin/env python3
"""KANDy example: spherical pendulum — a constraint force as the nonlinearity.

A point mass on a massless rod of unit length, free to move anywhere on the
unit sphere S^2 under gravity.  State is (q, p) in R^6 with q on the sphere and
p tangent to it.  Units are nondimensional: L = 1, m = 1, g = 1 (time is
measured in units of sqrt(L/g)).

    dq/dt = p
    dp/dt = -g*e3 + lambda*q

The Lagrange multiplier lambda is not free — it is whatever keeps the mass on
the sphere.  Differentiate the constraint twice:

    q.q = 1            =>   q.p = 0
    d/dt (q.p) = 0     =>   q.(dp/dt) + |p|^2 = 0
                       =>   -g*q3 + lambda*|q|^2 + |p|^2 = 0
                       =>   lambda = g*q3 - |p|^2                (using |q| = 1)

Writing s = p1^2 + p2^2 + p3^2 = |p|^2, the componentwise right-hand side is

    dq1/dt = p1                 dp1/dt = g*q1*q3 - q1*s
    dq2/dt = p2                 dp2/dt = g*q2*q3 - q2*s
    dq3/dt = p3                 dp3/dt = g*q3^2 - g - q3*s

Conserved along the true flow: |q| = 1,  q.p = 0,  E = |p|^2/2 + g*q3,
and the vertical angular momentum Lz = q1*p2 - q2*p1.

THE LESSON: lift the CONSTRAINT, and beware the manifold it lives on
------------------------------------------------------------------------------
1. THE NONLINEARITY COMES FROM THE CONSTRAINT FORCE.  Nothing in the physical
   setup is polynomial; gravity is linear and the rod is a hard constraint.
   Eliminating the constraint is what manufactures the nonlinear terms, and
   they have a shape no other example in this skill has: TRIPLE PRODUCTS
   q_i*s = q_i*(p1^2 + p2^2 + p3^2) — a position component times a squared
   speed.  Every one of them mixes DIFFERENT state variables (q_i with p_j), so
   by rule 1 of references/lifts.md every one of them must be a lift
   coordinate.  Leaving them out is a structural error: a separable KAN cannot
   manufacture q1*p2^2 from edges on q1 and p2 no matter how it is trained.

2. THE SAME EQUATION SHOWS THE OTHER HALF OF RULE 1.  dp3/dt contains both
   q3*s and g*q3^2.  They look equally nonlinear but they are treated
   completely differently:

     * q3*s is a product of DIFFERENT variables  ->  needs its own coordinate.
     * q3^2 is a power of a SINGLE variable      ->  must NOT get a coordinate.
       Supply the plain coordinate q3 and let its spline edge learn the whole
       parabola  g*q3^2 - g  by itself.  Adding q3^2 alongside q3 would make
       the decomposition non-unique and wreck the symbolic extraction (rule 2).

   So the minimal 9-feature lift is

       theta = [p1, p2, p3, q1*q3, q2*q3, q3, q1*s, q2*s, q3*s]

   and every RHS term above is a LINEAR combination of these, except the single
   q3 edge which is quadratic:
       dq_i/dt = theta_i
       dp1/dt  = g*theta_4 - theta_7        (theta_4 = q1*q3, theta_7 = q1*s)
       dp2/dt  = g*theta_5 - theta_8
       dp3/dt  = (g*theta_6^2 - g) - theta_9

3. THE CONDITION NUMBER IS NECESSARY BUT NOT SUFFICIENT.  Training data on a
   constraint manifold is exactly where mathbio/sir_example.py loses
   identifiability, so run the diagnostic from references/lifts.md:
   ``np.linalg.cond(np.column_stack([Theta, np.ones(len(Theta))]))``.  Here it
   comes out SMALL (~7) on the physical, on-manifold data — |q| = 1 and
   q.p = 0 are quadratic in the state, so unlike S + I + R = const they are not
   affine relations among the lifted columns.  And yet the on-manifold fit
   still mis-recovers dp3/dt, printing spurious q1q3^2 and q2q3^2 terms.

   The reason is that for a KAN the relevant notion of dependence is not
   linear.  Each edge is an arbitrary univariate function, so the fit is
   degenerate as soon as some sum of univariate functions of the features
   vanishes on the data.  On S^2 exactly that happens:

       (q1*q3)^2 + (q2*q3)^2 = q3^2*(q1^2 + q2^2) = q3^2 - q3^4
       i.e.   theta_4^2 + theta_5^2 - (theta_6^2 - theta_6^4) = 0

   — a separable relation among theta_4, theta_5, theta_6 with zero linear
   content, invisible to cond().  The edge decomposition of dp3/dt is therefore
   non-unique and the snapper picks a bad member of the family (R^2 ~ 0.9986).

   THE FIX is the same as in sir_example.py even though the diagnostic missed
   it: sample OFF the manifold.  Drawing q in a shell 0.7 <= |q| <= 1.3 with p
   no longer tangent breaks |q| = 1, kills the separable relation, and all six
   equations come out exact.  This script fits both models so the contrast is
   visible side by side; the on-manifold model is the one used for rollout
   validation, because that is the data a real experiment would give.

4. THE LEARNED FIELD IS NOT CONSTRAINED.  KANDy fits a generic vector field on
   R^6; nothing forces the rollout to stay on S^2.  So the interesting number
   is not just the rollout RMSE but the drift of |q| - 1, q.p and E along the
   LEARNED trajectory — that is what figure (b) shows.

Symbolic settings: 53 of the 54 edges are linear or zero and one (the q3 edge
into dp3/dt) is quadratic, so lib=["x", "x^2", "0"] with weight_simple=0.0 —
the default simplicity pressure snaps the real parabola to '0'.

Lift  phi: R^6 -> R^9   theta = [p1, p2, p3, q1*q3, q2*q3, q3, q1*s, q2*s, q3*s]
KAN:  width = [9, 6],  base_fun='zero'
"""

import copy
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
rng = np.random.default_rng(SEED)

G = 1.0          # nondimensional gravity (L = m = g = 1)

print(f"[MODEL] Spherical pendulum on S^2,  L = 1,  g = {G}")
print( "[MODEL] lambda = g*q3 - |p|^2  (from q.q = 1 -> q.p = 0 -> q.pdot + |p|^2 = 0)")


def sp_rhs_np(X: np.ndarray) -> np.ndarray:
    """Vectorised spherical-pendulum RHS, X = (N, 6) = [q1,q2,q3,p1,p2,p3].

    Off the sphere this is the polynomial EXTENSION of the constrained field
    (lambda = g*q3 - |p|^2 with |q|^2 already set to 1).  It agrees with the
    physical dynamics on S^2 and is the object whose coefficients we want to
    recover, so the off-manifold training set below targets the same function.
    """
    q1, q2, q3, p1, p2, p3 = (X[:, i] for i in range(6))
    s = p1**2 + p2**2 + p3**2
    lam = G * q3 - s                      # Lagrange multiplier
    return np.column_stack([p1, p2, p3, lam * q1, lam * q2, -G + lam * q3])


def sph(t, state):
    return sp_rhs_np(np.asarray(state)[None, :])[0]


def invariants(traj: np.ndarray):
    """(|q| - 1, q.p, E) along a trajectory."""
    q, p = traj[:, :3], traj[:, 3:]
    return (np.linalg.norm(q, axis=1) - 1.0,
            np.sum(q * p, axis=1),
            0.5 * np.sum(p * p, axis=1) + G * q[:, 2])


# ---------------------------------------------------------------------------
# 1. Training data
#
#    ON-MANIFOLD: q uniform on S^2 by Gaussian normalisation (as in
#    hopf_example.py), p drawn in the tangent plane at q (a random vector minus
#    its radial component) and rescaled over a range of speeds.  This is the
#    data a real experiment would give — it satisfies |q| = 1 and q.p = 0
#    exactly — and it is what the rollout validation uses.
#
#    OFF-MANIFOLD: q in a shell around the sphere, p with no tangency imposed.
#    Used for the identifiability contrast in section 5.
# ---------------------------------------------------------------------------
N_SAMPLES = 8000
SPEED_RANGE = (0.2, 2.5)
SHELL_RANGE = (0.7, 1.3)


def sample_on_manifold(n: int) -> np.ndarray:
    q = rng.standard_normal((n, 3))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    v = rng.standard_normal((n, 3))
    p = v - np.sum(v * q, axis=1, keepdims=True) * q          # project to T_q S^2
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    p *= rng.uniform(*SPEED_RANGE, size=(n, 1))
    return np.hstack([q, p])


def sample_off_manifold(n: int) -> np.ndarray:
    q = rng.standard_normal((n, 3))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q *= rng.uniform(*SHELL_RANGE, size=(n, 1))               # shell, |q| != 1
    p = rng.standard_normal((n, 3))                           # not tangent
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    p *= rng.uniform(*SPEED_RANGE, size=(n, 1))
    return np.hstack([q, p])


X = sample_on_manifold(N_SAMPLES)
X_dot = sp_rhs_np(X)
X_off = sample_off_manifold(N_SAMPLES)
X_dot_off = sp_rhs_np(X_off)

print(f"[DATA]  {N_SAMPLES} on-manifold samples;  "
      f"max ||q|-1| = {np.abs(np.linalg.norm(X[:, :3], axis=1) - 1).max():.2e},  "
      f"max |q.p| = {np.abs(np.sum(X[:, :3] * X[:, 3:], axis=1)).max():.2e}")
print(f"[DATA]  {N_SAMPLES} off-manifold samples; |q| in {SHELL_RANGE}, p not tangent")
print(f"[DATA]  speed range |p| in {SPEED_RANGE}")

# ---------------------------------------------------------------------------
# 2. Minimal lift — states/cross-products only, NO single-variable powers
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["p1", "p2", "p3", "q1q3", "q2q3", "q3", "q1s", "q2s", "q3s"]


def pendulum_lift(X: np.ndarray) -> np.ndarray:
    q1, q2, q3, p1, p2, p3 = (X[:, i] for i in range(6))
    s = p1**2 + p2**2 + p3**2                      # |p|^2
    return np.column_stack([
        p1, p2, p3,                                # dq/dt
        q1 * q3, q2 * q3,                          # gravity x constraint
        q3,                                        # edge learns g*q3^2 - g
        q1 * s, q2 * s, q3 * s,                    # constraint triple products
    ])


lift = CustomLift(fn=pendulum_lift, output_dim=9, name="spherical_pendulum_lift")

# --- Conditioning diagnostic (references/lifts.md, "Two design rules") ------
Theta, Theta_off = pendulum_lift(X), pendulum_lift(X_off)
cond_on = np.linalg.cond(np.column_stack([Theta, np.ones(len(Theta))]))
cond_off = np.linalg.cond(np.column_stack([Theta_off, np.ones(len(Theta_off))]))
print(f"[DATA]  cond(Theta | 1)  on-manifold  = {cond_on:7.2f}")
print(f"[DATA]  cond(Theta | 1)  off-manifold = {cond_off:7.2f}")
print("[DATA]  Both small -> no LINEAR degeneracy.  But a KAN edge is an")
print("[DATA]  arbitrary univariate function, so check for SEPARABLE relations")
print("[DATA]  too:  theta_4^2 + theta_5^2 - (theta_6^2 - theta_6^4) = 0 on S^2.")
sep = lambda T: T[:, 3]**2 + T[:, 4]**2 - (T[:, 5]**2 - T[:, 5]**4)
print(f"[DATA]  max |separable residual|  on-manifold  = {np.abs(sep(Theta)).max():.2e}")
print(f"[DATA]  max |separable residual|  off-manifold = {np.abs(sep(Theta_off)).max():.2e}")
print("[DATA]  It vanishes identically on-manifold -> dp3/dt is NOT identifiable")
print("[DATA]  from on-manifold data, even though cond() says the lift is fine.")

# ---------------------------------------------------------------------------
# 3. KANDy models — one per training set
# ---------------------------------------------------------------------------
print("\n--- Model ON: physical, on-manifold data (used for rollout) ---")
model = KANDy(lift=lift, grid=5, k=3, steps=300, seed=SEED, base_fun="zero")
model.fit(X=X, X_dot=X_dot, val_frac=0.15, test_frac=0.15, lamb=0.0, patience=50)

pred = model.predict(X)
raw_r2 = [
    1.0 - np.sum((X_dot[:, i] - pred[:, i]) ** 2)
        / np.sum((X_dot[:, i] - X_dot[:, i].mean()) ** 2)
    for i in range(6)
]
print(f"[EVAL]  Network R^2 per equation (on-manifold): {np.round(raw_r2, 6)}")

print("\n--- Model OFF: shell samples, for identifiability contrast ---")
lift_off = CustomLift(fn=pendulum_lift, output_dim=9, name="spherical_pendulum_lift")
model_off = KANDy(lift=lift_off, grid=5, k=3, steps=300, seed=SEED, base_fun="zero")
model_off.fit(X=X_off, X_dot=X_dot_off, val_frac=0.15, test_frac=0.15,
              lamb=0.0, patience=50)

# ---------------------------------------------------------------------------
# 4. Rollout validation — a precessing (non-planar) orbit
#
#    Started at a turning point of the polar motion with pure azimuthal
#    momentum, so the mass traces a precessing band rather than a planar swing.
#    Done BEFORE get_formula(), which rewrites the spline edges in place.
# ---------------------------------------------------------------------------
DT, N_STEPS = 0.01, 2000
t_eval = np.arange(N_STEPS) * DT

Q3_0 = -0.2                                     # start in the lower hemisphere
P_PHI = 1.364                                   # azimuthal speed -> precession
X0 = np.array([np.sqrt(1.0 - Q3_0**2), 0.0, Q3_0, 0.0, P_PHI, 0.0])
print(f"\n[EVAL]  x0 = {np.round(X0, 4)}   (q.p = {X0[:3] @ X0[3:]:.1e})")

sol = solve_ivp(sph, [t_eval[0], t_eval[-1]], X0, t_eval=t_eval,
                method="RK45", rtol=1e-12, atol=1e-12)
true_traj = sol.y.T
pred_traj = model.rollout(X0, T=len(true_traj), dt=DT, integrator="rk4")

rmse = np.sqrt(np.mean((pred_traj - true_traj) ** 2))
rmse_q = np.sqrt(np.mean((pred_traj[:, :3] - true_traj[:, :3]) ** 2))
rmse_p = np.sqrt(np.mean((pred_traj[:, 3:] - true_traj[:, 3:]) ** 2))
print(f"[EVAL]  Rollout RMSE over {len(true_traj)} steps (t = {t_eval[-1]:.1f}): "
      f"{rmse:.6f}   (q: {rmse_q:.6f}, p: {rmse_p:.6f})")
print(f"[EVAL]  q3 range true [{true_traj[:, 2].min():.3f}, {true_traj[:, 2].max():.3f}], "
      f"KANDy [{pred_traj[:, 2].min():.3f}, {pred_traj[:, 2].max():.3f}]")

# Geometric invariants: exact for the true flow, only approximate for the
# learned one, because nothing constrains the fitted vector field to S^2.
nq_t, qp_t, E_t = invariants(true_traj)
nq_l, qp_l, E_l = invariants(pred_traj)
E0 = E_t[0]
print(f"[EVAL]  Invariant drift, TRUE    rollout: "
      f"max||q|-1| = {np.abs(nq_t).max():.2e}, max|q.p| = {np.abs(qp_t).max():.2e}, "
      f"max|E-E0| = {np.abs(E_t - E0).max():.2e}")
print(f"[EVAL]  Invariant drift, LEARNED rollout: "
      f"max||q|-1| = {np.abs(nq_l).max():.2e}, max|q.p| = {np.abs(qp_l).max():.2e}, "
      f"max|E-E0| = {np.abs(E_l - E0).max():.2e}")

# get_formula() rewrites the spline edges in place, so keep an untouched copy
# of the trained network for the edge-activation figure (see references/api.md).
model_pre = copy.deepcopy(model)
train_theta = torch.tensor(Theta[:2048], dtype=torch.float32)

# ---------------------------------------------------------------------------
# 5. Symbolic extraction  (LAST — get_formula mutates model_ in place)
#
#    One edge (q3 -> dp3/dt) is a genuine parabola, so 'x^2' must be in the
#    library; weight_simple=0.0 removes the simplicity pressure that would
#    otherwise snap that parabola to '0'.  Same handling as
#    mathbio/lotka_volterra_competition_example.py.
# ---------------------------------------------------------------------------
LHS = ["dq1/dt", "dq2/dt", "dq3/dt", "dp1/dt", "dp2/dt", "dp3/dt"]


def polish(expr, tol: float = 1e-2, places: int = 3):
    """Expand a snapped formula and drop numerically negligible terms.

    PyKAN returns each edge in the affine-composed form c*f(a*x + b) + d, which
    hides the coefficients; expanding reveals them and pruning removes the tiny
    cross-products expansion generates.  Float exponents are rationalised so
    SymPy will expand them.
    """
    rationalise = lambda ex: ex.replace(
        lambda x: x.is_Pow and x.exp.is_Float,
        lambda x: sp.Pow(x.base, sp.Integer(round(float(x.exp)))),
    )
    e = sp.expand(rationalise(sp.expand(expr)))
    kept = [t for t in sp.Add.make_args(e)
            if abs(float(t.as_coeff_Mul()[0])) >= tol]
    e = sp.Add(*kept) if kept else sp.Integer(0)
    return e.xreplace({n: sp.Float(round(float(n), places)) for n in e.atoms(sp.Float)})


for tag, mdl, Xs, Ys in [("ON-MANIFOLD ", model, X, X_dot),
                         ("OFF-MANIFOLD", model_off, X_off, X_dot_off)]:
    print(f"\n[SYMBOLIC] {tag} model — extracting formulas ...")
    fs = mdl.get_formula(var_names=FEATURE_NAMES, round_places=6,
                         lib=["x", "x^2", "0"], r2_threshold=0.80,
                         weight_simple=0.0)
    for lab, f in zip(LHS, fs):
        print(f"  {lab} = {polish(f)}")
    r2 = mdl.score_formula(fs, Xs, Ys, var_names=FEATURE_NAMES)
    print(f"[SYMBOLIC] Formula R^2 per equation: {np.round(r2, 6)}")

print("\n[SYMBOLIC] True:  dq1/dt = p1,  dq2/dt = p2,  dq3/dt = p3")
print(f"[SYMBOLIC]        dp1/dt = {G}*q1q3 - 1.0*q1s")
print(f"[SYMBOLIC]        dp2/dt = {G}*q2q3 - 1.0*q2s")
print(f"[SYMBOLIC]        dp3/dt = {G}*q3**2 - 1.0*q3s - {G}")
print("[SYMBOLIC] The on-manifold model trades  theta_4^2 + theta_5^2  against")
print("[SYMBOLIC] theta_6^2 - theta_6^4 in dp3/dt; the off-manifold model cannot.")

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
use_pub_style()
OUT = "results/SphericalPendulum"
os.makedirs(OUT, exist_ok=True)

# 6a. Trajectory on the sphere — true vs learned, with a wireframe S^2
fig = plt.figure(figsize=(6.5, 6))
ax = fig.add_subplot(111, projection="3d")
u_ = np.linspace(0, 2 * np.pi, 48)
v_ = np.linspace(0, np.pi, 24)
ax.plot_wireframe(np.outer(np.cos(u_), np.sin(v_)),
                  np.outer(np.sin(u_), np.sin(v_)),
                  np.outer(np.ones_like(u_), np.cos(v_)),
                  color="0.75", lw=0.3, alpha=0.6)
ax.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2],
        color="#1f77b4", lw=1.3, label="true")
ax.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2],
        color="#d62728", lw=1.0, ls="--", label="KANDy")
ax.scatter(*X0[:3], color="k", s=25, label="$x_0$")
ax.set_xlabel("$q_1$"); ax.set_ylabel("$q_2$"); ax.set_zlabel("$q_3$")
ax.set_box_aspect((1, 1, 1))
ax.set_title("Spherical pendulum: precessing orbit on $S^2$")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/sphere_trajectory.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/sphere_trajectory.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6b. Invariant drift along the LEARNED rollout — the point of this example
fig, axes = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True)
for ax_, yt, yl, lab in [
    (axes[0], nq_t, nq_l, r"$|q| - 1$"),
    (axes[1], qp_t, qp_l, r"$q \cdot p$"),
    (axes[2], E_t - E0, E_l - E0, r"$E - E_0$"),
]:
    ax_.plot(t_eval, yt, color="#1f77b4", lw=1.0, label="true (RK45)")
    ax_.plot(t_eval, yl, color="#d62728", lw=1.0, ls="--", label="KANDy rollout")
    ax_.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax_.set_ylabel(lab)
    ax_.grid(alpha=0.3, ls="--")
axes[0].legend(loc="upper left", fontsize=8)
axes[0].set_title("Geometric invariants: the learned field is not constrained to $S^2$")
axes[2].set_xlabel("time")
fig.tight_layout()
fig.savefig(f"{OUT}/invariant_drift.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/invariant_drift.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6c. q3 time series
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(t_eval, true_traj[:, 2], color="#1f77b4", lw=1.3, label="$q_3$ true")
ax.plot(t_eval, pred_traj[:, 2], color="#d62728", lw=1.0, ls="--", label="$q_3$ KANDy")
ax.set_xlabel("time")
ax.set_ylabel("$q_3$ (height)")
ax.set_title("Polar oscillation")
ax.legend(loc="best", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/q3_timeseries.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6d. Loss curves
if getattr(model, "train_results_", None):
    fig, ax = plot_loss_curves(model.train_results_, save=f"{OUT}/loss_curves")
    plt.close(fig)

# 6e. Edge activations of the UNSNAPPED on-manifold network — the q3 edge is a
#     parabola, every other live edge is a straight line.
fig, axes = plot_all_edges(
    model_pre.model_, X=train_theta,
    in_var_names=FEATURE_NAMES,
    out_var_names=LHS,
    save=f"{OUT}/edge_activations",
)
plt.close(fig)

print(f"[FIGS]  Saved {OUT}/")
