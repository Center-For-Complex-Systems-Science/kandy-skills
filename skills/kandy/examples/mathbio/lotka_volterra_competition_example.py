#!/usr/bin/env python3
"""KANDy example: Lotka–Volterra two-species competition.

Two species competing for a shared resource:

    dN1/dt = r1 * N1 * (1 - (N1 + a12 * N2) / K1)
    dN2/dt = r2 * N2 * (1 - (N2 + a21 * N1) / K2)

Expanded:

    dN1/dt = r1*N1 - (r1/K1)*N1^2 - (r1*a12/K1)*N1*N2
    dN2/dt = r2*N2 - (r2/K2)*N2^2 - (r2*a21/K2)*N1*N2

Parameters satisfy a12*a21 < 1, so the species coexist at a stable equilibrium.

THE LESSON: put cross-terms in the lift, NOT powers
---------------------------------------------------
The RHS contains N1^2, N2^2 and N1*N2, so it is tempting to reach for
``PolynomialLift(degree=2)``, which supplies all five monomials
[N1, N2, N1^2, N1*N2, N2^2].  That fits the data perfectly but the symbolic
extraction comes out wrong, because it makes the edge decomposition
NON-UNIQUE:

  * A KAN edge applies an arbitrary univariate function to its input.  Given
    N1 on one edge and N1^2 on another, the term -0.53*N1^2 can be produced by
    either edge (or split across both) — nothing picks a canonical answer.
    The snapper then reports quartics like (a - b*N1^2)^2.

  * Supplying only [N1, N2, N1*N2] removes the ambiguity.  Each lifted
    coordinate is a functionally independent quantity, the N1 edge learns the
    whole quadratic  r1*N1 - (r1/K1)*N1^2  by itself, and the recovered
    coefficients match the truth to three decimals.

So the rule is narrower than "encode the nonlinearity": encode the terms the
KAN *cannot* build — products of DIFFERENT state variables.  Powers of a
single variable are exactly what a spline edge is already good at.  (This is
the opposite of SINDy, where every monomial must be in the dictionary.)

Lift  phi: R^2 -> R^3   theta = [N1, N2, N1*N2]
KAN:  width = [3, 2],  base_fun='zero'
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

R1, K1, A12 = 0.8, 1.5, 0.5     # species 1: growth, carrying capacity, competition
R2, K2, A21 = 0.6, 1.0, 0.4     # species 2

DENOM = 1.0 - A12 * A21
N1_EQ = (K1 - A12 * K2) / DENOM
N2_EQ = (K2 - A21 * K1) / DENOM
print(f"[MODEL] a12*a21 = {A12 * A21:.2f} < 1  ->  coexistence at "
      f"(N1*, N2*) = ({N1_EQ:.4f}, {N2_EQ:.4f})")


def lv_rhs_np(N1, N2):
    return np.column_stack([
        R1 * N1 * (1.0 - (N1 + A12 * N2) / K1),
        R2 * N2 * (1.0 - (N2 + A21 * N1) / K2),
    ])


def lv(t, state):
    return lv_rhs_np(np.array([state[0]]), np.array([state[1]]))[0]


# ---------------------------------------------------------------------------
# 1. Training data — independent samples over the positive quadrant
#
#    Sampling N1 and N2 independently (rather than along a single trajectory)
#    keeps the lifted features linearly independent, which is what makes the
#    coefficients identifiable.  See sir_example.py for the full explanation.
# ---------------------------------------------------------------------------
N_SAMPLES = 6000
rng = np.random.default_rng(SEED)

X = rng.uniform(0.0, 2.0, size=(N_SAMPLES, 2))
X_dot = lv_rhs_np(X[:, 0], X[:, 1])
print(f"[DATA]  {N_SAMPLES} independent state samples over [0, 2]^2")

# ---------------------------------------------------------------------------
# 2. Minimal lift — raw states plus the ONE genuine cross-term
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["N1", "N2", "N1*N2"]


def lv_lift(X: np.ndarray) -> np.ndarray:
    N1, N2 = X[:, 0], X[:, 1]
    return np.column_stack([N1, N2, N1 * N2])


lift = CustomLift(fn=lv_lift, output_dim=3, name="lv_lift")

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
# 4. Rollout validation — approach to the coexistence equilibrium
#
#    Do this BEFORE get_formula(): auto_symbolic rewrites the model's spline
#    edges in place, so everything after it sees the snapped surrogate rather
#    than the trained network.
# ---------------------------------------------------------------------------
DT     = 0.05
T_MAX  = 40.0
t_eval = np.arange(0.0, T_MAX, DT)
X0     = [0.15, 1.6]

sol = solve_ivp(lv, [t_eval[0], t_eval[-1]], X0, t_eval=t_eval,
                method="RK45", rtol=1e-10, atol=1e-12)
true_traj = sol.y.T
pred_traj = model.rollout(np.array(X0), T=len(true_traj), dt=DT, integrator="rk4")

rmse = np.sqrt(np.mean((pred_traj - true_traj) ** 2))
print(f"\n[EVAL]  Rollout RMSE (T={len(true_traj)} steps): {rmse:.6f}")
print(f"[EVAL]  Final state — true ({true_traj[-1, 0]:.4f}, {true_traj[-1, 1]:.4f}), "
      f"KANDy ({pred_traj[-1, 0]:.4f}, {pred_traj[-1, 1]:.4f}), "
      f"equilibrium ({N1_EQ:.4f}, {N2_EQ:.4f})")

# Phase-plane trajectories for the figure, computed before snapping
PHASE_STARTS = [(0.15, 1.6), (1.8, 0.15), (0.3, 0.2), (1.9, 1.8)]
phase_pairs = []
for x0 in PHASE_STARTS:
    s = solve_ivp(lv, [0, T_MAX], list(x0), t_eval=t_eval,
                  method="RK45", rtol=1e-10, atol=1e-12).y.T
    p = model.rollout(np.array(x0), T=len(s), dt=DT, integrator="rk4")
    phase_pairs.append((s, p))

# Edge activations, also captured before snapping
train_theta = torch.tensor(lv_lift(X[:2048]), dtype=torch.float32)

# ---------------------------------------------------------------------------
# 5. Symbolic extraction
#
#    Each edge is a QUADRATIC in its own input, so the library must contain
#    'x^2'.  weight_simple=0.0 removes the simplicity pressure that otherwise
#    snaps a real quadratic edge to '0' — with the default 0.8 the N1 and N2
#    edges are discarded and only the cross-term survives.
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Extracting formulas ...")
formulas = model.get_formula(
    var_names=FEATURE_NAMES,
    round_places=6,
    lib=["x", "x^2", "0"],
    r2_threshold=0.80,
    weight_simple=0.0,
)


def polish(expr, tol: float = 1e-2, places: int = 3):
    """Expand a snapped formula and drop numerically negligible terms.

    PyKAN returns each edge in the affine-composed form c*f(a*x + b) + d, e.g.
    -0.008*(6.13 - 8.17*N1)**2.0, which hides the coefficients.  Expanding
    reveals them; pruning removes the tiny cross-products that expansion
    generates.  Float exponents are rationalised so SymPy will expand them.
    """
    rationalise = lambda ex: ex.replace(
        lambda x: x.is_Pow and x.exp.is_Float,
        lambda x: sp.Pow(x.base, sp.Integer(round(float(x.exp)))),
    )
    e = sp.expand(rationalise(sp.expand(expr)))
    kept = [t for t in sp.Add.make_args(e)
            if abs(float(t.as_coeff_Mul()[0])) >= tol]
    e = sp.Add(*kept) if kept else sp.Integer(0)
    # Round only float coefficients — rounding Integers would turn the
    # rationalised exponent 2 back into 2.0.
    return e.xreplace({n: sp.Float(round(float(n), places)) for n in e.atoms(sp.Float)})


for lab, f in zip(["dN1/dt", "dN2/dt"], formulas):
    print(f"  {lab} = {polish(f)}")

r2 = model.score_formula(formulas, X, X_dot, var_names=FEATURE_NAMES)
print(f"[SYMBOLIC] Formula R^2 per equation: {np.round(r2, 6)}")
print(f"[SYMBOLIC] True:  dN1/dt = {R1}*N1 - {R1 / K1:.3f}*N1**2 - {R1 * A12 / K1:.3f}*N1*N2")
print(f"[SYMBOLIC]        dN2/dt = {R2}*N2 - {R2 / K2:.3f}*N2**2 - {R2 * A21 / K2:.3f}*N1*N2")

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/LotkaVolterraCompetition", exist_ok=True)

# 6a. Time series
fig, ax = plt.subplots(figsize=(9, 4))
for ci, (lab, c) in enumerate(zip(["N1", "N2"], ["#1f77b4", "#d62728"])):
    ax.plot(t_eval, true_traj[:, ci], color=c, lw=1.4, label=f"{lab} true")
    ax.plot(t_eval, pred_traj[:, ci], color=c, lw=1.0, ls="--", label=f"{lab} KANDy")
ax.axhline(N1_EQ, color="#1f77b4", lw=0.6, ls=":", alpha=0.7)
ax.axhline(N2_EQ, color="#d62728", lw=0.6, ls=":", alpha=0.7)
ax.set_xlabel("time")
ax.set_ylabel("population density")
ax.set_title("Lotka–Volterra competition: approach to coexistence")
ax.legend(loc="best", fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig("results/LotkaVolterraCompetition/timeseries.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6b. Phase plane — several initial conditions, true vs learned
fig, ax = plt.subplots(figsize=(5.5, 5))
for s, p in phase_pairs:
    ax.plot(s[:, 0], s[:, 1], color="#1f77b4", lw=1.4, alpha=0.7)
    ax.plot(p[:, 0], p[:, 1], color="#d62728", lw=1.0, ls="--")
ax.plot(N1_EQ, N2_EQ, "k*", ms=12, label="coexistence equilibrium")
ax.plot([], [], color="#1f77b4", lw=1.4, label="true")
ax.plot([], [], color="#d62728", lw=1.0, ls="--", label="KANDy")
ax.set_xlabel("$N_1$")
ax.set_ylabel("$N_2$")
ax.set_title("Phase plane")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("results/LotkaVolterraCompetition/phase_plane.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6c. Trajectory error
fig, ax = plot_trajectory_error(
    true_traj, pred_traj, t=t_eval,
    save="results/LotkaVolterraCompetition/trajectory_error",
)
plt.close(fig)

# 6d. Loss curves
if getattr(model, "train_results_", None):
    fig, ax = plot_loss_curves(
        model.train_results_, save="results/LotkaVolterraCompetition/loss_curves",
    )
    plt.close(fig)

# 6e. Edge activations — N1 and N2 edges are parabolas, the cross-term is a line
fig, axes = plot_all_edges(
    model.model_, X=train_theta,
    in_var_names=FEATURE_NAMES,
    out_var_names=["dN1/dt", "dN2/dt"],
    save="results/LotkaVolterraCompetition/edge_activations",
)
plt.close(fig)

print("[FIGS]  Saved results/LotkaVolterraCompetition/")
