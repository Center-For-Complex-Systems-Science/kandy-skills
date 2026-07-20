#!/usr/bin/env python3
"""KANDy example: Trefoil knot via Hopf fibration.

The Hopf fibration pi: S^3 -> S^2 maps unit quaternions to points on the
2-sphere.  Every output involves bilinear products x_i * x_j, so the map
is NOT separable in the raw coordinates -- a single-layer KAN computing
sum_i a_ij psi_i(x_i) cannot represent it (zero-set corollary: the zero
set of a separable function is a union of hyperplanes, but x1*x3 = 0 is
not).  KANDy resolves this by pre-computing all required cross-terms in a
Koopman lift phi, so the KAN only needs to learn a linear map on the
lifted features.

A (2,3)-torus knot (trefoil) is a closed curve on S^3 parametrised via
the Clifford torus decomposition of S^3 into two complex coordinates
(z1, z2) with |z1|^2 + |z2|^2 = 1:

    z1 = cos(alpha) * e^{i * 2t},   z2 = sin(alpha) * e^{i * 3t}

The real embedding is (Re z1, Im z1, Re z2, Im z2) in R^4.  A second
component with phase offsets gives a *linked pair* -- two trefoils that
cannot be separated without cutting, a topological invariant of S^3.

This script tests KANDy on data with nontrivial geometric structure:
trefoil knots are 1D submanifolds of S^3 that wind non-trivially, with
strong inter-coordinate correlations that could fool a naive learner.
The linked pair tests whether KANDy preserves the topological relationship
-- both knots' Hopf images should be distinct closed curves on S^2.

The Koopman lift phi: R^4 -> R^5 computes:

    theta_1 = x1*x3 + x2*x4       (= p1 / 2)
    theta_2 = x2*x3 - x1*x4       (= p2 / 2)
    theta_3 = x1^2 + x2^2 - x3^2 - x4^2   (= p3)
    theta_4 = x1^2 + x2^2          (|z1|^2, Hopf-fiber invariant)
    theta_5 = x3^2 + x4^2          (|z2|^2, Hopf-fiber invariant)

The first three are exactly the Hopf components (up to a factor of 2).
theta_4 and theta_5 are fiber invariants -- constant along each Hopf
fiber -- included for completeness but expected to be zeroed by the KAN.

Expected KAN structure (perfect sparsity):
  - 3 active edges (theta_1, theta_2, theta_3): all linear
  - 2 dead edges (theta_4, theta_5): zeroed out
  - Linear mixing matrix A recovers the factor of 2 on p1, p2
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kandy import KANDy, CustomLift
from kandy.plotting import (
    get_all_edge_activations,
    plot_all_edges,
    plot_loss_curves,
    use_pub_style,
)

# ---------------------------------------------------------------------------
# 0. Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cpu"

RESULTS_DIR = "results/TrefoilKnot"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Geometry helpers
# ---------------------------------------------------------------------------

def hopf_map(x4: np.ndarray) -> np.ndarray:
    """Hopf fibration pi: S^3 -> S^2.

    pi(x1,x2,x3,x4) = (2(x1*x3 + x2*x4),
                        2(x2*x3 - x1*x4),
                        x1^2 + x2^2 - x3^2 - x4^2)

    Equivalently, writing z1 = x1+ix2, z2 = x3+ix4:
        (2 Re(z1 conj(z2)),  2 Im(z1 conj(z2)),  |z1|^2 - |z2|^2)
    Every point on S^2 has a full circle (fiber) as its pre-image on S^3,
    and any two such fibers are linked -- this is the topological content
    of the fibration.
    """
    x1, x2, x3, x4v = x4[:, 0], x4[:, 1], x4[:, 2], x4[:, 3]
    p1 = 2.0 * (x1 * x3 + x2 * x4v)
    p2 = 2.0 * (x2 * x3 - x1 * x4v)
    p3 = x1**2 + x2**2 - x3**2 - x4v**2
    return np.column_stack([p1, p2, p3]).astype(np.float32)


def stereographic_s3_to_r3(x4: np.ndarray) -> np.ndarray:
    """Stereographic projection S^3 -> R^3 (from north pole x4=1).

    Used to visualise S^3 curves in 3D: projects (x1,x2,x3,x4) to
    (x1, x2, x3) / (1 - x4).  The trefoil knot, which lives on S^3,
    becomes a knotted curve in R^3.
    """
    denom = np.clip(1.0 - x4[:, 3], 1e-9, None)
    return x4[:, :3] / denom[:, None]


def torus_knot_s3(p: int, q: int, alpha: float, m: int,
                  phi1: float = 0.0, phi2: float = 0.0) -> np.ndarray:
    """(p,q)-torus knot on S^3 via Clifford torus parametrisation.

    The Clifford torus decomposes S^3 into pairs (z1, z2) in C^2 with
    |z1|^2 + |z2|^2 = 1.  Setting:

        z1 = cos(alpha) * e^{i*(p*t + phi1)}
        z2 = sin(alpha) * e^{i*(q*t + phi2)}

    traces a (p,q)-torus knot.  For (p,q) = (2,3) this is a trefoil.
    The parameter alpha controls the "latitude" on the Clifford torus;
    alpha = pi/4 places the knot on the symmetric torus where |z1| = |z2|.

    Returns (m, 4) real coordinates [Re z1, Im z1, Re z2, Im z2] on S^3.
    """
    t = np.linspace(0, 2 * np.pi, m, endpoint=False)
    z1 = np.cos(alpha) * np.exp(1j * (p * t + phi1))
    z2 = np.sin(alpha) * np.exp(1j * (q * t + phi2))
    return np.column_stack([z1.real, z1.imag, z2.real, z2.imag]).astype(np.float32)


def linked_trefoils(m: int, alpha: float = np.pi / 4,
                    delta_phi1: float = np.pi / 2,
                    delta_phi2: float = np.pi / 2):
    """Two linked (2,3)-trefoil knots on S^3.

    Phase offsets (delta_phi1, delta_phi2) shift the second component so
    that the two trefoils are geometrically linked -- they cannot be
    separated by a continuous deformation, reflecting the linking structure
    intrinsic to S^3.
    """
    K1 = torus_knot_s3(2, 3, alpha, m, phi1=0.0, phi2=0.0)
    K2 = torus_knot_s3(2, 3, alpha, m, phi1=delta_phi1, phi2=delta_phi2)
    return K1, K2


# ---------------------------------------------------------------------------
# 2. Generate data -- trefoils + random S^3 samples for training
# ---------------------------------------------------------------------------
M_KNOT = 2000

K1, K2 = linked_trefoils(M_KNOT)
print(f"[DATA]  Trefoil K1: {K1.shape}, K2: {K2.shape}")
print(f"        S^3 radius check: K1 max|q|={np.linalg.norm(K1, axis=1).max():.6f}, "
      f"K2 max|q|={np.linalg.norm(K2, axis=1).max():.6f}")

# Trefoils are 1D curves -- not enough to learn a map on all of S^3.
# Uniform random S^3 samples provide full-dimensional coverage.
N_TRAIN = 10_000
rng = np.random.default_rng(SEED)
Q_rand = rng.standard_normal((N_TRAIN, 4)).astype(np.float32)
Q_rand /= np.linalg.norm(Q_rand, axis=1, keepdims=True)

Q_all = np.concatenate([Q_rand, K1, K2], axis=0)
P_all = hopf_map(Q_all)

print(f"[DATA]  Total training samples: {len(Q_all)} "
      f"({N_TRAIN} random + {2*M_KNOT} trefoil)")

# ---------------------------------------------------------------------------
# 3. Koopman lift -- engineered bilinear features for Hopf map
#
# A single-layer KAN cannot represent x1*x3 from raw inputs (zero-set
# corollary).  The lift pre-computes all bilinear cross-terms so the KAN
# [5 -> 3] only needs to learn a linear map on the lifted coordinates.
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["x1x3+x2x4", "x2x3-x1x4", "x1sq+x2sq-x3sq-x4sq",
                 "x1sq+x2sq", "x3sq+x4sq"]


def hopf_lift_fn(X: np.ndarray) -> np.ndarray:
    """Koopman lift phi: R^4 -> R^5.

    theta_1 = x1*x3 + x2*x4           = p1/2  (Hopf component)
    theta_2 = x2*x3 - x1*x4           = p2/2  (Hopf component)
    theta_3 = x1^2+x2^2 - x3^2-x4^2  = p3    (Hopf component)
    theta_4 = x1^2 + x2^2             = |z1|^2 (fiber invariant)
    theta_5 = x3^2 + x4^2             = |z2|^2 (fiber invariant)

    theta_4 and theta_5 are constant along each Hopf fiber (they equal
    |z1|^2 and |z2|^2).  The KAN should learn to zero these edges.
    """
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    f1 = x1 * x3 + x2 * x4
    f2 = x2 * x3 - x1 * x4
    f3 = x1**2 + x2**2 - x3**2 - x4**2
    f4 = x1**2 + x2**2
    f5 = x3**2 + x4**2
    return np.column_stack([f1, f2, f3, f4, f5])


lift = CustomLift(fn=hopf_lift_fn, output_dim=5, name="hopf_bilinear")

# ---------------------------------------------------------------------------
# 4. Train KANDy
# ---------------------------------------------------------------------------
print("\n--- Training KANDy [5 -> 3] on Hopf map ---")
model = KANDy(lift=lift, grid=5, k=3, steps=200, seed=SEED, device=DEVICE)
model.fit(X=Q_all, X_dot=P_all, val_frac=0.15, test_frac=0.15, lamb=0.0)

# ---------------------------------------------------------------------------
# 5. Evaluate on trefoil data specifically
#
# Trefoil knots have strong inter-coordinate correlations (the four
# coordinates are locked to a 1D curve), so this is a harder test than
# random S^3 points -- a model that memorises isolated regions of S^3
# would fail here.
# ---------------------------------------------------------------------------
P_k1_true = hopf_map(K1)
P_k2_true = hopf_map(K2)
P_k1_pred = model.predict(K1)
P_k2_pred = model.predict(K2)

rmse_k1 = np.sqrt(np.mean((P_k1_pred - P_k1_true) ** 2))
rmse_k2 = np.sqrt(np.mean((P_k2_pred - P_k2_true) ** 2))
print(f"\n[EVAL]  Trefoil K1 RMSE: {rmse_k1:.6f}")
print(f"[EVAL]  Trefoil K2 RMSE: {rmse_k2:.6f}")

# ---------------------------------------------------------------------------
# 6. Symbolic extraction
#
# Expected: p1 = 2*theta_1,  p2 = 2*theta_2,  p3 = theta_3
# with theta_4, theta_5 absent (fiber invariants zeroed).
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Discovered Hopf map:")
try:
    formulas = model.get_formula(var_names=FEATURE_NAMES, round_places=3)
    for comp, f in zip(["p1", "p2", "p3"], formulas):
        print(f"  {comp} = {f}")
except Exception as exc:
    print(f"  Symbolic extraction failed: {exc}")

# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
use_pub_style()

# 7a. Three-panel figure: S^3 trefoils -> true S^2 image -> KANDy S^2 image
R3_k1 = stereographic_s3_to_r3(K1)
R3_k2 = stereographic_s3_to_r3(K2)

fig = plt.figure(figsize=(14, 5))

# Panel 1: linked trefoils in R^3 via stereographic projection
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
ax1.plot(R3_k1[:, 0], R3_k1[:, 1], R3_k1[:, 2], lw=1.8, alpha=0.9,
         color="#d62728", label="K1")
ax1.plot(R3_k2[:, 0], R3_k2[:, 1], R3_k2[:, 2], lw=1.8, alpha=0.9,
         color="#1f77b4", label="K2")
ax1.set_title("Linked trefoils in $\\mathbb{R}^3$\n(stereographic $S^3 \\to \\mathbb{R}^3$)")
ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
ax1.set_box_aspect((1, 1, 1))
ax1.view_init(elev=18, azim=55)
ax1.legend(fontsize=8)

# S^2 wireframe for panels 2-3
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 16)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))

# Panel 2: true Hopf images -- each trefoil maps to a closed curve on S^2
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
ax2.plot_wireframe(xs, ys, zs, rstride=2, cstride=2, lw=0.3, alpha=0.15, color="gray")
ax2.plot(P_k1_true[:, 0], P_k1_true[:, 1], P_k1_true[:, 2],
         lw=1.8, alpha=0.9, color="#d62728")
ax2.plot(P_k2_true[:, 0], P_k2_true[:, 1], P_k2_true[:, 2],
         lw=1.8, alpha=0.9, color="#1f77b4")
ax2.set_title("True Hopf image on $S^2$")
ax2.set_xlabel("$p_1$"); ax2.set_ylabel("$p_2$"); ax2.set_zlabel("$p_3$")
ax2.set_box_aspect((1, 1, 1))
ax2.view_init(elev=18, azim=55)

# Panel 3: KANDy predictions -- should reproduce the S^2 curves faithfully
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
ax3.plot_wireframe(xs, ys, zs, rstride=2, cstride=2, lw=0.3, alpha=0.15, color="gray")
ax3.plot(P_k1_pred[:, 0], P_k1_pred[:, 1], P_k1_pred[:, 2],
         lw=1.8, alpha=0.9, color="#d62728")
ax3.plot(P_k2_pred[:, 0], P_k2_pred[:, 1], P_k2_pred[:, 2],
         lw=1.8, alpha=0.9, color="#1f77b4")
ax3.set_title(f"KANDy prediction on $S^2$\nRMSE: {(rmse_k1+rmse_k2)/2:.2e}")
ax3.set_xlabel("$p_1$"); ax3.set_ylabel("$p_2$"); ax3.set_zlabel("$p_3$")
ax3.set_box_aspect((1, 1, 1))
ax3.view_init(elev=18, azim=55)

fig.tight_layout()
fig.savefig(f"{RESULTS_DIR}/trefoil_hopf.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{RESULTS_DIR}/trefoil_hopf.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7b. Edge activations -- expect 3 linear + 2 dead (perfect sparsity)
train_theta = torch.tensor(hopf_lift_fn(Q_all[:int(len(Q_all) * 0.70)]),
                           dtype=torch.float32)
fig, _ = plot_all_edges(
    model.model_,
    X=train_theta,
    in_var_names=FEATURE_NAMES,
    out_var_names=["p1", "p2", "p3"],
    save=f"{RESULTS_DIR}/edge_activations",
)
plt.close(fig)

# 7c. Loss curves
fig, _ = plot_loss_curves(model.train_results_, save=f"{RESULTS_DIR}/loss_curves")
plt.close(fig)

print(f"\n[FIGS]  Saved to {RESULTS_DIR}/")
