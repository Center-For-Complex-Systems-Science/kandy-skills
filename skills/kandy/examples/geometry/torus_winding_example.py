#!/usr/bin/env python3
"""KANDy example: the flat 2-torus in R^3 and its winding flows.

The torus T^2 = S^1 x S^1 is parametrised by two angles (theta1, theta2) in
(-pi, pi]^2.  Its standard embedding into R^3, with major radius R and minor
radius r, is

    X = (R + r*cos(theta2)) * cos(theta1)
    Y = (R + r*cos(theta2)) * sin(theta1)
    Z =  r*sin(theta2)

Expanded, the three ambient coordinates are LINEAR in exactly five features:

    X = R*cos(theta1) + r*cos(theta2)*cos(theta1)
    Y = R*sin(theta1) + r*cos(theta2)*sin(theta1)
    Z =                 r*sin(theta2)

    theta_lift = [ cos(t1), sin(t1), cos(t2)*cos(t1), cos(t2)*sin(t1), sin(t2) ]

so the true mixing matrix A (3 x 5) is

    [ R  0  r  0  0 ]
    [ 0  R  0  r  0 ]
    [ 0  0  0  0  r ]

Part 1 fits that embedding in MAP MODE (inputs -> outputs, no time
integration): X = angles, X_dot = ambient coordinates, exactly as
geometry/hopf_example.py does.  Part 2 pushes the linear winding flow

    dtheta1/dt = omega1,   dtheta2/dt = omega2

through the learned embedding: a rational ratio omega = (2, 3) gives a closed
(2,3)-torus knot (the trefoil family of geometry/trefoil_knot_example.py),
while an irrational ratio omega = (1, golden ratio) gives a dense line that
fills the surface.  Part 2 is illustration, not a second training run.

THE LESSON: trig x trig cross-terms
-----------------------------------
1. cos(theta2)*cos(theta1) is a product of functions of DIFFERENT angles, so
   it is precisely what a separable KAN cannot build: sum_i a_i psi_i(u_i)
   never produces a product of two of its inputs.  Supplying cos/sin of each
   angle SEPARATELY -- the angular analogue of a Fourier lift, and what
   KANDy's FourierLift gives for a periodic field -- is therefore NOT enough,
   no matter how many harmonics you add.  Model A below uses that harmonic
   lift and Model B the engineered five-feature lift; per-component RMSE shows
   Model A nailing Z = r*sin(theta2) (no cross-term) while missing the entire
   cross-term contribution to X and Y.  The best separable approximation of
   X is R*cos(theta1), because averaging the cross-term over theta2 kills it;
   the leftover is a structural error floor of about r/2 in RMS, not a
   training-budget problem.

2. Powers vs products, restated for trig.  You do NOT need cos^2(theta1) or
   any higher harmonic cos(k*theta1) in the lift: a spline edge already is an
   arbitrary univariate function, so it learns whatever function of cos(t1)
   the data demands.  Only the MIXED product needs its own coordinate.  This
   is the trig x trig case; unicycle-style kinematics give trig x linear and
   rigid-body (Euler top) dynamics give linear x linear.  Same rule each time.

3. Periodicity comes free with the lift.  The angles are sampled uniformly
   and independently over (-pi, pi]^2, and every lifted coordinate is built
   from cos/sin, so the learned map is automatically 2*pi-periodic in both
   angles -- it respects the topology of T^2 by construction and needs no
   samples near the seam to learn that theta = -pi and theta = +pi are the
   same point.  Feed raw angles to the network instead and the model has no
   way to know the domain wraps: it must infer the identification from data,
   it will not match derivatives across the seam, and any rollout that drifts
   outside the sampled interval leaves the spline grid entirely.  Choosing a
   lift is therefore also a way of declaring the geometry of the state space,
   not just the nonlinearity of the vector field.

Lift  phi: R^2 -> R^5   theta = [cos t1, sin t1, cos t2 cos t1, cos t2 sin t1, sin t2]
KAN:  width = [5, 3],  base_fun='zero'
"""

import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kandy import KANDy, CustomLift
from kandy.plotting import plot_all_edges, plot_loss_curves, use_pub_style

# ---------------------------------------------------------------------------
# 0. Reproducibility and geometry parameters
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

R_MAJOR = 2.0
R_MINOR = 0.8
RESULTS_DIR = "results/TorusWinding"

A_TRUE = np.array([
    [R_MAJOR, 0.0,     R_MINOR, 0.0,     0.0    ],
    [0.0,     R_MAJOR, 0.0,     R_MINOR, 0.0    ],
    [0.0,     0.0,     0.0,     0.0,     R_MINOR],
])

print(f"[MODEL] Torus T^2 -> R^3 with R = {R_MAJOR}, r = {R_MINOR}")
print(f"[MODEL] Aspect ratio R/r = {R_MAJOR / R_MINOR:.2f} > 1 -> embedded (no self-intersection)")


def wrap_pi(a: np.ndarray) -> np.ndarray:
    """Wrap angles into (-pi, pi]."""
    return -(np.mod(-a + np.pi, 2.0 * np.pi) - np.pi)


def torus_embedding(TH: np.ndarray) -> np.ndarray:
    """Exact embedding T^2 -> R^3 for angle pairs TH of shape (N, 2)."""
    t1, t2 = TH[:, 0], TH[:, 1]
    rho = R_MAJOR + R_MINOR * np.cos(t2)
    return np.column_stack([rho * np.cos(t1), rho * np.sin(t1), R_MINOR * np.sin(t2)])


def torus_residual(P: np.ndarray) -> np.ndarray:
    """Implicit surface residual (sqrt(X^2+Y^2) - R)^2 + Z^2 - r^2, zero on T^2."""
    rad = np.hypot(P[:, 0], P[:, 1])
    return (rad - R_MAJOR) ** 2 + P[:, 2] ** 2 - R_MINOR ** 2


# ---------------------------------------------------------------------------
# 1. Training data — independent uniform angle samples over (-pi, pi]^2
#
#    Random rather than gridded: a product grid makes cos(t1) and
#    cos(t2)*cos(t1) share a repeated factor pattern across rows, which
#    inflates the conditioning of the lifted design matrix.  Independent
#    draws keep the five features cleanly uncorrelated (E[cos t2] = 0 over a
#    full period is what orthogonalises cos(t1) against cos(t2)*cos(t1)).
# ---------------------------------------------------------------------------
N_TRAIN = 4000
N_TEST = 1500
rng = np.random.default_rng(SEED)

TH_train = wrap_pi(rng.uniform(-np.pi, np.pi, size=(N_TRAIN, 2)))
TH_test = wrap_pi(rng.uniform(-np.pi, np.pi, size=(N_TEST, 2)))
P_train = torus_embedding(TH_train)
P_test = torus_embedding(TH_test)

print(f"[DATA]  {N_TRAIN} train / {N_TEST} test angle pairs, uniform on (-pi, pi]^2")
print(f"[DATA]  ambient range: X,Y in [{P_train[:, 0].min():.2f}, {P_train[:, 0].max():.2f}], "
      f"Z in [{P_train[:, 2].min():.2f}, {P_train[:, 2].max():.2f}]")

# ---------------------------------------------------------------------------
# 2A. Model A lift — harmonics of each angle SEPARATELY (no cross-terms)
#
#     KANDy's FourierLift takes a periodic spatial FIELD u in R^{Nx} and
#     returns its leading Fourier coefficients; for a two-angle state the
#     honest analogue is this hand-built harmonic lift, which supplies
#     cos(k*t1), sin(k*t1), cos(k*t2), sin(k*t2) for k = 1..n_modes.  Every
#     coordinate is a function of ONE angle, so the whole lift is separable
#     and the cross-term is unreachable at any n_modes.
# ---------------------------------------------------------------------------
N_MODES = 3
HARM_FEATURE_NAMES = [
    f"{fn}({k}t{i})" for k in range(1, N_MODES + 1) for i in (1, 2) for fn in ("cos", "sin")
]


def harmonic_lift(TH: np.ndarray) -> np.ndarray:
    t1, t2 = TH[:, 0], TH[:, 1]
    cols = []
    for k in range(1, N_MODES + 1):
        cols += [np.cos(k * t1), np.sin(k * t1), np.cos(k * t2), np.sin(k * t2)]
    return np.column_stack(cols)


harm_lift = CustomLift(fn=harmonic_lift, output_dim=4 * N_MODES, name="angular_harmonics")

# ---------------------------------------------------------------------------
# 2B. Model B lift — the engineered five features
#
#     Note what is ABSENT: no cos^2(t1), no cos(2*t1), no sin(t1)*sin(t2).
#     Only the two genuine mixed products cos(t2)*cos(t1) and cos(t2)*sin(t1)
#     are added to the four first-harmonic terms that actually appear (and
#     cos(t2) alone is dropped too — nothing in the embedding uses it on its
#     own).  Five coordinates, and the KAN's job reduces to a linear map.
# ---------------------------------------------------------------------------
ENG_FEATURE_NAMES = ["cos_t1", "sin_t1", "cos_t2_cos_t1", "cos_t2_sin_t1", "sin_t2"]


def torus_lift(TH: np.ndarray) -> np.ndarray:
    t1, t2 = TH[:, 0], TH[:, 1]
    c1, s1, c2, s2 = np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2)
    return np.column_stack([c1, s1, c2 * c1, c2 * s1, s2])


eng_lift = CustomLift(fn=torus_lift, output_dim=5, name="torus_embedding_lift")

Theta_train = torus_lift(TH_train)
cond = np.linalg.cond(np.column_stack([Theta_train, np.ones(len(Theta_train))]))
print(f"[DATA]  cond([Theta, 1]) for the engineered lift = {cond:.3f}  (near 1 -> identifiable)")
cond_h = np.linalg.cond(np.column_stack([harmonic_lift(TH_train), np.ones(N_TRAIN)]))
print(f"[DATA]  cond([Theta, 1]) for the harmonic lift    = {cond_h:.3f}")

# ---------------------------------------------------------------------------
# 3. Train both models in MAP MODE (X = angles, X_dot = ambient coordinates)
# ---------------------------------------------------------------------------
print(f"\n--- Model A: angular harmonics, no cross-terms (KAN=[{4 * N_MODES},3]) ---")
model_harm = KANDy(lift=harm_lift, grid=5, k=3, steps=150, seed=SEED, base_fun="zero")
model_harm.fit(X=TH_train, X_dot=P_train, val_frac=0.15, test_frac=0.15,
               lamb=0.0, patience=50)

print("\n--- Model B: engineered torus lift (KAN=[5,3]) ---")
model_eng = KANDy(lift=eng_lift, grid=5, k=3, steps=150, seed=SEED, base_fun="zero")
model_eng.fit(X=TH_train, X_dot=P_train, val_frac=0.15, test_frac=0.15,
              lamb=0.0, patience=50)

# ---------------------------------------------------------------------------
# 4. Evaluation on held-out angles — per-component RMSE
# ---------------------------------------------------------------------------
pred_harm = model_harm.predict(TH_test)
pred_eng = model_eng.predict(TH_test)

rmse_harm = np.sqrt(np.mean((pred_harm - P_test) ** 2, axis=0))
rmse_eng = np.sqrt(np.mean((pred_eng - P_test) ** 2, axis=0))

print("\n[EVAL]  Held-out RMSE per ambient component")
print(f"          {'':<10}{'X':>12}{'Y':>12}{'Z':>12}{'total':>12}")
print(f"          {'Model A':<10}" + "".join(f"{v:>12.6f}" for v in rmse_harm)
      + f"{np.sqrt(np.mean((pred_harm - P_test) ** 2)):>12.6f}")
print(f"          {'Model B':<10}" + "".join(f"{v:>12.6f}" for v in rmse_eng)
      + f"{np.sqrt(np.mean((pred_eng - P_test) ** 2)):>12.6f}")
print(f"[EVAL]  Model A structural floor on X, Y: r/2 = {R_MINOR / 2:.4f} "
      "(RMS of the discarded cross-term)")
print(f"[EVAL]  Model A gets Z right ({rmse_harm[2]:.2e}) because Z = r*sin(t2) "
      "has no cross-term.")

# Does the learned image actually lie ON the torus?
res_eng = torus_residual(pred_eng)
res_harm = torus_residual(pred_harm)
print(f"\n[EVAL]  Implicit surface residual (sqrt(X^2+Y^2)-R)^2 + Z^2 - r^2, "
      "mean |.| over test points")
print(f"          Model A: {np.abs(res_harm).mean():.6f}   max {np.abs(res_harm).max():.6f}")
print(f"          Model B: {np.abs(res_eng).mean():.6f}   max {np.abs(res_eng).max():.6f}")

# Recovered mixing matrix.  get_A() returns the per-edge spline SCALE, whose
# split against the spline's own amplitude is a normalisation convention, so
# it shows the sparsity pattern rather than R and r.  The physical
# coefficients come from least-squares of the network output on the lifted
# features (and, below, from the symbolic formulas).
A_hat = model_eng.get_A()
Theta_test = torus_lift(TH_test)
A_eff = np.linalg.lstsq(Theta_test, pred_eng, rcond=None)[0].T

np.set_printoptions(precision=3, suppress=True)
print("\n[EVAL]  True A (3 x 5), columns "
      "[cos t1, sin t1, cos t2 cos t1, cos t2 sin t1, sin t2]:")
print(A_TRUE)
print("[EVAL]  model.get_A() — raw spline scales (sparsity pattern):")
print(A_hat)
print("[EVAL]  Effective A — least squares of network output on lifted features:")
print(A_eff)
print(f"[EVAL]  max |A_eff - A_true| = {np.abs(A_eff - A_TRUE).max():.6f}")

# ---------------------------------------------------------------------------
# 5. The winding flow — dtheta1/dt = omega1, dtheta2/dt = omega2
#
#    Linear in the angles, so no fitting is needed: integrating it is exact.
#    Rational omega1/omega2 = p/q closes after one period (a (p,q)-torus
#    knot); irrational never closes and the orbit is dense in T^2.  Both are
#    pushed through the LEARNED embedding to see the fit on curves it was
#    never shown as curves.
# ---------------------------------------------------------------------------
def winding(omega, t):
    return wrap_pi(np.column_stack([omega[0] * t, omega[1] * t]))


OMEGA_RAT = (2.0, 3.0)
PHI = 0.5 * (1.0 + np.sqrt(5.0))
OMEGA_IRR = (1.0, PHI)

t_rat = np.linspace(0.0, 2.0 * np.pi, 3000)
TH_rat = winding(OMEGA_RAT, t_rat)
P_rat_true = torus_embedding(TH_rat)
P_rat_pred = model_eng.predict(TH_rat)
rmse_rat = np.sqrt(np.mean((P_rat_pred - P_rat_true) ** 2))

t_irr = np.linspace(0.0, 300.0, 20000)
TH_irr = winding(OMEGA_IRR, t_irr)
P_irr_true = torus_embedding(TH_irr)
P_irr_pred = model_eng.predict(TH_irr)
rmse_irr = np.sqrt(np.mean((P_irr_pred - P_irr_true) ** 2))

closure_rat = np.linalg.norm(P_rat_true[-1] - P_rat_true[0])
closure_irr = np.linalg.norm(P_irr_true[-1] - P_irr_true[0])
print(f"\n[EVAL]  Rational winding omega = {OMEGA_RAT} -> (2,3) torus knot, "
      f"closes: |x(T)-x(0)| = {closure_rat:.2e}")
print(f"[EVAL]  Irrational winding omega = (1, phi={PHI:.6f}) -> dense line, "
      f"does not close: |x(T)-x(0)| = {closure_irr:.3f}")
print(f"[EVAL]  Embedding RMSE along the rational orbit:   {rmse_rat:.6f}")
print(f"[EVAL]  Embedding RMSE along the irrational orbit: {rmse_irr:.6f}")

# Edge activations are captured now, before symbolic extraction mutates the model.
train_theta_t = torch.tensor(Theta_train[:2048], dtype=torch.float32)

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs(RESULTS_DIR, exist_ok=True)

# Surface mesh for the 3D panels
gu = np.linspace(-np.pi, np.pi, 120)
gv = np.linspace(-np.pi, np.pi, 60)
GU, GV = np.meshgrid(gu, gv)
SX = (R_MAJOR + R_MINOR * np.cos(GV)) * np.cos(GU)
SY = (R_MAJOR + R_MINOR * np.cos(GV)) * np.sin(GU)
SZ = R_MINOR * np.sin(GV)


def draw_torus(ax):
    ax.plot_surface(SX, SY, SZ, rstride=2, cstride=2, color="#cccccc",
                    alpha=0.18, linewidth=0, antialiased=True, shade=True)
    ax.set_box_aspect((1, 1, 0.45))
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=32, azim=40)


# 6a. (2,3) winding curve — true vs KANDy
fig = plt.figure(figsize=(11, 5))
for i, (P, ttl, col) in enumerate([
    (P_rat_true, "True (2,3) winding", "#d62728"),
    (P_rat_pred, f"KANDy embedding (RMSE {rmse_rat:.2e})", "#1f77b4"),
]):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    draw_torus(ax)
    ax.plot(P[:, 0], P[:, 1], P[:, 2], lw=1.6, color=col)
    ax.set_title(ttl, fontsize=10)
fig.suptitle("Rational winding on $T^2$: a closed (2,3)-torus knot", fontsize=12)
fig.tight_layout()
fig.savefig(f"{RESULTS_DIR}/winding_rational.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{RESULTS_DIR}/winding_rational.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6b. Irrational winding — dense orbit through the learned embedding
fig = plt.figure(figsize=(10.5, 4.8))
ax = fig.add_subplot(1, 2, 1, projection="3d")
draw_torus(ax)
ax.plot(P_irr_pred[:, 0], P_irr_pred[:, 1], P_irr_pred[:, 2],
        lw=0.25, alpha=0.55, color="#2ca02c")
ax.set_title("KANDy embedding, $\\omega = (1, \\varphi)$", fontsize=10)
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(TH_irr[:, 0], TH_irr[:, 1], ".", ms=0.4, alpha=0.35, color="#2ca02c")
ax2.set_xlabel("$\\theta_1$"); ax2.set_ylabel("$\\theta_2$")
ax2.set_xlim(-np.pi, np.pi); ax2.set_ylim(-np.pi, np.pi)
ax2.set_aspect("equal")
ax2.set_title("Same orbit on the fundamental domain", fontsize=10)
fig.suptitle("Irrational winding: the orbit is dense in $T^2$", fontsize=12)
fig.tight_layout()
fig.savefig(f"{RESULTS_DIR}/winding_irrational.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{RESULTS_DIR}/winding_irrational.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6c. Per-component RMSE — harmonic lift vs engineered lift
fig, ax = plt.subplots(figsize=(5.2, 3.4))
idx = np.arange(3)
w = 0.36
ax.bar(idx - w / 2, rmse_harm, w, color="#d62728",
       label=f"harmonics only [{4 * N_MODES}→3]")
ax.bar(idx + w / 2, rmse_eng, w, color="#2ca02c", label="engineered [5→3]")
ax.axhline(R_MINOR / 2, color="k", ls=":", lw=0.8)
ax.text(2.35, R_MINOR / 2 * 1.15, "$r/2$", fontsize=8)
ax.set_yscale("log")
ax.set_xticks(idx); ax.set_xticklabels(["X", "Y", "Z"])
ax.set_ylabel("held-out RMSE")
ax.set_title("Missing cross-term = structural error on X and Y")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3, ls="--")
fig.tight_layout()
fig.savefig(f"{RESULTS_DIR}/rmse_per_component.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{RESULTS_DIR}/rmse_per_component.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6d. Loss curves for both models
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, res, ttl in [
    (axes[0], getattr(model_harm, "train_results_", None), "Model A: harmonics only"),
    (axes[1], getattr(model_eng, "train_results_", None), "Model B: engineered lift"),
]:
    if res:
        plot_loss_curves(res, ax=ax)
    ax.set_title(ttl, fontsize=10)
fig.tight_layout()
fig.savefig(f"{RESULTS_DIR}/loss_curves.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{RESULTS_DIR}/loss_curves.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6e. Edge activations for the engineered model — every edge should be a line
fig, _axes = plot_all_edges(
    model_eng.model_, X=train_theta_t,
    in_var_names=ENG_FEATURE_NAMES,
    out_var_names=["X", "Y", "Z"],
    save=f"{RESULTS_DIR}/edge_activations",
)
plt.close(fig)

print(f"\n[FIGS]  Saved {RESULTS_DIR}/")

# ---------------------------------------------------------------------------
# 7. Symbolic extraction — LAST, because get_formula() snaps every spline
#    edge to a surrogate in place and all later predictions use the surrogate.
#    Every edge here is linear, so the library is just {x, 0} and
#    weight_simple=0.0 keeps the simplicity prior from zeroing live edges.
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Extracting the embedding formulas ...")
formulas = model_eng.get_formula(
    var_names=ENG_FEATURE_NAMES, round_places=4,
    lib=["x", "0"], r2_threshold=0.80, weight_simple=0.0,
)
for comp, f in zip(["X", "Y", "Z"], formulas):
    print(f"  {comp} = {f}")

r2 = model_eng.score_formula(formulas, TH_test, P_test, var_names=ENG_FEATURE_NAMES)
print(f"[SYMBOLIC] Formula R^2 per component: {np.round(r2, 6)}")
print(f"[SYMBOLIC] True:  X = {R_MAJOR}*cos_t1 + {R_MINOR}*cos_t2_cos_t1")
print(f"[SYMBOLIC]        Y = {R_MAJOR}*sin_t1 + {R_MINOR}*cos_t2_sin_t1")
print(f"[SYMBOLIC]        Z = {R_MINOR}*sin_t2")
