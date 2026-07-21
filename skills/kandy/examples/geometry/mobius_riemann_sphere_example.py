#!/usr/bin/env python3
"""KANDy example: Mobius transformations on the Riemann sphere.

The Mobius transformations are the conformal automorphisms of the Riemann
sphere CP^1 = C u {infinity}:

    z_{n+1} = (a*z_n + b) / (c*z_n + d),     a, b, c, d in C,  a*d - b*c != 0

Two KANDy models are trained.

MODEL A — the Mobius MAP in real coordinates, z = x + i*y.
Multiplying numerator and denominator by conj(c*z + d) clears the complex
division:

    (a*z + b)/(c*z + d) = (a*z + b) * conj(c*z + d) / |c*z + d|^2
        = [ A*|z|^2 + B*z + C*conj(z) + D ] * u,
      A = a*conj(c),  B = a*conj(d),  C = b*conj(c),  D = b*conj(d),
      u = 1 / |c*z + d|^2 = 1 / ((c1*x - c2*y + d1)^2 + (c1*y + c2*x + d2)^2)

Taking real and imaginary parts, BOTH real outputs are LINEAR in four real
features theta4 = [u, x*u, y*u, r2*u] with r2 = x^2 + y^2:

    x_{n+1} = Re(D)*u + (Re(B)+Re(C))*x*u + (Im(C)-Im(B))*y*u + Re(A)*r2*u
    y_{n+1} = Im(D)*u + (Im(B)+Im(C))*x*u + (Re(B)-Re(C))*y*u + Im(A)*r2*u

The script verifies this against direct complex arithmetic to machine
precision before training — and then throws one of the four features away,
for the reason in point 3 below.

MODEL B — the geometric companion: stereographic projection R^2 -> S^2, the
chart that turns the plane into the Riemann sphere,

    p = ( 2x,  2y,  x^2 + y^2 - 1 ) / (1 + x^2 + y^2)

With w = 1/(1 + x^2 + y^2) this is p = (2*x*w, 2*y*w, 1 - 2*w): three outputs
linear in theta = [x*w, y*w, w].

THE LESSON: the skill's only RATIONAL lift
------------------------------------------
1.  Every other example lifts with polynomials, trigonometric terms or radial
    bases.  Here the essential feature is a DENOMINATOR, u = 1/|c*z + d|^2.
    The governing rule is unchanged — "encode what the KAN cannot build" — but
    what cannot be built is not a product this time, it is a RECIPROCAL THAT
    COUPLES BOTH VARIABLES.  A spline edge is an arbitrary univariate
    function, so 1/x of a single coordinate would be free; but
    1/((c1*x - c2*y + d1)^2 + (c1*y + c2*x + d2)^2) mixes x and y INSIDE the
    denominator and is therefore inseparable, exactly like x*y.  It has to be
    a lift coordinate.  Once it is, x*u, y*u and r2*u are the only other
    ingredients needed and the map is exactly linear in the lifted features.

2.  WHERE THE RECIPROCAL IS DANGEROUS.  At the pole z = -d/c the denominator
    vanishes and u -> infinity.  Sample near it and the lifted features have
    unbounded range: the spline grid stretches to cover a handful of enormous
    outliers, every ordinary sample collapses into one or two grid cells, and
    the fit diverges or flatlines.  This script handles it by REJECTION
    SAMPLING — uniform over the disc |z| <= R_SAMPLE minus the disc
    |z - pole| < POLE_EXCLUSION — and it prints the min/max of every lifted
    feature, |c*z + d| itself, and np.linalg.cond of the lifted matrix.  If a
    rational lift will not fit, read those diagnostics first: a feature range
    spanning several decades means you are sampling into the pole, not that
    the optimiser needs more steps.  The orbit seeds below are also chosen so
    that the whole ITERATED orbit stays inside the sampled disc — a map-mode
    rollout that wanders outside the training region is extrapolating.

3.  THE FOUR-FEATURE LIFT IS RANK-DEFICIENT, AND THE CONDITION NUMBER SAYS SO.
    Expanding the denominator gives the exact identity

        |c*z + d|^2 = |c|^2*r2 + 2*Re(c*conj(d))*x - 2*Im(c*conj(d))*y + |d|^2

    and multiplying through by u turns it into

        1 = |c|^2*(r2*u) + 2*g1*(x*u) - 2*g2*(y*u) + |d|^2*u,   g = c*conj(d)

    i.e. the CONSTANT FUNCTION lies in the span of theta4.  Since a KAN edge
    is affine-plus-spline, the network already has a bias, so [theta4 | 1] is
    exactly singular — cond ~ 1e16, printed below — and the coefficients are
    only defined up to adding multiples of that identity.  This is rule 2 of
    references/lifts.md ("lift coordinates must be functionally independent")
    in its sharpest form: the fit would be perfect and the recovered constants
    meaningless.  The cure is to solve the identity for u and DROP it:

        theta = [x*u, y*u, r2*u]      (cond ~ 10, coefficients unique)

    with the corresponding true coefficients derived analytically below.
    Model B has no such problem: its identity w + r2*w = 1 involves r2*w,
    which is NOT one of its lift coordinates, so [theta_B | 1] is benign.

4.  Once the pole is excluded and the redundant coordinate is dropped, both
    models are essentially exact (R^2 ~ 1) with STRAIGHT-LINE edges, because
    the lift has already done all the work.  The KAN is not approximating a
    Mobius transformation, it is reading off its coefficients — and the
    recovered 2x2 block on [x*u, y*u] comes out as a similarity matrix
    [[p, -q], [q, p]], which is conformality made visible.

Symbolic extraction runs LAST — get_formula() replaces the spline edges with
their snapped surrogates in place, so every numeric check and every figure is
produced before it is called.

Lift phi_A: R^2 -> R^3   theta = [x*u, y*u, (x^2+y^2)*u],  u = 1/|c*z+d|^2
KAN A: width = [3, 2],  base_fun='zero'
Lift phi_B: R^2 -> R^3   theta = [x*w, y*w, w],  w = 1/(1 + x^2 + y^2)
KAN B: width = [3, 3],  base_fun='zero'
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
# 0. Reproducibility and the Mobius transformation
#
#    Build a LOXODROMIC map by conjugating the multiplier w -> k*w (|k| != 1,
#    k not real) with the chart S(z) = (z - P_FIX)/(z - Q_FIX) that sends the
#    fixed points to 0 and infinity.  Orbits then spiral away from the
#    repelling fixed point Q_FIX and into the attracting one P_FIX.
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

P_FIX = 0.6 + 0.4j       # attracting fixed point
Q_FIX = -1.0 - 0.8j      # repelling fixed point
K_MULT = 0.86 * np.exp(1j * 0.62)   # multiplier: |k| < 1 and arg(k) != 0

A_C = P_FIX - K_MULT * Q_FIX
B_C = P_FIX * Q_FIX * (K_MULT - 1.0)
C_C = 1.0 - K_MULT
D_C = K_MULT * P_FIX - Q_FIX

DET = A_C * D_C - B_C * C_C
POLE = -D_C / C_C        # z where c*z + d = 0  ->  u blows up

print("[MODEL] Mobius map  z -> (a z + b)/(c z + d)")
print(f"[MODEL]   a = {A_C:.6f}")
print(f"[MODEL]   b = {B_C:.6f}")
print(f"[MODEL]   c = {C_C:.6f}")
print(f"[MODEL]   d = {D_C:.6f}")
print(f"[MODEL]   ad - bc = {DET:.6f}   (|ad - bc| = {abs(DET):.6f} != 0)")
print(f"[MODEL]   multiplier k = {K_MULT:.6f}  ->  LOXODROMIC (|k| = {abs(K_MULT):.3f})")
print(f"[MODEL]   fixed points: attracting {P_FIX}, repelling {Q_FIX}")
print(f"[MODEL]   POLE at z = -d/c = {POLE:.6f}   (|pole| = {abs(POLE):.4f})")


def mobius_complex(z: np.ndarray) -> np.ndarray:
    """Ground truth, by direct complex arithmetic."""
    return (A_C * z + B_C) / (C_C * z + D_C)


# --- analytic coefficients for the four-feature lift [u, x*u, y*u, r2*u] ----
_A = A_C * np.conj(C_C)
_B = A_C * np.conj(D_C)
_C = B_C * np.conj(C_C)
_D = B_C * np.conj(D_C)

M_TRUE_4 = np.array([
    [_D.real, _B.real + _C.real, _C.imag - _B.imag, _A.real],
    [_D.imag, _B.imag + _C.imag, _B.real - _C.real, _A.imag],
])

# --- eliminate u via  1 = |c|^2*r2u + 2*g1*xu - 2*g2*yu + |d|^2*u ----------
G_CD = C_C * np.conj(D_C)
ABS_C2 = abs(C_C) ** 2
ABS_D2 = abs(D_C) ** 2

M_TRUE_A = np.empty((2, 3))
M0_TRUE_A = np.empty(2)
for _i in range(2):
    _cu = M_TRUE_4[_i, 0]
    M0_TRUE_A[_i] = _cu / ABS_D2
    M_TRUE_A[_i, 0] = M_TRUE_4[_i, 1] - _cu * 2.0 * G_CD.real / ABS_D2
    M_TRUE_A[_i, 1] = M_TRUE_4[_i, 2] + _cu * 2.0 * G_CD.imag / ABS_D2
    M_TRUE_A[_i, 2] = M_TRUE_4[_i, 3] - _cu * ABS_C2 / ABS_D2

# ---------------------------------------------------------------------------
# 1. Lifts
# ---------------------------------------------------------------------------
FEATURES_4 = ["u", "x*u", "y*u", "r2*u"]
FEATURES_A = ["x*u", "y*u", "r2*u"]
FEATURES_B = ["x*w", "y*w", "w"]


def _u_of(X: np.ndarray) -> np.ndarray:
    """u = 1/|c*z + d|^2 in real coordinates."""
    x, y = X[:, 0], X[:, 1]
    dr = C_C.real * x - C_C.imag * y + D_C.real
    di = C_C.real * y + C_C.imag * x + D_C.imag
    return 1.0 / (dr * dr + di * di)


def mobius_lift4(X: np.ndarray) -> np.ndarray:
    """The full derivation, [u, x*u, y*u, r2*u] — used only for diagnostics."""
    x, y = X[:, 0], X[:, 1]
    u = _u_of(X)
    return np.column_stack([u, x * u, y * u, (x * x + y * y) * u])


def mobius_lift(X: np.ndarray) -> np.ndarray:
    """phi_A: [x, y] -> [x*u, y*u, (x^2+y^2)*u]  (u eliminated, see LESSON 3)."""
    x, y = X[:, 0], X[:, 1]
    u = _u_of(X)
    return np.column_stack([x * u, y * u, (x * x + y * y) * u])


def stereographic_lift(X: np.ndarray) -> np.ndarray:
    """phi_B: [x, y] -> [x*w, y*w, w],  w = 1/(1 + x^2 + y^2)."""
    x, y = X[:, 0], X[:, 1]
    w = 1.0 / (1.0 + x * x + y * y)
    return np.column_stack([x * w, y * w, w])


def stereographic(X: np.ndarray) -> np.ndarray:
    """Exact projection R^2 -> S^2 (ground truth for Model B and figure b)."""
    x, y = X[:, 0], X[:, 1]
    w = 1.0 / (1.0 + x * x + y * y)
    return np.column_stack([2.0 * x * w, 2.0 * y * w, 1.0 - 2.0 * w])


M_TRUE_B = np.array([[2.0, 0.0, 0.0],
                     [0.0, 2.0, 0.0],
                     [0.0, 0.0, -2.0]])
M0_TRUE_B = np.array([0.0, 0.0, 1.0])

# ---------------------------------------------------------------------------
# 2. Verify both derivations BEFORE training
#
#    If these do not pass at ~1e-12 the algebra is wrong and no amount of
#    training will rescue the fit.
# ---------------------------------------------------------------------------
_rng = np.random.default_rng(SEED)
_zc = _rng.normal(scale=1.2, size=4000) + 1j * _rng.normal(scale=1.2, size=4000)
_Xc = np.column_stack([_zc.real, _zc.imag])
_direct = np.column_stack([mobius_complex(_zc).real, mobius_complex(_zc).imag])

_e4 = np.abs(mobius_lift4(_Xc) @ M_TRUE_4.T - _direct).max()
_e3 = np.abs(mobius_lift(_Xc) @ M_TRUE_A.T + M0_TRUE_A - _direct).max()
_eb = np.abs(stereographic_lift(_Xc) @ M_TRUE_B.T + M0_TRUE_B
             - stereographic(_Xc)).max()
print(f"\n[MODEL] 4-feature lifted-linear form vs complex arithmetic: "
      f"max abs err = {_e4:.3e}")
print(f"[MODEL] 3-feature reduced form   vs complex arithmetic: "
      f"max abs err = {_e3:.3e}")
print(f"[MODEL] stereographic lifted-linear form:                "
      f"max abs err = {_eb:.3e}")
assert max(_e4, _e3, _eb) < 1e-9, "lift derivation is wrong"

# ---------------------------------------------------------------------------
# 3. Training data — uniform on a disc with a neighbourhood of the pole removed
# ---------------------------------------------------------------------------
N_SAMPLES = 6000
R_SAMPLE = 2.6           # sample the disc |z| <= R_SAMPLE ...
POLE_EXCLUSION = 1.2     # ... minus the disc |z - pole| < POLE_EXCLUSION

rng = np.random.default_rng(SEED)
kept, n_seen, n_rej = [], 0, 0
while sum(len(p) for p in kept) < N_SAMPLES:
    n = 4 * N_SAMPLES
    cand = (R_SAMPLE * np.sqrt(rng.uniform(size=n))
            * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=n)))
    ok = np.abs(cand - POLE) >= POLE_EXCLUSION
    n_seen += n
    n_rej += int((~ok).sum())
    kept.append(cand[ok])
Z = np.concatenate(kept)[:N_SAMPLES]
X = np.column_stack([Z.real, Z.imag])

_next = mobius_complex(Z)
Y_A = np.column_stack([_next.real, _next.imag])   # next state (map mode)
Y_B = stereographic(X)                            # point on S^2

Theta_4 = mobius_lift4(X)
Theta_A = mobius_lift(X)
Theta_B = stereographic_lift(X)

print(f"\n[DATA]  {N_SAMPLES} independent samples, uniform on the disc "
      f"|z| <= {R_SAMPLE} MINUS the disc |z - pole| < {POLE_EXCLUSION}")
print(f"[DATA]  rejection rate {100.0 * n_rej / n_seen:.2f}%   "
      f"closest retained sample to the pole: |z - pole| = "
      f"{np.abs(Z - POLE).min():.4f}")
print(f"[DATA]  |c*z + d| in [{np.sqrt(1.0 / Theta_4[:, 0]).min():.4f}, "
      f"{np.sqrt(1.0 / Theta_4[:, 0]).max():.4f}]   (0 would be the pole)")
for j, nm in enumerate(FEATURES_4):
    print(f"[DATA]    theta4[{nm:>4}] in "
          f"[{Theta_4[:, j].min():>9.4f}, {Theta_4[:, j].max():>9.4f}]")
for j, nm in enumerate(FEATURES_B):
    print(f"[DATA]    thetaB[{nm:>4}] in "
          f"[{Theta_B[:, j].min():>9.4f}, {Theta_B[:, j].max():>9.4f}]")

# The rank diagnostic of LESSON 3, made concrete.
ident = (ABS_C2 * Theta_4[:, 3] + 2.0 * G_CD.real * Theta_4[:, 1]
         - 2.0 * G_CD.imag * Theta_4[:, 2] + ABS_D2 * Theta_4[:, 0])
cond_4 = np.linalg.cond(np.column_stack([Theta_4, np.ones(len(Theta_4))]))
cond_A = np.linalg.cond(np.column_stack([Theta_A, np.ones(len(Theta_A))]))
cond_B = np.linalg.cond(np.column_stack([Theta_B, np.ones(len(Theta_B))]))
print(f"[DATA]  identity check  max |{ABS_C2:.4f}*r2u + {2 * G_CD.real:.4f}*xu "
      f"- {2 * G_CD.imag:.4f}*yu + {ABS_D2:.4f}*u - 1| = "
      f"{np.abs(ident - 1.0).max():.3e}")
print(f"[DATA]  cond(theta4 | 1) = {cond_4:.3e}   <-- SINGULAR: drop u")
print(f"[DATA]  cond(thetaA | 1) = {cond_A:.3e}   <-- usable lift")
print(f"[DATA]  cond(thetaB | 1) = {cond_B:.3e}")

lift_A = CustomLift(fn=mobius_lift, output_dim=3, name="mobius_rational_lift")
lift_B = CustomLift(fn=stereographic_lift, output_dim=3, name="stereographic_lift")

# ---------------------------------------------------------------------------
# 4. Train both models  (MAP MODE: X = current state, X_dot = target/next state)
# ---------------------------------------------------------------------------
print("\n--- Model A: Mobius map, rational lift (KAN=[3,2]) ---")
model_A = KANDy(lift=lift_A, grid=5, k=3, steps=200, seed=SEED, base_fun="zero")
model_A.fit(X=X, X_dot=Y_A, val_frac=0.15, test_frac=0.15, lamb=0.0, patience=40)

print("\n--- Model B: stereographic projection (KAN=[3,3]) ---")
model_B = KANDy(lift=lift_B, grid=5, k=3, steps=200, seed=SEED, base_fun="zero")
model_B.fit(X=X, X_dot=Y_B, val_frac=0.15, test_frac=0.15, lamb=0.0, patience=40)

# ---------------------------------------------------------------------------
# 5. Evaluation — all of it BEFORE symbolic extraction
# ---------------------------------------------------------------------------
n_test = int(len(X) * 0.15)
X_test = X[len(X) - n_test:]
YA_test = Y_A[len(X) - n_test:]
YB_test = Y_B[len(X) - n_test:]

pred_A = model_A.predict(X_test)
pred_B = model_B.predict(X_test)
rmse_A = np.sqrt(np.mean((pred_A - YA_test) ** 2))
rmse_B = np.sqrt(np.mean((pred_B - YB_test) ** 2))


def r2_per_output(y_true, y_pred):
    return [1.0 - np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
            / np.sum((y_true[:, i] - y_true[:, i].mean()) ** 2)
            for i in range(y_true.shape[1])]


print(f"\n[EVAL]  Model A test RMSE: {rmse_A:.3e}   "
      f"R^2 = {np.round(r2_per_output(YA_test, pred_A), 6)}")
print(f"[EVAL]  Model B test RMSE: {rmse_B:.3e}   "
      f"R^2 = {np.round(r2_per_output(YB_test, pred_B), 6)}")


def effective_coefficients(model, Theta, X_raw):
    """Read the network out as an affine map  y = M @ theta + m0.

    Every edge is (should be) a straight line, so the whole network collapses
    to an affine function of the lifted features.  Regressing the NETWORK
    OUTPUT on theta therefore recovers the coefficients the KAN actually
    learned, which is what we compare against the analytic matrices.  This is
    only well posed because [Theta | 1] is well conditioned — see LESSON 3.
    ``get_A()`` returns PyKAN's per-edge spline SCALE factors, a different and
    un-normalised quantity, so it is printed alongside rather than instead.
    """
    G = np.column_stack([Theta, np.ones(len(Theta))])
    coef, *_ = np.linalg.lstsq(G, model.predict(X_raw), rcond=None)
    return coef[:-1].T, coef[-1]


M_hat_A, m0_hat_A = effective_coefficients(model_A, Theta_A, X)
M_hat_B, m0_hat_B = effective_coefficients(model_B, Theta_B, X)

np.set_printoptions(precision=4, suppress=True, linewidth=120)
print(f"\n[EVAL]  Model A   theta = {FEATURES_A}   (rows: x_next, y_next)")
print("[EVAL]    true M_A    =\n", M_TRUE_A)
print("[EVAL]    learned M_A =\n", M_hat_A)
print("[EVAL]    intercept   true", M0_TRUE_A, " learned", m0_hat_A)
print(f"[EVAL]    max |M_A_hat - M_A| = {np.abs(M_hat_A - M_TRUE_A).max():.3e}   "
      f"max |m0_hat - m0| = {np.abs(m0_hat_A - M0_TRUE_A).max():.3e}")
print("[EVAL]    model_A.get_A()  (PyKAN per-edge spline scales) =\n",
      model_A.get_A())

print(f"\n[EVAL]  Model B   theta = {FEATURES_B}   (rows: p1, p2, p3)")
print("[EVAL]    true M_B    =\n", M_TRUE_B)
print("[EVAL]    learned M_B =\n", M_hat_B)
print("[EVAL]    intercept   true", M0_TRUE_B, " learned", m0_hat_B)
print(f"[EVAL]    max |M_B_hat - M_B| = {np.abs(M_hat_B - M_TRUE_B).max():.3e}   "
      f"max |m0_hat - m0| = {np.abs(m0_hat_B - M0_TRUE_B).max():.3e}")
print("[EVAL]    model_B.get_A()  (PyKAN per-edge spline scales) =\n",
      model_B.get_A())
print(f"[EVAL]    conformality: learned [x*u, y*u] block = "
      f"[[{M_hat_A[0, 0]:.4f}, {M_hat_A[0, 1]:.4f}], "
      f"[{M_hat_A[1, 0]:.4f}, {M_hat_A[1, 1]:.4f}]]  -> similarity matrix")

# --- Orbit validation for Model A: iterate the map (map mode) ---------------
#     Seeds sit on a circle around the attracting fixed point that is small
#     enough for the whole forward orbit to stay inside the sampled disc.
N_ORBITS = 6
N_STEPS = 80
_phi = np.linspace(0.0, 2.0 * np.pi, N_ORBITS, endpoint=False)
_seed_z = P_FIX + 0.95 * np.exp(1j * _phi)
orbit_seeds = np.column_stack([_seed_z.real, _seed_z.imag])

true_orbits = np.zeros((N_STEPS + 1, N_ORBITS, 2))
pred_orbits = np.zeros((N_STEPS + 1, N_ORBITS, 2))
true_orbits[0] = orbit_seeds
pred_orbits[0] = orbit_seeds
zt = _seed_z.copy()
xp = orbit_seeds.copy()
for n in range(N_STEPS):
    zt = mobius_complex(zt)
    true_orbits[n + 1] = np.column_stack([zt.real, zt.imag])
    xp = model_A.predict(xp)          # map mode: iterate predict
    pred_orbits[n + 1] = xp

step_rmse = np.sqrt(np.mean((pred_orbits - true_orbits) ** 2, axis=(1, 2)))
divergence = np.linalg.norm(pred_orbits - true_orbits, axis=2).max(axis=1)
orbit_radius = np.linalg.norm(true_orbits, axis=2).max()
print(f"\n[EVAL]  Orbit iteration: {N_ORBITS} seeds on |z - P| = 0.95, "
      f"{N_STEPS} steps")
print(f"[EVAL]    true orbits stay within |z| <= {orbit_radius:.3f} "
      f"(sampled disc radius {R_SAMPLE}) — no extrapolation")
print(f"[EVAL]    per-step RMSE   mean {step_rmse.mean():.3e}   "
      f"final {step_rmse[-1]:.3e}   max {step_rmse.max():.3e}")
print(f"[EVAL]    max orbit divergence over all seeds/steps: {divergence.max():.3e}")
print(f"[EVAL]    endpoint — true {true_orbits[-1, 0]}, learned "
      f"{pred_orbits[-1, 0]}, fixed point [{P_FIX.real} {P_FIX.imag}]")

# ---------------------------------------------------------------------------
# 6. Figures — also before symbolic extraction, so the edges plotted are the
#    trained splines rather than their snapped surrogates.
# ---------------------------------------------------------------------------
use_pub_style()
OUT = "results/MobiusRiemannSphere"
os.makedirs(OUT, exist_ok=True)

# 6a. Plane orbits, true vs learned, with the two fixed points marked
fig, ax = plt.subplots(figsize=(5.8, 5.6))
for j in range(N_ORBITS):
    ax.plot(true_orbits[:, j, 0], true_orbits[:, j, 1],
            color="#1f77b4", lw=1.3, alpha=0.85,
            marker="o", ms=2.0, mfc="#1f77b4", mec="none")
    ax.plot(pred_orbits[:, j, 0], pred_orbits[:, j, 1],
            color="#d62728", lw=1.0, ls="--")
ax.plot(P_FIX.real, P_FIX.imag, "k*", ms=14, label="attracting fixed point")
ax.plot(Q_FIX.real, Q_FIX.imag, "kx", ms=10, mew=2, label="repelling fixed point")
ax.plot(POLE.real, POLE.imag, "o", ms=7, mfc="none", mec="#7f7f7f", mew=1.6,
        label=r"pole $z=-d/c$")
ax.add_patch(plt.Circle((POLE.real, POLE.imag), POLE_EXCLUSION, fill=False,
                        ls=":", ec="#7f7f7f", lw=1.0))
ax.add_patch(plt.Circle((0.0, 0.0), R_SAMPLE, fill=False, ls=":",
                        ec="#2ca02c", lw=1.0))
ax.plot([], [], color="#1f77b4", lw=1.3, label="true Mobius")
ax.plot([], [], color="#d62728", lw=1.0, ls="--", label="KANDy")
ax.set_xlabel(r"$x = \mathrm{Re}\,z$")
ax.set_ylabel(r"$y = \mathrm{Im}\,z$")
ax.set_title("Loxodromic Mobius orbits in the plane")
ax.set_aspect("equal", adjustable="box")
ax.legend(loc="lower left", fontsize=7.5)
fig.tight_layout()
fig.savefig(f"{OUT}/plane_orbits.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/plane_orbits.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6b. The same orbits pushed onto S^2 by stereographic projection
fig = plt.figure(figsize=(6.6, 6.2))
ax = fig.add_subplot(111, projection="3d")
_uu, _vv = np.mgrid[0:2 * np.pi:25j, 0:np.pi:13j]
ax.plot_wireframe(np.cos(_uu) * np.sin(_vv), np.sin(_uu) * np.sin(_vv),
                  np.cos(_vv), color="#d5d5d5", lw=0.4, alpha=0.6)
for j in range(N_ORBITS):
    st = stereographic(true_orbits[:, j, :])
    sp_ = stereographic(pred_orbits[:, j, :])
    ax.plot(st[:, 0], st[:, 1], st[:, 2], color="#1f77b4", lw=1.5,
            marker="o", ms=1.8, mfc="#1f77b4", mec="none")
    ax.plot(sp_[:, 0], sp_[:, 1], sp_[:, 2], color="#d62728", lw=1.1, ls="--")
fp = stereographic(np.array([[P_FIX.real, P_FIX.imag],
                             [Q_FIX.real, Q_FIX.imag]]))
ax.scatter(fp[0, 0], fp[0, 1], fp[0, 2], c="k", marker="*", s=150, depthshade=False)
ax.scatter(fp[1, 0], fp[1, 1], fp[1, 2], c="k", marker="x", s=80, depthshade=False)
ax.plot([], [], color="#1f77b4", lw=1.6, label="true Mobius")
ax.plot([], [], color="#d62728", lw=1.1, ls="--", label="KANDy")
ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=24, azim=-58)
ax.set_xlabel("$p_1$"); ax.set_ylabel("$p_2$"); ax.set_zlabel("$p_3$")
ax.set_title(r"Mobius orbits on the Riemann sphere $S^2$")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/sphere_orbits.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/sphere_orbits.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6c. Model B — true vs predicted points on S^2
fig = plt.figure(figsize=(10, 4.4))
for idx, (pts, ttl) in enumerate([
    (YB_test, "True stereographic projection"),
    (pred_B, "KANDy (Model B)"),
]):
    ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.2, alpha=0.3,
               c=pts[:, 2], cmap="viridis", rasterized=True)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(ttl, fontsize=10)
    ax.set_xlabel("$p_1$"); ax.set_ylabel("$p_2$"); ax.set_zlabel("$p_3$")
fig.suptitle(r"Stereographic chart $\mathbb{R}^2 \to S^2$", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/stereographic_scatter.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT}/stereographic_scatter.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 6d. Loss curves
for tag, mdl in [("A", model_A), ("B", model_B)]:
    if getattr(mdl, "train_results_", None):
        fig, _ax = plot_loss_curves(mdl.train_results_,
                                    save=f"{OUT}/loss_curves_model{tag}")
        plt.close(fig)

# 6e. Edge activations — every edge should be a straight line
fig, _axes = plot_all_edges(
    model_A.model_,
    X=torch.tensor(Theta_A[:4096], dtype=torch.float32),
    in_var_names=FEATURES_A,
    out_var_names=["x_next", "y_next"],
    save=f"{OUT}/edges_modelA",
)
plt.close(fig)

fig, _axes = plot_all_edges(
    model_B.model_,
    X=torch.tensor(Theta_B[:4096], dtype=torch.float32),
    in_var_names=FEATURES_B,
    out_var_names=["p1", "p2", "p3"],
    save=f"{OUT}/edges_modelB",
)
plt.close(fig)

print(f"\n[FIGS]  Saved {OUT}/")

# ---------------------------------------------------------------------------
# 7. Symbolic extraction — LAST, because get_formula() mutates the model
#
#    The lift has linearised the problem, so the only library entries needed
#    are 'x' and '0'.  weight_simple=0.0 removes the simplicity pressure that
#    would otherwise snap a genuinely small coefficient to zero.
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Model A — Mobius map (expect straight lines in theta):")
formulas_A = model_A.get_formula(var_names=FEATURES_A, round_places=4,
                                 lib=["x", "0"], r2_threshold=0.80,
                                 weight_simple=0.0)
for lab, f in zip(["x_{n+1}", "y_{n+1}"], formulas_A):
    print(f"  {lab} = {f}")
print(f"[SYMBOLIC] Model A formula R^2: "
      f"{np.round(model_A.score_formula(formulas_A, X, Y_A, var_names=FEATURES_A), 6)}")
print(f"[SYMBOLIC] True: x_next = {M_TRUE_A[0, 0]:.4f}*x*u + {M_TRUE_A[0, 1]:.4f}*y*u "
      f"+ {M_TRUE_A[0, 2]:.4f}*r2*u + {M0_TRUE_A[0]:.4f}")
print(f"[SYMBOLIC]       y_next = {M_TRUE_A[1, 0]:.4f}*x*u + {M_TRUE_A[1, 1]:.4f}*y*u "
      f"+ {M_TRUE_A[1, 2]:.4f}*r2*u + {M0_TRUE_A[1]:.4f}")

print("\n[SYMBOLIC] Model B — stereographic projection:")
formulas_B = model_B.get_formula(var_names=FEATURES_B, round_places=4,
                                 lib=["x", "0"], r2_threshold=0.80,
                                 weight_simple=0.0)
for lab, f in zip(["p1", "p2", "p3"], formulas_B):
    print(f"  {lab} = {f}")
print(f"[SYMBOLIC] Model B formula R^2: "
      f"{np.round(model_B.score_formula(formulas_B, X, Y_B, var_names=FEATURES_B), 6)}")
print("[SYMBOLIC] True: p1 = 2*x*w,  p2 = 2*y*w,  p3 = 1 - 2*w")
