#!/usr/bin/env python3
"""KANDy example: Orszag-Tang vortex — recovering the 2D MHD equations.

The Orszag-Tang vortex (Orszag & Tang, JFM 1979) is the standard entry point to
MHD turbulence: a smooth, doubly periodic initial condition whose velocity and
magnetic field have DIFFERENT topologies, so the flow immediately steepens into
current sheets and reconnects.  In the incompressible 2D formulation the state
is the vorticity w and the magnetic vector potential A:

    dw/dt = -(u*w_x + v*w_y) + (Bx*j_x + By*j_y) + nu * lap(w)
    dA/dt = -(u*A_x + v*A_y)                     + eta * lap(A)

with the streamfunction and the fields recovered by inversion,

    lap(psi) = -w,   u =  dpsi/dy,  v = -dpsi/dx
    j        = -lap(A),  Bx = dA/dy,  By = -dA/dx.

Initial condition (the Orszag-Tang state):

    u = (-sin y,  sin x)        <=>  w = cos x + cos y
    B = (-sin y,  sin 2x)       <=>  A = cos y + cos(2x)/2

LESSON 1: pick state variables that make the constraint structural
-----------------------------------------------------------------
MHD carries the constraint div(B) = 0.  A model fitted to (Bx, By) as
independent fields has no reason to respect it, and a rollout that violates it
produces spurious magnetic forces that no amount of extra training fixes.
Evolving the POTENTIAL A instead makes div(B) = 0 an identity of the
parameterisation: B = (A_y, -A_x) is divergence-free for ANY A, including a
badly learned one.  The constraint stops being something the fit has to
discover and becomes something it cannot break.  This is the same reasoning
that puts cross-terms in the lift — encode structure you already know instead
of hoping the optimiser rediscovers it.

LESSON 2: the decoy features are the actual experiment
------------------------------------------------------
The lift below contains three features that are dimensionally plausible and
NOT in the equations:  Bx*w_x, By*w_y  ("advect the vorticity by B") and
u*j_x  ("advect the current by u").  They are not strawmen: in the Orszag-Tang
state u and B are strongly correlated at t=0, so these decoys are genuinely
predictive early on.  A fit that lands weight on them has learned a correlation
in the initial data rather than the equations, and it will diverge under
rollout.  Driving them to zero is the evidence that the recovered operator is
the MHD operator and not a look-alike.

Two diffusivities, deliberately unequal: nu != eta (magnetic Prandtl number
Pm = nu/eta = 0.4), so recovering them is two independent measurements rather
than the same number twice.

    theta = [w, A,                                     (redundant, must vanish)
             u*w_x, v*w_y, Bx*j_x, By*j_y,             (vorticity equation)
             u*A_x, v*A_y,                             (induction equation)
             Bx*w_x, By*w_y, u*j_x,                    (decoys, must vanish)
             lap_w, lap_A]                             (dissipation)

Note what is NOT in that list: the current density j.  It is the most natural
local plasma field there is, and adding it as a bare feature makes eta
non-identifiable, because j = -lap(A) IDENTICALLY.  The two columns are exactly
antiparallel, so the resistive coefficient splits arbitrarily between them: an
earlier run of this script recovered "-0.0032*j + 0.0018*lap_A" — the sum is
eta = 0.005 to three digits, but neither number means anything on its own, at
R^2 = 1.  Lift coordinates must be functionally INDEPENDENT; j enters the
physics only through the cross-terms Bx*j_x and By*j_y, and that is where it
belongs.

KAN:  width = [13, 2],  base_fun='zero'
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from _utils import ensure_outdir, print_check, save_metrics

from kandy import KANDy, CustomLift
from kandy.plotting import plot_all_edges, plot_loss_curves, use_pub_style

# ---------------------------------------------------------------------------
# 0. Reproducibility and physical parameters
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

N_GRID = 128          # grid points per dimension, domain [0, 2*pi]^2
NU = 2.0e-3           # kinematic viscosity
ETA = 5.0e-3          # magnetic diffusivity (Pm = nu/eta = 0.4)
DT = 2.0e-3
N_STEPS = 500         # to t = 1.0, past the first current-sheet formation
SNAP_EVERY = 50

print(f"[MODEL] Orszag-Tang vortex, incompressible 2D MHD")
print(f"[MODEL] nu = {NU}, eta = {ETA}, magnetic Prandtl number Pm = {NU / ETA:.2f}")

# ---------------------------------------------------------------------------
# 1. Pseudo-spectral 2D MHD solver in (vorticity, vector potential) form
# ---------------------------------------------------------------------------
KX = np.fft.fftfreq(N_GRID, d=1.0 / N_GRID).reshape(-1, 1)
KY = np.fft.fftfreq(N_GRID, d=1.0 / N_GRID).reshape(1, -1)
K2 = KX ** 2 + KY ** 2
K2_INV = np.where(K2 == 0, 1.0, K2)
DEALIAS = (np.abs(KX) < N_GRID / 3) & (np.abs(KY) < N_GRID / 3)


def local_fields(w_hat: np.ndarray, A_hat: np.ndarray) -> dict:
    """Physical-space fields and their derivatives from (w_hat, A_hat).

    Every spatial derivative is spectral, which is exact for periodic data.
    """
    psi_hat = np.where(K2 == 0, 0.0, w_hat / K2_INV)      # lap(psi) = -w
    ifft = lambda z: np.real(np.fft.ifft2(z))
    return {
        "w": ifft(w_hat),
        "A": ifft(A_hat),
        "j": ifft(K2 * A_hat),                            # j = -lap(A)
        "u": ifft(1j * KY * psi_hat),
        "v": ifft(-1j * KX * psi_hat),
        "Bx": ifft(1j * KY * A_hat),
        "By": ifft(-1j * KX * A_hat),
        "w_x": ifft(1j * KX * w_hat),
        "w_y": ifft(1j * KY * w_hat),
        "A_x": ifft(1j * KX * A_hat),
        "A_y": ifft(1j * KY * A_hat),
        "j_x": ifft(1j * KX * K2 * A_hat),
        "j_y": ifft(1j * KY * K2 * A_hat),
        "lap_w": ifft(-K2 * w_hat),
        "lap_A": ifft(-K2 * A_hat),
    }


def true_tendencies(f: dict) -> tuple[np.ndarray, np.ndarray]:
    """(dw/dt, dA/dt) in physical space from the local fields."""
    dw = (-(f["u"] * f["w_x"] + f["v"] * f["w_y"])
          + (f["Bx"] * f["j_x"] + f["By"] * f["j_y"])
          + NU * f["lap_w"])
    dA = -(f["u"] * f["A_x"] + f["v"] * f["A_y"]) + ETA * f["lap_A"]
    return dw, dA


def rhs_hat(w_hat: np.ndarray, A_hat: np.ndarray):
    """Spectral RHS with 2/3-rule dealiasing of the quadratic terms."""
    f = local_fields(w_hat, A_hat)
    dw_nl = -(f["u"] * f["w_x"] + f["v"] * f["w_y"]) + (
        f["Bx"] * f["j_x"] + f["By"] * f["j_y"])
    dA_nl = -(f["u"] * f["A_x"] + f["v"] * f["A_y"])
    return (np.fft.fft2(dw_nl) * DEALIAS - NU * K2 * w_hat,
            np.fft.fft2(dA_nl) * DEALIAS - ETA * K2 * A_hat)


def rk4_step(w_hat, A_hat, dt, rhs=rhs_hat):
    k1w, k1a = rhs(w_hat, A_hat)
    k2w, k2a = rhs(w_hat + 0.5 * dt * k1w, A_hat + 0.5 * dt * k1a)
    k3w, k3a = rhs(w_hat + 0.5 * dt * k2w, A_hat + 0.5 * dt * k2a)
    k4w, k4a = rhs(w_hat + dt * k3w, A_hat + dt * k3a)
    return (w_hat + (dt / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w),
            A_hat + (dt / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a))


x = np.arange(N_GRID) * 2 * np.pi / N_GRID
XX, YY = np.meshgrid(x, x, indexing="ij")
w0 = np.cos(XX) + np.cos(YY)                    # u = (-sin y, sin x)
A0 = np.cos(YY) + 0.5 * np.cos(2 * XX)          # B = (-sin y, sin 2x)
w_hat, A_hat = np.fft.fft2(w0), np.fft.fft2(A0)

print(f"[SIM]   {N_GRID}x{N_GRID} spectral, dt={DT}, {N_STEPS} steps "
      f"(t_end = {N_STEPS * DT:.2f})")

# ---------------------------------------------------------------------------
# 2. Local features — one training sample per grid point per snapshot
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "w", "A",
    "u*w_x", "v*w_y", "Bx*j_x", "By*j_y",
    "u*A_x", "v*A_y",
    "Bx*w_x", "By*w_y", "u*j_x",
    "lap_w", "lap_A",
]
N_FEATURES = len(FEATURE_NAMES)
FI = {name: i for i, name in enumerate(FEATURE_NAMES)}
DECOY_IDX = [FI[n] for n in ("Bx*w_x", "By*w_y", "u*j_x")]
REDUNDANT_IDX = [FI[n] for n in ("w", "A")]
ADVECT_IDX = [FI["u*w_x"], FI["v*w_y"]]
LORENTZ_IDX = [FI["Bx*j_x"], FI["By*j_y"]]
INDUCTION_IDX = [FI["u*A_x"], FI["v*A_y"]]


def build_features(f: dict) -> np.ndarray:
    """Lift one (w, A) snapshot to (N_GRID^2, 13) local features."""
    return np.stack([
        f["w"], f["A"],
        f["u"] * f["w_x"], f["v"] * f["w_y"],
        f["Bx"] * f["j_x"], f["By"] * f["j_y"],
        f["u"] * f["A_x"], f["v"] * f["A_y"],
        f["Bx"] * f["w_x"], f["By"] * f["w_y"], f["u"] * f["j_x"],
        f["lap_w"], f["lap_A"],
    ], axis=-1).reshape(-1, N_FEATURES)


feature_pool, target_pool, snapshots, energies = [], [], [], []
w_run, A_run = w_hat.copy(), A_hat.copy()
for i in range(N_STEPS + 1):
    if i % SNAP_EVERY == 0:
        f = local_fields(w_run, A_run)
        dw, dA = true_tendencies(f)
        feature_pool.append(build_features(f))
        target_pool.append(np.stack([dw.reshape(-1), dA.reshape(-1)], axis=-1))
        snapshots.append((f["w"].copy(), f["A"].copy(), f["j"].copy()))
        energies.append((0.5 * np.mean(f["u"] ** 2 + f["v"] ** 2),
                         0.5 * np.mean(f["Bx"] ** 2 + f["By"] ** 2)))
    w_run, A_run = rk4_step(w_run, A_run, DT)

feature_pool = np.concatenate(feature_pool)
target_pool = np.concatenate(target_pool)
print(f"[DATA]  {len(snapshots)} snapshots -> {feature_pool.shape[0]} grid samples")

N_TRAIN = 12000
sel = np.random.default_rng(SEED + 1).choice(len(feature_pool), N_TRAIN, replace=False)
Theta = feature_pool[sel]
Y = target_pool[sel]

cond = np.linalg.cond(np.column_stack([Theta, np.ones(len(Theta))]))
print(f"[DATA]  {N_TRAIN} training points, cond(Theta) = {cond:.1f}   "
      f"(adding j as a bare feature makes Theta exactly rank-deficient: j = -lap_A)")

# How predictive are the decoys on their own?  If they were useless the test
# would be trivial; this prints the correlation that makes it a real test.
for k in DECOY_IDX:
    r = np.corrcoef(Theta[:, k], Y[:, 0])[0, 1]
    print(f"[DATA]  decoy {FEATURE_NAMES[k]:>8s} correlates with dw/dt at "
          f"r = {r:+.3f}  (a look-alike, not a strawman)")

# ---------------------------------------------------------------------------
# 3. KANDy model — features are pre-computed, so the lift is the identity
# ---------------------------------------------------------------------------
lift = CustomLift(fn=lambda X: X, output_dim=N_FEATURES, name="mhd2d_local")
model = KANDy(lift=lift, grid=5, k=3, steps=200, seed=SEED, base_fun="zero")
model.fit(X=Theta, X_dot=Y, val_frac=0.15, test_frac=0.15, lamb=0.0, patience=40)

pred = model.predict(Theta)
raw_r2 = [
    1.0 - np.sum((Y[:, i] - pred[:, i]) ** 2) / np.sum((Y[:, i] - Y[:, i].mean()) ** 2)
    for i in range(2)
]
print(f"[EVAL]  Network R^2 — dw/dt {raw_r2[0]:.6f}, dA/dt {raw_r2[1]:.6f}")

# ---------------------------------------------------------------------------
# 4. Field-level check and PDE rollout of the LEARNED operator
#
#    All of this runs BEFORE get_formula(), which rewrites the spline edges in
#    place; anything measured afterwards would be the snapped surrogate.
# ---------------------------------------------------------------------------
def learned_rhs_hat(w_hat_s: np.ndarray, A_hat_s: np.ndarray):
    """Spectral tendencies from the learned pointwise model."""
    f = local_fields(w_hat_s, A_hat_s)
    out = model.predict(build_features(f))
    dw = out[:, 0].reshape(N_GRID, N_GRID)
    dA = out[:, 1].reshape(N_GRID, N_GRID)
    return np.fft.fft2(dw) * DEALIAS, np.fft.fft2(dA) * DEALIAS


mid = len(snapshots) // 2
w_t = np.fft.fft2(snapshots[mid][0])
A_t = np.fft.fft2(snapshots[mid][1])
f_t = local_fields(w_t, A_t)
dw_true, dA_true = true_tendencies(f_t)
out_t = model.predict(build_features(f_t))
dw_pred = out_t[:, 0].reshape(N_GRID, N_GRID)
dA_pred = out_t[:, 1].reshape(N_GRID, N_GRID)


def _r2(a, b):
    return float(1.0 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2))


field_r2 = [_r2(dw_true, dw_pred), _r2(dA_true, dA_pred)]
print(f"[EVAL]  Field R^2 on a held-out snapshot — dw/dt {field_r2[0]:.6f}, "
      f"dA/dt {field_r2[1]:.6f}")

ROLL_STEPS = 250
w_roll, A_roll = np.fft.fft2(snapshots[0][0]), np.fft.fft2(snapshots[0][1])
w_ref, A_ref = w_roll.copy(), A_roll.copy()
for _ in range(ROLL_STEPS):
    w_roll, A_roll = rk4_step(w_roll, A_roll, DT, rhs=learned_rhs_hat)
    w_ref, A_ref = rk4_step(w_ref, A_ref, DT)

w_roll_p, A_roll_p = np.real(np.fft.ifft2(w_roll)), np.real(np.fft.ifft2(A_roll))
w_ref_p, A_ref_p = np.real(np.fft.ifft2(w_ref)), np.real(np.fft.ifft2(A_ref))
roll_rel_w = float(np.sqrt(np.mean((w_roll_p - w_ref_p) ** 2)) / np.std(w_ref_p))
roll_rel_A = float(np.sqrt(np.mean((A_roll_p - A_ref_p) ** 2)) / np.std(A_ref_p))
j_roll = np.real(np.fft.ifft2(K2 * A_roll))
j_ref = np.real(np.fft.ifft2(K2 * A_ref))
print(f"[EVAL]  PDE rollout ({ROLL_STEPS} steps, t={ROLL_STEPS * DT:.2f}): "
      f"relative RMSE — w {100 * roll_rel_w:.3f}%, A {100 * roll_rel_A:.3f}%")

# Label-free: nothing in training mentioned energy, yet the learned operator
# must reproduce the kinetic <-> magnetic exchange that defines this test.
def _energies(w_h, A_h):
    f = local_fields(w_h, A_h)
    return (0.5 * np.mean(f["u"] ** 2 + f["v"] ** 2),
            0.5 * np.mean(f["Bx"] ** 2 + f["By"] ** 2))


ek_roll, em_roll = _energies(w_roll, A_roll)
ek_ref, em_ref = _energies(w_ref, A_ref)
energy_err = float(max(abs(ek_roll - ek_ref) / ek_ref, abs(em_roll - em_ref) / em_ref))
print(f"[EVAL]  Energies at t={ROLL_STEPS * DT:.2f} — kinetic {ek_roll:.5f} vs "
      f"{ek_ref:.5f}, magnetic {em_roll:.5f} vs {em_ref:.5f} "
      f"(max rel err {100 * energy_err:.2f}%)")

theta_t = torch.tensor(Theta[:2048], dtype=torch.float32)
model_out_pre_snap = model.predict(Theta)

# ---------------------------------------------------------------------------
# 5. Linear readout — the RHS is linear in these features by construction
# ---------------------------------------------------------------------------
A_TRUE = np.zeros((2, N_FEATURES))
A_TRUE[0, ADVECT_IDX] = -1.0
A_TRUE[0, LORENTZ_IDX] = 1.0
A_TRUE[0, FI["lap_w"]] = NU
A_TRUE[1, INDUCTION_IDX] = -1.0
A_TRUE[1, FI["lap_A"]] = ETA

A_hat_fit, *_ = np.linalg.lstsq(Theta, model_out_pre_snap, rcond=None)
A_hat_fit = A_hat_fit.T

nu_hat = float(A_hat_fit[0, FI["lap_w"]])
eta_hat = float(A_hat_fit[1, FI["lap_A"]])
nu_rel_err = abs(nu_hat - NU) / NU
eta_rel_err = abs(eta_hat - ETA) / ETA
advect_err = float(np.abs(A_hat_fit[0, ADVECT_IDX] - (-1.0)).max())
lorentz_err = float(np.abs(A_hat_fit[0, LORENTZ_IDX] - 1.0).max())
induction_err = float(np.abs(A_hat_fit[1, INDUCTION_IDX] - (-1.0)).max())
decoy_max = float(np.abs(A_hat_fit[:, DECOY_IDX]).max())
redundant_max = float(np.abs(A_hat_fit[:, REDUNDANT_IDX]).max())

np.set_printoptions(precision=4, suppress=True)
print(f"\n[EVAL]  Recovered readout (rows dw/dt, dA/dt):\n{A_hat_fit}")
print(f"[EVAL]  nu  recovered {nu_hat:.5f} vs true {NU}  ({100 * nu_rel_err:.2f}% error)")
print(f"[EVAL]  eta recovered {eta_hat:.5f} vs true {ETA} ({100 * eta_rel_err:.2f}% error)")
print(f"[EVAL]  Pm recovered {nu_hat / eta_hat:.3f} vs true {NU / ETA:.3f}")
print(f"[EVAL]  advection err {advect_err:.4f}, Lorentz err {lorentz_err:.4f}, "
      f"induction err {induction_err:.4f}")
print(f"[EVAL]  largest weight on a DECOY feature: {decoy_max:.4f}   "
      f"(Bx*w_x, By*w_y, u*j_x are not in the equations)")
print(f"[EVAL]  largest weight on a redundant feature (w, A): {redundant_max:.4f}")

# ---------------------------------------------------------------------------
# 6. Symbolic extraction  (LAST — get_formula mutates the model in place)
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Extracting the 2D MHD equations ...")
formulas = model.get_formula(
    var_names=FEATURE_NAMES, round_places=5,
    lib=["x", "0"], r2_threshold=0.80, weight_simple=0.0,
)
for lab, f_ in zip(["dw/dt", "dA/dt"], formulas):
    print(f"  {lab} = {f_}")
r2 = model.score_formula(formulas, Theta, Y, var_names=FEATURE_NAMES)
print(f"[SYMBOLIC] Formula R^2 per equation: {np.round(r2, 6)}")
print(f"[SYMBOLIC] True:  dw/dt = -u*w_x - v*w_y + Bx*j_x + By*j_y + {NU}*lap_w")
print(f"[SYMBOLIC]        dA/dt = -u*A_x - v*A_y + {ETA}*lap_A")

# ---------------------------------------------------------------------------
# 7. Verification
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "network_r2_min": 0.99,
    "field_r2_min": 0.99,
    "nu_rel_err_max": 0.15,
    "eta_rel_err_max": 0.15,
    "advection_coef_abs_err_max": 0.05,
    "lorentz_coef_abs_err_max": 0.05,
    "induction_coef_abs_err_max": 0.05,
    "decoy_weight_max": 0.05,
    "redundant_weight_max": 0.05,
    "rollout_rel_rmse_max": 0.05,
    "energy_rel_err_max": 0.05,
    "formula_r2_min": 0.99,
}
passed = print_check(
    min(raw_r2) >= THRESHOLDS["network_r2_min"]
    and min(field_r2) >= THRESHOLDS["field_r2_min"]
    and nu_rel_err <= THRESHOLDS["nu_rel_err_max"]
    and eta_rel_err <= THRESHOLDS["eta_rel_err_max"]
    and advect_err <= THRESHOLDS["advection_coef_abs_err_max"]
    and lorentz_err <= THRESHOLDS["lorentz_coef_abs_err_max"]
    and induction_err <= THRESHOLDS["induction_coef_abs_err_max"]
    and decoy_max <= THRESHOLDS["decoy_weight_max"]
    and redundant_max <= THRESHOLDS["redundant_weight_max"]
    and max(roll_rel_w, roll_rel_A) <= THRESHOLDS["rollout_rel_rmse_max"]
    and energy_err <= THRESHOLDS["energy_rel_err_max"]
    and float(np.min(r2)) >= THRESHOLDS["formula_r2_min"],
    "Orszag-Tang: 2D MHD operator recovered with nu, eta and no decoy terms",
)
RESULT_DIR = ensure_outdir("results/OrszagTang")
save_metrics(RESULT_DIR / "metrics.json", {
    "task": "OrszagTang",
    "category": "plasma",
    "seed": SEED,
    "sampling": "pseudo-spectral 2D MHD DNS snapshots, one sample per grid point",
    "grid": N_GRID,
    "nu_true": NU, "nu_recovered": nu_hat, "nu_rel_err": nu_rel_err,
    "eta_true": ETA, "eta_recovered": eta_hat, "eta_rel_err": eta_rel_err,
    "prandtl_true": NU / ETA, "prandtl_recovered": nu_hat / eta_hat,
    "feature_names": FEATURE_NAMES,
    "true_readout": A_TRUE.tolist(),
    "linear_readout": A_hat_fit.tolist(),
    "advection_coef_abs_err": advect_err,
    "lorentz_coef_abs_err": lorentz_err,
    "induction_coef_abs_err": induction_err,
    "decoy_weight_max": decoy_max,
    "redundant_weight_max": redundant_max,
    "cond_theta": float(cond),
    "network_r2": [float(v) for v in raw_r2],
    "field_r2": field_r2,
    "rollout_steps": ROLL_STEPS,
    "rollout_rel_rmse_w": roll_rel_w,
    "rollout_rel_rmse_A": roll_rel_A,
    "energy_rel_err": energy_err,
    "formula_r2": np.asarray(r2).tolist(),
    "thresholds": THRESHOLDS,
    "passed": passed,
})


# ---------------------------------------------------------------------------
# 8. Figures
# ---------------------------------------------------------------------------
EXTENT = [0, 2 * np.pi, 0, 2 * np.pi]


def plot_current_sheets(outdir: str) -> None:
    """The signature of the Orszag-Tang test: current sheets, true vs learned."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    s = np.abs(j_ref).max()
    panels = [
        (snapshots[0][2], "initial current $j$", np.abs(snapshots[0][2]).max()),
        (j_ref, f"true, $t={ROLL_STEPS * DT:.2f}$", s),
        (j_roll, f"KANDy rollout, $t={ROLL_STEPS * DT:.2f}$", s),
        (j_roll - j_ref, "error", np.abs(j_roll - j_ref).max() + 1e-12),
    ]
    for ax, (field, title, scale) in zip(axes, panels):
        im = ax.imshow(field.T, origin="lower", extent=EXTENT, cmap="RdBu_r",
                       vmin=-scale, vmax=scale)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Orszag-Tang vortex: current density from the learned MHD operator",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{outdir}/current_sheets.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tendencies(outdir: str) -> None:
    """One-step tendency fields, true vs predicted, for both equations."""
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5))
    rows = [(dw_true, dw_pred, r"$\partial_t\omega$", field_r2[0]),
            (dA_true, dA_pred, r"$\partial_t A$", field_r2[1])]
    for row, (true, pred_, name, r2v) in zip(axes, rows):
        s = np.abs(true).max()
        for ax, (field, title) in zip(row, [(true, f"true {name}"),
                                            (pred_, f"KANDy {name}"),
                                            (pred_ - true, "error")]):
            im = ax.imshow(field.T, origin="lower", extent=EXTENT, cmap="RdBu_r",
                           vmin=-s, vmax=s)
            ax.set_title(f"{title}   ($R^2$={r2v:.5f})" if "error" not in title
                         else title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Tendency fields on a held-out snapshot", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{outdir}/tendency_fields.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_readout(outdir: str) -> None:
    """Recovered vs true coefficients — the decoy columns are the point."""
    fig, ax = plt.subplots(figsize=(11, 3.6))
    idx = np.arange(N_FEATURES)
    ax.bar(idx - 0.2, A_TRUE[0], width=0.18, color="#1f77b4", label=r"true $\partial_t\omega$")
    ax.bar(idx - 0.02, A_hat_fit[0], width=0.18, color="#7fbde8", label=r"KANDy $\partial_t\omega$")
    ax.bar(idx + 0.16, A_TRUE[1], width=0.18, color="#d62728", label=r"true $\partial_t A$")
    ax.bar(idx + 0.34, A_hat_fit[1], width=0.18, color="#f0928f", label=r"KANDy $\partial_t A$")
    for k in DECOY_IDX:
        ax.axvspan(k - 0.45, k + 0.55, color="0.85", zorder=0)
    ax.set_xticks(idx)
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("coefficient")
    ax.set_title("Recovered MHD operator (shaded columns are decoy features)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{outdir}/readout_coefficients.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{outdir}/readout_coefficients.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_energy_exchange(outdir: str) -> None:
    """Kinetic <-> magnetic energy exchange, the physics of this test case."""
    fig, ax = plt.subplots(figsize=(6, 4))
    t_snap = np.arange(len(energies)) * SNAP_EVERY * DT
    ek, em = zip(*energies)
    ax.plot(t_snap, ek, "o-", color="#1f77b4", ms=3, lw=1.2, label="kinetic (DNS)")
    ax.plot(t_snap, em, "s-", color="#d62728", ms=3, lw=1.2, label="magnetic (DNS)")
    t_r = ROLL_STEPS * DT
    ax.plot([t_r], [ek_roll], "*", color="#1f77b4", ms=14, mec="k", mew=0.5,
            label="kinetic (KANDy rollout)")
    ax.plot([t_r], [em_roll], "*", color="#d62728", ms=14, mec="k", mew=0.5,
            label="magnetic (KANDy rollout)")
    ax.set_xlabel("time"); ax.set_ylabel("energy density")
    ax.set_title("Orszag-Tang: kinetic / magnetic energy exchange")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{outdir}/energy_exchange.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


use_pub_style()
OUT = str(RESULT_DIR)
plot_current_sheets(OUT)
plot_tendencies(OUT)
plot_readout(OUT)
plot_energy_exchange(OUT)
if getattr(model, "train_results_", None):
    fig, _ = plot_loss_curves(model.train_results_, save=f"{OUT}/loss_curves")
    plt.close(fig)
fig, _ = plot_all_edges(
    model.model_, X=theta_t,
    in_var_names=FEATURE_NAMES,
    out_var_names=["dw/dt", "dA/dt"],
    save=f"{OUT}/edge_activations",
)
plt.close(fig)
print(f"[FIGS]  Saved {OUT}/")
