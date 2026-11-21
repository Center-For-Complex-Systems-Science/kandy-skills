#!/usr/bin/env python3
"""KANDy example: Brio-Wu shock tube — discovering the ideal-MHD flux function.

The Brio-Wu problem (Brio & Wu, JCP 1988) is the MHD Riemann problem every
compressible plasma code is validated against.  With gamma = 2 and a constant
normal field Bx = 0.75 the initial state is

    left  (x < 0.5):  rho = 1.000,  p = 1.0,  By = +1,  u = v = 0
    right (x > 0.5):  rho = 0.125,  p = 0.1,  By = -1,  u = v = 0

and by t = 0.1 the solution contains a fast rarefaction, a slow COMPOUND wave
(a shock welded to a rarefaction — an intermediate wave with no gas-dynamic
analogue), a contact discontinuity, a slow shock and a fast rarefaction.

The 1D ideal-MHD system is dU/dt = -dF/dx with

    U = [rho, rho*u, rho*v, By, E]
    F = [rho*u,
         rho*u^2 + p + By^2/2 - Bx^2/2,
         rho*u*v - Bx*By,
         u*By - v*Bx,
         (E + p + B^2/2)*u - Bx*(u*Bx + v*By)]

THE TASK: recover F, not the RHS
--------------------------------
The lift holds the x-derivative of twelve CANDIDATE flux functions, so each
equation is linear in the features with coefficients in {0, -1, +1}:

    theta = dx[ rho*u,  rho*u^2,  p,  By^2/2,  rho*u*v,  Bx*By,
                u*By,   v*Bx,     F_E,  Bx*(u*Bx+v*By),
                rho*v,  rho*u*By ]                     <- last two are decoys

Recovering it means recovering which flux term belongs to which conservation
law and with what sign — in particular the magnetic TENSION term -Bx*By in the
transverse momentum equation and the induction equation d(By)/dt = -dx(u*By -
v*Bx), where the MHD-specific physics lives.

LESSON 1: one Riemann problem cannot identify a flux function
-------------------------------------------------------------
A Riemann problem is SELF-SIMILAR: its entire solution is a one-parameter
family of states in x/t, so every candidate flux evaluated along it is a
function of that single parameter and the twelve feature columns become nearly
dependent.  Fitting the Brio-Wu tube alone reaches R^2 = 0.999 and returns
coefficients off by more than 1.0 — including large weight on the decoys.  This
is the conserved-quantity trap of mathbio/sir_example.py in another costume:
the data lies on a low-dimensional set, so the decomposition is not unique and
a high R^2 certifies nothing.

The fix is not more data from the same tube; it is data from DIFFERENT tubes.
This script trains on an ensemble of randomised MHD Riemann problems (varying
Bx too, which is what separates Bx*By from By and v*Bx from v) and keeps the
canonical Brio-Wu problem entirely out of training, as the validation case.
The single-problem fit is run anyway, as a measured control.

LESSON 2: shock cells are outliers, and an MSE fit does not know that
---------------------------------------------------------------------
The targets are what real data affords — dU/dt from CENTRED TIME DIFFERENCES of
stored snapshots.  In smooth regions that estimate and the centred flux
divergence agree to truncation error.  Across a captured discontinuity both
break down: the snapshots carry the solver's numerical dissipation and the
centred derivatives ring, so the residual there is ~100x larger than anywhere
else.  Since the loss is an unweighted MSE, a few percent of shock cells
dominate the gradient and drag every coefficient off {0, ±1}.  A jump detector
marks them; the count discarded is reported, not silently dropped.

VALIDATION: the discovered flux is put back into the solver.  The recovered
coefficient matrix defines a flux function; running the SAME finite-volume
scheme with it reproduces the held-out Brio-Wu profile, compound wave included.
That is the honest rollout — centred-differencing the learned RHS directly
would ring itself apart at the shock.

KAN:  width = [12, 5],  base_fun='zero'
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

GAMMA = 2.0            # Brio-Wu uses gamma = 2
CFL = 0.4

BW_BX = 0.75           # canonical Brio-Wu: normal field, held out of training
BW_LEFT = [1.000, 0.0, 0.0, 1.0, +1.0]      # [rho, u, v, p, By]
BW_RIGHT = [0.125, 0.0, 0.0, 0.1, -1.0]
BW_CELLS, BW_T_END = 800, 0.10

N_PROBLEMS = 10        # randomised training Riemann problems
TR_CELLS, TR_T_END = 800, 0.06
# Snapshots are stored FINELY so the central time difference is accurate, but
# only every SAMPLE_EVERY-th one contributes training rows.  Accuracy of the
# derivative estimate and size of the dataset are separate knobs; tying them
# together is what makes "more data" fail to help.
DT_SNAP = 1.0e-4
SAMPLE_EVERY = 8

print("[MODEL] Brio-Wu MHD shock tube, gamma = 2, Bx = 0.75 (validation case)")

# ---------------------------------------------------------------------------
# 1. A MUSCL / HLL finite-volume solver for 1D ideal MHD
#
#    Primitive W = [rho, u, v, p, By];  conserved U = [rho, rho u, rho v, By, E]
#    Bx is constant within a problem (div(B) = 0 in 1D) but varies between them.
# ---------------------------------------------------------------------------
def prim2cons(W: np.ndarray, bx: float) -> np.ndarray:
    rho, u, v, p, By = (W[..., i] for i in range(5))
    E = p / (GAMMA - 1.0) + 0.5 * rho * (u ** 2 + v ** 2) + 0.5 * (bx ** 2 + By ** 2)
    return np.stack([rho, rho * u, rho * v, By, E], axis=-1)


def cons2prim(U: np.ndarray, bx: float) -> np.ndarray:
    rho, mx, my, By, E = (U[..., i] for i in range(5))
    rho = np.maximum(rho, 1e-8)
    u, v = mx / rho, my / rho
    p = (GAMMA - 1.0) * (E - 0.5 * rho * (u ** 2 + v ** 2) - 0.5 * (bx ** 2 + By ** 2))
    return np.stack([rho, u, v, np.maximum(p, 1e-8), By], axis=-1)


def flux_true(W: np.ndarray, bx: float) -> np.ndarray:
    """Exact ideal-MHD x-flux from primitive variables."""
    rho, u, v, p, By = (W[..., i] for i in range(5))
    b2 = bx ** 2 + By ** 2
    E = p / (GAMMA - 1.0) + 0.5 * rho * (u ** 2 + v ** 2) + 0.5 * b2
    return np.stack([
        rho * u,
        rho * u ** 2 + p + 0.5 * By ** 2 - 0.5 * bx ** 2,
        rho * u * v - bx * By,
        u * By - v * bx,
        (E + p + 0.5 * b2) * u - bx * (u * bx + v * By),
    ], axis=-1)


def fast_speed(W: np.ndarray, bx: float) -> np.ndarray:
    """Fast magnetosonic speed — a property of the Riemann solver, not of the
    physics being discovered, so the same estimator serves every flux."""
    rho, _, _, p, By = (W[..., i] for i in range(5))
    a2 = GAMMA * p / rho
    ca2 = (bx ** 2 + By ** 2) / rho
    cax2 = bx ** 2 / rho
    disc = np.maximum((a2 + ca2) ** 2 - 4.0 * a2 * cax2, 0.0)
    return np.sqrt(np.maximum(0.5 * (a2 + ca2 + np.sqrt(disc)), 1e-12))


def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a * b > 0, np.where(np.abs(a) < np.abs(b), a, b), 0.0)


def hll_flux(WL, WR, flux_fn, bx):
    UL, UR = prim2cons(WL, bx), prim2cons(WR, bx)
    FL, FR = flux_fn(WL, bx), flux_fn(WR, bx)
    cL, cR = fast_speed(WL, bx), fast_speed(WR, bx)
    SL = np.minimum(WL[..., 1] - cL, WR[..., 1] - cR)[:, None]
    SR = np.maximum(WL[..., 1] + cL, WR[..., 1] + cR)[:, None]
    SL, SR = np.minimum(SL, 0.0), np.maximum(SR, 0.0)
    return (SR * FL - SL * FR + SL * SR * (UR - UL)) / (SR - SL)


def divergence(U, flux_fn, bx, dx):
    """-dF/dx with MUSCL(minmod) reconstruction and outflow boundaries."""
    W = cons2prim(U, bx)
    Wg = np.concatenate([W[:1], W[:1], W, W[-1:], W[-1:]], axis=0)   # 2 ghosts
    slope = minmod(Wg[1:-1] - Wg[:-2], Wg[2:] - Wg[1:-1])
    WL = (Wg[1:-1] + 0.5 * slope)[:-1]
    WR = (Wg[1:-1] - 0.5 * slope)[1:]
    return -(np.diff(hll_flux(WL, WR, flux_fn, bx), axis=0)) / dx


def evolve(U, t_end, flux_fn, bx, dx, snap_times=None):
    """SSP-RK2 in time; returns the final state and snapshots at requested times."""
    t, snaps = 0.0, []
    targets = list(snap_times) if snap_times is not None else []
    while t < t_end - 1e-12:
        W = cons2prim(U, bx)
        dt = CFL * dx / np.max(np.abs(W[:, 1]) + fast_speed(W, bx))
        if targets and t + dt > targets[0] - 1e-12:
            dt = targets[0] - t
        U1 = U + dt * divergence(U, flux_fn, bx, dx)
        U = 0.5 * (U + U1 + dt * divergence(U1, flux_fn, bx, dx))
        t += dt
        if targets and t >= targets[0] - 1e-12:
            snaps.append(U.copy())
            targets.pop(0)
    return U, snaps


# ---------------------------------------------------------------------------
# 2. Candidate flux functions and the derived training set
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "dx_rho_u", "dx_rho_uu", "dx_p", "dx_By2_2", "dx_rho_uv", "dx_Bx_By",
    "dx_u_By", "dx_v_Bx", "dx_F_energy", "dx_Bx_uBx_vBy",
    "dx_rho_v", "dx_rho_u_By",
]
N_FEATURES = len(FEATURE_NAMES)
FI = {n: i for i, n in enumerate(FEATURE_NAMES)}
DECOY_IDX = [FI["dx_rho_v"], FI["dx_rho_u_By"]]
EQ_NAMES = ["d(rho)/dt", "d(rho u)/dt", "d(rho v)/dt", "d(By)/dt", "d(E)/dt"]

GUARD = 3              # cells dropped at each boundary
JUMP_TOL = 0.15        # relative jump marking a captured discontinuity
DILATE = 3             # cells on each side also treated as contaminated
ACTIVE_TOL = 0.02      # a cell is uninformative if nothing is happening in it


def candidate_fluxes(W: np.ndarray, bx: float) -> np.ndarray:
    """The twelve candidate flux functions evaluated at cell centres."""
    rho, u, v, p, By = (W[:, i] for i in range(5))
    b2 = bx ** 2 + By ** 2
    E = p / (GAMMA - 1.0) + 0.5 * rho * (u ** 2 + v ** 2) + 0.5 * b2
    return np.stack([
        rho * u,                                  # mass flux
        rho * u ** 2,                             # ram pressure
        p,                                        # thermal pressure
        0.5 * By ** 2,                            # magnetic pressure
        rho * u * v,                              # transverse momentum advection
        bx * By,                                  # magnetic tension
        u * By,                                   # field advection
        v * bx,                                   # field stretching
        (E + p + 0.5 * b2) * u,                   # total energy advection
        bx * (u * bx + v * By),                   # Poynting flux
        rho * v,                                  # decoy
        rho * u * By,                             # decoy
    ], axis=-1)


# True flux composition C (F_k = sum_m C[k, m] g_m), so the readout on
# dU/dt = -dF/dx is A = -C.
C_TRUE = np.zeros((5, N_FEATURES))
C_TRUE[0, FI["dx_rho_u"]] = 1.0
C_TRUE[1, [FI["dx_rho_uu"], FI["dx_p"], FI["dx_By2_2"]]] = 1.0
C_TRUE[2, [FI["dx_rho_uv"], FI["dx_Bx_By"]]] = [1.0, -1.0]
C_TRUE[3, [FI["dx_u_By"], FI["dx_v_Bx"]]] = [1.0, -1.0]
C_TRUE[4, [FI["dx_F_energy"], FI["dx_Bx_uBx_vBy"]]] = [1.0, -1.0]
A_TRUE = -C_TRUE


def run_problem(left, right, bx, n_cells, t_end):
    """Solve one Riemann problem and turn its snapshots into a training set."""
    dx = 1.0 / n_cells
    xc = (np.arange(n_cells) + 0.5) * dx
    W0 = np.where((xc < 0.5)[:, None], np.array(left), np.array(right))
    U0 = prim2cons(W0, bx)
    snap_times = list(np.arange(1, int(round(t_end / DT_SNAP)) + 1) * DT_SNAP)
    U_final, snaps = evolve(U0.copy(), t_end, flux_true, bx, dx, snap_times)

    feats, targs, smooths, actives, sample_idx = [], [], [], [], []
    for n in range(1, len(snaps) - 1):
        if n % SAMPLE_EVERY:
            continue
        U_n = snaps[n]
        W_n = cons2prim(U_n, bx)
        dUdt = (snaps[n + 1] - snaps[n - 1]) / (2.0 * DT_SNAP)
        g = candidate_fluxes(W_n, bx)
        f = np.zeros_like(g)
        f[1:-1] = (g[2:] - g[:-2]) / (2.0 * dx)          # central x-derivative

        scale = np.ptp(U_n, axis=0)
        scale[scale == 0] = 1.0
        jump = np.zeros(n_cells)
        jump[1:-1] = np.max(np.abs(U_n[2:] - U_n[:-2]) / scale, axis=1)
        bad = jump > JUMP_TOL
        for _ in range(DILATE):
            bad = bad | np.roll(bad, 1) | np.roll(bad, -1)

        # A cell in the undisturbed left or right state carries no information;
        # including it only pulls the spline grids towards zero.
        tscale = np.maximum(np.abs(dUdt).max(axis=0), 1e-12)
        act = (np.abs(dUdt) / tscale).max(axis=1) > ACTIVE_TOL

        sl = slice(GUARD, n_cells - GUARD)
        feats.append(f[sl]); targs.append(dUdt[sl])
        smooths.append(~bad[sl]); actives.append(act[sl])
        sample_idx.append(n)

    return {
        "bx": bx, "dx": dx, "xc": xc, "U_final": U_final, "snaps": snaps,
        "Theta": np.concatenate(feats), "Y": np.concatenate(targs),
        "smooth": np.concatenate(smooths), "active": np.concatenate(actives),
        "per_snap": (feats, targs, smooths, sample_idx),
    }


# --- the held-out validation problem ---------------------------------------
bw = run_problem(BW_LEFT, BW_RIGHT, BW_BX, BW_CELLS, BW_T_END)
XC = bw["xc"]
print(f"[SIM]   canonical Brio-Wu: {BW_CELLS} cells to t={BW_T_END}, "
      f"{len(bw['snaps'])} snapshots  (VALIDATION ONLY)")

# --- the randomised training ensemble --------------------------------------
rng = np.random.default_rng(SEED)


def random_state():
    return [rng.uniform(0.15, 1.4),      # rho
            rng.uniform(-0.3, 0.3),      # u
            rng.uniform(-0.3, 0.3),      # v
            rng.uniform(0.15, 1.4),      # p
            rng.uniform(-1.2, 1.2)]      # By


problems = []
for i in range(N_PROBLEMS):
    bx_i = float(rng.uniform(0.4, 1.1))
    problems.append(run_problem(random_state(), random_state(), bx_i,
                                TR_CELLS, TR_T_END))
print(f"[SIM]   {N_PROBLEMS} randomised Riemann problems, "
      f"Bx in [0.4, 1.1] — varying Bx is what separates Bx*By from By")

Theta_ens = np.concatenate([p["Theta"] for p in problems])
Y_ens = np.concatenate([p["Y"] for p in problems])
smooth_ens = np.concatenate([p["smooth"] for p in problems])
active_ens = np.concatenate([p["active"] for p in problems])
good_ens = smooth_ens & active_ens
print(f"[DATA]  ensemble: {len(Theta_ens)} cell samples; "
      f"{100 * (~smooth_ens).mean():.1f}% shock-contaminated "
      f"(jump > {JUMP_TOL}, dilated {DILATE}), "
      f"{100 * (~active_ens).mean():.1f}% in an undisturbed state")

good_bw = bw["smooth"] & bw["active"]
print(f"[DATA]  Brio-Wu alone: {len(bw['Theta'])} cell samples, "
      f"{100 * good_bw.mean():.1f}% usable")


def _cond(T):
    return float(np.linalg.cond(np.column_stack([T, np.ones(len(T))])))


cond_ens = _cond(Theta_ens[good_ens])
cond_single = _cond(bw["Theta"][good_bw])
print(f"[DATA]  cond(Theta): one Riemann problem {cond_single:.1f}, "
      f"ensemble of {N_PROBLEMS} {cond_ens:.1f}")

# The residual of the TRUE equations, stated before any fit runs: this is why
# shock cells have to go.
resid = bw["Y"] - bw["Theta"] @ A_TRUE.T
r_smooth = float(np.sqrt(np.mean(resid[bw["smooth"] & bw["active"]] ** 2)))
r_shock = float(np.sqrt(np.mean(resid[~bw["smooth"]] ** 2)))
print(f"[DATA]  residual of the TRUE equations: {r_smooth:.4e} on smooth cells, "
      f"{r_shock:.4e} on shock cells ({r_shock / r_smooth:.0f}x)")

# ---------------------------------------------------------------------------
# 3. Three fits: the model, and two controls that isolate each lesson
#
#    The five equations differ by orders of magnitude in scale (the energy flux
#    dwarfs the mass flux) and the loss is an unweighted MSE, so each target is
#    divided by its own standard deviation and the coefficients rescaled
#    afterwards — the per-equation scaling of mathbio/epileptor_example.py.
# ---------------------------------------------------------------------------
N_TRAIN = 16000
Y_SCALE = Y_ens[good_ens].std(axis=0)
print(f"[DATA]  per-equation target scales: {np.array2string(Y_SCALE, precision=3)}")


def fit_on(Theta, Y, mask, tag):
    idx = np.flatnonzero(mask)
    idx = rng.choice(idx, min(N_TRAIN, len(idx)), replace=False)
    X, Yt = Theta[idx], Y[idx] / Y_SCALE
    lift = CustomLift(fn=lambda Z: Z, output_dim=N_FEATURES, name="mhd_flux_lift")
    m = KANDy(lift=lift, grid=5, k=3, steps=200, seed=SEED, base_fun="zero")
    m.fit(X=X, X_dot=Yt, val_frac=0.15, test_frac=0.15, lamb=0.0, patience=40)
    A_scaled, *_ = np.linalg.lstsq(X, m.predict(X), rcond=None)
    A_phys = A_scaled.T * Y_SCALE[:, None]              # undo the target scaling
    err = float(np.abs(A_phys - A_TRUE).max())
    print(f"[EVAL]  {tag}: {len(X)} rows, max |coefficient error| = {err:.4f}")
    return m, A_phys, err, X, Yt


print("\n--- Model A: ensemble of Riemann problems, smooth cells ---")
model, A_hat, coef_err, X_train, Y_train = fit_on(
    Theta_ens, Y_ens, good_ens, "ensemble + smooth cells")

print("\n--- Control 1: the Brio-Wu problem ALONE (self-similar data) ---")
model_single, A_single, coef_err_single, _, _ = fit_on(
    bw["Theta"], bw["Y"], good_bw, "single Riemann problem")

print("\n--- Control 2: the ensemble, but keeping shock cells ---")
model_shock, A_shock, coef_err_shock, _, _ = fit_on(
    Theta_ens, Y_ens, active_ens, "ensemble + shock cells")

similarity_penalty = coef_err_single / max(coef_err, 1e-12)
shock_penalty = coef_err_shock / max(coef_err, 1e-12)
print(f"\n[EVAL]  Same lift, same optimiser, only the SAMPLES differ:")
print(f"[EVAL]    one self-similar problem -> {similarity_penalty:.1f}x the "
      f"coefficient error")
print(f"[EVAL]    shock cells left in       -> {shock_penalty:.1f}x the "
      f"coefficient error")

np.set_printoptions(precision=3, suppress=True)
print(f"\n[EVAL]  Recovered readout A (rows {EQ_NAMES}):\n{A_hat}")
decoy_max = float(np.abs(A_hat[:, DECOY_IDX]).max())
decoy_max_single = float(np.abs(A_single[:, DECOY_IDX]).max())
tension_hat = float(A_hat[2, FI["dx_Bx_By"]])
induction_hat = A_hat[3, [FI["dx_u_By"], FI["dx_v_Bx"]]]
print(f"[EVAL]  magnetic tension, +dx[Bx*By] in d(rho v)/dt: {tension_hat:+.4f} "
      f"(true +1)")
print(f"[EVAL]  induction:  d(By)/dt = {induction_hat[0]:+.4f}*dx[u*By] "
      f"{induction_hat[1]:+.4f}*dx[v*Bx]   (true -1, +1)")
print(f"[EVAL]  largest weight on a decoy flux: {decoy_max:.4f}   "
      f"(single-problem fit put {decoy_max_single:.4f} there)")

pred = model.predict(X_train)
raw_r2 = [float(1.0 - np.sum((Y_train[:, i] - pred[:, i]) ** 2)
                / np.sum((Y_train[:, i] - Y_train[:, i].mean()) ** 2))
          for i in range(5)]
print(f"[EVAL]  Network R^2 per equation: {np.round(raw_r2, 6)}")

# ---------------------------------------------------------------------------
# 4. Validation — put the discovered flux back into the solver and re-run the
#    held-out Brio-Wu problem.  Before get_formula(), which mutates the model.
# ---------------------------------------------------------------------------
def make_flux(A_mat: np.ndarray):
    """Turn a recovered readout into a flux function the solver can use."""
    C = -A_mat

    def _flux(W: np.ndarray, bx: float) -> np.ndarray:
        return candidate_fluxes(W, bx) @ C.T

    return _flux


W0_bw = np.where((XC < 0.5)[:, None], np.array(BW_LEFT), np.array(BW_RIGHT))
U0_bw = prim2cons(W0_bw, BW_BX)
DX_BW = 1.0 / BW_CELLS
W_ref = cons2prim(bw["U_final"], BW_BX)

rollouts = {}
for tag, A_mat in [("ensemble", A_hat), ("single-problem", A_single),
                   ("with-shocks", A_shock)]:
    U_l, _ = evolve(U0_bw.copy(), BW_T_END, make_flux(A_mat), BW_BX, DX_BW)
    rollouts[tag] = cons2prim(U_l, BW_BX)

VAR_NAMES = ["rho", "u", "v", "p", "By"]


def rel_l1(W_pred):
    """Relative L1 error — the norm convergence is measured in for conservation
    laws with discontinuities.  An L2/RMSE comparison of shock-tube profiles is
    dominated by sub-cell shock POSITION: shifting a jump of O(1) by two cells
    out of 800 already costs several percent in RMSE while the wave structure
    is visually identical.  L1 is reported as the gate, L2 alongside it."""
    return {n: float(np.abs(W_pred[:, i] - W_ref[:, i]).sum()
                     / np.abs(W_ref[:, i] - W_ref[:, i].mean()).sum())
            for i, n in enumerate(VAR_NAMES)}


def rel_rmse(W_pred):
    return {n: float(np.sqrt(np.mean((W_pred[:, i] - W_ref[:, i]) ** 2))
                     / np.std(W_ref[:, i])) for i, n in enumerate(VAR_NAMES)}


roll_rel = rel_l1(rollouts["ensemble"])
roll_l2 = rel_rmse(rollouts["ensemble"])
roll_worst = max(roll_rel.values())
print(f"\n[EVAL]  Held-out Brio-Wu at t={BW_T_END} with the DISCOVERED flux, "
      f"relative L1: " + ", ".join(f"{k} {100 * v:.2f}%" for k, v in roll_rel.items()))
print(f"[EVAL]    (relative L2 for reference: "
      + ", ".join(f"{k} {100 * v:.2f}%" for k, v in roll_l2.items()) + ")")
for tag in ("single-problem", "with-shocks"):
    print(f"[EVAL]    same run, {tag} fit: worst variable "
          f"{100 * max(rel_l1(rollouts[tag]).values()):.2f}% (L1)")

theta_t = torch.tensor(X_train[:2048], dtype=torch.float32)

# ---------------------------------------------------------------------------
# 5. Symbolic extraction  (LAST — get_formula mutates the model in place)
#
#    The formulas are for the SCALED tendencies (dU/dt divided by Y_SCALE),
#    which is what the network was trained on; multiply an equation through by
#    its scale to read it in physical units.
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Extracting the ideal-MHD flux divergence ...")
formulas = model.get_formula(
    var_names=FEATURE_NAMES, round_places=4,
    lib=["x", "0"], r2_threshold=0.80, weight_simple=0.0,
)
for lab, scale_i, f_ in zip(EQ_NAMES, Y_SCALE, formulas):
    print(f"  {lab} / {scale_i:.4g} = {f_}")
r2 = model.score_formula(formulas, X_train, Y_train, var_names=FEATURE_NAMES)
print(f"[SYMBOLIC] Formula R^2 per equation: {np.round(r2, 6)}")
print("[SYMBOLIC] True (physical units):")
print("  d(rho)/dt   = -dx[rho*u]")
print("  d(rho u)/dt = -dx[rho*u^2] - dx[p] - dx[By^2/2]")
print("  d(rho v)/dt = -dx[rho*u*v] + dx[Bx*By]")
print("  d(By)/dt    = -dx[u*By] + dx[v*Bx]")
print("  d(E)/dt     = -dx[F_energy] + dx[Bx*(u*Bx + v*By)]")

# ---------------------------------------------------------------------------
# 6. Verification
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "coef_abs_err_max": 0.10,
    "decoy_weight_max": 0.10,
    "network_r2_min": 0.99,
    # The snapped formulas are slightly worse than the network they come from:
    # auto_symbolic refits each edge in isolation, and on the transverse
    # momentum equation that costs ~3% of R^2 even though the least-squares
    # readout of the same edge is within 0.2% of -1.  The readout, not the
    # snapped formula, is what this example claims to recover.
    "formula_r2_min": 0.95,
    "rollout_rel_l1_max": 0.05,
    "self_similarity_penalty_min": 3.0,
    "shock_contamination_penalty_min": 2.0,
    "shock_residual_ratio_min": 10.0,
}
passed = print_check(
    coef_err <= THRESHOLDS["coef_abs_err_max"]
    and decoy_max <= THRESHOLDS["decoy_weight_max"]
    and min(raw_r2) >= THRESHOLDS["network_r2_min"]
    and float(np.min(r2)) >= THRESHOLDS["formula_r2_min"]
    and roll_worst <= THRESHOLDS["rollout_rel_l1_max"]
    and similarity_penalty >= THRESHOLDS["self_similarity_penalty_min"]
    and shock_penalty >= THRESHOLDS["shock_contamination_penalty_min"]
    and (r_shock / r_smooth) >= THRESHOLDS["shock_residual_ratio_min"],
    "Brio-Wu: ideal-MHD flux recovered from an ensemble; both traps measured",
)
RESULT_DIR = ensure_outdir("results/BrioWu")
save_metrics(RESULT_DIR / "metrics.json", {
    "task": "BrioWu",
    "category": "plasma",
    "seed": SEED,
    "sampling": ("MUSCL-HLL snapshots from an ensemble of randomised Riemann "
                 "problems; central time differences; canonical Brio-Wu held out"),
    "gamma": GAMMA, "brio_wu_Bx": BW_BX, "brio_wu_cells": BW_CELLS,
    "brio_wu_t_end": BW_T_END, "n_training_problems": N_PROBLEMS,
    "feature_names": FEATURE_NAMES, "equation_names": EQ_NAMES,
    "true_readout": A_TRUE.tolist(),
    "recovered_readout": A_hat.tolist(),
    "recovered_readout_single_problem": A_single.tolist(),
    "recovered_readout_with_shocks": A_shock.tolist(),
    "coef_abs_err": coef_err,
    "coef_abs_err_single_problem": coef_err_single,
    "coef_abs_err_with_shocks": coef_err_shock,
    "self_similarity_penalty": float(similarity_penalty),
    "shock_contamination_penalty": float(shock_penalty),
    "cond_theta_ensemble": cond_ens,
    "cond_theta_single_problem": cond_single,
    "shock_cell_fraction": float((~smooth_ens).mean()),
    "shock_residual_ratio": float(r_shock / r_smooth),
    "decoy_weight_max": decoy_max,
    "decoy_weight_max_single_problem": decoy_max_single,
    "magnetic_tension_coef": tension_hat,
    "induction_coefs": induction_hat.tolist(),
    "network_r2": raw_r2,
    "formula_r2": np.asarray(r2).tolist(),
    "target_scales": Y_SCALE.tolist(),
    "rollout_rel_l1": roll_rel,
    "rollout_rel_l2": roll_l2,
    "rollout_rel_l1_single_problem": rel_l1(rollouts["single-problem"]),
    "rollout_rel_l1_with_shocks": rel_l1(rollouts["with-shocks"]),
    "thresholds": THRESHOLDS,
    "passed": passed,
})


# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
def plot_shock_tube(outdir: str) -> None:
    """Brio-Wu at t = T_END, integrated with each discovered flux function."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [(0, r"density $\rho$"), (4, r"transverse field $B_y$"),
              (1, r"velocity $u$"), (3, r"pressure $p$")]
    for ax, (i, name) in zip(axes.ravel(), panels):
        ax.plot(XC, W_ref[:, i], color="#1f77b4", lw=1.7, label="reference solver")
        ax.plot(XC, rollouts["ensemble"][:, i], color="#d62728", lw=1.1, ls="--",
                label="KANDy flux (ensemble fit)")
        ax.plot(XC, rollouts["single-problem"][:, i], color="#7f7f7f", lw=0.9,
                ls=":", label="KANDy flux (single-problem fit)")
        ax.set_xlabel("x"); ax.set_ylabel(name)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].annotate("compound wave", xy=(0.44, 0.72), xycoords="axes fraction",
                        fontsize=8, color="0.3")
    fig.suptitle(f"Held-out Brio-Wu shock tube at $t={BW_T_END}$, integrated with "
                 "the discovered ideal-MHD flux", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{outdir}/shock_tube_profiles.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{outdir}/shock_tube_profiles.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_shock_mask(outdir: str) -> None:
    """Which cells the jump detector discards, and the residual that justifies it."""
    feats, targs, smooths, sample_idx = bw["per_snap"]
    n = len(smooths) // 2
    W_n = cons2prim(bw["snaps"][sample_idx[n]], BW_BX)
    bad = ~smooths[n]
    xk = XC[GUARD:BW_CELLS - GUARD]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    axes[0].plot(xk, W_n[GUARD:-GUARD, 0], color="#1f77b4", lw=1.4, label=r"$\rho$")
    axes[0].plot(xk, W_n[GUARD:-GUARD, 4], color="#2ca02c", lw=1.4, label=r"$B_y$")
    for xb in xk[bad]:
        axes[0].axvline(xb, color="#d62728", alpha=0.10, lw=1.2)
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("state")
    axes[0].set_title(f"Cells discarded at $t={sample_idx[n] * DT_SNAP:.3f}$ "
                      f"({100 * bad.mean():.1f}% of the domain, shaded)")
    res = np.abs(targs[n] - feats[n] @ A_TRUE.T).max(axis=1)
    axes[1].semilogy(xk, np.maximum(res, 1e-12), color="0.3", lw=0.9)
    axes[1].set_ylabel("residual of the\ntrue equations")
    axes[1].set_xlabel("x")
    fig.tight_layout()
    fig.savefig(f"{outdir}/shock_cell_mask.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_readout(outdir: str) -> None:
    """Recovered flux composition for all three fits, against the truth."""
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    panels = [(A_TRUE, "true readout"), (A_hat, "ensemble, smooth cells"),
              (A_single, "single Riemann problem"), (A_shock, "ensemble + shocks")]
    for ax, (M, title) in zip(axes, panels):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
        ax.set_xticks(range(N_FEATURES))
        ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=7)
        ax.set_yticks(range(5)); ax.set_yticklabels(EQ_NAMES, fontsize=8)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Ideal-MHD flux composition: the lift is identical in all "
                 "three fits, only the samples differ", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{outdir}/readout_matrix.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{outdir}/readout_matrix.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


use_pub_style()
OUT = str(RESULT_DIR)
plot_shock_tube(OUT)
plot_shock_mask(OUT)
plot_readout(OUT)
if getattr(model, "train_results_", None):
    fig, _ = plot_loss_curves(model.train_results_, save=f"{OUT}/loss_curves")
    plt.close(fig)
fig, _ = plot_all_edges(
    model.model_, X=theta_t,
    in_var_names=FEATURE_NAMES, out_var_names=EQ_NAMES,
    save=f"{OUT}/edge_activations",
)
plt.close(fig)
print(f"[FIGS]  Saved {OUT}/")
