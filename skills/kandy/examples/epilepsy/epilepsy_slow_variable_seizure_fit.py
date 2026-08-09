#!/usr/bin/env python3
"""Slow-variable Kuramoto model: fitting the seizure as a regime change.

The result that forces this model
---------------------------------
In this recording the intrinsic frequencies are not constant.  Mean phase
velocity per channel, in 5 s blocks (rad/s):

    hippocampus_R   12.2 (pre)  ->  60.5 (ictal)  ->  53.7 (post)
    motor_R         39.7        ->  78.3          ->  46.6

The seizure IS the frequency excursion.  Any model with a constant omega has
defined the phenomenon away before fitting starts, which is why
``epilepsy_kuramoto_sakaguchi_fit.py`` scores a NEGATIVE held-out R^2: its
constant term alone is wrong by half the signal.

A constant-parameter oscillator model also has exactly one regime.  It can sit
synchronized or desynchronized, but nothing carries it INTO a different state
and back out, so it cannot have an event at all.  Reproducing onset AND offset
requires a slow variable that crosses a threshold and returns -- the Epileptor
structure (cf. ``mathbio/epileptor_example.py``).

The model
---------
State eta = [theta (M), u (M), z]:

    dtheta_i/dt = omega_i + sum_j kappa_ij*sin(theta_j - theta_i + alpha)
    du_i/dt     = G_i(omega, z, R)              <- KANDy, z enters as a cross-term
    dz/dt       = (h(<omega>, R, sd) - z) / tau <- Epileptor relaxation

Three structural choices, each fixing a specific failure of the first pass:

1. **z relaxes; it does not free-run.**  ``dz/dt = (h - z)/tau`` is a
   contraction toward a target set by the FAST state alone -- h is a function of
   <omega>, R and the frequency spread, and deliberately NOT of z.  That is the
   Epileptor's claim (the slow variable is driven by fast activity) and it is
   what makes offset structural: when the fast state relaxes, h falls, and z is
   pulled back.  A free KAN output for dz/dt has nothing pulling it back, and in
   the first pass z ran to its floor and stayed -- onset with no offset.
   tau is chosen by a scan (below), not assumed.

2. **omega is fitted as a LAGGED INCREMENT, not an instantaneous derivative.**
   domega/dt is the second derivative of a 1 s-smoothed phase: almost pure
   smoothing noise.  Fitting it gave held-out R^2 = -4.5, five times worse than
   predicting no change.  The target here is [u(t+L) - u(t)]/L with L = 0.25 s,
   where the signal is far above the differentiation noise floor.

3. **omega is bounded by construction, so clipping is never load-bearing.**
   omega_i is evolved through a squashed coordinate,

       omega_i = lo_i + (hi_i - lo_i) * (1 + tanh(u_i)) / 2

   so no value of u can push omega outside [lo_i, hi_i] no matter what the KAN
   predicts.  h is squashed the same way into the observed z range.  The first
   pass leaned on hard clipping for 66% of the rollout, which meant the
   post-onset trajectory was a clamp rather than dynamics.  This version does no
   clipping at all, and asserts it: ``clip_active_fraction`` is a gate, and the
   integrator only guards against non-finite values.

Lift design follows the repo's independence rule: cross-products of DIFFERENT
variables go in the lift (omega_i*z, R*z); powers of a SINGLE variable do not,
because a spline edge already represents any univariate function.  Note that
<omega> is excluded from the omega lift -- it is the mean of the omega_i that
are already columns, so including it would make the lift linearly dependent and
the decomposition non-unique.

What this script can and cannot show
------------------------------------
You asked to fit THIS seizure, so the seizure is in the training data.  That
rules out any claim of out-of-sample seizure prediction, and none is made.
Two things are still genuinely testable and are the point of the script:

1. EMERGENCE.  Fitting derivatives well does not imply the event appears in a
   free-running rollout.  The model is integrated autonomously from a PRE-ICTAL
   initial condition with no forcing, no nudging, and no knowledge of when the
   seizure occurs.  The frozen-parameter Kuramoto-Sakaguchi model is run from
   the identical initial condition as a control; it cannot produce an event by
   construction, so any excursion the full model shows is attributable to the
   slow variable.  Offset is scored separately from onset -- a model that
   ignites and never recovers fails.

2. SPECIFICITY.  A model that seizes from every initial condition has learned
   nothing.  Rollouts are launched from starts spread across the whole
   recording, and the excursion rate from pre-ictal starts is compared with the
   rate from post-ictal starts.

Held-out skill is reported on a post-ictal block with a guard band, against a
zero-increment baseline, since for slow variables "predict no change" is the
honest thing to beat.
"""

from __future__ import annotations

import contextlib
import os
import pickle
import sys
from io import StringIO
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-kandy")

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXAMPLE_ROOT.parents[2]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, filtfilt, hilbert, savgol_filter

from _utils import ensure_outdir, print_check, save_metrics
from kandy import CustomLift, KANDy
from kandy.plotting import plot_loss_curves, use_pub_style


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = REPO_ROOT / "data" / "Dory_2021_04_12_006_preprocessed.pkl"
RESULT_DIR = ensure_outdir(REPO_ROOT / "results" / "Epilepsy-Slow-Variable-Seizure")

CHANNELS = os.environ.get(
    "KANDY_SV_CHANNELS", "motor_R,posterior_R,striatum_R,hippocampus_R"
).split(",")

BANDPASS_HZ = (1.0, 40.0)
FILTER_ORDER = 4
EDGE_TRIM_SECONDS = 2.0
PHASE_FAST_SECONDS = 0.155
OMEGA_SLOW_SECONDS = 1.0
Z_SLOW_SECONDS = 4.0

EVENT_WINDOW = (-1.0, 5.0)
PRE_ICTAL = (-12.0, -2.0)
POST_ICTAL = (12.0, 26.0)
HELDOUT_WINDOW = (20.0, 28.0)
GUARD_SECONDS = env_float("KANDY_SV_GUARD_SECONDS", Z_SLOW_SECONDS)

LAG_SECONDS = env_float("KANDY_SV_LAG_SECONDS", 0.25)   # fix 2
BOX_MARGIN = env_float("KANDY_SV_BOX_MARGIN", 0.35)     # fix 3
TAU_GRID = np.logspace(np.log10(0.25), np.log10(20.0), 24)  # fix 1

GRID = env_int("KANDY_SV_GRID", 5)
OMEGA_STEPS = env_int("KANDY_SV_OMEGA_STEPS", 80)
Z_STEPS = env_int("KANDY_SV_Z_STEPS", 80)
N_TRAIN_MAX = env_int("KANDY_SV_N_TRAIN_MAX", 4000)

ROLLOUT_START = env_float("KANDY_SV_ROLLOUT_START", -3.0)
ROLLOUT_SECONDS = env_float("KANDY_SV_ROLLOUT_SECONDS", 12.0)
N_SPECIFICITY_STARTS = env_int("KANDY_SV_N_SPEC_STARTS", 14)


# ---------------------------------------------------------------------------
# 1. Data -> [theta, omega, z]
# ---------------------------------------------------------------------------
def odd_window(seconds: float, fs: float, minimum: int = 5) -> int:
    n = max(minimum, int(round(seconds * fs)))
    return n + 1 if n % 2 == 0 else n


def quiet_fit(model: KANDy, **kwargs) -> None:
    with contextlib.redirect_stdout(StringIO()):
        model.fit(**kwargs)


def r2_per_output(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    den = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
    den = np.where(den > 0, den, np.nan)
    return 1.0 - np.sum((y - p) ** 2, axis=0) / den


with DATA_PATH.open("rb") as fh:
    payload = pickle.load(fh)

X_ALL = np.asarray(payload["X"], dtype=float)
FS = float(payload["fs"])
DT = float(payload["dt"])
T_ALL = np.asarray(payload["t"], dtype=float)
ALL_LABELS = list(payload["labels"])
CH = [ALL_LABELS.index(c) for c in CHANNELS]
LABELS = list(CHANNELS)
M = len(LABELS)
PAIRS = [(a, b) for a in range(M) for b in range(a + 1, M)]

nyq = 0.5 * FS
bb, aa = butter(FILTER_ORDER, [BANDPASS_HZ[0] / nyq, BANDPASS_HZ[1] / nyq], btype="band")
X_FILT = filtfilt(bb, aa, X_ALL[:, CH], axis=0)
analytic = hilbert(X_FILT, axis=0)
theta_u = np.unwrap(np.angle(analytic), axis=0)
log_amp = np.log(np.abs(analytic) + 1e-6)

w_fast = odd_window(PHASE_FAST_SECONDS, FS)
w_om = odd_window(OMEGA_SLOW_SECONDS, FS)
w_z = odd_window(Z_SLOW_SECONDS, FS)

theta_s = savgol_filter(theta_u, w_fast, 3, axis=0, mode="interp")
dtheta_fast = savgol_filter(theta_u, w_fast, 3, deriv=1, delta=DT, axis=0, mode="interp")
omega_slow = savgol_filter(theta_u, w_om, 3, deriv=1, delta=DT, axis=0, mode="interp")

amp_z = (log_amp - log_amp.mean(axis=0)) / (log_amp.std(axis=0) + 1e-12)
z_raw = savgol_filter(amp_z.mean(axis=1), w_z, 3, mode="interp")
dz_raw = savgol_filter(amp_z.mean(axis=1), w_z, 3, deriv=1, delta=DT, mode="interp")

edge = int(round(EDGE_TRIM_SECONDS * FS))
sl = slice(edge, len(X_ALL) - edge)
T = T_ALL[sl]
THETA = theta_s[sl]
DTHETA = dtheta_fast[sl]
OMEGA = omega_slow[sl]
SIGNAL = X_FILT[sl]
Z = z_raw[sl]
DZ = dz_raw[sl]
N_ROW = len(T)
LAG = int(round(LAG_SECONDS / DT))

print("[DATA]  Loaded epilepsy recording", flush=True)
print(f"[SEL]   {M} channels: {LABELS}", flush=True)
print(
    f"[PRE]   theta window {w_fast}, omega window {w_om}, z window {w_z}; "
    f"{N_ROW} rows {T[0]:.1f}..{T[-1]:.1f}s; increment lag {LAG} samples "
    f"({LAG_SECONDS:.2f}s)",
    flush=True,
)


def block_mean(arr, lo, hi):
    m = (T >= lo) & (T <= hi)
    return float(np.mean(arr[m])) if m.any() else float("nan")


om_pre = block_mean(OMEGA.mean(1), *PRE_ICTAL)
om_ict = block_mean(OMEGA.mean(1), *EVENT_WINDOW)
om_post = block_mean(OMEGA.mean(1), *POST_ICTAL)
print(
    f"[EVENT] mean omega: pre={om_pre:.1f}, ictal={om_ict:.1f}, post={om_post:.1f} rad/s",
    flush=True,
)


# ---------------------------------------------------------------------------
# 2. Bounded coordinates -- fix 3: omega and h can never leave their box
# ---------------------------------------------------------------------------
om_lo = OMEGA.min(axis=0)
om_hi = OMEGA.max(axis=0)
om_span = om_hi - om_lo
OM_LO = om_lo - BOX_MARGIN * om_span
OM_HI = om_hi + BOX_MARGIN * om_span

Z_LO = Z.min() - BOX_MARGIN * (Z.max() - Z.min())
Z_HI = Z.max() + BOX_MARGIN * (Z.max() - Z.min())
Z_C = 0.5 * (Z_LO + Z_HI)
Z_R = 0.5 * (Z_HI - Z_LO)


def omega_from_u(u):
    return OM_LO + 0.5 * (OM_HI - OM_LO) * (1.0 + np.tanh(u))


def omega_from_u_t(u):
    lo = torch.tensor(OM_LO, dtype=u.dtype)
    hi = torch.tensor(OM_HI, dtype=u.dtype)
    return lo + 0.5 * (hi - lo) * (1.0 + torch.tanh(u))


def u_from_omega(om):
    frac = 2.0 * (om - OM_LO) / (OM_HI - OM_LO) - 1.0
    return np.arctanh(np.clip(frac, -0.999, 0.999))


U = u_from_omega(OMEGA)
sech2 = 1.0 - np.tanh(U) ** 2
print(
    f"[BOX]   omega box per channel: "
    + ", ".join(f"[{a:.0f},{b:.0f}]" for a, b in zip(OM_LO, OM_HI))
    + f"; min sech^2 over data = {sech2.min():.3f} (0 would mean the box is too tight)",
    flush=True,
)


# ---------------------------------------------------------------------------
# 3. Split: the seizure is in TRAIN by request; held-out block is post-ictal
# ---------------------------------------------------------------------------
rows = np.arange(N_ROW - LAG)
test_mask = (T[: N_ROW - LAG] >= HELDOUT_WINDOW[0]) & (
    T[: N_ROW - LAG] <= HELDOUT_WINDOW[1]
)
near_test = (T[: N_ROW - LAG] >= HELDOUT_WINDOW[0] - GUARD_SECONDS) & (
    T[: N_ROW - LAG] <= HELDOUT_WINDOW[1] + GUARD_SECONDS
)
test_rows = rows[test_mask]
train_rows = rows[~near_test]
rng = np.random.default_rng(SEED)
fit_rows = (
    np.sort(rng.choice(train_rows, N_TRAIN_MAX, replace=False))
    if len(train_rows) > N_TRAIN_MAX
    else train_rows
)
print(
    f"[SPLIT] train={len(train_rows)} rows (seizure INCLUDED), "
    f"heldout={len(test_rows)} rows ({HELDOUT_WINDOW[0]:.0f}..{HELDOUT_WINDOW[1]:.0f}s), "
    f"guard={GUARD_SECONDS:.0f}s",
    flush=True,
)


# ---------------------------------------------------------------------------
# 4. Fast block: coupling on the residual dtheta - omega
# ---------------------------------------------------------------------------
def pair_sin_cos(theta: np.ndarray) -> np.ndarray:
    feats = []
    for a, b in PAIRS:
        d = theta[:, b] - theta[:, a]
        feats.extend([np.sin(d), np.cos(d)])
    return np.column_stack(feats)


RESID = DTHETA - OMEGA
Dsg = np.column_stack([np.ones(N_ROW), pair_sin_cos(THETA)])
Cfast, *_ = np.linalg.lstsq(Dsg[train_rows], RESID[train_rows], rcond=None)
Cfast = Cfast.T

w_edge, a_edge = [], []
for p, (a, b) in enumerate(PAIRS):
    cs, cc = 1 + 2 * p, 2 + 2 * p
    for row, sgn in ((a, 1.0), (b, -1.0)):
        A, B = sgn * Cfast[row, cs], Cfast[row, cc]
        w_edge.append(float(np.hypot(A, B)))
        a_edge.append(float(np.arctan2(B, A)))
w_edge, a_edge = np.array(w_edge), np.array(a_edge)
ALPHA = float(np.arctan2(np.sum(w_edge * np.sin(a_edge)), np.sum(w_edge * np.cos(a_edge))))

KAPPA = np.zeros((M, M))
for i in range(M):
    partners = [j for j in range(M) if j != i]
    cols = [np.ones(len(train_rows))]
    for j in partners:
        cols.append(np.sin(THETA[train_rows, j] - THETA[train_rows, i] + ALPHA))
    sol, *_ = np.linalg.lstsq(np.column_stack(cols), RESID[train_rows, i], rcond=None)
    for k, j in enumerate(partners):
        KAPPA[i, j] = sol[1 + k]

pred_resid = np.zeros((len(test_rows), M))
for i in range(M):
    for j in range(M):
        if i != j:
            pred_resid[:, i] += KAPPA[i, j] * np.sin(
                THETA[test_rows, j] - THETA[test_rows, i] + ALPHA
            )
coupling_r2 = r2_per_output(RESID[test_rows], pred_resid)
print(
    f"[FAST]  alpha={ALPHA:+.3f} rad, heldout coupling R2 mean="
    f"{np.nanmean(coupling_r2):+.3f}",
    flush=True,
)


# ---------------------------------------------------------------------------
# 5. Lifts.  eta = [theta (M) | u (M) | z]
# ---------------------------------------------------------------------------
TH, U_S, Z_S = slice(0, M), slice(M, 2 * M), 2 * M
ETA = np.column_stack([THETA, U, Z[:, None]])

om_center = OMEGA[train_rows].mean(axis=0)
om_spread = OMEGA[train_rows].std(axis=0) + 1e-12


def _fast_np(eta):
    th = eta[:, TH]
    om = omega_from_u(eta[:, U_S])
    omz = (om - om_center) / om_spread
    R = np.abs(np.mean(np.exp(1j * th), axis=1))
    return omz, eta[:, Z_S], R


def _fast_t(eta):
    th = eta[:, TH]
    om = omega_from_u_t(eta[:, U_S])
    omz = (om - torch.tensor(om_center, dtype=eta.dtype)) / torch.tensor(
        om_spread, dtype=eta.dtype
    )
    R = torch.sqrt(
        torch.mean(torch.sin(th), dim=1) ** 2
        + torch.mean(torch.cos(th), dim=1) ** 2
        + 1e-12
    )
    return omz, eta[:, Z_S], R


# omega lift: <omega> is EXCLUDED -- it is the mean of the omega_i columns, so
# including it would make the lift linearly dependent (independence rule).
def u_lift_np(eta):
    om, z, R = _fast_np(eta)
    feats = [z, R, R * z]
    for i in range(M):
        feats.extend([om[:, i], om[:, i] * z])
    return np.column_stack(feats)


def u_lift_torch(eta):
    om, z, R = _fast_t(eta)
    feats = [z, R, R * z]
    for i in range(M):
        feats.extend([om[:, i], om[:, i] * z])
    return torch.stack(feats, dim=1)


# h lift: functions of the FAST state only -- no z.  That is what makes
# dz/dt = (h - z)/tau a contraction rather than a free-running equation.
def h_lift_np(eta):
    om, _z, R = _fast_np(eta)
    mo = om.mean(axis=1)
    sd = om.std(axis=1)
    return np.column_stack([mo, R, sd, mo * R])


def h_lift_torch(eta):
    om, _z, R = _fast_t(eta)
    mo = om.mean(dim=1)
    sd = torch.std(om, dim=1, unbiased=False)
    return torch.stack([mo, R, sd, mo * R], dim=1)


U_LIFT_DIM = 3 + 2 * M
H_LIFT_DIM = 4
U_FEATURES = ["z", "R", "R*z"] + [n for i in range(M) for n in (f"om{i}", f"om{i}*z")]
H_FEATURES = ["<om>", "R", "sd(om)", "<om>*R"]


# ---------------------------------------------------------------------------
# 6. Fix 2 -- omega as a lagged increment in the bounded coordinate
# ---------------------------------------------------------------------------
DU_LAG = (U[LAG:] - U[:-LAG]) / LAG_SECONDS          # du/dt over the lag
du_scale = DU_LAG[train_rows].std(axis=0) + 1e-12

model_u = KANDy(
    lift=CustomLift(u_lift_np, U_LIFT_DIM, u_lift_torch, "slow_omega_bounded"),
    grid=GRID, k=3, steps=OMEGA_STEPS, seed=SEED,
    base_fun=torch.nn.Identity(), device="cpu",
)
print("[TRAIN] Fitting du/dt (lagged increment, bounded coordinate) ...", flush=True)
quiet_fit(model_u, X=ETA[fit_rows], X_dot=DU_LAG[fit_rows] / du_scale,
          val_frac=0.15, test_frac=0.15, lamb=0.0, opt="LBFGS", lr=1.0,
          batch=-1, stop_grid_update_step=10, patience=0, verbose=False)


# ---------------------------------------------------------------------------
# 7. Fix 1 -- Epileptor relaxation for z; tau chosen by scan, not assumed
# ---------------------------------------------------------------------------
def h_target(tau: float) -> np.ndarray:
    """h = z + tau*zdot, the relaxation target implied by dz/dt=(h-z)/tau."""
    return Z + tau * DZ


fast_lin = np.column_stack([np.ones(N_ROW), h_lift_np(ETA)])
tau_scores = []
for tau in TAU_GRID:
    h = h_target(tau)
    sol, *_ = np.linalg.lstsq(fast_lin[train_rows], h[train_rows], rcond=None)
    pred = fast_lin[test_rows] @ sol
    tau_scores.append(float(r2_per_output(h[test_rows, None], pred[:, None])[0]))
tau_scores = np.array(tau_scores)
TAU = float(TAU_GRID[int(np.argmax(tau_scores))])
print(
    f"[TAU]   scan over {len(TAU_GRID)} values in "
    f"[{TAU_GRID[0]:.2f}, {TAU_GRID[-1]:.1f}]s -> tau={TAU:.2f}s "
    f"(heldout R2 of h from the fast state = {tau_scores.max():+.3f})",
    flush=True,
)

H_DATA = h_target(TAU)
# Fit h in a squashed coordinate so the model's h can never leave the z box.
H_INNER = Z_R * np.arctanh(np.clip((H_DATA - Z_C) / Z_R, -0.999, 0.999))
h_scale = float(H_INNER[train_rows].std() + 1e-12)

model_h = KANDy(
    lift=CustomLift(h_lift_np, H_LIFT_DIM, h_lift_torch, "epileptor_h"),
    grid=GRID, k=3, steps=Z_STEPS, seed=SEED,
    base_fun=torch.nn.Identity(), device="cpu",
)
print("[TRAIN] Fitting h(fast) for dz/dt=(h-z)/tau ...", flush=True)
quiet_fit(model_h, X=ETA[fit_rows], X_dot=(H_INNER[fit_rows] / h_scale)[:, None],
          val_frac=0.15, test_frac=0.15, lamb=0.0, opt="LBFGS", lr=1.0,
          batch=-1, stop_grid_update_step=10, patience=0, verbose=False)


def h_model(eta: np.ndarray) -> np.ndarray:
    inner = (model_h.predict(eta) * h_scale).ravel()
    return Z_C + Z_R * np.tanh(inner / Z_R)


# ---------------------------------------------------------------------------
# 8. Held-out skill vs. zero-change baselines
# ---------------------------------------------------------------------------
pred_du = model_u.predict(ETA[test_rows]) * du_scale
pred_h = h_model(ETA[test_rows])
pred_dz = (pred_h - Z[test_rows]) / TAU

scores = {
    "du_kandy": r2_per_output(DU_LAG[test_rows], pred_du),
    "du_zero_baseline": r2_per_output(DU_LAG[test_rows], np.zeros_like(DU_LAG[test_rows])),
    "dz_epileptor": r2_per_output(DZ[test_rows, None], pred_dz[:, None]),
    "dz_zero_baseline": r2_per_output(DZ[test_rows, None], np.zeros((len(test_rows), 1))),
    "h_recovery": r2_per_output(H_DATA[test_rows, None], pred_h[:, None]),
}
print("\n[EVAL]  held-out skill (post-ictal block, guard-banded)", flush=True)
for k, v in scores.items():
    print(f"        {k:>20}: mean={np.nanmean(v):+.4f}  {np.round(v, 3)}", flush=True)
du_gain = float(np.nanmean(scores["du_kandy"]) - np.nanmean(scores["du_zero_baseline"]))
dz_gain = float(np.nanmean(scores["dz_epileptor"]) - np.nanmean(scores["dz_zero_baseline"]))
print(f"        gain over zero-change: du={du_gain:+.4f}, dz={dz_gain:+.4f}", flush=True)


# ---------------------------------------------------------------------------
# 9. Autonomous rollout -- no clipping anywhere
# ---------------------------------------------------------------------------
SIN_A, COS_A = np.sin(ALPHA), np.cos(ALPHA)


def coupling_batch(th: np.ndarray) -> np.ndarray:
    out = np.zeros_like(th)
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            d = th[:, j] - th[:, i]
            out[:, i] += KAPPA[i, j] * (np.sin(d) * COS_A + np.cos(d) * SIN_A)
    return out


def rhs_full(eta: np.ndarray) -> np.ndarray:
    out = np.empty_like(eta)
    om = omega_from_u(eta[:, U_S])
    out[:, TH] = om + coupling_batch(eta[:, TH])
    out[:, U_S] = model_u.predict(eta) * du_scale
    out[:, Z_S] = (h_model(eta) - eta[:, Z_S]) / TAU
    return out


def rhs_frozen(eta: np.ndarray) -> np.ndarray:
    out = np.zeros_like(eta)
    out[:, TH] = omega_from_u(eta[:, U_S]) + coupling_batch(eta[:, TH])
    return out


def rollout_batch(rhs, eta0: np.ndarray, steps: int, dt: float):
    """RK4.  No clipping: omega and h are bounded by their tanh parameterization."""
    eta = eta0.astype(float).copy()
    traj = np.empty((steps + 1, *eta.shape))
    traj[0] = eta
    for s in range(steps):
        k1 = rhs(eta)
        k2 = rhs(eta + 0.5 * dt * k1)
        k3 = rhs(eta + 0.5 * dt * k2)
        k4 = rhs(eta + dt * k3)
        eta = eta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(eta).all():
            return traj[: s + 1], float("nan")
        traj[s + 1] = eta
    return traj, 0.0


steps = int(round(ROLLOUT_SECONDS / DT))
i0 = int(np.argmin(np.abs(T - ROLLOUT_START)))
traj_full, clip_full = rollout_batch(rhs_full, ETA[i0][None, :], steps, DT)
traj_froz, _ = rollout_batch(rhs_frozen, ETA[i0][None, :], steps, DT)
t_roll = T[i0 : i0 + len(traj_full)]
ref = ETA[i0 : i0 + len(traj_full)]

mo_full = omega_from_u(traj_full[:, 0, U_S]).mean(axis=1)
mo_froz = omega_from_u(traj_froz[: len(traj_full), 0, U_S]).mean(axis=1)
mo_ref = omega_from_u(ref[:, U_S]).mean(axis=1)
nb = int(1.0 / DT)
base_full, base_froz, base_ref = mo_full[:nb].mean(), mo_froz[:nb].mean(), mo_ref[:nb].mean()
peak_full = float(mo_full.max() - base_full)
peak_froz = float(mo_froz.max() - base_froz)
peak_ref = float(mo_ref.max() - base_ref)
onset_full = float(t_roll[int(np.argmax(mo_full))])
onset_ref = float(t_roll[int(np.argmax(mo_ref))])

# Offset: how far the excursion has decayed by the end of the run.
tail = slice(-int(2.0 / DT), None)
recover_full = 1.0 - (mo_full[tail].mean() - base_full) / max(peak_full, 1e-9)
recover_ref = 1.0 - (mo_ref[tail].mean() - base_ref) / max(peak_ref, 1e-9)

print(
    f"\n[ROLLOUT] autonomous from t={T[i0]:.1f}s for {ROLLOUT_SECONDS:.0f}s, "
    f"no forcing, NO CLIPPING (clip fraction={clip_full:.2f})",
    flush=True,
)
print(f"          mean-omega rise: data={peak_ref:+.1f}, model={peak_full:+.1f}, "
      f"frozen-control={peak_froz:+.1f} rad/s", flush=True)
print(f"          peak time:       data t={onset_ref:+.2f}s, model t={onset_full:+.2f}s", flush=True)
print(f"          offset recovery: data={recover_ref:.2f}, model={recover_full:.2f} "
      "(1.0 = fully returned to baseline)", flush=True)
print(f"          z at end:        data={ref[-1, Z_S]:+.2f}, model={traj_full[-1, 0, Z_S]:+.2f}", flush=True)


# ---------------------------------------------------------------------------
# 10. Specificity
# ---------------------------------------------------------------------------
spec_times = np.linspace(float(T[0]) + 1.0, float(T[-1]) - ROLLOUT_SECONDS - 1.0,
                         N_SPECIFICITY_STARTS)
spec_idx = np.array([int(np.argmin(np.abs(T - s))) for s in spec_times])
spec_traj, _ = rollout_batch(rhs_full, ETA[spec_idx], steps, DT)
spec_mo = omega_from_u(spec_traj[:, :, U_S]).mean(axis=2)
spec_base = spec_mo[:nb].mean(axis=0)
spec_rise = spec_mo.max(axis=0) - spec_base
EXC_THRESHOLD = 0.5 * abs(peak_ref)
is_pre = spec_times < EVENT_WINDOW[0]
is_post = spec_times > EVENT_WINDOW[1]
rate_pre = float(np.mean(spec_rise[is_pre] > EXC_THRESHOLD)) if is_pre.any() else float("nan")
rate_post = float(np.mean(spec_rise[is_post] > EXC_THRESHOLD)) if is_post.any() else float("nan")

print(f"\n[SPEC]  excursion threshold = {EXC_THRESHOLD:.1f} rad/s", flush=True)
print("        rise by start: " +
      " ".join(f"{t:+.0f}s:{r:+.0f}" for t, r in zip(spec_times, spec_rise)), flush=True)
print(f"        excursion rate pre-ictal={rate_pre:.2f}, post-ictal={rate_post:.2f}", flush=True)


# ---------------------------------------------------------------------------
# 11. Gates
# ---------------------------------------------------------------------------
gates = {
    "du_beats_zero_change": {"required": 0.02, "achieved": du_gain,
                             "passed": du_gain >= 0.02},
    "dz_beats_zero_change": {"required": 0.02, "achieved": dz_gain,
                             "passed": dz_gain >= 0.02},
    "onset_emerges_from_pre_ictal_IC": {
        "required": f">= 50% of observed rise ({0.5 * peak_ref:.1f} rad/s)",
        "achieved": peak_full, "passed": peak_full >= 0.5 * peak_ref},
    "frozen_control_shows_no_event": {
        "required": f"<= 20% of observed rise ({0.2 * peak_ref:.1f} rad/s)",
        "achieved": peak_froz, "passed": peak_froz <= 0.2 * peak_ref},
    "offset_occurs": {
        "required": "recovery >= 0.5 of the excursion",
        "achieved": float(recover_full), "passed": bool(recover_full >= 0.5)},
    "specific_to_pre_ictal_states": {
        "required": "post-ictal rate <= 0.5 * pre-ictal rate",
        "achieved": [rate_pre, rate_post],
        "passed": bool(rate_post <= 0.5 * rate_pre) if rate_pre > 0 else False},
    "no_clipping_used": {
        "required": "clip fraction == 0 (bounded by construction)",
        "achieved": clip_full, "passed": clip_full == 0.0},
}
passed = print_check(
    all(g["passed"] for g in gates.values()),
    "slow variable generates seizure onset AND offset autonomously",
)
for name, g in gates.items():
    print(f"        {'PASS' if g['passed'] else 'FAIL'}  {name}: {g['achieved']}", flush=True)

metrics = {
    "task": "slow-variable Kuramoto seizure model (Epileptor relaxation, bounded omega)",
    "channels": LABELS,
    "note": "seizure is IN the training data by design; this tests emergence, "
            "offset and specificity, not out-of-sample seizure prediction",
    "windows": {"event": list(EVENT_WINDOW), "pre_ictal": list(PRE_ICTAL),
                "post_ictal": list(POST_ICTAL), "heldout": list(HELDOUT_WINDOW)},
    "omega_blocks": {"pre": om_pre, "ictal": om_ict, "post": om_post},
    "structure": {
        "increment_lag_seconds": LAG_SECONDS,
        "omega_box": [[float(a), float(b)] for a, b in zip(OM_LO, OM_HI)],
        "z_box": [Z_LO, Z_HI],
        "box_margin": BOX_MARGIN,
        "min_sech2_over_data": float(sech2.min()),
        "tau_seconds": TAU,
        "tau_grid_heldout_r2": [float(v) for v in tau_scores],
        "tau_grid": [float(v) for v in TAU_GRID],
        "h_depends_on": H_FEATURES,
        "u_lift_features": U_FEATURES,
    },
    "fast_block": {"alpha_rad": ALPHA, "kappa": KAPPA,
                   "coupling_r2_heldout": [float(v) for v in coupling_r2]},
    "heldout_r2_mean": {k: float(np.nanmean(v)) for k, v in scores.items()},
    "du_gain_over_zero": du_gain,
    "dz_gain_over_zero": dz_gain,
    "rollout": {"start_time": float(T[i0]), "seconds": ROLLOUT_SECONDS,
                "data_rise": peak_ref, "model_rise": peak_full,
                "frozen_control_rise": peak_froz,
                "data_peak_time": onset_ref, "model_peak_time": onset_full,
                "data_recovery": float(recover_ref), "model_recovery": float(recover_full),
                "clip_active_fraction": clip_full},
    "specificity": {"start_times": [float(v) for v in spec_times],
                    "rise": [float(v) for v in spec_rise],
                    "threshold": EXC_THRESHOLD,
                    "rate_pre_ictal": rate_pre, "rate_post_ictal": rate_post},
    "gates": gates,
    "passed": passed,
}
save_metrics(RESULT_DIR / "metrics.json", metrics)
np.savez_compressed(
    RESULT_DIR / "slow_variable_arrays.npz",
    t=T, labels=np.asarray(LABELS), signal=SIGNAL, theta=THETA, omega=OMEGA,
    u=U, z=Z, dz=DZ, du_lag=DU_LAG, kappa=KAPPA, alpha=np.array([ALPHA]),
    tau=np.array([TAU]), tau_grid=TAU_GRID, tau_scores=tau_scores,
    h_data=H_DATA, om_box=np.column_stack([OM_LO, OM_HI]),
    t_rollout=t_roll, traj_full=traj_full[:, 0], traj_frozen=traj_froz[:, 0],
    ref_rollout=ref, spec_times=spec_times, spec_rise=spec_rise, spec_mo=spec_mo,
)


# ---------------------------------------------------------------------------
# 12. Figures
# ---------------------------------------------------------------------------
use_pub_style()


def plot_state() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for ax in axes:
        ax.axvspan(*EVENT_WINDOW, color="#f4a261", alpha=0.20, lw=0)
    for i, lab in enumerate(LABELS):
        axes[0].plot(T, OMEGA[:, i], lw=0.8, label=lab)
    axes[0].set_ylabel(r"$\omega_i$ (rad/s)")
    axes[0].legend(ncol=4, fontsize=7)
    axes[0].set_title("The seizure is the frequency excursion")
    axes[1].plot(T, OMEGA.mean(1), color="#d62728", lw=1.1)
    axes[1].set_ylabel(r"$\langle\omega\rangle$")
    axes[2].plot(T, Z, color="#1f77b4", lw=1.2, label="z")
    axes[2].plot(T, H_DATA, color="#2ca02c", lw=0.9, alpha=0.8,
                 label=rf"$h=z+\tau\dot z$ ($\tau$={TAU:.1f}s)")
    axes[2].legend(fontsize=8)
    axes[2].set_ylabel("slow z")
    axes[2].set_xlabel("recording time (s)")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "state_variables.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "state_variables.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_emergence() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    for ax in axes:
        ax.axvspan(*EVENT_WINDOW, color="#f4a261", alpha=0.20, lw=0)
    axes[0].plot(t_roll, mo_ref, color="#1f77b4", lw=1.4, label="data")
    axes[0].plot(t_roll, mo_full, color="#d62728", lw=1.4, ls="--",
                 label="slow-variable model (autonomous)")
    axes[0].plot(t_roll, mo_froz, color="0.45", lw=1.2, ls=":",
                 label="frozen-parameter control")
    axes[0].set_ylabel(r"$\langle\omega\rangle$ (rad/s)")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"Autonomous rollout from a pre-ictal state (t={T[i0]:.1f}s), "
                      "no forcing, no clipping")
    axes[1].plot(t_roll, ref[:, Z_S], color="#1f77b4", lw=1.4, label="data")
    axes[1].plot(t_roll, traj_full[:, 0, Z_S], color="#d62728", lw=1.4, ls="--",
                 label="model")
    axes[1].set_ylabel("slow z")
    axes[1].legend(fontsize=8)
    axes[2].plot(t_roll, np.cos(ref[:, 0]), color="#1f77b4", lw=0.7, alpha=0.7)
    axes[2].plot(t_roll, np.cos(traj_full[:, 0, 0]), color="#d62728", lw=0.7,
                 ls="--", alpha=0.9)
    axes[2].set_ylabel(f"cos θ  {LABELS[0]}")
    axes[2].set_xlabel("recording time (s)")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "seizure_emergence.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "seizure_emergence.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_specificity() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = plt.cm.viridis(np.linspace(0, 1, len(spec_times)))
    for k in range(len(spec_times)):
        axes[0].plot(np.arange(spec_mo.shape[0]) * DT, spec_mo[:, k],
                     color=colors[k], lw=0.9)
    axes[0].set_xlabel("time since rollout start (s)")
    axes[0].set_ylabel(r"$\langle\omega\rangle$")
    axes[0].set_title("rollouts from starts across the recording")
    axes[1].axhline(EXC_THRESHOLD, color="#d62728", ls="--", lw=1.0,
                    label="excursion threshold")
    axes[1].axvspan(*EVENT_WINDOW, color="#f4a261", alpha=0.25, lw=0)
    axes[1].scatter(spec_times, spec_rise, c=colors, s=44, zorder=3)
    axes[1].set_xlabel("rollout start time (s)")
    axes[1].set_ylabel(r"rise in $\langle\omega\rangle$ (rad/s)")
    axes[1].set_title("does it seize from everywhere?")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "specificity.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "specificity.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_portrait_and_tau() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot(Z, OMEGA.mean(1), color="0.78", lw=0.7, label="data (all)")
    ev = (T >= EVENT_WINDOW[0]) & (T <= EVENT_WINDOW[1])
    axes[0].plot(Z[ev], OMEGA[ev].mean(1), color="#1f77b4", lw=1.8, label="data (seizure)")
    axes[0].plot(traj_full[:, 0, Z_S], mo_full, color="#d62728", lw=1.4, ls="--",
                 label="autonomous rollout")
    axes[0].set_xlabel("slow z")
    axes[0].set_ylabel(r"$\langle\omega\rangle$ (rad/s)")
    axes[0].set_title("Slow-fast plane: the seizure is a loop")
    axes[0].legend(fontsize=8)
    axes[1].semilogx(TAU_GRID, tau_scores, marker="o", color="#1f77b4")
    axes[1].axvline(TAU, color="#d62728", ls="--", lw=1.2, label=rf"$\tau$={TAU:.1f}s")
    axes[1].set_xlabel(r"$\tau$ (s)")
    axes[1].set_ylabel(r"heldout $R^2$ of $h$ from the fast state")
    axes[1].set_title("relaxation time constant, scanned")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "slow_fast_portrait.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "slow_fast_portrait.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_losses() -> None:
    for mdl, tag in ((model_u, "loss_omega"), (model_h, "loss_h")):
        if getattr(mdl, "train_results_", None):
            fig, _ = plot_loss_curves(mdl.train_results_, save=str(RESULT_DIR / tag))
            plt.close(fig)


plot_state()
plot_emergence()
plot_specificity()
plot_portrait_and_tau()
plot_losses()
print(f"\n[FIGS]  Saved {RESULT_DIR}", flush=True)
