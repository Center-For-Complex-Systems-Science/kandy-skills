#!/usr/bin/env python3
"""Corrected adaptive-Kuramoto model with the data assimilation swept, and with
omega switchable between a constant and a state.

Relationship to ``epilepsy_adaptive_kuramoto_fit.py``
-----------------------------------------------------
Same pair lifts, same kappa proxy, same signal reconstruction.  Corrections:

1. **Explicit omega, fitted outside the KAN.**  The original phase lift is 75
   pair columns with no constant and no omega, and the state is [theta, kappa]
   with no frequency in it at all, so the intrinsic frequency had to be
   synthesized from the offsets of 75 coupling splines.  That makes omega and
   coupling mutually unidentifiable -- you cannot ask how much of dtheta/dt is
   frequency and how much is interaction, because they live in the same
   parameters.  Here dtheta_i/dt = omega_i + Phi_i(pair lift), with the KAN fit
   to the residual dtheta_i/dt - omega_i.

2. **base_fun = 0.**  Pure spline edges, no scale_base*x passthrough.  Being
   precise about what this does NOT do, since it is easy to assume otherwise:
   zeroing the base does not stop the model carrying a constant.  Measured
   directly -- a KAN with base_fun=0 over a lift of only sin/cos columns
   reproduces a target of mean 40.29 with pred_mean 40.289.  The spline
   coefficients carry the offset.  Item 1 is what gets the constant out, by
   giving the KAN a residual that already has zero mean.

3. **Chronological split with a guard band** wider than the widest filter,
   replacing a stride-5 interleave that let held-out rows share
   Savitzky-Golay support with training rows.

The assimilation is swept, not chosen
-------------------------------------
    dx/dt = f_model(x) + a * q_data(t) + lam * (x_data(t) - x)

The original runs a=1, lam=20 and reports the resulting correlation as a
result.  Every setting here reports both the replay quality AND the
assimilation share, || a*q + lam*(x_data - x) || / || total dx/dt ||, in
per-output-scaled units: instead of a correlation whose meaning depends on a
hidden lam, you get a curve and can say "corr X with share Y".  lam=0, a=0 is
the anchor -- that row is the model alone.

What the constant-omega run established
---------------------------------------
Replay quality is a STEP in lam, not a graded curve: corr stayed at 0.01-0.03
for lam <= 10 and jumped to 0.72 at lam = 20.  The reason is exact -- uniformly
random phase error has RMS pi/sqrt(3) = 1.814 rad, and every theta RMSE at
lam <= 10 sat at that ceiling (1.72-1.80).  Those replays are phase-decorrelated;
there is no "light assimilation" regime, only off and on.

Under a nudge the steady-state phase error is ~ dOmega/lam, where dOmega is the
model's frequency error.  So the threshold is set by dOmega and by nothing the
model gets right.  With constant omega, lam=20 gave 0.598 rad, implying
dOmega ~ 12 rad/s.

The experiment this script runs
-------------------------------
``KANDY_AK_OMEGA_MODE`` selects:

    constant   omega_i is a per-channel constant (the run above)
    state      omega_i is a STATE, evolved in a bounded coordinate u with
               omega_i = squash(u_i) and a relaxational law
                   du_i/dt = (U_target(theta, kappa, omega) - u_i) / tau_u
               tau_u scanned on a validation block.  Relaxation rather than a
               free KAN output is deliberate: a free du/dt has nothing pulling
               it back, and a small constant bias in a fitted derivative
               integrates into u -> +-inf, at which point sech^2 -> 0 and omega
               is pinned at the box wall -- bounded but dynamically dead.

Everything else is held fixed, so the two runs differ in one thing.  The
prediction being tested is quantitative, not directional:

    lam_min ~ dOmega

If promoting omega to a state cuts dOmega, lam_min must fall in proportion.
Both quantities are measured here -- dOmega directly as the RMS phase-velocity
error of the RHS on measured states over the event window, lam_min as the
smallest swept lam whose theta RMSE clears the decorrelation ceiling -- so the
ratio is reported and the claim can fail.
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

from _utils import ensure_outdir, print_check, save_metrics, wrap_pi
from kandy import CustomLift, KANDy
from kandy.plotting import plot_loss_curves, use_pub_style


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

OMEGA_MODE = os.environ.get("KANDY_AK_OMEGA_MODE", "state").strip().lower()
if OMEGA_MODE not in {"constant", "state"}:
    raise ValueError("KANDY_AK_OMEGA_MODE must be 'constant' or 'state'")
OMEGA_IS_STATE = OMEGA_MODE == "state"

DATA_PATH = REPO_ROOT / "data" / "Dory_2021_04_12_006_preprocessed.pkl"
RESULT_DIR = ensure_outdir(
    REPO_ROOT / "results" / f"Epilepsy-Adaptive-Assimilation-{OMEGA_MODE}")

BANDPASS_HZ = (1.0, 40.0)
FILTER_ORDER = 4
EDGE_TRIM_SECONDS = 2.0
PHASE_SAVGOL_SECONDS = 0.155
OMEGA_SLOW_SECONDS = 1.0
KAPPA_SAVGOL_SECONDS = 0.505
FORCING_SAVGOL_SECONDS = 0.055

GRID = env_int("KANDY_AK_GRID", 4)
FIT_STEPS = env_int("KANDY_AK_STEPS", 40)
U_STEPS = env_int("KANDY_AK_U_STEPS", 60)
N_TRAIN_MAX = env_int("KANDY_AK_N_TRAIN_MAX", 4000)
TRAIN_FRAC = env_float("KANDY_AK_TRAIN_FRAC", 0.70)
VAL_FRAC_OF_TRAIN = env_float("KANDY_AK_VAL_FRAC", 0.15)
GUARD_SECONDS = env_float("KANDY_AK_GUARD_SECONDS", 1.0)
LAG_SECONDS = env_float("KANDY_AK_LAG_SECONDS", 0.25)
BOX_MARGIN = env_float("KANDY_AK_BOX_MARGIN", 0.35)
TAU_U_GRID = np.logspace(np.log10(0.05), np.log10(10.0), 24)

EVENT_WINDOW = (-1.0, 4.0)
LAM_GRID = [float(v) for v in os.environ.get(
    "KANDY_AK_LAM_GRID", "0,1,2,3,5,7,10,14,20,28,40").split(",")]
Q_GRID = [float(v) for v in os.environ.get("KANDY_AK_Q_GRID", "0,1").split(",")]
TARGET_CORR = env_float("KANDY_AK_TARGET_CORR", 0.50)
# theta RMSE below this counts as "phase is being tracked at all".  The
# decorrelation ceiling (uniform phase error) is pi/sqrt(3) = 1.8138.
DECORRELATED_RMSE = float(np.pi / np.sqrt(3.0))
TRACKING_RMSE = env_float("KANDY_AK_TRACKING_RMSE", 1.0)


class ZeroBase(torch.nn.Module):
    """base_fun = 0: edges are pure splines, no scale_base*x passthrough."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


# ---------------------------------------------------------------------------
# 1. Data
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

X_RAW = np.asarray(payload["X"], dtype=float)
FS = float(payload["fs"])
DT = float(payload["dt"])
T_ALL = np.asarray(payload["t"], dtype=float)
LABELS = list(payload["labels"])
M = X_RAW.shape[1]

PAIRS = [(i, j) for i in range(M) for j in range(i + 1, M)]
PAIR_INDEX = {p: k for k, p in enumerate(PAIRS)}
P = len(PAIRS)

nyq = 0.5 * FS
bb, aa = butter(FILTER_ORDER, [BANDPASS_HZ[0] / nyq, BANDPASS_HZ[1] / nyq], btype="band")
X_FILT = filtfilt(bb, aa, X_RAW, axis=0)
analytic = hilbert(X_FILT, axis=0)
theta_u = np.unwrap(np.angle(analytic), axis=0)

log_amp_raw = np.log(np.abs(analytic) + 1e-6)
amp_mean, amp_std = log_amp_raw.mean(axis=0), log_amp_raw.std(axis=0) + 1e-12
log_amp = (log_amp_raw - amp_mean) / amp_std

w_theta = odd_window(PHASE_SAVGOL_SECONDS, FS)
w_omega = odd_window(OMEGA_SLOW_SECONDS, FS)
w_kappa = odd_window(KAPPA_SAVGOL_SECONDS, FS)
theta_s = savgol_filter(theta_u, w_theta, 3, axis=0, mode="interp")
theta_dot = savgol_filter(theta_u, w_theta, 3, deriv=1, delta=DT, axis=0, mode="interp")
omega_slow = savgol_filter(theta_u, w_omega, 3, deriv=1, delta=DT, axis=0, mode="interp")

pair_log_amp = np.column_stack([0.5 * (log_amp[:, i] + log_amp[:, j]) for i, j in PAIRS])
kappa_raw = savgol_filter(pair_log_amp, w_kappa, 3, axis=0, mode="interp")
kappa_dot_raw = savgol_filter(pair_log_amp, w_kappa, 3, deriv=1, delta=DT,
                              axis=0, mode="interp")
kappa_mean, kappa_std = kappa_raw.mean(axis=0), kappa_raw.std(axis=0) + 1e-12
kappa_z = (kappa_raw - kappa_mean) / kappa_std
kappa_dot = kappa_dot_raw / kappa_std

edge = int(round(EDGE_TRIM_SECONDS * FS))
sl = slice(edge, len(X_RAW) - edge)
T_FULL = T_ALL[sl]
SIGNAL_FULL = X_FILT[sl]
THETA_FULL = theta_s[sl]
DTHETA_FULL = theta_dot[sl]
OMEGA_SLOW_FULL = omega_slow[sl]
KAPPA_FULL = kappa_z[sl]
DKAPPA_FULL = kappa_dot[sl]

LAG = int(round(LAG_SECONDS / DT))
TRIM = LAG if OMEGA_IS_STATE else 0     # lagged du/dt target needs headroom
KEEP = len(T_FULL) - TRIM

T = T_FULL[:KEEP]
SIGNAL = SIGNAL_FULL[:KEEP]
THETA = THETA_FULL[:KEEP]
DTHETA = DTHETA_FULL[:KEEP]
KAPPA = KAPPA_FULL[:KEEP]
DKAPPA = DKAPPA_FULL[:KEEP]
N_ROW = len(T)

print(f"[MODE]  omega is a {'STATE' if OMEGA_IS_STATE else 'CONSTANT'}", flush=True)
print("[DATA]  Loaded epilepsy recording", flush=True)
print(f"        {N_ROW} rows {T[0]:.1f}..{T[-1]:.1f}s, {M} channels, {P} pairs",
      flush=True)


# ---------------------------------------------------------------------------
# 2. State layout
# ---------------------------------------------------------------------------
def make_box(arr):
    lo, hi = arr.min(axis=0), arr.max(axis=0)
    span = hi - lo
    return lo - BOX_MARGIN * span, hi + BOX_MARGIN * span


def squash(x, lo, hi):
    return lo + 0.5 * (hi - lo) * (1.0 + np.tanh(x))


def squash_t(x, lo, hi):
    lo_t = torch.as_tensor(lo, dtype=x.dtype)
    hi_t = torch.as_tensor(hi, dtype=x.dtype)
    return lo_t + 0.5 * (hi_t - lo_t) * (1.0 + torch.tanh(x))


def unsquash(y, lo, hi):
    return np.arctanh(np.clip(2.0 * (y - lo) / (hi - lo) - 1.0, -0.999, 0.999))


TH_S = slice(0, M)
if OMEGA_IS_STATE:
    U_S = slice(M, 2 * M)
    KP_S = slice(2 * M, 2 * M + P)
    N_STATE = 2 * M + P
    OM_LO, OM_HI = make_box(OMEGA_SLOW_FULL)
    U_FULL = unsquash(OMEGA_SLOW_FULL, OM_LO, OM_HI)
    U = U_FULL[:KEEP]
    DU_LAG = (U_FULL[LAG:] - U_FULL[:-LAG])[:KEEP] / LAG_SECONDS
    U_LO, U_HI = make_box(U)
    U_C, U_R = 0.5 * (U_LO + U_HI), 0.5 * (U_HI - U_LO)
    STATE = np.column_stack([THETA, U, KAPPA])
    print(f"[BOX]   omega box "
          + ", ".join(f"[{a:.0f},{b:.0f}]" for a, b in zip(OM_LO, OM_HI))
          + f"; min sech^2 over data = {float((1 - np.tanh(U) ** 2).min()):.3f}",
          flush=True)
else:
    KP_S = slice(M, M + P)
    N_STATE = M + P
    STATE = np.column_stack([THETA, KAPPA])


# ---------------------------------------------------------------------------
# 3. Chronological split; a validation slice is carved out of TRAIN for tau_u
# ---------------------------------------------------------------------------
guard = max(int(round(GUARD_SECONDS * FS)), w_kappa)
rows = np.arange(N_ROW)
n_tr = int(N_ROW * TRAIN_FRAC)
train_all = rows[: n_tr - guard]
n_val = int(len(train_all) * VAL_FRAC_OF_TRAIN)
val_rows = train_all[-n_val:]
train_rows = train_all[: -(n_val + guard)]
test_rows = rows[n_tr + guard:]
rng = np.random.default_rng(SEED)
fit_rows = (np.sort(rng.choice(train_rows, N_TRAIN_MAX, replace=False))
            if len(train_rows) > N_TRAIN_MAX else train_rows)
event_in_train = bool(np.any((T[train_rows] >= EVENT_WINDOW[0])
                             & (T[train_rows] <= EVENT_WINDOW[1])))
print(
    f"[SPLIT] train={len(train_rows)} ({T[train_rows[0]]:.1f}..{T[train_rows[-1]]:.1f}s), "
    f"val={len(val_rows)} ({T[val_rows[0]]:.1f}..{T[val_rows[-1]]:.1f}s), "
    f"test={len(test_rows)} ({T[test_rows[0]]:.1f}..{T[test_rows[-1]]:.1f}s); "
    f"seizure in TRAIN: {event_in_train}",
    flush=True,
)


# ---------------------------------------------------------------------------
# 4. Lifts -- pair lifts unchanged from the original adaptive model
# ---------------------------------------------------------------------------
def phase_lift_np(state):
    th, kp = state[:, TH_S], state[:, KP_S]
    feats = []
    for i, j in PAIRS:
        k = kp[:, PAIR_INDEX[(i, j)]]
        d = th[:, j] - th[:, i]
        s, c = np.sin(d), np.cos(d)
        feats.extend([k * s, k * c, s, c, k])
    return np.column_stack(feats)


def phase_lift_torch(state):
    th, kp = state[:, TH_S], state[:, KP_S]
    feats = []
    for i, j in PAIRS:
        k = kp[:, PAIR_INDEX[(i, j)]]
        d = th[:, j] - th[:, i]
        s, c = torch.sin(d), torch.cos(d)
        feats.extend([k * s, k * c, s, c, k])
    return torch.stack(feats, dim=1)


def kappa_lift_np(state):
    th, kp = state[:, TH_S], state[:, KP_S]
    sins, coss = [], []
    for i, j in PAIRS:
        d = th[:, j] - th[:, i]
        sins.append(np.sin(d))
        coss.append(np.cos(d))
    return np.column_stack(sins + coss + [kp[:, k] for k in range(P)])


def kappa_lift_torch(state):
    th, kp = state[:, TH_S], state[:, KP_S]
    sins, coss = [], []
    for i, j in PAIRS:
        d = th[:, j] - th[:, i]
        sins.append(torch.sin(d))
        coss.append(torch.cos(d))
    return torch.stack(sins + coss + [kp[:, k] for k in range(P)], dim=1)


if OMEGA_IS_STATE:
    om_center = OMEGA_SLOW_FULL[train_rows].mean(axis=0)
    om_spread = OMEGA_SLOW_FULL[train_rows].std(axis=0) + 1e-12

    # Cross-terms between DIFFERENT variables are explicit; <kappa> is a scalar
    # summary and the individual kappa_p are not in this lift, so no linear
    # dependence is introduced.
    def u_lift_np(state):
        th = state[:, TH_S]
        om = (squash(state[:, U_S], OM_LO, OM_HI) - om_center) / om_spread
        kp = state[:, KP_S]
        R = np.abs(np.mean(np.exp(1j * th), axis=1))
        mk = kp.mean(axis=1)
        feats = [np.ones(len(state)), R, mk, R * mk]
        for i in range(M):
            feats.extend([om[:, i], om[:, i] * R, om[:, i] * mk])
        return np.column_stack(feats)

    def u_lift_torch(state):
        th = state[:, TH_S]
        om = (squash_t(state[:, U_S], OM_LO, OM_HI)
              - torch.as_tensor(om_center, dtype=state.dtype)) / \
            torch.as_tensor(om_spread, dtype=state.dtype)
        kp = state[:, KP_S]
        R = torch.sqrt(torch.mean(torch.sin(th), dim=1) ** 2
                       + torch.mean(torch.cos(th), dim=1) ** 2 + 1e-12)
        mk = torch.mean(kp, dim=1)
        feats = [torch.ones_like(R), R, mk, R * mk]
        for i in range(M):
            feats.extend([om[:, i], om[:, i] * R, om[:, i] * mk])
        return torch.stack(feats, dim=1)

    U_LIFT_DIM = 4 + 3 * M


# ---------------------------------------------------------------------------
# 5. Explicit frequency term, then KANs on the residuals
# ---------------------------------------------------------------------------
if OMEGA_IS_STATE:
    OMEGA_TERM = OMEGA_SLOW_FULL[:KEEP]      # time-varying, supplied by the u state
else:
    OMEGA_TERM = np.tile(DTHETA[train_rows].mean(axis=0), (N_ROW, 1))

OMEGA_REPORT = OMEGA_TERM[train_rows].mean(axis=0)
C_KAPPA = DKAPPA[train_rows].mean(axis=0)
THETA_RESID = DTHETA - OMEGA_TERM
KAPPA_RESID = DKAPPA - C_KAPPA
th_scale = THETA_RESID[train_rows].std(axis=0) + 1e-12
kp_scale = KAPPA_RESID[train_rows].std(axis=0) + 1e-12

print("[OMEGA] mean intrinsic frequency on train rows:", flush=True)
for i, lab in enumerate(LABELS):
    print(f"        {lab:>14}: {OMEGA_REPORT[i]:+7.2f} rad/s "
          f"({OMEGA_REPORT[i] / (2 * np.pi):5.2f} Hz)", flush=True)

model_phase = KANDy(
    lift=CustomLift(phase_lift_np, 5 * P, phase_lift_torch, "adaptive_phase"),
    grid=GRID, k=3, steps=FIT_STEPS, seed=SEED, base_fun=ZeroBase(), device="cpu",
)
model_kappa = KANDy(
    lift=CustomLift(kappa_lift_np, 3 * P, kappa_lift_torch, "adaptive_kappa"),
    grid=GRID, k=3, steps=FIT_STEPS, seed=SEED, base_fun=ZeroBase(), device="cpu",
)
print("[TRAIN] Fitting phase residual (base_fun=0, omega pulled out) ...", flush=True)
quiet_fit(model_phase, X=STATE[fit_rows], X_dot=THETA_RESID[fit_rows] / th_scale,
          val_frac=0.1, test_frac=0.1, lamb=0.0, opt="LBFGS", lr=1.0, batch=-1,
          stop_grid_update_step=5, patience=0, verbose=False)
print("[TRAIN] Fitting kappa residual ...", flush=True)
quiet_fit(model_kappa, X=STATE[fit_rows], X_dot=KAPPA_RESID[fit_rows] / kp_scale,
          val_frac=0.1, test_frac=0.1, lamb=0.0, opt="LBFGS", lr=1.0, batch=-1,
          stop_grid_update_step=5, patience=0, verbose=False)

TAU_U = float("nan")
if OMEGA_IS_STATE:
    u_lift_vals = u_lift_np(STATE)
    tau_scores = []
    for tau in TAU_U_GRID:
        tgt = U + tau * DU_LAG
        sol, *_ = np.linalg.lstsq(u_lift_vals[train_rows], tgt[train_rows], rcond=None)
        pred = np.clip(u_lift_vals[val_rows] @ sol, U_C - U_R, U_C + U_R)
        rate = (pred - U[val_rows]) / tau
        tau_scores.append(float(np.nanmean(r2_per_output(DU_LAG[val_rows], rate))))
    tau_scores = np.array(tau_scores)
    TAU_U = float(TAU_U_GRID[int(np.argmax(tau_scores))])
    print(f"[TAU]   tau_u={TAU_U:.2f}s (validation R2 of du/dt = {tau_scores.max():+.3f})"
          + ("   [on grid boundary]"
             if TAU_U in (TAU_U_GRID[0], TAU_U_GRID[-1]) else ""), flush=True)

    U_INNER = U_R * np.arctanh(np.clip((U + TAU_U * DU_LAG - U_C) / U_R, -0.999, 0.999))
    u_scale = U_INNER[train_rows].std(axis=0) + 1e-12
    model_u = KANDy(
        lift=CustomLift(u_lift_np, U_LIFT_DIM, u_lift_torch, "omega_state"),
        grid=GRID, k=3, steps=U_STEPS, seed=SEED, base_fun=ZeroBase(), device="cpu",
    )
    print("[TRAIN] Fitting du/dt (relaxational, bounded coordinate) ...", flush=True)
    quiet_fit(model_u, X=STATE[fit_rows], X_dot=U_INNER[fit_rows] / u_scale,
              val_frac=0.1, test_frac=0.1, lamb=0.0, opt="LBFGS", lr=1.0, batch=-1,
              stop_grid_update_step=5, patience=0, verbose=False)


def f_model(state: np.ndarray) -> np.ndarray:
    s = np.atleast_2d(state)
    if OMEGA_IS_STATE:
        om = squash(s[:, U_S], OM_LO, OM_HI)
        dth = om + model_phase.predict(s) * th_scale
        tgt = U_C + U_R * np.tanh((model_u.predict(s) * u_scale) / U_R)
        du = (tgt - s[:, U_S]) / TAU_U
        dkp = C_KAPPA + model_kappa.predict(s) * kp_scale
        return np.column_stack([dth, du, dkp])
    dth = OMEGA_TERM[0] + model_phase.predict(s) * th_scale
    dkp = C_KAPPA + model_kappa.predict(s) * kp_scale
    return np.column_stack([dth, dkp])


coupling_rms = float(np.sqrt(np.mean(
    (model_phase.predict(STATE[train_rows]) * th_scale) ** 2)))
omega_rms = float(np.sqrt(np.mean(OMEGA_REPORT ** 2)))
print(f"[COUPLE] rms coupling = {coupling_rms:.3f} rad/s vs rms omega = "
      f"{omega_rms:.1f} rad/s (ratio {coupling_rms / max(omega_rms, 1e-12):.3f})",
      flush=True)


# ---------------------------------------------------------------------------
# 6. dOmega: the RHS phase-velocity error on measured states.  This is the
#    quantity predicted to set the nudge threshold.
# ---------------------------------------------------------------------------
STATE_DOT = (np.column_stack([DTHETA, DU_LAG, DKAPPA]) if OMEGA_IS_STATE
             else np.column_stack([DTHETA, DKAPPA]))
F_ON_DATA = f_model(STATE)
ev_mask = (T >= EVENT_WINDOW[0]) & (T <= EVENT_WINDOW[1])
DELTA_OMEGA = float(np.sqrt(np.mean(
    (F_ON_DATA[ev_mask][:, TH_S] - DTHETA[ev_mask]) ** 2)))
DELTA_OMEGA_ALL = float(np.sqrt(np.mean((F_ON_DATA[:, TH_S] - DTHETA) ** 2)))
print(f"\n[dOMEGA] RHS phase-velocity error: event window={DELTA_OMEGA:.2f} rad/s, "
      f"whole recording={DELTA_OMEGA_ALL:.2f} rad/s", flush=True)

pred = F_ON_DATA[test_rows]
base = np.zeros_like(pred)
base[:, TH_S] = OMEGA_TERM[test_rows]
base[:, KP_S] = C_KAPPA
truth = STATE_DOT[test_rows]
r2_model = r2_per_output(truth, pred)
r2_base = r2_per_output(truth, base)
phase_gain = float(np.nanmean(r2_model[TH_S]) - np.nanmean(r2_base[TH_S]))
print("[EVAL]  held-out derivative R2 (chronological test block)", flush=True)
print(f"        phase: model={np.nanmean(r2_model[TH_S]):+.4f}, "
      f"omega-only={np.nanmean(r2_base[TH_S]):+.4f}, gain={phase_gain:+.4f}", flush=True)


# ---------------------------------------------------------------------------
# 7. Residual drive q(t)
# ---------------------------------------------------------------------------
STATE_SCALE = np.std(STATE_DOT, axis=0) + 1e-12
resid = STATE_DOT - F_ON_DATA
w_q = odd_window(FORCING_SAVGOL_SECONDS, FS)
Q_DOT = savgol_filter(resid, w_q, 3, axis=0, mode="interp")
q_share_static = float(np.mean(np.abs(Q_DOT / STATE_SCALE))
                       / max(np.mean(np.abs(STATE_DOT / STATE_SCALE)), 1e-12))
print(f"[Q]     |q|/|dx/dt| = {q_share_static:.3f} in scaled units", flush=True)


# ---------------------------------------------------------------------------
# 8. Assimilation sweep
# ---------------------------------------------------------------------------
def half_grid(t0, t1):
    steps = int(round((t1 - t0) / DT))
    half_t = t0 + np.arange(2 * steps + 1) * (0.5 * DT)
    ref = np.column_stack([np.interp(half_t, T, STATE[:, c]) for c in range(N_STATE)])
    qq = np.column_stack([np.interp(half_t, T, Q_DOT[:, c]) for c in range(N_STATE)])
    return steps, half_t, ref, qq


def replay(lam, a, grid):
    steps, half_t, ref, qq = grid
    times = half_t[::2]
    x = ref[0].copy()
    out = np.empty((steps + 1, N_STATE))
    out[0] = x
    model_mag = assim_mag = 0.0

    def rhs(state, h, accumulate=False):
        nonlocal model_mag, assim_mag
        m = f_model(state)[0]
        ext = a * qq[h] + lam * (ref[h] - state)
        if accumulate:
            model_mag += float(np.mean(np.abs(m / STATE_SCALE)))
            assim_mag += float(np.mean(np.abs(ext / STATE_SCALE)))
        return m + ext

    for s in range(steps):
        h = 2 * s
        k1 = rhs(x, h, accumulate=True)
        k2 = rhs(x + 0.5 * DT * k1, h + 1)
        k3 = rhs(x + 0.5 * DT * k2, h + 1)
        k4 = rhs(x + DT * k3, h + 2)
        x = x + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(x).all():
            return times[: s + 1], out[: s + 1], float("nan")
        out[s + 1] = x
    return times, out, assim_mag / max(model_mag + assim_mag, 1e-12)


def state_to_signal(state):
    s = np.atleast_2d(state)
    pair_avg = np.zeros((P, M))
    for r, (i, j) in enumerate(PAIRS):
        pair_avg[r, i] = 0.5
        pair_avg[r, j] = 0.5
    pair_log = s[:, KP_S] * kappa_std + kappa_mean
    chan_log = np.clip(pair_log @ np.linalg.pinv(pair_avg).T, -5.0, 5.0)
    return np.exp(chan_log * amp_std + amp_mean) * np.cos(s[:, TH_S])


def score(times, traj):
    ref_state = np.column_stack([np.interp(times, T, STATE[:, c])
                                 for c in range(N_STATE)])
    ref_sig = np.column_stack([np.interp(times, T, SIGNAL[:, c]) for c in range(M)])
    pred_sig = state_to_signal(traj)
    corr = [float(np.corrcoef(pred_sig[:, c], ref_sig[:, c])[0, 1])
            if np.std(pred_sig[:, c]) > 1e-12 else float("nan") for c in range(M)]
    r_pred = np.abs(np.mean(np.exp(1j * traj[:, TH_S]), axis=1))
    r_ref = np.abs(np.mean(np.exp(1j * ref_state[:, TH_S]), axis=1))
    return {
        "signal_corr_mean": float(np.nanmean(corr)),
        "signal_rmse": float(np.sqrt(np.mean((pred_sig - ref_sig) ** 2))),
        "theta_rmse_wrapped": float(np.sqrt(np.mean(
            wrap_pi(traj[:, TH_S] - ref_state[:, TH_S]) ** 2))),
        "order_rmse": float(np.sqrt(np.mean((r_pred - r_ref) ** 2))),
        "samples": int(len(times)),
    }


ev0 = max(EVENT_WINDOW[0], float(T[0]))
ev1 = min(EVENT_WINDOW[1], float(T[-1]))
print(f"\n[SWEEP] seizure window {ev0:.1f}..{ev1:.1f}s, "
      f"{len(LAM_GRID)} nudge rates x {len(Q_GRID)} q gains", flush=True)
EVENT_GRID = half_grid(ev0, ev1)
sweep_rows, traces = [], {}
for a in Q_GRID:
    for lam in LAM_GRID:
        times, traj, share = replay(lam, a, EVENT_GRID)
        row = {"lam": lam, "q_gain": a, "assimilation_share": share, **score(times, traj)}
        sweep_rows.append(row)
        traces[(a, lam)] = (times, traj)
        print(f"        lam={lam:5.1f} a={a:.0f}  corr={row['signal_corr_mean']:+.3f}  "
              f"theta_rmse={row['theta_rmse_wrapped']:.3f}  share={share:.3f}", flush=True)

anchor = next(r for r in sweep_rows if r["lam"] == 0.0 and r["q_gain"] == 0.0)
print(f"\n[ANCHOR] model alone (lam=0, a=0): corr={anchor['signal_corr_mean']:+.3f}, "
      f"theta_rmse={anchor['theta_rmse_wrapped']:.3f}", flush=True)


def lam_min_for(a: float):
    ok = [r["lam"] for r in sweep_rows
          if r["q_gain"] == a and r["theta_rmse_wrapped"] <= TRACKING_RMSE]
    return float(min(ok)) if ok else float("nan")


LAM_MIN = {a: lam_min_for(a) for a in Q_GRID}
lam_min_headline = LAM_MIN.get(0.0, float("nan"))
ratio = lam_min_headline / DELTA_OMEGA if DELTA_OMEGA > 0 else float("nan")
print(
    f"[THRESH] lam_min (theta RMSE <= {TRACKING_RMSE:.1f} rad, decorrelation "
    f"ceiling {DECORRELATED_RMSE:.3f}): "
    + ", ".join(f"a={a:g} -> {LAM_MIN[a]:.3g}" for a in Q_GRID),
    flush=True,
)
print(f"[THRESH] dOmega={DELTA_OMEGA:.2f} rad/s, lam_min(a=0)/dOmega = {ratio:.2f} "
      "(the prediction is that this ratio is ~1 and stable across omega modes)",
      flush=True)

qualified = [r for r in sweep_rows if r["signal_corr_mean"] >= TARGET_CORR]
lightest = min(qualified, key=lambda r: r["assimilation_share"]) if qualified else None
if lightest:
    print(f"[LIGHT] lightest reaching corr >= {TARGET_CORR:.2f}: lam={lightest['lam']:g}, "
          f"a={lightest['q_gain']:g} -> corr={lightest['signal_corr_mean']:+.3f} with "
          f"{lightest['assimilation_share']:.1%} of dx/dt supplied externally", flush=True)
else:
    print(f"[LIGHT] no setting reached corr >= {TARGET_CORR:.2f}", flush=True)

show = lightest or max(sweep_rows, key=lambda r: r["signal_corr_mean"])
SHOW_KEY = (show["q_gain"], show["lam"])


# ---------------------------------------------------------------------------
# 9. Seizure-disruption sweep, at a FIXED assimilation setting
# ---------------------------------------------------------------------------
#
# With lam pinned at the minimal tracking value the nudge is actively pulling
# the trajectory onto the measured seizure, so a model parameter that removes
# the event has to overpower the assimilation.  That makes this a different and
# stricter question than the autonomous one: not "can the model avoid seizing?"
# (it never seizes on its own -- the anchor correlation is ~0) but "which model
# parameters can suppress an event the data is actively imposing?"
#
# Two failure modes have to be excluded or the answer is meaningless:
#
#   breaking the model everywhere also removes the seizure.  So every cell must
#   retain PRE-ICTAL fidelity, scored against the baseline cell's own pre-ictal
#   correlation, before it can count as a disruptor.
#
#   a trajectory that saturates, dies or leaves the physiological frequency band
#   is not seizure-free, it is broken.  Viability is calibrated on the RECORDING
#   -- the measured trajectory is pushed through the same test -- because a band
#   calibrated on the pre-ictal baseline would reject the real seizure and then
#   silently classify every genuinely seizing cell as non-viable.
#
DISRUPT = bool(env_int("KANDY_AK_DISRUPT", 1)) and OMEGA_IS_STATE
if DISRUPT:
    DIS_LAM = env_float("KANDY_AK_DISRUPT_LAM", float(lam_min_headline)
                        if np.isfinite(lam_min_headline) else 20.0)
    DIS_Q = env_float("KANDY_AK_DISRUPT_Q", 0.0)
    N_G = env_int("KANDY_AK_N_G", 9)
    N_DU = env_int("KANDY_AK_N_DU", 9)
    N_1D = env_int("KANDY_AK_N_1D", 9)
    PRE_W, EV_W, POST_W = (-6.0, -1.0), (-1.0, 4.0), (4.0, 8.0)

    G_GRID = np.linspace(0.0, 3.0, N_G)
    DU_GRID = np.linspace(-2.0, 2.0, N_DU) * float(np.mean(U_R))
    P_GRID = np.linspace(0.0, 3.0, N_1D)
    TG_GRID = np.logspace(-1.0, 1.0, N_1D)

    def f_model_p(state, g, du, pl, tg):
        """Parameterized RHS.  g=1, du=0, pl=1, tg=1 reproduces the fit."""
        om = squash(state[:, U_S], OM_LO, OM_HI)
        dth = om + g[:, None] * (model_phase.predict(state) * th_scale)
        inner = model_u.predict(state) * u_scale
        tgt = U_C + U_R * np.tanh((inner + du[:, None]) / U_R)
        dudt = tg[:, None] * (tgt - state[:, U_S]) / TAU_U
        dkp = C_KAPPA + pl[:, None] * (model_kappa.predict(state) * kp_scale)
        return np.column_stack([dth, dudt, dkp])

    # Nudge the OBSERVABLES only.  omega/u is part of the state vector, so a
    # full-state nudge drags omega straight onto the measured excursion: the
    # seizure is then injected by the assimilation and no model parameter can
    # possibly disrupt it (measured: baseline rise 55.9 vs data 56.6).  With u
    # unnudged, theta and kappa are pinned to the recording and u is INFERRED
    # from them through the u-lift, which depends on theta only via R and on
    # kappa via <kappa>.  Any frequency excursion is then produced by the model
    # from observed input, which is the only version of the question that has
    # an answer.
    NUDGE_MASK = np.ones(N_STATE)
    if not env_int("KANDY_AK_DISRUPT_NUDGE_U", 0):
        NUDGE_MASK[U_S] = 0.0

    def replay_batch(g, du, pl, tg, t0, t1, lam, a):
        steps = int(round((t1 - t0) / DT))
        half_t = t0 + np.arange(2 * steps + 1) * (0.5 * DT)
        ref = np.column_stack([np.interp(half_t, T, STATE[:, c])
                               for c in range(N_STATE)])
        qq = np.column_stack([np.interp(half_t, T, Q_DOT[:, c])
                              for c in range(N_STATE)])
        n = len(g)
        x = np.tile(ref[0], (n, 1))
        traj = np.empty((steps + 1, n, N_STATE))
        traj[0] = x

        def rhs(state, h):
            return (f_model_p(state, g, du, pl, tg)
                    + a * qq[h] + lam * NUDGE_MASK * (ref[h] - state))

        for s in range(steps):
            h = 2 * s
            k1 = rhs(x, h)
            k2 = rhs(x + 0.5 * DT * k1, h + 1)
            k3 = rhs(x + 0.5 * DT * k2, h + 1)
            k4 = rhs(x + DT * k3, h + 2)
            x = x + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if not np.isfinite(x).all():
                x = np.nan_to_num(x, nan=np.inf, posinf=np.inf, neginf=-np.inf)
            traj[s + 1] = x
        return half_t[::2], traj

    dis_g = np.concatenate([np.repeat(G_GRID, N_DU), np.ones(N_1D), np.ones(N_1D)])
    dis_du = np.concatenate([np.tile(DU_GRID, N_G), np.zeros(N_1D), np.zeros(N_1D)])
    dis_pl = np.concatenate([np.ones(N_G * N_DU), P_GRID, np.ones(N_1D)])
    dis_tg = np.concatenate([np.ones(N_G * N_DU), np.ones(N_1D), TG_GRID])
    n_plane = N_G * N_DU

    _nudged = ("ALL states" if NUDGE_MASK[U_S].max() > 0
               else "theta+kappa only (omega inferred by the model)")
    print(f"\n[DISRUPT] assimilation lam={DIS_LAM:g}, a={DIS_Q:g} on {_nudged}; "
          f"{N_G} coupling gains x {N_DU} omega offsets "
          f"(+ {N_1D} plasticity, + {N_1D} tau_u gains) = {len(dis_g)} replays "
          f"over {PRE_W[0]:.0f}..{POST_W[1]:.0f}s", flush=True)
    dis_t, dis_traj = replay_batch(dis_g, dis_du, dis_pl, dis_tg,
                                   PRE_W[0], POST_W[1], DIS_LAM, DIS_Q)

    pre_m = (dis_t >= PRE_W[0]) & (dis_t <= PRE_W[1])
    ev_m = (dis_t >= EV_W[0]) & (dis_t <= EV_W[1])
    ref_sig_dis = np.column_stack([np.interp(dis_t, T, SIGNAL[:, c]) for c in range(M)])
    om_traj = squash(dis_traj[:, :, U_S], OM_LO, OM_HI).mean(axis=2)   # (steps, cells)
    kp_traj = dis_traj[:, :, KP_S].mean(axis=2)
    sech_traj = (1 - np.tanh(dis_traj[:, :, U_S]) ** 2).min(axis=2)

    om_data = np.column_stack([np.interp(dis_t, T, OMEGA_SLOW_FULL[:KEEP][:, c])
                               for c in range(M)]).mean(axis=1)
    kp_data = np.column_stack([np.interp(dis_t, T, KAPPA[:, c])
                               for c in range(P)]).mean(axis=1)
    DATA_RISE = float(om_data[ev_m].max() - om_data[pre_m].mean())

    OM_BAND = (0.75 * float(np.min(OMEGA_SLOW_FULL.mean(axis=1))),
               1.25 * float(np.max(OMEGA_SLOW_FULL.mean(axis=1))))
    AMP_FLOOR = float(np.percentile(KAPPA.mean(axis=1), 5))
    data_viable = bool(OM_BAND[0] <= om_data.mean() <= OM_BAND[1]
                       and kp_data.mean() >= AMP_FLOOR)
    print(f"[DISRUPT] data rise={DATA_RISE:.1f} rad/s; viability band "
          f"{OM_BAND[0]:.0f}..{OM_BAND[1]:.0f} rad/s; measured trajectory "
          f"viable={data_viable}"
          + ("" if data_viable else "   [CALIBRATION ERROR]"), flush=True)

    def corr_window(cell, mask):
        sig = state_to_signal(dis_traj[mask][:, cell, :])
        vals = [np.corrcoef(sig[:, c], ref_sig_dis[mask][:, c])[0, 1]
                if np.std(sig[:, c]) > 1e-12 else np.nan for c in range(M)]
        return float(np.nanmean(vals))

    rise = om_traj[ev_m].max(axis=0) - om_traj[pre_m].mean(axis=0)
    om_mean = om_traj.mean(axis=0)
    kp_mean = kp_traj.mean(axis=0)
    finite = np.isfinite(dis_traj).all(axis=(0, 2))
    viable = (finite & (sech_traj.min(axis=0) > 1e-3)
              & (om_mean >= OM_BAND[0]) & (om_mean <= OM_BAND[1])
              & (kp_mean >= AMP_FLOOR))
    pre_corr = np.array([corr_window(c, pre_m) if viable[c] else np.nan
                         for c in range(len(dis_g))])

    base_cell = int(np.argmin((dis_g - 1.0) ** 2 + (dis_du / max(np.mean(U_R), 1e-9)) ** 2
                              + (dis_pl - 1.0) ** 2 + (dis_tg - 1.0) ** 2))
    BASE_PRE = float(pre_corr[base_cell])
    BASE_RISE = float(rise[base_cell])
    print(f"[DISRUPT] baseline cell (g=1, du=0, p=1, tau gain=1): "
          f"rise={BASE_RISE:.1f} rad/s, pre-ictal corr={BASE_PRE:+.3f}", flush=True)

    SUPPRESS_FRAC = env_float("KANDY_AK_SUPPRESS_FRAC", 0.5)
    FIDELITY_FRAC = env_float("KANDY_AK_FIDELITY_FRAC", 0.8)
    # Suppression is measured against the BASELINE REPLAY, not against the
    # data: the question is which parameters remove the event the fitted model
    # actually produces.  Whether that baseline event resembles the real one is
    # a separate question, gated explicitly below -- if the baseline replay does
    # not reproduce the excursion there is nothing to disrupt and the sweep is
    # vacuous, which must be reported rather than hidden behind a high
    # disruptor count.
    suppressed = rise < SUPPRESS_FRAC * BASE_RISE
    baseline_reproduces = bool(BASE_RISE >= 0.5 * DATA_RISE)
    print(f"[DISRUPT] baseline replay recovers {BASE_RISE / max(DATA_RISE, 1e-9):.0%} "
          f"of the measured excursion -> the disruption question is "
          f"{'well posed' if baseline_reproduces else 'VACUOUS (nothing to disrupt)'}",
          flush=True)
    keeps_fidelity = pre_corr >= FIDELITY_FRAC * BASE_PRE
    disruptor = viable & suppressed & keeps_fidelity

    rise_map = rise[:n_plane].reshape(N_G, N_DU)
    pre_map = pre_corr[:n_plane].reshape(N_G, N_DU)
    dis_map = disruptor[:n_plane].reshape(N_G, N_DU)
    viable_map = viable[:n_plane].reshape(N_G, N_DU)
    print(
        f"[DISRUPT] plane: viable {int(viable_map.sum())}/{viable_map.size}, "
        f"suppressed {int((suppressed[:n_plane]).sum())}/{n_plane}, "
        f"DISRUPTORS (suppressed + fidelity + viable) "
        f"{int(dis_map.sum())}/{dis_map.size}", flush=True)
    for name, sl_ in (("plasticity p", slice(n_plane, n_plane + N_1D)),
                      ("tau_u gain", slice(n_plane + N_1D, None))):
        print(f"          {name}: disruptors {int(disruptor[sl_].sum())}/{N_1D}, "
              f"rise range {rise[sl_].min():.1f}..{rise[sl_].max():.1f} rad/s",
              flush=True)

    dis_idx = np.where(disruptor)[0]
    if len(dis_idx):
        cost = ((dis_g[dis_idx] - 1.0) / 3.0) ** 2 + \
               (dis_du[dis_idx] / max(DU_GRID.max() - DU_GRID.min(), 1e-9)) ** 2 + \
               ((dis_pl[dis_idx] - 1.0) / 3.0) ** 2 + \
               (np.log10(dis_tg[dis_idx]) / 2.0) ** 2
        best = int(dis_idx[int(np.argmin(cost))])
        print(f"[DISRUPT] smallest disruptor: g={dis_g[best]:.2f}, "
              f"delta_u={dis_du[best]:+.3f}, p={dis_pl[best]:.2f}, "
              f"tau_gain={dis_tg[best]:.2f} -> rise={rise[best]:.1f} rad/s "
              f"(data {DATA_RISE:.1f}), pre-ictal corr={pre_corr[best]:+.3f} "
              f"(baseline {BASE_PRE:+.3f})", flush=True)
    else:
        best = base_cell
        print("[DISRUPT] no cell suppressed the event while keeping fidelity",
              flush=True)


# ---------------------------------------------------------------------------
# 10. Gates, metrics, figures
# ---------------------------------------------------------------------------
shares0 = [next(r["assimilation_share"] for r in sweep_rows
                if r["lam"] == lam and r["q_gain"] == 0.0) for lam in LAM_GRID] \
    if 0.0 in Q_GRID else []
monotone = bool(not shares0 or all(np.diff(shares0) >= -1e-6))
gates = {
    "coupling_beats_omega_only_offblock": {
        "required": 0.01, "achieved": phase_gain, "passed": bool(phase_gain >= 0.01)},
    "autonomous_anchor_measured": {
        "required": "lam=0,a=0 finite", "achieved": anchor["signal_corr_mean"],
        "passed": bool(np.isfinite(anchor["signal_corr_mean"]))},
    "assimilation_share_increases_with_lam_at_a0": {
        "required": "monotone", "achieved": monotone, "passed": monotone},
    "nudge_threshold_located": {
        "required": "some lam clears the decorrelation ceiling",
        "achieved": lam_min_headline, "passed": bool(np.isfinite(lam_min_headline))},
}
passed = print_check(
    all(g["passed"] for g in gates.values()),
    f"omega-as-{OMEGA_MODE} model fitted; nudge threshold and dOmega both measured",
)
for name, g in gates.items():
    print(f"        {'PASS' if g['passed'] else 'FAIL'}  {name}: {g['achieved']}", flush=True)

metrics = {
    "task": f"corrected adaptive Kuramoto (omega={OMEGA_MODE}) with swept assimilation",
    "omega_mode": OMEGA_MODE,
    "channels": LABELS,
    "state_dim": N_STATE,
    "split": {"train": int(len(train_rows)), "val": int(len(val_rows)),
              "test": int(len(test_rows)), "seizure_in_train": event_in_train,
              "guard_seconds": float(guard * DT)},
    "omega_rad_s": [float(v) for v in OMEGA_REPORT],
    "omega_hz": [float(v / (2 * np.pi)) for v in OMEGA_REPORT],
    "tau_u_seconds": TAU_U,
    "coupling_magnitude": {"rms_coupling_rad_s": coupling_rms,
                           "rms_omega_rad_s": omega_rms,
                           "ratio": coupling_rms / max(omega_rms, 1e-12)},
    "delta_omega_event_rad_s": DELTA_OMEGA,
    "delta_omega_all_rad_s": DELTA_OMEGA_ALL,
    "phase_gain_over_omega_only": phase_gain,
    "q_static_share": q_share_static,
    "assimilation_sweep": sweep_rows,
    "autonomous_anchor": anchor,
    "lam_min": {str(a): LAM_MIN[a] for a in Q_GRID},
    "lam_min_over_delta_omega": ratio,
    "tracking_rmse_criterion": TRACKING_RMSE,
    "decorrelation_ceiling": DECORRELATED_RMSE,
    "lightest_reaching_target": lightest,
    "gates": gates,
    "passed": passed,
}
if DISRUPT:
    metrics["disruption_sweep"] = {
        "assimilation": {"lam": DIS_LAM, "q_gain": DIS_Q,
                         "nudged_states": "theta+kappa" if NUDGE_MASK[U_S].max() == 0
                         else "all"},
        "windows": {"pre": list(PRE_W), "event": list(EV_W), "post": list(POST_W)},
        "data_rise_rad_s": DATA_RISE,
        "baseline": {"rise": BASE_RISE, "pre_corr": BASE_PRE,
                     "fraction_of_measured_rise": BASE_RISE / max(DATA_RISE, 1e-9),
                     "reproduces_the_event": baseline_reproduces},
        "criteria": {"suppress_frac": SUPPRESS_FRAC, "fidelity_frac": FIDELITY_FRAC,
                     "omega_band": list(OM_BAND), "amp_floor": AMP_FLOOR,
                     "measured_trajectory_viable": data_viable},
        "g_grid": [float(v) for v in G_GRID],
        "delta_u_grid": [float(v) for v in DU_GRID],
        "plasticity_grid": [float(v) for v in P_GRID],
        "tau_gain_grid": [float(v) for v in TG_GRID],
        "rise_map": rise_map, "pre_corr_map": pre_map,
        "viable_map": viable_map.astype(int), "disruptor_map": dis_map.astype(int),
        "n_disruptors_plane": int(dis_map.sum()),
        "n_disruptors_total": int(disruptor.sum()),
        "smallest_disruptor": {"g": float(dis_g[best]), "delta_u": float(dis_du[best]),
                               "plasticity": float(dis_pl[best]),
                               "tau_gain": float(dis_tg[best]),
                               "rise": float(rise[best]),
                               "pre_corr": float(pre_corr[best]),
                               "found": bool(len(dis_idx))},
    }
    gates["baseline_replay_reproduces_the_event"] = {
        "required": "baseline rise >= 50% of the measured rise",
        "achieved": [BASE_RISE, DATA_RISE], "passed": baseline_reproduces}
    gates["disruptor_exists"] = {
        "required": ">=1 viable cell suppressing the event while keeping fidelity",
        "achieved": int(disruptor.sum()), "passed": bool(disruptor.sum() > 0)}
    gates["viability_admits_the_real_seizure"] = {
        "required": "measured trajectory viable", "achieved": data_viable,
        "passed": bool(data_viable)}
    metrics["gates"] = gates
    metrics["passed"] = bool(all(g["passed"] for g in gates.values()))
save_metrics(RESULT_DIR / "metrics.json", metrics)

times_show, traj_show = traces[SHOW_KEY]
ref_sig_show = np.column_stack([np.interp(times_show, T, SIGNAL[:, c]) for c in range(M)])
np.savez_compressed(
    RESULT_DIR / "assimilation_arrays.npz",
    t=T, labels=np.asarray(LABELS), signal=SIGNAL, state=STATE, state_dot=STATE_DOT,
    q_dot=Q_DOT, omega=OMEGA_REPORT, lam_grid=np.asarray(LAM_GRID),
    q_grid=np.asarray(Q_GRID),
    corr=np.array([r["signal_corr_mean"] for r in sweep_rows]),
    share=np.array([r["assimilation_share"] for r in sweep_rows]),
    theta_rmse=np.array([r["theta_rmse_wrapped"] for r in sweep_rows]),
    show_time=times_show, show_pred=state_to_signal(traj_show), show_ref=ref_sig_show,
)

use_pub_style()


def plot_sweep() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for a, marker in zip(Q_GRID, ("o", "s")):
        sel = [r for r in sweep_rows if r["q_gain"] == a]
        lam = [r["lam"] for r in sel]
        axes[0].plot(lam, [r["signal_corr_mean"] for r in sel], marker=marker,
                     label=f"a={a:g}")
        axes[1].plot(lam, [r["theta_rmse_wrapped"] for r in sel], marker=marker,
                     label=f"a={a:g}")
        axes[2].plot([r["assimilation_share"] for r in sel],
                     [r["signal_corr_mean"] for r in sel], marker=marker, label=f"a={a:g}")
    axes[0].axhline(anchor["signal_corr_mean"], color="0.4", ls=":", lw=1.0,
                    label="model alone")
    axes[0].set_ylabel("seizure-window signal corr")
    axes[1].axhline(DECORRELATED_RMSE, color="#d62728", ls="--", lw=1.0,
                    label="decorrelated ceiling")
    axes[1].axhline(TRACKING_RMSE, color="#2ca02c", ls=":", lw=1.0, label="tracking")
    if np.isfinite(lam_min_headline):
        axes[1].axvline(lam_min_headline, color="#2ca02c", lw=1.0, alpha=0.5)
    axes[1].set_ylabel(r"$\theta$ RMSE (rad)")
    axes[2].set_xlabel("assimilation share")
    axes[2].set_ylabel("signal corr")
    for ax in axes[:2]:
        ax.set_xlabel(r"nudge rate $\lambda$ (s$^{-1}$)")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle(rf"$\omega$ = {OMEGA_MODE}:  $\Delta\omega$={DELTA_OMEGA:.1f} rad/s, "
                 rf"$\lambda_{{\min}}$={lam_min_headline:.3g}, "
                 rf"ratio={ratio:.2f}")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"assimilation_sweep.{e}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_replay() -> None:
    pred_sig = state_to_signal(traj_show)
    fig, axes = plt.subplots(M + 1, 1, figsize=(11, 1.35 * (M + 1)), sharex=True)
    for c, ax in enumerate(axes[:M]):
        ax.plot(times_show, ref_sig_show[:, c], lw=0.65, color="#1f77b4")
        ax.plot(times_show, pred_sig[:, c], lw=0.8, ls="--", color="#d62728")
        ax.axvline(0.0, color="0.2", lw=0.8, ls=":")
        ax.set_ylabel(LABELS[c], rotation=0, ha="right", va="center", fontsize=8)
    ref_state = np.column_stack([np.interp(times_show, T, STATE[:, c])
                                 for c in range(N_STATE)])
    axes[-1].plot(times_show, np.abs(np.mean(np.exp(1j * ref_state[:, TH_S]), axis=1)),
                  lw=1.0, color="#1f77b4", label="data")
    axes[-1].plot(times_show, np.abs(np.mean(np.exp(1j * traj_show[:, TH_S]), axis=1)),
                  lw=1.0, ls="--", color="#d62728", label="model + assimilation")
    axes[-1].set_ylabel("order", rotation=0, ha="right", va="center", fontsize=8)
    axes[-1].set_ylim(0, 1.05)
    axes[-1].set_xlabel("recording time (s)")
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"omega={OMEGA_MODE}: replay at $\\lambda$={show['lam']:g}, "
                 f"a={show['q_gain']:g}  |  corr={show['signal_corr_mean']:.3f}, "
                 f"{show['assimilation_share']:.1%} of $dx/dt$ external")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"seizure_replay.{e}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_losses() -> None:
    mdls = [(model_phase, "loss_phase"), (model_kappa, "loss_kappa")]
    if OMEGA_IS_STATE:
        mdls.append((model_u, "loss_omega"))
    for mdl, tag in mdls:
        if getattr(mdl, "train_results_", None):
            fig, _ = plot_loss_curves(mdl.train_results_, save=str(RESULT_DIR / tag))
            plt.close(fig)


def plot_disruption() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))
    ext = [DU_GRID[0], DU_GRID[-1], G_GRID[0], G_GRID[-1]]
    im = axes[0].imshow(rise_map, origin="lower", aspect="auto", extent=ext, cmap="magma")
    axes[0].contour(DU_GRID, G_GRID, rise_map,
                    levels=[SUPPRESS_FRAC * BASE_RISE], colors="w", linewidths=1.6)
    axes[0].set_title(rf"event rise in $\langle\omega\rangle$ (data {DATA_RISE:.0f})")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(pre_map, origin="lower", aspect="auto", extent=ext,
                         cmap="viridis")
    axes[1].set_title("pre-ictal fidelity (corr)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    klass = np.where(~viable_map, 0, np.where(dis_map, 2, 1))
    axes[2].imshow(klass, origin="lower", aspect="auto", extent=ext, vmin=0, vmax=2,
                   cmap=matplotlib.colors.ListedColormap(["0.75", "#d62728", "#2ca02c"]))
    axes[2].set_title("non-viable / seizes / DISRUPTED")
    for ax in axes[:3]:
        ax.scatter([0.0], [1.0], marker="*", s=170, c="w", edgecolors="k", zorder=5)
        ax.set_xlabel(r"$\omega$ offset $\delta_u$")
        ax.set_ylabel("coupling gain $g$")
    axes[3].plot(dis_t, om_data, color="#1f77b4", lw=1.4, label="data")
    axes[3].plot(dis_t, om_traj[:, base_cell], color="#d62728", lw=1.3, ls="--",
                 label="baseline replay")
    axes[3].plot(dis_t, om_traj[:, best], color="#2ca02c", lw=1.5,
                 label="smallest disruptor")
    axes[3].axvspan(*EV_W, color="#f4a261", alpha=0.2, lw=0)
    axes[3].set_xlabel("recording time (s)")
    axes[3].set_ylabel(r"$\langle\omega\rangle$ (rad/s)")
    axes[3].set_title("the event, and its suppression")
    axes[3].legend(fontsize=7)
    fig.suptitle(rf"Disrupting the seizure at fixed assimilation "
                 rf"($\lambda$={DIS_LAM:g}, a={DIS_Q:g})")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"seizure_disruption.{e}", dpi=250, bbox_inches="tight")
    plt.close(fig)


plot_sweep()
plot_replay()
if DISRUPT:
    plot_disruption()
plot_losses()
print(f"\n[FIGS]  Saved {RESULT_DIR}", flush=True)
