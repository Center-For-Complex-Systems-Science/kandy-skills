#!/usr/bin/env python3
"""Simulated DBS on an adaptive-Kuramoto seizure model driven by PRE-ICTAL q(t).

Why this script exists
----------------------
``epilepsy_adaptive_kuramoto_fit.py`` reproduces the seizure by integrating

    dx/dt = f_KANDy(x) + q_data(t) + 20 * (x_data(t) - x)

where ``q_data`` is the model's OWN residual taken from the whole recording.
The seizure is therefore supplied to the model rather than produced by it, and
``epilepsy_adaptive_kuramoto_seizure_freedom.py`` measured exactly that: with
q=0 and lambda=0 the replay correlation is 0.005, and 0 of 14 pre-ictal initial
conditions produce an event.

That makes a stimulation experiment on the original RHS uninterpretable.  If
q(t) still carries the seizure through the ictal window, then "DBS terminated
the seizure" only measures whether the stimulus amplitude exceeded a forcing
term we wrote down ourselves.  Two changes fix the experiment:

1. **q(t) is estimated on the pre-ictal segment only** and frozen at its
   pre-ictal mean from onset onward, so no ictal information reaches the RHS.
   Frozen q is a constant vector, which acts as a fixed excitability offset on
   each block -- that is stated in the output rather than hidden.
2. **lambda = 0 everywhere.**  A nudge onto measured data drags the trajectory
   through the seizure whatever the stimulator does, which would make every
   termination result vacuous.

Model
-----
State eta = [theta (M), u (M), v (P), z (1)], the bounded parameterization from
``epilepsy_adaptive_kuramoto_seizure_freedom.py``: omega and kappa live in tanh
boxes via inner coordinates u, v, and every slow block is an Epileptor-style
contraction toward a squashed target, so no rollout can run away or be clipped.

    dtheta_i/dt = omega_i(u) + s(t) * Phi_i(eta) + q_theta + DBS_theta
    du_i/dt     = (U_tgt(eta) - u_i) / tau_u + q_u
    dv_p/dt     = (V_tgt(eta) - v_p) / tau_v + q_v
    dz/dt       = (H(eta)     - z)   / tau_z + q_z + DBS_z

Stimulation
-----------
Four mechanisms, each with its own amplitude axis, swept one at a time:

    M1 hf_entrain      dtheta_i/dt += -A * w(t) * sin(theta_i - 2 pi f_stim t)
    M2 couple_suppress s(t) = 1 - a * w(t)          (functional disconnection)
    M3 pulse_reset     theta_i <- theta_i - g * sin(theta_i - theta_ref)
                                                    at 1/f_pulse intervals
    M4 slow_drive      dz/dt += -A_z * w(t)

applied to TARGET channels only (default: the seizure focus).  w(t) in [0, 1]
is the protocol gate:

    open    w = 1 on [STIM_ON, STIM_OFF]
    closed  w latches on when <omega> rises past d_hi, off when it falls
            below d_lo  (RNS-style responsive stimulation)
    sham    the open-loop window shifted into the pre-ictal segment -- the
            specificity control.  If sham "terminates" too, the effect is not
            about stimulating during the event.

What is gated
-------------
The decisive gate is ``baseline_seizes``: the unstimulated model, driven by
pre-ictal q alone, must produce an event.  If it does not, there is nothing to
terminate and every efficacy number below is reported as UNMEASURABLE rather
than as a result.  This is the honest failure mode for this recording and the
script is built to report it clearly rather than to avoid it.

Run:
    cd skills/kandy/examples && python epilepsy/epilepsy_dbs_termination.py
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


# ---------------------------------------------------------------------------
# 0. Parameters
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = REPO_ROOT / "data" / "Dory_2021_04_12_006_preprocessed.pkl"
RESULT_DIR = ensure_outdir(REPO_ROOT / "results" / "Epilepsy-DBS-Termination")

BANDPASS_HZ = (1.0, 40.0)
FILTER_ORDER = 4
EDGE_TRIM_SECONDS = 2.0
PHASE_FAST_SECONDS = 0.155
OMEGA_SLOW_SECONDS = 1.0
KAPPA_SECONDS = 0.505
Z_SLOW_SECONDS = 4.0

EVENT_WINDOW = (-1.0, 5.0)
PRE_ICTAL = (-12.0, -2.0)
POST_ICTAL = (12.0, 26.0)
VAL_WINDOW = (8.0, 14.0)
HELDOUT_WINDOW = (20.0, 28.0)
GUARD_SECONDS = env_float("KANDY_DBS_GUARD_SECONDS", Z_SLOW_SECONDS)

LAG_SECONDS = env_float("KANDY_DBS_LAG_SECONDS", 0.25)
Z_LAG_SECONDS = env_float("KANDY_DBS_Z_LAG_SECONDS", 2.0)
BOX_MARGIN = env_float("KANDY_DBS_BOX_MARGIN", 0.35)
TAU_U_GRID = np.logspace(np.log10(0.05), np.log10(10.0), 24)
TAU_V_GRID = np.logspace(np.log10(0.05), np.log10(10.0), 24)
TAU_Z_GRID = np.logspace(np.log10(0.25), np.log10(40.0), 28)
LAMB_GRID = [float(v) for v in os.environ.get(
    "KANDY_DBS_LAMB_GRID", "0.0,0.001,0.01,0.1").split(",")]

GRID = env_int("KANDY_DBS_GRID", 5)
THETA_STEPS = env_int("KANDY_DBS_THETA_STEPS", 60)
U_STEPS = env_int("KANDY_DBS_U_STEPS", 80)
V_STEPS = env_int("KANDY_DBS_V_STEPS", 60)
H_STEPS = env_int("KANDY_DBS_H_STEPS", 80)
N_TRAIN_MAX = env_int("KANDY_DBS_N_TRAIN_MAX", 4000)

# --- the q(t) treatment this script exists to test -------------------------
Q_FREEZE_TIME = env_float("KANDY_DBS_Q_FREEZE_TIME", 0.0)   # seizure onset
Q_SMOOTH_SECONDS = env_float("KANDY_DBS_Q_SMOOTH_SECONDS", 0.055)
Q_GAIN = env_float("KANDY_DBS_Q_GAIN", 1.0)

# --- rollout ---------------------------------------------------------------
ROLLOUT_START = env_float("KANDY_DBS_ROLLOUT_START", -3.0)
ROLLOUT_SECONDS = env_float("KANDY_DBS_ROLLOUT_SECONDS", 12.0)

# --- stimulation -----------------------------------------------------------
TARGETS = os.environ.get("KANDY_DBS_TARGETS", "hippocampus_R").strip()
#
# Clinical DBS runs at 130-180 Hz.  This recording is sampled at 200 Hz, so a
# 130 Hz carrier is 1.5 samples per cycle: an RK4 step of 5 ms cannot represent
# it, and integrating it anyway produces an aliased drive whose apparent
# frequency is an artefact of the step size, not a model of stimulation.  The
# defaults below are the highest frequencies this sampling rate supports, which
# are still well above the 5-7 Hz natural rhythm and so still act as
# supra-natural entrainment.  Raising them past the guard in section 8 is
# allowed but is recorded as a failed gate.
#
F_STIM = env_float("KANDY_DBS_F_STIM", 20.0)
F_PULSE = env_float("KANDY_DBS_F_PULSE", 40.0)
THETA_REF = env_float("KANDY_DBS_THETA_REF", 0.0)
STIM_ON = env_float("KANDY_DBS_STIM_ON", 0.5)        # 0.5 s after onset
STIM_OFF = env_float("KANDY_DBS_STIM_OFF", 6.0)
SHAM_SHIFT = env_float("KANDY_DBS_SHAM_SHIFT", -6.0)  # window moved pre-ictal
DETECT_HI = env_float("KANDY_DBS_DETECT_HI", 0.35)   # fraction of observed rise
DETECT_LO = env_float("KANDY_DBS_DETECT_LO", 0.15)
N_AMP = env_int("KANDY_DBS_N_AMP", 9)
BURDEN_CUT = env_float("KANDY_DBS_BURDEN_CUT", 0.5)  # >=50% burden reduction

MECHANISMS = ["hf_entrain", "couple_suppress", "pulse_reset", "slow_drive"]
PROTOCOLS = ["open", "closed", "sham"]


# ---------------------------------------------------------------------------
# 1. Data -> theta, omega, kappa, z
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
LABELS = list(payload["labels"])
M = len(LABELS)
PAIRS = [(a, b) for a in range(M) for b in range(a + 1, M)]
P = len(PAIRS)

target_names = [c.strip() for c in TARGETS.split(",") if c.strip()]
missing = [n for n in target_names if n not in LABELS]
if missing:
    raise ValueError(f"Unknown stimulation target {missing}; recording has {LABELS}")
TARGET_IDX = [LABELS.index(n) for n in target_names]
TARGET_MASK = np.zeros(M)
TARGET_MASK[TARGET_IDX] = 1.0

nyq = 0.5 * FS
bb, aa = butter(FILTER_ORDER, [BANDPASS_HZ[0] / nyq, BANDPASS_HZ[1] / nyq], btype="band")
X_FILT = filtfilt(bb, aa, X_ALL, axis=0)
analytic = hilbert(X_FILT, axis=0)
theta_u = np.unwrap(np.angle(analytic), axis=0)
log_amp_raw = np.log(np.abs(analytic) + 1e-6)
amp_mean = log_amp_raw.mean(axis=0)
amp_std = log_amp_raw.std(axis=0) + 1e-12
log_amp = (log_amp_raw - amp_mean) / amp_std

w_fast = odd_window(PHASE_FAST_SECONDS, FS)
w_om = odd_window(OMEGA_SLOW_SECONDS, FS)
w_kp = odd_window(KAPPA_SECONDS, FS)
w_z = odd_window(Z_SLOW_SECONDS, FS)

theta_s = savgol_filter(theta_u, w_fast, 3, axis=0, mode="interp")
dtheta_fast = savgol_filter(theta_u, w_fast, 3, deriv=1, delta=DT, axis=0, mode="interp")
omega_slow = savgol_filter(theta_u, w_om, 3, deriv=1, delta=DT, axis=0, mode="interp")

pair_log_amp = np.column_stack([0.5 * (log_amp[:, a] + log_amp[:, b]) for a, b in PAIRS])
kappa_raw = savgol_filter(pair_log_amp, w_kp, 3, axis=0, mode="interp")
kappa_mean = kappa_raw.mean(axis=0)
kappa_std = kappa_raw.std(axis=0) + 1e-12
kappa_z = (kappa_raw - kappa_mean) / kappa_std

z_raw = savgol_filter(log_amp.mean(axis=1), w_z, 3, mode="interp")

edge = int(round(EDGE_TRIM_SECONDS * FS))
sl = slice(edge, len(X_ALL) - edge)
T = T_ALL[sl]
THETA = theta_s[sl]
DTHETA = dtheta_fast[sl]
OMEGA = omega_slow[sl]
KAPPA = kappa_z[sl]
SIGNAL = X_FILT[sl]
Z = z_raw[sl]
N_ROW = len(T)
LAG = int(round(LAG_SECONDS / DT))
Z_LAG = int(round(Z_LAG_SECONDS / DT))
MAX_LAG = max(LAG, Z_LAG)
DZ = (Z[Z_LAG:] - Z[:-Z_LAG]) / Z_LAG_SECONDS

print("[DATA]  Loaded epilepsy recording", flush=True)
print(f"[SEL]   {M} channels {LABELS}; {P} pairs", flush=True)
print(f"[STIM]  targets = {target_names} (indices {TARGET_IDX})", flush=True)
print(
    f"[PRE]   {N_ROW} rows {T[0]:.1f}..{T[-1]:.1f}s; windows theta={w_fast}, "
    f"omega={w_om}, kappa={w_kp}, z={w_z}",
    flush=True,
)


def block_mean(arr, lo, hi):
    m = (T >= lo) & (T <= hi)
    return float(np.mean(arr[m])) if m.any() else float("nan")


print(
    "[EVENT] mean omega: "
    f"pre={block_mean(OMEGA.mean(1), *PRE_ICTAL):.1f}, "
    f"ictal={block_mean(OMEGA.mean(1), *EVENT_WINDOW):.1f}, "
    f"post={block_mean(OMEGA.mean(1), *POST_ICTAL):.1f} rad/s",
    flush=True,
)


# ---------------------------------------------------------------------------
# 2. Bounded coordinates
# ---------------------------------------------------------------------------
def make_box(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = arr.min(axis=0), arr.max(axis=0)
    span = hi - lo
    return lo - BOX_MARGIN * span, hi + BOX_MARGIN * span


OM_LO, OM_HI = make_box(OMEGA)
KP_LO, KP_HI = make_box(KAPPA)
Z_LO, Z_HI = make_box(Z[:, None])
Z_LO, Z_HI = float(Z_LO[0]), float(Z_HI[0])
Z_C, Z_R = 0.5 * (Z_LO + Z_HI), 0.5 * (Z_HI - Z_LO)


def squash(x, lo, hi):
    return lo + 0.5 * (hi - lo) * (1.0 + np.tanh(x))


def squash_t(x, lo, hi):
    lo_t = torch.as_tensor(lo, dtype=x.dtype)
    hi_t = torch.as_tensor(hi, dtype=x.dtype)
    return lo_t + 0.5 * (hi_t - lo_t) * (1.0 + torch.tanh(x))


def unsquash(y, lo, hi):
    frac = 2.0 * (y - lo) / (hi - lo) - 1.0
    return np.arctanh(np.clip(frac, -0.999, 0.999))


U = unsquash(OMEGA, OM_LO, OM_HI)
V = unsquash(KAPPA, KP_LO, KP_HI)
U_LO, U_HI = make_box(U)
V_LO, V_HI = make_box(V)
U_C, U_R = 0.5 * (U_LO + U_HI), 0.5 * (U_HI - U_LO)
V_C, V_R = 0.5 * (V_LO + V_HI), 0.5 * (V_HI - V_LO)


def to_inner(target, c, r):
    return r * np.arctanh(np.clip((target - c) / r, -0.999, 0.999))


def from_inner(inner, c, r, delta=0.0):
    return c + r * np.tanh((inner + delta) / r)


TH_S = slice(0, M)
U_S = slice(M, 2 * M)
V_S = slice(2 * M, 2 * M + P)
Z_S = 2 * M + P
N_ETA = 2 * M + P + 1
ETA = np.column_stack([THETA, U, V, Z[:, None]])


# ---------------------------------------------------------------------------
# 3. Split: three time blocks, guard-banded
# ---------------------------------------------------------------------------
rows = np.arange(N_ROW - MAX_LAG)
t_rows = T[: N_ROW - MAX_LAG]


def block(window):
    inside = (t_rows >= window[0]) & (t_rows <= window[1])
    near = (t_rows >= window[0] - GUARD_SECONDS) & (t_rows <= window[1] + GUARD_SECONDS)
    return rows[inside], near


val_rows, near_val = block(VAL_WINDOW)
test_rows, near_test = block(HELDOUT_WINDOW)
train_rows = rows[~(near_val | near_test)]
pre_rows = rows[(t_rows >= PRE_ICTAL[0]) & (t_rows <= PRE_ICTAL[1])]
rng = np.random.default_rng(SEED)
fit_rows = (
    np.sort(rng.choice(train_rows, N_TRAIN_MAX, replace=False))
    if len(train_rows) > N_TRAIN_MAX
    else train_rows
)
print(
    f"[SPLIT] train={len(train_rows)} (seizure INCLUDED), val={len(val_rows)} "
    f"({VAL_WINDOW[0]:.0f}..{VAL_WINDOW[1]:.0f}s, lamb only), test={len(test_rows)} "
    f"({HELDOUT_WINDOW[0]:.0f}..{HELDOUT_WINDOW[1]:.0f}s), guard={GUARD_SECONDS:.0f}s; "
    f"pre-ictal rows for q = {len(pre_rows)}",
    flush=True,
)


# ---------------------------------------------------------------------------
# 4. Lifts
# ---------------------------------------------------------------------------
om_center = OMEGA[train_rows].mean(axis=0)
om_spread = OMEGA[train_rows].std(axis=0) + 1e-12
kp_center = KAPPA[train_rows].mean(axis=0)
kp_spread = KAPPA[train_rows].std(axis=0) + 1e-12


def fast_np(eta):
    th = eta[:, TH_S]
    om = (squash(eta[:, U_S], OM_LO, OM_HI) - om_center) / om_spread
    kp = (squash(eta[:, V_S], KP_LO, KP_HI) - kp_center) / kp_spread
    R = np.abs(np.mean(np.exp(1j * th), axis=1))
    return th, om, kp, eta[:, Z_S], R


def fast_t(eta):
    th = eta[:, TH_S]
    om = (squash_t(eta[:, U_S], OM_LO, OM_HI)
          - torch.as_tensor(om_center, dtype=eta.dtype)) / \
        torch.as_tensor(om_spread, dtype=eta.dtype)
    kp = (squash_t(eta[:, V_S], KP_LO, KP_HI)
          - torch.as_tensor(kp_center, dtype=eta.dtype)) / \
        torch.as_tensor(kp_spread, dtype=eta.dtype)
    R = torch.sqrt(
        torch.mean(torch.sin(th), dim=1) ** 2
        + torch.mean(torch.cos(th), dim=1) ** 2 + 1e-12
    )
    return th, om, kp, eta[:, Z_S], R


def theta_lift_np(eta):
    th, om, kp, _z, _R = fast_np(eta)
    feats = [np.ones(len(eta))] + [om[:, i] for i in range(M)]
    for p, (a, b) in enumerate(PAIRS):
        d = th[:, b] - th[:, a]
        s, c, k = np.sin(d), np.cos(d), kp[:, p]
        feats.extend([k * s, k * c, s, c, k])
    return np.column_stack(feats)


def theta_lift_torch(eta):
    th, om, kp, _z, _R = fast_t(eta)
    feats = [torch.ones_like(om[:, 0])] + [om[:, i] for i in range(M)]
    for p, (a, b) in enumerate(PAIRS):
        d = th[:, b] - th[:, a]
        s, c, k = torch.sin(d), torch.cos(d), kp[:, p]
        feats.extend([k * s, k * c, s, c, k])
    return torch.stack(feats, dim=1)


def u_lift_np(eta):
    _th, om, kp, z, R = fast_np(eta)
    mk = kp.mean(axis=1)
    feats = [np.ones(len(eta)), z, R, R * z, mk, mk * z]
    for i in range(M):
        feats.extend([om[:, i], om[:, i] * z])
    return np.column_stack(feats)


def u_lift_torch(eta):
    _th, om, kp, z, R = fast_t(eta)
    mk = torch.mean(kp, dim=1)
    feats = [torch.ones_like(z), z, R, R * z, mk, mk * z]
    for i in range(M):
        feats.extend([om[:, i], om[:, i] * z])
    return torch.stack(feats, dim=1)


def v_lift_np(eta):
    th, _om, kp, z, R = fast_np(eta)
    feats = [np.ones(len(eta)), z, R, R * z]
    for p, (a, b) in enumerate(PAIRS):
        d = th[:, b] - th[:, a]
        feats.extend([np.sin(d), np.cos(d), kp[:, p], kp[:, p] * z])
    return np.column_stack(feats)


def v_lift_torch(eta):
    th, _om, kp, z, R = fast_t(eta)
    feats = [torch.ones_like(z), z, R, R * z]
    for p, (a, b) in enumerate(PAIRS):
        d = th[:, b] - th[:, a]
        feats.extend([torch.sin(d), torch.cos(d), kp[:, p], kp[:, p] * z])
    return torch.stack(feats, dim=1)


def h_lift_np(eta):
    _th, om, kp, _z, R = fast_np(eta)
    mo, so = om.mean(axis=1), om.std(axis=1)
    mk, sk = kp.mean(axis=1), kp.std(axis=1)
    return np.column_stack([np.ones(len(eta)), mo, so, R, mk, sk, mo * R, mk * R])


def h_lift_torch(eta):
    _th, om, kp, _z, R = fast_t(eta)
    mo, so = torch.mean(om, dim=1), torch.std(om, dim=1, unbiased=False)
    mk, sk = torch.mean(kp, dim=1), torch.std(kp, dim=1, unbiased=False)
    return torch.stack([torch.ones_like(R), mo, so, R, mk, sk, mo * R, mk * R], dim=1)


THETA_LIFT_DIM = 1 + M + 5 * P
U_LIFT_DIM = 6 + 2 * M
V_LIFT_DIM = 4 + 4 * P
H_LIFT_DIM = 8
print(
    f"[LIFT]  theta={THETA_LIFT_DIM}, u={U_LIFT_DIM}, v={V_LIFT_DIM}, h={H_LIFT_DIM}",
    flush=True,
)


# ---------------------------------------------------------------------------
# 5. Fit
# ---------------------------------------------------------------------------
lamb_choices: dict[str, dict] = {}
tau_choices: dict[str, dict] = {}


def fit_block(name, lift_np, lift_dim, lift_torch, steps, target, scale, score_fn):
    """Fit once per lamb; keep the model that wins on the VAL time block."""
    best, table = None, []
    for lamb in LAMB_GRID:
        m = KANDy(
            lift=CustomLift(lift_np, lift_dim, lift_torch, name),
            grid=GRID, k=3, steps=steps, seed=SEED,
            base_fun=torch.nn.Identity(), device="cpu",
        )
        quiet_fit(m, X=ETA[fit_rows], X_dot=target[fit_rows] / scale,
                  val_frac=0.15, test_frac=0.15, lamb=lamb, opt="LBFGS", lr=1.0,
                  batch=-1, stop_grid_update_step=10, patience=0, verbose=False)
        score = float(score_fn(m))
        table.append({"lamb": lamb, "val_r2_mean": score})
        if best is None or score > best[1]:
            best = (m, score, lamb)
    lamb_choices[name] = {"grid": table, "selected": best[2], "val_r2_mean": best[1]}
    print(
        f"[TRAIN] {name}: "
        + ", ".join(f"{r['lamb']:g}->{r['val_r2_mean']:+.3f}" for r in table)
        + f"  | selected lamb={best[2]:g}",
        flush=True,
    )
    return best[0]


def tau_scan(name, lift_vals, state, rate, c, r, grid):
    """Pick a relaxation timescale by how well (target - y)/tau predicts dy/dt."""
    scores = []
    for tau in grid:
        tgt = state[: len(rate)] + tau * rate
        sol, *_ = np.linalg.lstsq(lift_vals[train_rows],
                                  np.atleast_2d(tgt[train_rows].T).T, rcond=None)
        pred = np.clip(lift_vals[val_rows] @ sol, c - r, c + r)
        pred_rate = (pred - np.atleast_2d(state[val_rows].T).T) / tau
        scores.append(float(np.nanmean(
            r2_per_output(np.atleast_2d(rate[val_rows].T).T, pred_rate)
        )))
    scores = np.array(scores)
    tau = float(grid[int(np.argmax(scores))])
    at_edge = bool(tau in (grid[0], grid[-1]))
    tau_choices[name] = {"tau": tau, "val_r2": float(scores.max()),
                         "on_grid_boundary": at_edge}
    print(
        f"[TAU]   {name}: tau={tau:.2f}s (val R2 of the rate = {scores.max():+.3f})"
        + ("   [WARNING: on the grid boundary -- not identified by the recording]"
           if at_edge else ""),
        flush=True,
    )
    return tau


THETA_RESID = DTHETA - OMEGA
th_scale = THETA_RESID[train_rows].std(axis=0) + 1e-12
DU_LAG = (U[LAG:] - U[:-LAG]) / LAG_SECONDS
DV_LAG = (V[LAG:] - V[:-LAG]) / LAG_SECONDS

model_theta = fit_block(
    "adaptive_kuramoto_omega", theta_lift_np, THETA_LIFT_DIM, theta_lift_torch,
    THETA_STEPS, THETA_RESID, th_scale,
    lambda m: np.nanmean(r2_per_output(
        THETA_RESID[val_rows], m.predict(ETA[val_rows]) * th_scale)),
)

coupling_rms = float(np.sqrt(np.mean(
    (model_theta.predict(ETA[train_rows]) * th_scale) ** 2)))
omega_rms = float(np.sqrt(np.mean(OMEGA[train_rows] ** 2)))
coupling_ratio = coupling_rms / max(omega_rms, 1e-12)
COUPLING_ALIVE = coupling_ratio >= 1e-3
print(
    f"[COUPLE] rms coupling = {coupling_rms:.3f} rad/s vs rms omega = {omega_rms:.1f} "
    f"rad/s (ratio {coupling_ratio:.2e})"
    + ("" if COUPLING_ALIVE else
       "   [coupling regularized to ~0: mechanism 'couple_suppress' has almost "
       "nothing to suppress and its axis will be flat BY CONSTRUCTION]"),
    flush=True,
)

TAU_U = tau_scan("omega", u_lift_np(ETA), U, DU_LAG, U_C, U_R, TAU_U_GRID)
TAU_V = tau_scan("kappa", v_lift_np(ETA), V, DV_LAG, V_C, V_R, TAU_V_GRID)

U_INNER = to_inner(U[: len(DU_LAG)] + TAU_U * DU_LAG, U_C, U_R)
V_INNER = to_inner(V[: len(DV_LAG)] + TAU_V * DV_LAG, V_C, V_R)
u_scale = U_INNER[train_rows].std(axis=0) + 1e-12
v_scale = V_INNER[train_rows].std(axis=0) + 1e-12


def du_of(model, idx):
    return (from_inner(model.predict(ETA[idx]) * u_scale, U_C, U_R) - U[idx]) / TAU_U


def dv_of(model, idx):
    return (from_inner(model.predict(ETA[idx]) * v_scale, V_C, V_R) - V[idx]) / TAU_V


model_u = fit_block(
    "slow_omega_bounded", u_lift_np, U_LIFT_DIM, u_lift_torch, U_STEPS,
    U_INNER, u_scale,
    lambda m: np.nanmean(r2_per_output(DU_LAG[val_rows], du_of(m, val_rows))),
)
model_v = fit_block(
    "adaptive_kappa_bounded", v_lift_np, V_LIFT_DIM, v_lift_torch, V_STEPS,
    V_INNER, v_scale,
    lambda m: np.nanmean(r2_per_output(DV_LAG[val_rows], dv_of(m, val_rows))),
)

TAU_Z = tau_scan("z", h_lift_np(ETA), Z, DZ, Z_C, Z_R, TAU_Z_GRID)
H_INNER = to_inner(Z[: len(DZ)] + TAU_Z * DZ, Z_C, Z_R)
h_scale = float(H_INNER[train_rows].std() + 1e-12)


def dz_of(model, idx):
    inner = (model.predict(ETA[idx]) * h_scale).ravel()
    return (from_inner(inner, Z_C, Z_R) - Z[idx]) / TAU_Z


model_h = fit_block(
    "epileptor_h", h_lift_np, H_LIFT_DIM, h_lift_torch, H_STEPS,
    H_INNER[:, None], h_scale,
    lambda m: r2_per_output(DZ[val_rows, None], dz_of(m, val_rows)[:, None])[0],
)


# ---------------------------------------------------------------------------
# 6. Held-out skill against the baseline each block has to beat
# ---------------------------------------------------------------------------
scores = {
    "theta_coupling_kandy": r2_per_output(
        THETA_RESID[test_rows], model_theta.predict(ETA[test_rows]) * th_scale),
    "theta_omega_only_baseline": r2_per_output(
        THETA_RESID[test_rows], np.zeros_like(THETA_RESID[test_rows])),
    "du_kandy": r2_per_output(DU_LAG[test_rows], du_of(model_u, test_rows)),
    "du_zero_baseline": r2_per_output(
        DU_LAG[test_rows], np.zeros_like(DU_LAG[test_rows])),
    "dv_kandy": r2_per_output(DV_LAG[test_rows], dv_of(model_v, test_rows)),
    "dv_zero_baseline": r2_per_output(
        DV_LAG[test_rows], np.zeros_like(DV_LAG[test_rows])),
    "dz_epileptor": r2_per_output(DZ[test_rows, None], dz_of(model_h, test_rows)[:, None]),
    "dz_zero_baseline": r2_per_output(
        DZ[test_rows, None], np.zeros((len(test_rows), 1))),
}
print("\n[EVAL]  held-out skill (post-ictal block, guard-banded)", flush=True)
for k, v in scores.items():
    print(f"        {k:>26}: mean={np.nanmean(v):+.4f}", flush=True)


# ---------------------------------------------------------------------------
# 7. The residual drive q(t), estimated on the PRE-ICTAL segment only
# ---------------------------------------------------------------------------
#
# q is the part of the measured rate the fitted blocks do not explain.  Taken
# over the whole recording it contains the seizure, which is what made the
# original replay circular.  Here it is built the same way but then TRUNCATED:
# the time-varying residual is used only for t < Q_FREEZE_TIME, and from onset
# onward q is held at its pre-ictal mean.  A constant q is an excitability
# offset -- a fixed shift of each block's fixed point -- so its magnitude is
# reported next to the model's own rate rather than left implicit.
#
Q_FULL = np.zeros((N_ROW, N_ETA))
Q_FULL[rows, TH_S] = THETA_RESID[rows] - model_theta.predict(ETA[rows]) * th_scale
Q_FULL[rows, U_S] = DU_LAG[rows] - du_of(model_u, rows)
Q_FULL[rows, V_S] = DV_LAG[rows] - dv_of(model_v, rows)
Q_FULL[rows, Z_S] = DZ[rows] - dz_of(model_h, rows)
Q_FULL[len(rows):] = Q_FULL[len(rows) - 1]

w_q = odd_window(Q_SMOOTH_SECONDS, FS)
Q_FULL = savgol_filter(Q_FULL, w_q, 3, axis=0, mode="interp")

Q_PRE_MEAN = Q_FULL[pre_rows].mean(axis=0)
Q_FREEZE_IDX = int(np.argmin(np.abs(T - Q_FREEZE_TIME)))

# The control drive actually integrated: measured residual before onset, frozen
# pre-ictal mean after it.
Q_CTRL = Q_FULL.copy()
Q_CTRL[Q_FREEZE_IDX:] = Q_PRE_MEAN
Q_CTRL *= Q_GAIN

# How strong is the drive, in the units each block is integrated in?
MODEL_RATE = np.zeros((N_ROW, N_ETA))
MODEL_RATE[rows, TH_S] = OMEGA[rows] + model_theta.predict(ETA[rows]) * th_scale
MODEL_RATE[rows, U_S] = du_of(model_u, rows)
MODEL_RATE[rows, V_S] = dv_of(model_v, rows)
MODEL_RATE[rows, Z_S] = dz_of(model_h, rows)


def drive_share(q_rows, idx):
    num = np.sqrt(np.mean(q_rows[idx] ** 2))
    den = np.sqrt(np.mean((MODEL_RATE[idx] + q_rows[idx]) ** 2))
    return float(num / max(den, 1e-12))


event_rows = rows[(t_rows >= EVENT_WINDOW[0]) & (t_rows <= EVENT_WINDOW[1])]
share_full_event = drive_share(Q_FULL, event_rows)
share_ctrl_event = drive_share(Q_CTRL, event_rows)
share_ctrl_pre = drive_share(Q_CTRL, pre_rows)
q_pre_rms = float(np.sqrt(np.mean(Q_FULL[pre_rows] ** 2)))
q_event_rms = float(np.sqrt(np.mean(Q_FULL[event_rows] ** 2)))

print(
    f"\n[Q]     residual drive: pre-ictal rms={q_pre_rms:.3f}, ictal rms="
    f"{q_event_rms:.3f} (ratio {q_event_rms / max(q_pre_rms, 1e-12):.2f}x)",
    flush=True,
)
print(
    f"[Q]     share of the integrated rate carried by q over the event window: "
    f"full-record q={share_full_event:.3f}  ->  pre-ictal-only q={share_ctrl_event:.3f}",
    flush=True,
)
print(
    f"[Q]     frozen from t={Q_FREEZE_TIME:.1f}s at the pre-ictal mean; "
    f"|q_theta|={np.abs(Q_PRE_MEAN[TH_S]).mean():.3f} rad/s, "
    f"q_z={Q_PRE_MEAN[Z_S]:+.4f}/s  (a constant q is an excitability offset)",
    flush=True,
)


Q_FROZEN = Q_PRE_MEAN * Q_GAIN


def q_at(t_value: float) -> np.ndarray:
    """Pre-ictal residual before onset, frozen pre-ictal mean after.

    Vectorized over the state blocks: this is evaluated four times per RK4 step
    for the whole sweep, so a per-column np.interp loop is the hot path.
    """
    if t_value >= Q_FREEZE_TIME:
        return Q_FROZEN
    j = int(np.clip(np.searchsorted(T, t_value) - 1, 0, N_ROW - 2))
    frac = (t_value - T[j]) / (T[j + 1] - T[j])
    return Q_CTRL[j] + frac * (Q_CTRL[j + 1] - Q_CTRL[j])


# ---------------------------------------------------------------------------
# 8. Stimulation: mechanisms, protocols, and the driven RHS
# ---------------------------------------------------------------------------
MEAN_OMEGA = float(OMEGA.mean())
STD_Z = float(np.std(Z))

# Amplitude grids.  Units differ per mechanism, so efficacy curves are
# comparable WITHIN a mechanism and charge is never compared ACROSS them.
AMP_GRIDS = {
    "hf_entrain": np.linspace(0.0, 1.5, N_AMP) * MEAN_OMEGA,        # rad/s
    "couple_suppress": np.linspace(0.0, 1.0, N_AMP),                # dimensionless
    "pulse_reset": np.linspace(0.0, np.pi, N_AMP),                  # rad per pulse
    "slow_drive": np.linspace(0.0, 3.0, N_AMP) * STD_Z / TAU_Z,     # z-units/s
}

STEPS = int(round(ROLLOUT_SECONDS / DT))
i0 = int(np.argmin(np.abs(T - ROLLOUT_START)))
t_roll = T[i0] + np.arange(STEPS + 1) * DT

# Detector thresholds, calibrated from the recording.
ref_slice = slice(i0, i0 + STEPS + 1)
mo_ref = OMEGA[ref_slice].mean(axis=1)
R_ref = np.abs(np.mean(np.exp(1j * THETA[ref_slice]), axis=1))
mk_ref = KAPPA[ref_slice].mean(axis=1)
z_ref = Z[ref_slice]
NB = int(round(1.0 / DT))
OM_BASE = float(mo_ref[:NB].mean())
peak_ref = float(mo_ref.max() - OM_BASE)
D_HI = OM_BASE + DETECT_HI * peak_ref
D_LO = OM_BASE + DETECT_LO * peak_ref
print(
    f"\n[DETECT] closed-loop trigger on <omega>: on above {D_HI:.1f}, off below "
    f"{D_LO:.1f} rad/s (pre-ictal baseline {OM_BASE:.1f}, observed rise {peak_ref:+.1f})",
    flush=True,
)

# What stimulation frequency can a 5 ms step actually carry?
CARRIER_LIMIT = 1.0 / (10.0 * DT)   # 10 RK4 steps per carrier cycle
PULSE_LIMIT = 1.0 / (4.0 * DT)      # 4 steps between pulses
CARRIER_RESOLVED = bool(F_STIM <= CARRIER_LIMIT and F_PULSE <= PULSE_LIMIT)
print(
    f"[CARRIER] f_stim={F_STIM:.0f} Hz (limit {CARRIER_LIMIT:.0f} Hz), "
    f"f_pulse={F_PULSE:.0f} Hz (limit {PULSE_LIMIT:.0f} Hz) at dt={1000 * DT:.1f} ms"
    + ("" if CARRIER_RESOLVED else
       "\n          [ALIASED: the carrier is faster than the integrator can "
       "represent, so the\n           delivered waveform is a step-size artefact. "
       "Clinical 130-180 Hz DBS is\n           NOT representable at fs=200 Hz -- "
       "that is a limit of this recording.]"),
    flush=True,
)


def protocol_gate(proto_id: np.ndarray, t_value: float,
                  latch: np.ndarray) -> np.ndarray:
    """w(t) in {0, 1} per row.  The closed-loop latch is updated once per
    integration step and held constant across the four RK4 stages -- standard
    for a switched system, and it keeps the stimulator causal in the same way a
    real device is (it cannot react inside its own sample interval)."""
    w = np.zeros(len(proto_id))
    open_on = (t_value >= STIM_ON) & (t_value <= STIM_OFF)
    sham_on = (t_value >= STIM_ON + SHAM_SHIFT) & (t_value <= STIM_OFF + SHAM_SHIFT)
    w[proto_id == 0] = 1.0 if open_on else 0.0
    w[proto_id == 2] = 1.0 if sham_on else 0.0
    closed = proto_id == 1
    w[closed] = latch[closed].astype(float)
    return w


def update_latch(latch: np.ndarray, mean_omega: np.ndarray) -> np.ndarray:
    """Schmitt trigger on mean omega."""
    out = latch.copy()
    out[mean_omega > D_HI] = True
    out[mean_omega < D_LO] = False
    return out


def rhs(eta, t_value, w, amp_hf, amp_sup, amp_z):
    out = np.zeros_like(eta)
    q = q_at(t_value)
    om = squash(eta[:, U_S], OM_LO, OM_HI)

    supp = 1.0 - amp_sup * w
    coup = model_theta.predict(eta) * th_scale
    out[:, TH_S] = om + supp[:, None] * coup + q[TH_S]
    if np.any(amp_hf * w):
        drive = (amp_hf * w)[:, None] * np.sin(eta[:, TH_S] - 2 * np.pi * F_STIM * t_value)
        out[:, TH_S] -= drive * TARGET_MASK[None, :]

    u_tgt = from_inner(model_u.predict(eta) * u_scale, U_C, U_R)
    out[:, U_S] = (u_tgt - eta[:, U_S]) / TAU_U + q[U_S]

    v_tgt = from_inner(model_v.predict(eta) * v_scale, V_C, V_R)
    out[:, V_S] = (v_tgt - eta[:, V_S]) / TAU_V + q[V_S]

    h = from_inner((model_h.predict(eta) * h_scale).ravel(), Z_C, Z_R)
    out[:, Z_S] = (h - eta[:, Z_S]) / TAU_Z + q[Z_S] - amp_z * w
    return out


def rollout(eta0, mech_id, amp, proto_id, keep_full=False):
    """Batched RK4 over rows that may differ in mechanism, amplitude and protocol.

    Nothing is clipped: omega and kappa are bounded by their tanh
    parameterization and every slow block relaxes toward a squashed target.
    """
    eta = np.atleast_2d(eta0).astype(float).copy()
    n = len(eta)
    mech_id = np.broadcast_to(np.atleast_1d(mech_id), (n,)).astype(int)
    amp = np.broadcast_to(np.atleast_1d(amp), (n,)).astype(float)
    proto_id = np.broadcast_to(np.atleast_1d(proto_id), (n,)).astype(int)

    amp_hf = np.where(mech_id == 0, amp, 0.0)
    amp_sup = np.where(mech_id == 1, amp, 0.0)
    amp_pulse = np.where(mech_id == 2, amp, 0.0)
    amp_z = np.where(mech_id == 3, amp, 0.0)

    mo = np.empty((STEPS + 1, n))
    Rt = np.empty((STEPS + 1, n))
    mk = np.empty((STEPS + 1, n))
    zt = np.empty((STEPS + 1, n))
    wt = np.empty((STEPS + 1, n))
    sech = np.empty((STEPS + 1, n))
    full = np.empty((STEPS + 1, n, N_ETA)) if keep_full else None

    def record(idx, e, w):
        om = squash(e[:, U_S], OM_LO, OM_HI)
        kp = squash(e[:, V_S], KP_LO, KP_HI)
        mo[idx] = om.mean(axis=1)
        Rt[idx] = np.abs(np.mean(np.exp(1j * e[:, TH_S]), axis=1))
        mk[idx] = kp.mean(axis=1)
        zt[idx] = e[:, Z_S]
        wt[idx] = w
        sech[idx] = np.minimum((1 - np.tanh(e[:, U_S]) ** 2).min(axis=1),
                               (1 - np.tanh(e[:, V_S]) ** 2).min(axis=1))
        if keep_full:
            full[idx] = e

    latch = np.zeros(n, dtype=bool)
    pulse_period = 1.0 / F_PULSE
    next_pulse = t_roll[0] + pulse_period
    latch = update_latch(latch, squash(eta[:, U_S], OM_LO, OM_HI).mean(axis=1))
    record(0, eta, protocol_gate(proto_id, t_roll[0], latch))

    done = STEPS
    for s in range(STEPS):
        t_value = t_roll[s]
        latch = update_latch(latch, squash(eta[:, U_S], OM_LO, OM_HI).mean(axis=1))
        w = protocol_gate(proto_id, t_value, latch)

        k1 = rhs(eta, t_value, w, amp_hf, amp_sup, amp_z)
        k2 = rhs(eta + 0.5 * DT * k1, t_value + 0.5 * DT, w, amp_hf, amp_sup, amp_z)
        k3 = rhs(eta + 0.5 * DT * k2, t_value + 0.5 * DT, w, amp_hf, amp_sup, amp_z)
        k4 = rhs(eta + DT * k3, t_value + DT, w, amp_hf, amp_sup, amp_z)
        eta = eta + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Phase-resetting pulses are a discrete map applied between steps.
        if t_roll[s + 1] >= next_pulse:
            kick = (amp_pulse * w)[:, None] * np.sin(eta[:, TH_S] - THETA_REF)
            eta[:, TH_S] -= kick * TARGET_MASK[None, :]
            while next_pulse <= t_roll[s + 1]:
                next_pulse += pulse_period

        if not np.isfinite(eta).all():
            done = s
            break
        record(s + 1, eta, w)

    return {"mean_omega": mo[: done + 1], "order": Rt[: done + 1],
            "mean_kappa": mk[: done + 1], "z": zt[: done + 1],
            "gate": wt[: done + 1], "sech2": sech[: done + 1], "steps": done,
            "eta": full[: done + 1] if keep_full else None}


# ---------------------------------------------------------------------------
# 9. Outcome measures
# ---------------------------------------------------------------------------
EXC_THRESHOLD = 0.5 * abs(peak_ref)
AMP_FLOOR = float(np.percentile(KAPPA.mean(axis=1), 5))
OM_BAND = (0.75 * float(np.min(OMEGA.mean(axis=1))),
           1.25 * float(np.max(OMEGA.mean(axis=1))))
ev_mask = (t_roll >= EVENT_WINDOW[0]) & (t_roll <= EVENT_WINDOW[1])


def burden_of(mean_omega: np.ndarray) -> float:
    """Seizure burden: area of <omega> above the pre-ictal baseline over the
    event window, in rad.  Termination is about removing area, not about
    shaving the peak -- a stimulus that delays onset by 2 s and then lets the
    full event run should not score as a success."""
    m = ev_mask[: len(mean_omega)]
    if not m.any():
        return float("nan")
    return float(np.sum(np.maximum(mean_omega[m] - OM_BASE, 0.0)) * DT)


def is_viable(s: dict) -> bool:
    """Seizure-free must mean 'still a working oscillator', not 'the model died'."""
    return bool(
        s["complete"] and s["min_sech2"] > 1e-3
        and OM_BAND[0] <= s["omega_mean"] <= OM_BAND[1]
        and s["kappa_mean"] >= AMP_FLOOR
    )


def summarize(res, idx, amp):
    mo = res["mean_omega"][:, idx]
    R = res["order"][:, idx]
    w = res["gate"][:, idx]
    return {
        "omega_rise": float(mo.max() - mo[:NB].mean()),
        "omega_mean": float(mo.mean()),
        "order_rise": float(R.max() - R[:NB].mean()),
        "kappa_mean": float(res["mean_kappa"][:, idx].mean()),
        "z_end": float(res["z"][-1, idx]),
        "burden": burden_of(mo),
        "charge": float(amp * np.sum(w) * DT),
        "duty": float(np.mean(w)),
        "min_sech2": float(res["sech2"][:, idx].min()),
        "complete": bool(len(mo) == STEPS + 1),
    }


data_stats = {"complete": True, "min_sech2": 1.0, "omega_mean": float(mo_ref.mean()),
              "kappa_mean": float(mk_ref.mean())}
data_viable = is_viable(data_stats)
burden_data = burden_of(mo_ref)
print(
    f"[VIABLE] band {OM_BAND[0]:.1f}..{OM_BAND[1]:.1f} rad/s, amplitude floor "
    f"{AMP_FLOOR:+.2f}; measured trajectory viable={data_viable}, "
    f"measured burden={burden_data:.1f} rad"
    + ("" if data_viable else
       "   [CALIBRATION ERROR: the criterion rejects the real event]"),
    flush=True,
)


# ---------------------------------------------------------------------------
# 10. Baseline: does the model seize at all under pre-ictal q?
# ---------------------------------------------------------------------------
print(
    f"\n[BASE]  unstimulated rollout from t={T[i0]:.1f}s for {ROLLOUT_SECONDS:.0f}s "
    "-- pre-ictal q only, lambda=0, no clipping",
    flush=True,
)
base_res = rollout(ETA[i0], 0, 0.0, 0, keep_full=True)
base_stats = summarize(base_res, 0, 0.0)
BASELINE_SEIZES = bool(base_stats["omega_rise"] >= EXC_THRESHOLD)
print(
    f"        mean-omega rise: data={peak_ref:+.1f}, model={base_stats['omega_rise']:+.1f} "
    f"rad/s (seizure threshold {EXC_THRESHOLD:.1f})",
    flush=True,
)
print(
    f"        burden: data={burden_data:.1f}, model={base_stats['burden']:.1f} rad; "
    f"order rise: data={float(R_ref.max() - R_ref[:NB].mean()):+.3f}, "
    f"model={base_stats['order_rise']:+.3f}",
    flush=True,
)
print(
    f"        -> unstimulated model {'SEIZES' if BASELINE_SEIZES else 'DOES NOT SEIZE'}",
    flush=True,
)
if not BASELINE_SEIZES:
    print(
        "        [UNMEASURABLE] With no event in the uncontrolled run there is\n"
        "        nothing for stimulation to terminate.  The sweep below still runs\n"
        "        and is still reported, but every efficacy number in it is a\n"
        "        perturbation of a non-event and must NOT be read as seizure\n"
        "        control.  See the gates block for what this rules out.",
        flush=True,
    )

BASE_BURDEN = base_stats["burden"]


# ---------------------------------------------------------------------------
# 11. The stimulation sweep
# ---------------------------------------------------------------------------
sweep_mech, sweep_amp, sweep_proto = [], [], []
for mi, mech in enumerate(MECHANISMS):
    for pi, _proto in enumerate(PROTOCOLS):
        for a in AMP_GRIDS[mech]:
            sweep_mech.append(mi)
            sweep_amp.append(float(a))
            sweep_proto.append(pi)
sweep_mech = np.array(sweep_mech)
sweep_amp = np.array(sweep_amp)
sweep_proto = np.array(sweep_proto)
N_RUN = len(sweep_mech)

print(
    f"\n[SWEEP] {len(MECHANISMS)} mechanisms x {len(PROTOCOLS)} protocols x {N_AMP} "
    f"amplitudes = {N_RUN} rollouts from the same pre-ictal state",
    flush=True,
)
sweep = rollout(np.tile(ETA[i0], (N_RUN, 1)), sweep_mech, sweep_amp, sweep_proto)
sweep_stats = [summarize(sweep, i, sweep_amp[i]) for i in range(N_RUN)]

for s in sweep_stats:
    s["viable"] = is_viable(s)
    s["burden_ratio"] = float(s["burden"] / max(BASE_BURDEN, 1e-9))
    s["terminated"] = bool(s["viable"] and s["burden_ratio"] <= (1.0 - BURDEN_CUT))

results_rows = []
for i in range(N_RUN):
    s = sweep_stats[i]
    results_rows.append({
        "mechanism": MECHANISMS[sweep_mech[i]],
        "protocol": PROTOCOLS[sweep_proto[i]],
        "amplitude": float(sweep_amp[i]),
        **{k: s[k] for k in ("burden", "burden_ratio", "omega_rise", "order_rise",
                             "charge", "duty", "min_sech2", "viable", "terminated",
                             "complete")},
    })

print("        burden ratio vs baseline (1.00 = unchanged, lower = suppressed)", flush=True)
for mi, mech in enumerate(MECHANISMS):
    for pi, proto in enumerate(PROTOCOLS):
        sel = (sweep_mech == mi) & (sweep_proto == pi)
        ratios = np.array([sweep_stats[i]["burden_ratio"] for i in np.where(sel)[0]])
        viab = np.array([sweep_stats[i]["viable"] for i in np.where(sel)[0]])
        term = int(sum(sweep_stats[i]["terminated"] for i in np.where(sel)[0]))
        print(
            f"          {mech:>16} {proto:>6}: "
            + " ".join(f"{r:5.2f}" for r in ratios)
            + f"  | viable {int(viab.sum())}/{len(viab)}, terminated {term}/{len(ratios)}",
            flush=True,
        )


def best_of(mech_i, proto_i):
    """Smallest amplitude that terminates; else the lowest-burden viable run.

    The no-termination tie-break is the LARGEST amplitude, not the smallest.
    When a mechanism does nothing every burden ratio is 1.00, and a
    smallest-amplitude tie-break would then select A=0 -- which is the
    unstimulated run, so the trace plot would show 'stimulation' that was never
    delivered.  Picking the strongest viable run instead shows what the
    mechanism does at full drive, which is the informative negative result.
    """
    sel = np.where((sweep_mech == mech_i) & (sweep_proto == proto_i))[0]
    term = [i for i in sel if sweep_stats[i]["terminated"]]
    if term:
        return int(min(term, key=lambda i: sweep_amp[i])), True
    viable = [i for i in sel if sweep_stats[i]["viable"]]
    if viable:
        return int(min(viable, key=lambda i: (sweep_stats[i]["burden"],
                                              -sweep_amp[i]))), False
    return int(sel[0]), False


best_index, best_terminates = {}, {}
for mi, mech in enumerate(MECHANISMS):
    for pi, proto in enumerate(PROTOCOLS):
        idx, ok = best_of(mi, pi)
        best_index[(mech, proto)] = idx
        best_terminates[(mech, proto)] = ok

print("\n[BEST]  smallest terminating amplitude per mechanism (open loop / closed loop)",
      flush=True)
for mech in MECHANISMS:
    parts = []
    for proto in ("open", "closed"):
        i = best_index[(mech, proto)]
        tag = "TERMINATES" if best_terminates[(mech, proto)] else "no termination"
        parts.append(
            f"{proto}: A={sweep_amp[i]:.3g}, burden={sweep_stats[i]['burden_ratio']:.2f}, "
            f"charge={sweep_stats[i]['charge']:.3g} [{tag}]"
        )
    print(f"          {mech:>16}  " + "   |   ".join(parts), flush=True)

sham_terminations = sum(
    1 for i in range(N_RUN)
    if sweep_proto[i] == 2 and sweep_stats[i]["terminated"]
)
open_terminations = sum(
    1 for i in range(N_RUN)
    if sweep_proto[i] == 0 and sweep_stats[i]["terminated"]
)
closed_terminations = sum(
    1 for i in range(N_RUN)
    if sweep_proto[i] == 1 and sweep_stats[i]["terminated"]
)
print(
    f"[SPEC]  terminations: open={open_terminations}, closed={closed_terminations}, "
    f"sham={sham_terminations} (sham stimulates the pre-ictal window only -- any "
    f"sham termination means the effect is not about stimulating during the event)",
    flush=True,
)

zero_rows = [i for i in range(N_RUN) if sweep_amp[i] == 0.0]
zero_burden_dev = float(np.max([
    abs(sweep_stats[i]["burden"] - BASE_BURDEN) for i in zero_rows
])) if zero_rows else float("nan")


# ---------------------------------------------------------------------------
# 12. Charge efficiency: open vs closed loop at matched efficacy
# ---------------------------------------------------------------------------
charge_compare = []
for mech in MECHANISMS:
    io, ic = best_index[(mech, "open")], best_index[(mech, "closed")]
    charge_compare.append({
        "mechanism": mech,
        "open_amplitude": float(sweep_amp[io]),
        "open_burden_ratio": sweep_stats[io]["burden_ratio"],
        "open_charge": sweep_stats[io]["charge"],
        "open_duty": sweep_stats[io]["duty"],
        "open_terminates": best_terminates[(mech, "open")],
        "closed_amplitude": float(sweep_amp[ic]),
        "closed_burden_ratio": sweep_stats[ic]["burden_ratio"],
        "closed_charge": sweep_stats[ic]["charge"],
        "closed_duty": sweep_stats[ic]["duty"],
        "closed_terminates": best_terminates[(mech, "closed")],
        # A saving is only defined when the open-loop arm actually delivered
        # charge AND both arms reached the same endpoint.  Comparing a
        # terminating run against a non-terminating one is not a saving.
        "charge_saving": (
            float(1.0 - sweep_stats[ic]["charge"] / sweep_stats[io]["charge"])
            if sweep_stats[io]["charge"] > 0
            and best_terminates[(mech, "open")] == best_terminates[(mech, "closed")]
            else None),
    })
print("\n[CHARGE] closed-loop duty cycle and charge relative to open loop", flush=True)
for row in charge_compare:
    saving = ("n/a (endpoints differ or no charge delivered)"
              if row["charge_saving"] is None
              else f"{100 * row['charge_saving']:+.0f}%")
    print(
        f"          {row['mechanism']:>16}: duty {row['open_duty']:.2f} -> "
        f"{row['closed_duty']:.2f}, charge {row['open_charge']:.3g} -> "
        f"{row['closed_charge']:.3g} ({saving})",
        flush=True,
    )


# ---------------------------------------------------------------------------
# 13. Gates
# ---------------------------------------------------------------------------
any_termination = open_terminations + closed_terminations > 0
termination_viable = all(
    sweep_stats[i]["viable"] for i in range(N_RUN) if sweep_stats[i]["terminated"]
)
gates = {
    "baseline_seizes_under_pre_ictal_q": {
        "required": f">= 50% of observed rise ({EXC_THRESHOLD:.1f} rad/s)",
        "achieved": base_stats["omega_rise"],
        "passed": BASELINE_SEIZES,
        "note": "DECISIVE: with no event in the uncontrolled run, every efficacy "
                "number below is a perturbation of a non-event.",
    },
    "measured_trajectory_is_viable": {
        "required": "the recorded seizure must score viable",
        "achieved": data_viable, "passed": bool(data_viable),
    },
    "zero_amplitude_reproduces_baseline": {
        "required": "< 1e-6 rad burden deviation",
        "achieved": zero_burden_dev, "passed": bool(zero_burden_dev < 1e-6),
    },
    "some_mechanism_terminates": {
        "required": f">= 1 open- or closed-loop run with burden reduced "
                    f">= {100 * BURDEN_CUT:.0f}% while viable",
        "achieved": open_terminations + closed_terminations,
        "passed": bool(any_termination),
    },
    "terminations_preserve_viability": {
        "required": "every terminating run stays a working oscillator",
        "achieved": termination_viable, "passed": bool(termination_viable),
    },
    "sham_does_not_terminate": {
        "required": "0 sham terminations (pre-ictal-window stimulation only)",
        "achieved": sham_terminations, "passed": bool(sham_terminations == 0),
    },
    "coupling_axis_is_informative": {
        "required": "rms coupling / rms omega >= 1e-3",
        "achieved": coupling_ratio, "passed": bool(COUPLING_ALIVE),
        "note": "if this fails, couple_suppress is flat BY CONSTRUCTION -- the "
                "coupling term was regularized to ~0, so there is nothing to suppress.",
    },
    "stim_carrier_is_resolved": {
        "required": f"f_stim <= {CARRIER_LIMIT:.0f} Hz and f_pulse <= {PULSE_LIMIT:.0f} Hz "
                    f"at dt={1000 * DT:.1f} ms",
        "achieved": [F_STIM, F_PULSE], "passed": CARRIER_RESOLVED,
        "note": "clinical 130-180 Hz DBS is not representable at fs=200 Hz; the "
                "waveform delivered here is a lower-frequency stand-in.",
    },
}
print("\n[GATES]", flush=True)
for name, g in gates.items():
    print(f"        {'PASS' if g['passed'] else 'FAIL'}  {name}: "
          f"achieved={g['achieved']}, required={g['required']}", flush=True)

passed = print_check(
    gates["baseline_seizes_under_pre_ictal_q"]["passed"]
    and gates["measured_trajectory_is_viable"]["passed"]
    and gates["zero_amplitude_reproduces_baseline"]["passed"]
    and gates["some_mechanism_terminates"]["passed"]
    and gates["terminations_preserve_viability"]["passed"]
    and gates["sham_does_not_terminate"]["passed"]
    and gates["stim_carrier_is_resolved"]["passed"],
    "simulated DBS terminates a seizure the model generates on its own",
)
if not BASELINE_SEIZES:
    print(
        "[VERDICT] The model does not generate a seizure from a pre-ictal state once\n"
        "          q(t) stops carrying ictal information.  That is the answer to the\n"
        "          question asked: restricting q to the pre-ictal segment does not\n"
        "          weaken the seizure, it removes it, because the seizure was never\n"
        "          in the learned dynamics.  No DBS conclusion can be drawn from this\n"
        "          model as fitted.",
        flush=True,
    )


# ---------------------------------------------------------------------------
# 14. Metrics and arrays
# ---------------------------------------------------------------------------
metrics = {
    "task": "simulated DBS on an adaptive-Kuramoto seizure model with pre-ictal q(t)",
    "data_path": str(DATA_PATH),
    "channels": LABELS,
    "stim_targets": target_names,
    "state": {"dimension": N_ETA, "theta": M, "u_omega": M, "v_kappa": P, "z": 1},
    "q_treatment": {
        "form": "q(t) = fitted residual for t < freeze; frozen pre-ictal mean after",
        "freeze_time_s": Q_FREEZE_TIME,
        "gain": Q_GAIN,
        "smooth_window_samples": int(w_q),
        "nudge_lambda": 0.0,
        "pre_ictal_rms": q_pre_rms,
        "ictal_rms": q_event_rms,
        "share_of_rate_event_window_full_record_q": share_full_event,
        "share_of_rate_event_window_pre_ictal_q": share_ctrl_event,
        "share_of_rate_pre_ictal_window": share_ctrl_pre,
        "frozen_vector_theta_mean_abs": float(np.abs(Q_PRE_MEAN[TH_S]).mean()),
        "frozen_vector_z": float(Q_PRE_MEAN[Z_S]),
    },
    "fit": {
        "grid": GRID, "k": 3, "lamb_grid": LAMB_GRID,
        "lamb_selected": {k: v["selected"] for k, v in lamb_choices.items()},
        "lamb_val_r2": {k: v["val_r2_mean"] for k, v in lamb_choices.items()},
        "tau": {k: v["tau"] for k, v in tau_choices.items()},
        "tau_on_grid_boundary": {k: v["on_grid_boundary"] for k, v in tau_choices.items()},
        "n_train": int(len(train_rows)), "n_val": int(len(val_rows)),
        "n_test": int(len(test_rows)),
        "coupling_rms_rad_s": coupling_rms, "omega_rms_rad_s": omega_rms,
        "coupling_ratio": coupling_ratio,
    },
    "heldout_r2_mean": {k: float(np.nanmean(v)) for k, v in scores.items()},
    "stimulation": {
        "mechanisms": MECHANISMS, "protocols": PROTOCOLS,
        "f_stim_hz": F_STIM, "f_pulse_hz": F_PULSE, "theta_ref": THETA_REF,
        "open_window_s": [STIM_ON, STIM_OFF],
        "sham_window_s": [STIM_ON + SHAM_SHIFT, STIM_OFF + SHAM_SHIFT],
        "closed_loop_detector": "mean omega, Schmitt trigger",
        "detect_hi_rad_s": D_HI, "detect_lo_rad_s": D_LO,
        "amplitude_grids": {k: [float(v) for v in g] for k, g in AMP_GRIDS.items()},
        "amplitude_units": {
            "hf_entrain": "rad/s", "couple_suppress": "dimensionless fraction",
            "pulse_reset": "rad per pulse", "slow_drive": "z-units/s",
        },
    },
    "outcome": {
        "burden_definition": "integral of max(<omega> - pre-ictal baseline, 0) over "
                             f"{EVENT_WINDOW[0]}..{EVENT_WINDOW[1]}s, in rad",
        "burden_cut_for_termination": BURDEN_CUT,
        "excursion_threshold_rad_s": EXC_THRESHOLD,
        "observed_rise_rad_s": peak_ref,
        "observed_burden_rad": burden_data,
        "baseline_burden_rad": BASE_BURDEN,
        "baseline_omega_rise_rad_s": base_stats["omega_rise"],
        "baseline_seizes": BASELINE_SEIZES,
        "viability_band_rad_s": list(OM_BAND),
        "amplitude_floor": AMP_FLOOR,
    },
    "runs": results_rows,
    "charge_comparison": charge_compare,
    "terminations": {"open": open_terminations, "closed": closed_terminations,
                     "sham": sham_terminations},
    "gates": gates,
    "passed": passed,
}
save_metrics(RESULT_DIR / "metrics.json", metrics)

np.savez_compressed(
    RESULT_DIR / "dbs_arrays.npz",
    t_roll=t_roll, t=T, labels=np.asarray(LABELS),
    q_full=Q_FULL, q_ctrl=Q_CTRL, q_pre_mean=Q_PRE_MEAN, model_rate=MODEL_RATE,
    omega_data=OMEGA, order_data=np.abs(np.mean(np.exp(1j * THETA), axis=1)),
    kappa_data=KAPPA, z_data=Z,
    base_mean_omega=base_res["mean_omega"][:, 0], base_order=base_res["order"][:, 0],
    base_z=base_res["z"][:, 0], base_mean_kappa=base_res["mean_kappa"][:, 0],
    sweep_mech=sweep_mech, sweep_amp=sweep_amp, sweep_proto=sweep_proto,
    sweep_burden=np.array([s["burden"] for s in sweep_stats]),
    sweep_burden_ratio=np.array([s["burden_ratio"] for s in sweep_stats]),
    sweep_omega_rise=np.array([s["omega_rise"] for s in sweep_stats]),
    sweep_charge=np.array([s["charge"] for s in sweep_stats]),
    sweep_viable=np.array([s["viable"] for s in sweep_stats]),
    sweep_terminated=np.array([s["terminated"] for s in sweep_stats]),
    ref_mean_omega=mo_ref, ref_order=R_ref, ref_z=z_ref, ref_mean_kappa=mk_ref,
)


# ---------------------------------------------------------------------------
# 15. Figures
# ---------------------------------------------------------------------------
use_pub_style()
PROTO_STYLE = {"open": ("#1f77b4", "-"), "closed": ("#d62728", "--"),
               "sham": ("0.55", ":")}


def plot_q_treatment() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    axes[0].plot(T, np.sqrt(np.mean(Q_FULL[:, TH_S] ** 2, axis=1)), lw=0.8,
                 color="0.4", label="full-record q (rejected)")
    axes[0].plot(T, np.sqrt(np.mean(Q_CTRL[:, TH_S] ** 2, axis=1)), lw=1.1,
                 color="#1f77b4", label="pre-ictal q, frozen at onset")
    axes[0].axvline(Q_FREEZE_TIME, color="0.2", lw=0.9, ls=":")
    axes[0].axvspan(*EVENT_WINDOW, color="#ffe9e9", zorder=-1)
    axes[0].set_ylabel(r"rms $q_\theta$ (rad/s)")
    axes[0].set_title("The residual drive, before and after truncation to the pre-ictal segment")
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].plot(T, Q_FULL[:, Z_S], lw=0.8, color="0.4", label="full-record $q_z$")
    axes[1].plot(T, Q_CTRL[:, Z_S], lw=1.1, color="#1f77b4", label="pre-ictal $q_z$")
    axes[1].axvline(Q_FREEZE_TIME, color="0.2", lw=0.9, ls=":")
    axes[1].axvspan(*EVENT_WINDOW, color="#ffe9e9", zorder=-1)
    axes[1].set_ylabel(r"$q_z$ (z-units/s)")
    axes[1].legend(fontsize=8, loc="upper right")

    axes[2].plot(T, OMEGA.mean(axis=1), lw=1.0, color="#2ca02c", label="measured")
    axes[2].plot(t_roll[: len(base_res["mean_omega"])],
                 base_res["mean_omega"][:, 0], lw=1.2, ls="--", color="#d62728",
                 label="model, pre-ictal q, no stim")
    axes[2].axhline(OM_BASE + EXC_THRESHOLD, color="0.3", lw=0.8, ls="-.",
                    label="seizure threshold")
    axes[2].axvline(Q_FREEZE_TIME, color="0.2", lw=0.9, ls=":")
    axes[2].axvspan(*EVENT_WINDOW, color="#ffe9e9", zorder=-1)
    axes[2].set_ylabel(r"$\langle\omega\rangle$ (rad/s)")
    axes[2].set_xlabel("recording time (s)")
    axes[2].legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"q_treatment.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_dose_response() -> None:
    fig, axes = plt.subplots(1, len(MECHANISMS), figsize=(4.0 * len(MECHANISMS), 3.8),
                             sharey=True)
    for mi, (mech, ax) in enumerate(zip(MECHANISMS, np.atleast_1d(axes))):
        grid = AMP_GRIDS[mech]
        norm = grid / max(grid.max(), 1e-12)
        for pi, proto in enumerate(PROTOCOLS):
            sel = np.where((sweep_mech == mi) & (sweep_proto == pi))[0]
            ratio = np.array([sweep_stats[i]["burden_ratio"] for i in sel])
            viable = np.array([sweep_stats[i]["viable"] for i in sel])
            color, ls = PROTO_STYLE[proto]
            ax.plot(norm, ratio, color=color, ls=ls, lw=1.3, marker="o", ms=3,
                    label=proto)
            if (~viable).any():
                ax.plot(norm[~viable], ratio[~viable], "x", color="k", ms=7,
                        mew=1.4, label="not viable" if pi == 0 else None)
        ax.axhline(1.0 - BURDEN_CUT, color="0.3", lw=0.9, ls="-.")
        ax.axhline(1.0, color="0.7", lw=0.8)
        ax.set_ylim(-0.05, 1.6)
        ax.set_xlabel("amplitude / max")
        ax.set_title(mech, fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("seizure burden / baseline")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(
        "Dose-response. Dash-dot line = termination criterion"
        + ("" if BASELINE_SEIZES else
           "   [BASELINE DOES NOT SEIZE -- these curves perturb a non-event]"),
        fontsize=11,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"dose_response.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_best_traces() -> None:
    fig, axes = plt.subplots(len(MECHANISMS), 3,
                             figsize=(13, 2.5 * len(MECHANISMS)), sharex=True)
    axes = np.atleast_2d(axes)
    for mi, mech in enumerate(MECHANISMS):
        ax_o, ax_r, ax_z = axes[mi]
        n_base = len(base_res["mean_omega"])
        ax_o.plot(t_roll[:n_base], base_res["mean_omega"][:, 0], color="0.25", lw=1.2,
                  label="no stim")
        ax_r.plot(t_roll[:n_base], base_res["order"][:, 0], color="0.25", lw=1.2)
        ax_z.plot(t_roll[:n_base], base_res["z"][:, 0], color="0.25", lw=1.2)
        ax_o.plot(t_roll[: len(mo_ref)], mo_ref, color="#2ca02c", lw=1.0, alpha=0.8,
                  label="data")
        ax_r.plot(t_roll[: len(R_ref)], R_ref, color="#2ca02c", lw=1.0, alpha=0.8)
        ax_z.plot(t_roll[: len(z_ref)], z_ref, color="#2ca02c", lw=1.0, alpha=0.8)

        n_sw = len(sweep["mean_omega"])
        for proto in ("open", "closed"):
            i = best_index[(mech, proto)]
            color, ls = PROTO_STYLE[proto]
            ax_o.plot(t_roll[:n_sw], sweep["mean_omega"][:, i], color=color, ls=ls,
                      lw=1.2, label=f"{proto} A={sweep_amp[i]:.2g}")
            ax_r.plot(t_roll[:n_sw], sweep["order"][:, i], color=color, ls=ls, lw=1.2)
            ax_z.plot(t_roll[:n_sw], sweep["z"][:, i], color=color, ls=ls, lw=1.2)
            lo, hi = ax_o.get_ylim()
            ax_o.fill_between(t_roll[:n_sw], lo, hi,
                              where=sweep["gate"][:, i] > 0.5, color=color,
                              alpha=0.08, zorder=-2)

        for ax in (ax_o, ax_r, ax_z):
            ax.axvspan(*EVENT_WINDOW, color="#ffe9e9", zorder=-3)
        ax_o.set_ylabel(f"{mech}\n" + r"$\langle\omega\rangle$", fontsize=8)
        if mi == 0:
            ax_o.set_title(r"mean frequency $\langle\omega\rangle$ (rad/s)", fontsize=10)
            ax_r.set_title("order parameter", fontsize=10)
            ax_z.set_title("slow variable z", fontsize=10)
            ax_o.legend(fontsize=6.5, loc="upper right", ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("recording time (s)")
    fig.suptitle("Best run per mechanism vs the unstimulated model (shading = stim on)",
                 fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"best_traces.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_charge_efficacy() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    markers = ["o", "s", "^", "D"]
    for mi, mech in enumerate(MECHANISMS):
        for pi, proto in enumerate(("open", "closed")):
            sel = np.where((sweep_mech == mi) & (sweep_proto == pi))[0]
            charge = np.array([sweep_stats[i]["charge"] for i in sel])
            ratio = np.array([sweep_stats[i]["burden_ratio"] for i in sel])
            grid = AMP_GRIDS[mech]
            color, _ls = PROTO_STYLE[proto]
            ax.scatter(charge / max(np.max(charge), 1e-12), ratio, s=34,
                       marker=markers[mi % len(markers)],
                       facecolor=color if proto == "closed" else "none",
                       edgecolor=color, lw=1.2,
                       label=f"{mech} ({proto})")
    ax.axhline(1.0 - BURDEN_CUT, color="0.3", lw=0.9, ls="-.")
    ax.set_xlabel("delivered charge / mechanism max (units differ per mechanism)")
    ax.set_ylabel("seizure burden / baseline")
    ax.set_title("Charge vs efficacy: filled = closed loop, open = open loop")
    ax.legend(fontsize=6.5, ncol=2, loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"charge_vs_efficacy.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_losses() -> None:
    for name, m in (("theta", model_theta), ("u", model_u),
                    ("v", model_v), ("h", model_h)):
        if getattr(m, "train_results_", None):
            fig, _ax = plot_loss_curves(m.train_results_,
                                        save=str(RESULT_DIR / f"loss_{name}"))
            plt.close(fig)


plot_q_treatment()
plot_dose_response()
plot_best_traces()
plot_charge_efficacy()
plot_losses()
print(f"[FIGS]  Saved {RESULT_DIR}", flush=True)
