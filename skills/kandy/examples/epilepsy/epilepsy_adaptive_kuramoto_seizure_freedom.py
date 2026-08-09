#!/usr/bin/env python3
"""Adaptive-Kuramoto seizure model with explicit omega, a constant column and a
slow variable -- fitted for autonomous replay, then swept for seizure freedom.

What this changes relative to ``epilepsy_adaptive_kuramoto_fit.py``
-------------------------------------------------------------------
That script fits

    dtheta_i/dt = F_i(kappa*sin d, kappa*cos d, sin d, cos d, kappa)
    dkappa_p/dt = G_p(sin d, cos d, kappa)

with no constant term, no explicit omega, and no slow variable, and it
reproduces the seizure by integrating

    dx/dt = f_KANDy(x) + q_data(t) + 20*(x_data(t) - x).

The last two terms are the problem for the question asked here.  Under a
lambda=20 nudge the trajectory tracks the measurement whatever the parameters
are, so a seizure-freedom sweep on that RHS returns "no change" for every knob:
you would be sweeping parameters of a model that is not driving the trajectory.
Seizure freedom is a statement about a system's OWN dynamics, so the forcing
and the nudge are both removed here and every rollout below is autonomous.

Three structural additions, all requested:

1. **Explicit omega_i and a constant column.**  The phase equation is restored
   to Kuramoto form,

       dtheta_i/dt = omega_i + g * Phi_i(theta, kappa, omega)

   with omega_i an exact passthrough and Phi the KAN, fitted to the residual
   dtheta_i/dt - omega_i.  The lift for Phi still carries a constant column and
   all six omega_i columns, so any residual frequency dependence or systematic
   offset is representable rather than smeared into the coupling splines.
   omega_i is itself a state, evolved through a bounded coordinate (below).

2. **kappa stays adaptive but becomes bounded.**  As before kappa_p is a
   pairwise log-amplitude envelope with its own equation, but it is evolved in
   a squashed coordinate so no rollout can push it outside the observed box.

3. **A slow variable z, and Epileptor relaxation for EVERY slow block.**  A
   constant-parameter oscillator model has exactly one regime and structurally
   cannot have an event, so z (the 4 s envelope of mean log amplitude) is added
   as a state.  More importantly, all three slow blocks take the relaxational
   form rather than free-running:

       dy/dt = (Y_target(rest of the state) - y) / tau_y

   with Y_target squashed into the observed box for y.  The reference model
   applies this to z alone and lets u free-run, and that is exactly what breaks
   in practice: a tanh box stops omega leaving its range but does not stop the
   inner coordinate u from running to +-inf, and once it does sech^2 -> 0 and
   omega is pinned at the wall -- bounded, but dynamically dead.  A small
   constant bias in a fitted derivative integrates into precisely that over a
   12 s rollout.  Measured here: with free-running u and v, 0 of 99 swept
   parameter cells produced a viable trajectory; with relaxation, all of them
   do, and the held-out gains for du, dv and dz go from -1.55/-0.12/+0.04 to
   positive.  Each tau_y is scanned, and scored on the RATE the rollout
   integrates rather than on the intermediate target, because y + tau*ydot
   changes with tau and an R2 on it always rewards the largest tau on the grid.

State eta = [theta (M) | u (M) | v (P) | z], with

    omega_i = OM_LO_i + (OM_HI_i - OM_LO_i) * (1 + tanh u_i)/2
    kappa_p = KP_LO_p + (KP_HI_p - KP_LO_p) * (1 + tanh v_p)/2

Bounded targets plus contraction means the integrator needs no clipping at all,
so a post-onset trajectory is dynamics and not a clamp.  ``sech2`` along every
rollout is recorded to prove the trajectory never sits on a boundary.

The four knobs
--------------
Because a KANDy model is A*Psi(phi(x)) with Psi separable, each fitted block is
a clean multiplicative or additive target:

    g        coupling gain            dtheta/dt = omega + g * Phi
    delta_z  z excitability offset    dz/dt = (h + delta_z - z)/tau_z
    delta_u  omega excitability       du/dt = (U_target + delta_u - u)/tau_u
    p        plasticity gain          dv/dt = p * (V_target - v)/tau_v

g=1, delta_z=0, delta_u=0, p=1 reproduces the fitted model exactly.  delta_z is
the analogue of the Epileptor's excitability parameter: it moves where the slow
variable wants to sit without touching any fast equation.

Seizure freedom is then a question about the (delta_z, g) plane: from a FIXED
pre-ictal initial condition, which parameter combinations fail to produce the
frequency excursion, while still producing a viable oscillating state?  A model
that goes quiet, saturates, or leaves the physiological band is not
seizure-free, it is broken, so every candidate is checked for viability before
it can be called free.  The reported operating point is the smallest normalized
move from the fitted parameters that abolishes the event.

delta_u is deliberately NOT in that plane.  It is the direct knob -- it pushes
down the target of the very equation that produces the excursion -- so letting
it win by default would reduce the result to "stop omega from rising and omega
does not rise".  It is scanned separately and used only as a labelled fallback
if the plane contains no seizure-free cell.

Regularization is not cosmetic here.  Unregularized, these blocks score far
below a zero-change baseline on a held-out time block (the coupling block:
-17.5), so lamb is chosen per block on a VALIDATION time window that is
separate from the test window.  KANDy's internal ``val_frac`` cannot do this
job -- it splits the fit rows at random, so its validation rows are interleaved
with training rows and share Savitzky-Golay support with them, and that kind of
split cannot see a temporal extrapolation failure.  If the selected lamb drives
the coupling term to ~0, the reported coupling/omega ratio says so and the g
axis of the sweep is flat by construction, which is a finding about the
recording rather than a property of the dynamics.

The sweep reports whichever question is actually answerable, and says which:

    fitted model seizes  -> ABOLITION, the question as posed: the smallest move
                            that removes the event.
    fitted model stable  -> INDUCTION: "which parameters give freedom?" is
                            vacuous when the fitted model is already free, so
                            the informative number becomes the DISTANCE TO THE
                            BOUNDARY -- the smallest move that produces an
                            event.  Reporting "100% of cells seizure-free"
                            without saying this would be true and completely
                            misleading.

Grids are two-sided for the same reason: a sweep that only lowers excitability
cannot locate a boundary it never crosses.

Viability is calibrated on the RECORDING, not on the pre-ictal baseline.  This
seizure takes <omega> from 30 to 60 rad/s, so a plausible-looking band of
0.5-1.5x the pre-ictal mean rejects the real event as non-viable and silently
discards every genuinely seizing cell before it can be counted.  The measured
trajectory is pushed through the criterion and the result is gated: a criterion
that rejects the ground truth is not a criterion.

What this can and cannot show
-----------------------------
The seizure is in the training data by request, so no claim of out-of-sample
seizure prediction is made.  What is testable, and is tested: whether the event
EMERGES autonomously from a pre-ictal state, whether it TERMINATES, whether a
frozen-parameter control stays quiet from the identical initial condition,
whether the model seizes from everywhere (specificity), and how the excursion
depends on the parameters.  Seizure freedom here is a property of a fitted
surrogate of one seizure in one recording; g, delta_z, delta_u and p are model
coefficients, not stimulation or drug parameters, and nothing here is a
treatment claim.

With a single event, the pre-ictal and post-ictal states are close in these
observables while only one of them leads into a seizure, so an autonomous model
on these coordinates is weakly identified for emergence no matter how it is
fitted.  Every gate below is reported whether it passes or fails; the failures
are the informative part.
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
RESULT_DIR = ensure_outdir(REPO_ROOT / "results" / "Epilepsy-Seizure-Freedom")

CHANNELS = os.environ.get("KANDY_SF_CHANNELS", "").strip()

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
GUARD_SECONDS = env_float("KANDY_SF_GUARD_SECONDS", Z_SLOW_SECONDS)

LAG_SECONDS = env_float("KANDY_SF_LAG_SECONDS", 0.25)
Z_LAG_SECONDS = env_float("KANDY_SF_Z_LAG_SECONDS", 2.0)
BOX_MARGIN = env_float("KANDY_SF_BOX_MARGIN", 0.35)
TAU_U_GRID = np.logspace(np.log10(0.05), np.log10(10.0), 24)
TAU_V_GRID = np.logspace(np.log10(0.05), np.log10(10.0), 24)
TAU_Z_GRID = np.logspace(np.log10(0.25), np.log10(40.0), 28)
LAMB_GRID = [float(v) for v in os.environ.get(
    "KANDY_SF_LAMB_GRID", "0.0,0.001,0.01,0.1").split(",")]

GRID = env_int("KANDY_SF_GRID", 5)
THETA_STEPS = env_int("KANDY_SF_THETA_STEPS", 60)
U_STEPS = env_int("KANDY_SF_U_STEPS", 80)
V_STEPS = env_int("KANDY_SF_V_STEPS", 60)
H_STEPS = env_int("KANDY_SF_H_STEPS", 80)
N_TRAIN_MAX = env_int("KANDY_SF_N_TRAIN_MAX", 4000)

ROLLOUT_START = env_float("KANDY_SF_ROLLOUT_START", -3.0)
ROLLOUT_SECONDS = env_float("KANDY_SF_ROLLOUT_SECONDS", 12.0)
N_GAIN = env_int("KANDY_SF_N_GAIN", 9)
N_DELTA = env_int("KANDY_SF_N_DELTA", 11)
N_PLAST = env_int("KANDY_SF_N_PLAST", 9)
N_DELTA_U = env_int("KANDY_SF_N_DELTA_U", 9)
N_IC = env_int("KANDY_SF_N_IC", 14)

# Grids are two-sided on purpose.  A one-sided sweep that only reduces
# excitability can answer "is the seizure abolished?" but not "where is the
# boundary?", and if the fitted model turns out to sit in the stable regime a
# one-sided sweep reports 100% freedom while never locating the transition it
# is supposed to be mapping.
GAIN_GRID = np.linspace(0.0, 3.0, N_GAIN)
PLAST_GRID = np.linspace(0.0, 3.0, N_PLAST)


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
ALL_LABELS = list(payload["labels"])

if CHANNELS:
    names = [c.strip() for c in CHANNELS.split(",") if c.strip()]
    missing = [n for n in names if n not in ALL_LABELS]
    if missing:
        raise ValueError(f"Unknown channels {missing}; recording has {ALL_LABELS}")
    CH = [ALL_LABELS.index(n) for n in names]
    LABELS = names
else:
    CH = list(range(len(ALL_LABELS)))
    LABELS = list(ALL_LABELS)

M = len(LABELS)
PAIRS = [(a, b) for a in range(M) for b in range(a + 1, M)]
P = len(PAIRS)

nyq = 0.5 * FS
bb, aa = butter(FILTER_ORDER, [BANDPASS_HZ[0] / nyq, BANDPASS_HZ[1] / nyq], btype="band")
X_FILT = filtfilt(bb, aa, X_ALL[:, CH], axis=0)
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

# Every slow target is a LAGGED INCREMENT, never an instantaneous derivative.
# d/dt of a 4 s-smoothed envelope is almost pure smoothing noise; over a lag
# matched to the variable's own timescale the signal sits well above it.  The
# same argument is why omega is fitted this way; z needs it at least as much.
DZ = (Z[Z_LAG:] - Z[:-Z_LAG]) / Z_LAG_SECONDS

print("[DATA]  Loaded epilepsy recording", flush=True)
print(f"[SEL]   {M} channels: {LABELS}; {P} pairs", flush=True)
print(
    f"[PRE]   windows theta={w_fast}, omega={w_om}, kappa={w_kp}, z={w_z}; "
    f"{N_ROW} rows {T[0]:.1f}..{T[-1]:.1f}s; increment lags "
    f"fast={LAG} ({LAG_SECONDS:.2f}s), slow={Z_LAG} ({Z_LAG_SECONDS:.2f}s)",
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
# 2. Bounded coordinates: omega and kappa can never leave their boxes
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
print(
    f"[BOX]   min sech^2 over data: omega={float((1 - np.tanh(U) ** 2).min()):.3f}, "
    f"kappa={float((1 - np.tanh(V) ** 2).min()):.3f} "
    "(0 would mean the box is too tight)",
    flush=True,
)

#
# Every slow variable relaxes; none free-runs.
#
# A tanh box stops omega leaving [OM_LO, OM_HI], but it does NOT stop the inner
# coordinate u from running to +-inf, and once it does sech^2 -> 0 and omega is
# pinned at the wall: bounded, but dynamically dead, which is the same failure
# as hard clipping wearing a different hat.  A free KAN output for du/dt has
# nothing pulling it back, and a small constant bias in a fitted derivative
# integrates into exactly that over a 12 s rollout.
#
# So all three slow blocks take the Epileptor form the reference model uses for
# z alone:
#
#     dy/dt = (Y_target(rest of the state) - y) / tau_y
#
# with Y_target squashed into the observed box.  That is a contraction toward a
# bounded target, so no rollout can saturate a coordinate, and offset is
# structural for omega and kappa in the same way it is for z.
#
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
# 3. Split: the seizure is in TRAIN by design; held-out block is post-ictal
# ---------------------------------------------------------------------------
#
# Three time blocks, not two.  Hyperparameters have to be chosen somewhere, and
# choosing them on the block the model is scored on would make that score
# meaningless.  KANDy's internal val_frac cannot do this job: it splits
# fit_rows at random, so its validation rows are interleaved with training rows
# and share Savitzky-Golay support with them.  That kind of validation cannot
# see the extrapolation failure that matters here, which is temporal.
#
rows = np.arange(N_ROW - MAX_LAG)
t_rows = T[: N_ROW - MAX_LAG]


def block(window):
    inside = (t_rows >= window[0]) & (t_rows <= window[1])
    near = (t_rows >= window[0] - GUARD_SECONDS) & (t_rows <= window[1] + GUARD_SECONDS)
    return rows[inside], near


val_rows, near_val = block(VAL_WINDOW)
test_rows, near_test = block(HELDOUT_WINDOW)
train_rows = rows[~(near_val | near_test)]
rng = np.random.default_rng(SEED)
fit_rows = (
    np.sort(rng.choice(train_rows, N_TRAIN_MAX, replace=False))
    if len(train_rows) > N_TRAIN_MAX
    else train_rows
)
print(
    f"[SPLIT] train={len(train_rows)} rows (seizure INCLUDED), "
    f"val={len(val_rows)} ({VAL_WINDOW[0]:.0f}..{VAL_WINDOW[1]:.0f}s) for lamb only, "
    f"test={len(test_rows)} ({HELDOUT_WINDOW[0]:.0f}..{HELDOUT_WINDOW[1]:.0f}s), "
    f"guard={GUARD_SECONDS:.0f}s",
    flush=True,
)


# ---------------------------------------------------------------------------
# 4. Lifts.  Cross-terms between DIFFERENT variables are explicit; powers of a
#    single variable are not (a spline edge already covers those).
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
    om = (squash_t(eta[:, U_S], OM_LO, OM_HI) - torch.as_tensor(om_center, dtype=eta.dtype)) / \
        torch.as_tensor(om_spread, dtype=eta.dtype)
    kp = (squash_t(eta[:, V_S], KP_LO, KP_HI) - torch.as_tensor(kp_center, dtype=eta.dtype)) / \
        torch.as_tensor(kp_spread, dtype=eta.dtype)
    R = torch.sqrt(
        torch.mean(torch.sin(th), dim=1) ** 2 + torch.mean(torch.cos(th), dim=1) ** 2 + 1e-12
    )
    return th, om, kp, eta[:, Z_S], R


# --- phase lift: the adaptive-Kuramoto lift PLUS a constant and omega_i ------
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


# --- omega lift: z enters as an explicit cross-term with every omega_i -------
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


# --- kappa lift: plasticity driven by phase relations, kappa itself and z ----
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


# --- h lift: FAST state only.  No z, which is what makes dz/dt a contraction.
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
    f"[LIFT]  theta={THETA_LIFT_DIM} (1 const + {M} omega + {5 * P} adaptive), "
    f"u={U_LIFT_DIM}, v={V_LIFT_DIM}, h={H_LIFT_DIM}",
    flush=True,
)


# ---------------------------------------------------------------------------
# 5. Fit.  Phase on the coupling residual; slow blocks on lagged increments.
# ---------------------------------------------------------------------------
lamb_choices: dict[str, dict] = {}
tau_choices: dict[str, dict] = {}


def fit_block(name, lift_np, lift_dim, lift_torch, steps, target, scale, score_fn):
    """Fit one block once per lamb and keep the model that wins on VAL_WINDOW.

    Selecting on a held-out time block is the whole point: unregularized KAN
    fits on this recording score far below a zero-change baseline off-block, so
    lamb is not cosmetic here, it is the difference between a model and a
    memorization of the training window.  score_fn always scores the RATE the
    rollout integrates, never the intermediate target, so the selected model is
    the one that is best at the job it is actually given.
    """
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
        f"[TRAIN] {name}: lamb sweep "
        + ", ".join(f"{r['lamb']:g}->{r['val_r2_mean']:+.3f}" for r in table)
        + f"  | selected lamb={best[2]:g}",
        flush=True,
    )
    return best[0]


def tau_scan(name, lift_vals, state, rate, c, r, grid):
    """Pick a relaxation timescale by how well (target - y)/tau predicts dy/dt.

    Scoring the rate rather than the target keeps the criterion comparable
    across tau: the target y + tau*ydot changes with tau, so an R2 on it
    silently switches question as tau grows and always rewards the largest
    value on the grid.
    """
    scores = []
    for tau in grid:
        tgt = state[: len(rate)] + tau * rate
        sol, *_ = np.linalg.lstsq(lift_vals[train_rows], np.atleast_2d(tgt[train_rows].T).T,
                                  rcond=None)
        pred = np.clip(lift_vals[val_rows] @ sol, c - r, c + r)
        pred_rate = (pred - np.atleast_2d(state[val_rows].T).T) / tau
        scores.append(float(np.nanmean(
            r2_per_output(np.atleast_2d(rate[val_rows].T).T, pred_rate)
        )))
    scores = np.array(scores)
    tau = float(grid[int(np.argmax(scores))])
    at_edge = bool(tau in (grid[0], grid[-1]))
    tau_choices[name] = {"tau": tau, "val_r2": float(scores.max()),
                         "on_grid_boundary": at_edge,
                         "grid": [float(v) for v in grid],
                         "scores": [float(v) for v in scores]}
    print(
        f"[TAU]   {name}: tau={tau:.2f}s over [{grid[0]:.2f}, {grid[-1]:.0f}]s "
        f"(validation R2 of the rate = {scores.max():+.3f})"
        + ("   [WARNING: on the grid boundary, so this timescale is not "
           "identified by the recording]" if at_edge else ""),
        flush=True,
    )
    return tau, scores


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

# How much coupling actually survived regularization?  If lamb drives the
# coupling term toward zero -- which is the honest outcome when a term has no
# held-out skill -- then dtheta/dt collapses to omega and the coupling-gain
# axis of the sweep is flat BY CONSTRUCTION.  That has to be visible, or a flat
# axis reads as "coupling does not matter dynamically" when it really means
# "this recording did not support fitting a coupling term at all".
coupling_rms = float(np.sqrt(np.mean(
    (model_theta.predict(ETA[train_rows]) * th_scale) ** 2)))
omega_rms = float(np.sqrt(np.mean(OMEGA[train_rows] ** 2)))
coupling_ratio = coupling_rms / max(omega_rms, 1e-12)
print(
    f"[COUPLE] rms coupling term = {coupling_rms:.3f} rad/s vs rms omega = "
    f"{omega_rms:.1f} rad/s (ratio {coupling_ratio:.2e})"
    + ("   [coupling regularized to ~0: the gain axis g will be flat]"
       if coupling_ratio < 1e-3 else ""),
    flush=True,
)

u_lift_vals = u_lift_np(ETA)
v_lift_vals = v_lift_np(ETA)
TAU_U, tau_u_scores = tau_scan("omega", u_lift_vals, U, DU_LAG, U_C, U_R, TAU_U_GRID)
TAU_V, tau_v_scores = tau_scan("kappa", v_lift_vals, V, DV_LAG, V_C, V_R, TAU_V_GRID)

U_INNER = to_inner(U[: len(DU_LAG)] + TAU_U * DU_LAG, U_C, U_R)
V_INNER = to_inner(V[: len(DV_LAG)] + TAU_V * DV_LAG, V_C, V_R)
u_scale = U_INNER[train_rows].std(axis=0) + 1e-12
v_scale = V_INNER[train_rows].std(axis=0) + 1e-12


def du_of(model, idx, delta=0.0):
    tgt = from_inner(model.predict(ETA[idx]) * u_scale, U_C, U_R, delta)
    return (tgt - U[idx]) / TAU_U


def dv_of(model, idx):
    tgt = from_inner(model.predict(ETA[idx]) * v_scale, V_C, V_R)
    return (tgt - V[idx]) / TAU_V


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

# tau by scan.  Score the quantity the rollout actually integrates, dz/dt, not
# h itself: h = z + tau*zdot changes with tau, so an R2 on h is not comparable
# across the grid (large tau makes h ~ tau*zdot and the score silently switches
# to a different question).  dz/dt is the same target for every tau.
fast_lin = h_lift_np(ETA)
TAU_Z, tau_z_scores = tau_scan("z", fast_lin, Z, DZ, Z_C, Z_R, TAU_Z_GRID)

H_DATA = Z[: len(DZ)] + TAU_Z * DZ
H_INNER = to_inner(H_DATA, Z_C, Z_R)
h_scale = float(H_INNER[train_rows].std() + 1e-12)


def dz_of(model, idx, delta=0.0):
    inner = (model.predict(ETA[idx]) * h_scale).ravel()
    return (from_inner(inner, Z_C, Z_R, delta) - Z[idx]) / TAU_Z


model_h = fit_block(
    "epileptor_h", h_lift_np, H_LIFT_DIM, h_lift_torch, H_STEPS,
    H_INNER[:, None], h_scale,
    lambda m: r2_per_output(DZ[val_rows, None], dz_of(m, val_rows)[:, None])[0],
)


# ---------------------------------------------------------------------------
# 6. Held-out skill against the baseline each block actually has to beat
# ---------------------------------------------------------------------------
pred_theta = model_theta.predict(ETA[test_rows]) * th_scale
pred_du = du_of(model_u, test_rows)
pred_dv = dv_of(model_v, test_rows)
pred_dz = dz_of(model_h, test_rows)
zero_t = np.zeros_like(THETA_RESID[test_rows])

scores = {
    "theta_coupling_kandy": r2_per_output(THETA_RESID[test_rows], pred_theta),
    "theta_omega_only_baseline": r2_per_output(THETA_RESID[test_rows], zero_t),
    "du_kandy": r2_per_output(DU_LAG[test_rows], pred_du),
    "du_zero_baseline": r2_per_output(DU_LAG[test_rows], np.zeros_like(DU_LAG[test_rows])),
    "dv_kandy": r2_per_output(DV_LAG[test_rows], pred_dv),
    "dv_zero_baseline": r2_per_output(DV_LAG[test_rows], np.zeros_like(DV_LAG[test_rows])),
    "dz_epileptor": r2_per_output(DZ[test_rows, None], pred_dz[:, None]),
    "dz_zero_baseline": r2_per_output(DZ[test_rows, None], np.zeros((len(test_rows), 1))),
}
print("\n[EVAL]  held-out skill (post-ictal block, guard-banded)", flush=True)
for k, v in scores.items():
    print(f"        {k:>26}: mean={np.nanmean(v):+.4f}", flush=True)

theta_gain = float(
    np.nanmean(scores["theta_coupling_kandy"]) - np.nanmean(scores["theta_omega_only_baseline"])
)
du_gain = float(np.nanmean(scores["du_kandy"]) - np.nanmean(scores["du_zero_baseline"]))
dv_gain = float(np.nanmean(scores["dv_kandy"]) - np.nanmean(scores["dv_zero_baseline"]))
dz_gain = float(np.nanmean(scores["dz_epileptor"]) - np.nanmean(scores["dz_zero_baseline"]))
print(
    f"        gain over baseline: coupling={theta_gain:+.4f}, du={du_gain:+.4f}, "
    f"dv={dv_gain:+.4f}, dz={dz_gain:+.4f}",
    flush=True,
)


# ---------------------------------------------------------------------------
# 7. Autonomous, parameterized, batched rollout.  No clipping anywhere.
# ---------------------------------------------------------------------------
def rhs(eta, gain, delta_z, delta_u, plast, frozen=False):
    """Every slow block is a contraction toward a squashed target, so nothing
    here can run away and nothing is clipped."""
    out = np.zeros_like(eta)
    om = squash(eta[:, U_S], OM_LO, OM_HI)
    out[:, TH_S] = om + gain[:, None] * (model_theta.predict(eta) * th_scale)
    if not frozen:
        u_tgt = from_inner(model_u.predict(eta) * u_scale, U_C, U_R, delta_u[:, None])
        out[:, U_S] = (u_tgt - eta[:, U_S]) / TAU_U
        v_tgt = from_inner(model_v.predict(eta) * v_scale, V_C, V_R)
        out[:, V_S] = plast[:, None] * (v_tgt - eta[:, V_S]) / TAU_V
        h = from_inner((model_h.predict(eta) * h_scale).ravel(), Z_C, Z_R, delta_z)
        out[:, Z_S] = (h - eta[:, Z_S]) / TAU_Z
    return out


def amp_from_kappa(kappa_phys: np.ndarray) -> np.ndarray:
    """Pair log-amplitudes -> per-channel amplitude (pseudo-inverse of the mean)."""
    pair_avg = np.zeros((P, M))
    for r, (a, b) in enumerate(PAIRS):
        pair_avg[r, a] = 0.5
        pair_avg[r, b] = 0.5
    pair_log = kappa_phys * kappa_std + kappa_mean
    chan_log = np.clip(pair_log @ np.linalg.pinv(pair_avg).T, -5.0, 5.0)
    return np.exp(chan_log * amp_std + amp_mean)


def rollout(eta0, steps, gain, delta_z, delta_u=0.0, plast=1.0,
            frozen=False, keep_full=False):
    """RK4.  omega and kappa are bounded by their tanh parameterization and
    relax toward bounded targets, so the integrator only guards against
    non-finite values -- it never clips."""
    eta = np.atleast_2d(eta0).astype(float).copy()
    n = len(eta)
    gain = np.broadcast_to(np.atleast_1d(gain), (n,)).astype(float)
    delta_z = np.broadcast_to(np.atleast_1d(delta_z), (n,)).astype(float)
    delta_u = np.broadcast_to(np.atleast_1d(delta_u), (n,)).astype(float)
    plast = np.broadcast_to(np.atleast_1d(plast), (n,)).astype(float)

    mo = np.empty((steps + 1, n))
    Rt = np.empty((steps + 1, n))
    mk = np.empty((steps + 1, n))
    zt = np.empty((steps + 1, n))
    sech = np.empty((steps + 1, n))
    full = np.empty((steps + 1, n, N_ETA)) if keep_full else None

    def record(idx, e):
        om = squash(e[:, U_S], OM_LO, OM_HI)
        kp = squash(e[:, V_S], KP_LO, KP_HI)
        mo[idx] = om.mean(axis=1)
        Rt[idx] = np.abs(np.mean(np.exp(1j * e[:, TH_S]), axis=1))
        mk[idx] = kp.mean(axis=1)
        zt[idx] = e[:, Z_S]
        sech[idx] = np.minimum(
            (1 - np.tanh(e[:, U_S]) ** 2).min(axis=1),
            (1 - np.tanh(e[:, V_S]) ** 2).min(axis=1),
        )
        if keep_full:
            full[idx] = e

    record(0, eta)
    done = steps
    for s in range(steps):
        k1 = rhs(eta, gain, delta_z, delta_u, plast, frozen)
        k2 = rhs(eta + 0.5 * DT * k1, gain, delta_z, delta_u, plast, frozen)
        k3 = rhs(eta + 0.5 * DT * k2, gain, delta_z, delta_u, plast, frozen)
        k4 = rhs(eta + DT * k3, gain, delta_z, delta_u, plast, frozen)
        eta = eta + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(eta).all():
            done = s
            break
        record(s + 1, eta)
    out = {"mean_omega": mo[: done + 1], "order": Rt[: done + 1],
           "mean_kappa": mk[: done + 1], "z": zt[: done + 1],
           "sech2": sech[: done + 1], "steps": done}
    if keep_full:
        out["eta"] = full[: done + 1]
    return out


STEPS = int(round(ROLLOUT_SECONDS / DT))
NB = int(round(1.0 / DT))          # 1 s baseline
TAIL = int(round(2.0 / DT))        # last 2 s for the offset test
i0 = int(np.argmin(np.abs(T - ROLLOUT_START)))
ref_slice = slice(i0, i0 + STEPS + 1)
t_roll = T[ref_slice]
mo_ref = OMEGA[ref_slice].mean(axis=1)
R_ref = np.abs(np.mean(np.exp(1j * THETA[ref_slice]), axis=1))
mk_ref = KAPPA[ref_slice].mean(axis=1)
z_ref = Z[ref_slice]
peak_ref = float(mo_ref.max() - mo_ref[:NB].mean())
R_rise_ref = float(R_ref.max() - R_ref[:NB].mean())
recover_ref = 1.0 - (mo_ref[-TAIL:].mean() - mo_ref[:NB].mean()) / max(peak_ref, 1e-9)


def summarize(res, idx=0):
    mo, R, mk, z = res["mean_omega"][:, idx], res["order"][:, idx], \
        res["mean_kappa"][:, idx], res["z"][:, idx]
    base = mo[:NB].mean()
    rise = float(mo.max() - base)
    tail = mo[-TAIL:].mean() if len(mo) >= TAIL else mo[-1:].mean()
    return {
        "omega_rise": rise,
        "omega_baseline": float(base),
        "omega_mean": float(mo.mean()),
        "order_rise": float(R.max() - R[:NB].mean()),
        "kappa_mean": float(mk.mean()),
        "z_end": float(z[-1]),
        "recovery": float(1.0 - (tail - base) / max(rise, 1e-9)),
        "min_sech2": float(res["sech2"][:, idx].min()),
        "complete": bool(len(mo) == STEPS + 1),
    }


print(
    f"\n[ROLLOUT] autonomous RK4 from t={T[i0]:.1f}s for {ROLLOUT_SECONDS:.0f}s, "
    "no forcing, no nudging, no clipping",
    flush=True,
)
main = rollout(ETA[i0], STEPS, 1.0, 0.0, 0.0, 1.0, keep_full=True)
froz = rollout(ETA[i0], STEPS, 1.0, 0.0, 0.0, 1.0, frozen=True, keep_full=True)
fit_stats = summarize(main)
froz_stats = summarize(froz)
print(
    f"          mean-omega rise: data={peak_ref:+.1f}, model={fit_stats['omega_rise']:+.1f}, "
    f"frozen-control={froz_stats['omega_rise']:+.1f} rad/s",
    flush=True,
)
print(
    f"          offset recovery: data={recover_ref:.2f}, model={fit_stats['recovery']:.2f} "
    "(1.0 = fully returned to baseline)",
    flush=True,
)
print(f"          min sech^2 along rollout = {fit_stats['min_sech2']:.4f} "
      "(> 0 means the trajectory never sat on a boundary)", flush=True)

EXC_THRESHOLD = 0.5 * abs(peak_ref)
AMP_FLOOR = float(np.percentile(KAPPA.mean(axis=1), 5))

#
# The viability band is calibrated from the RECORDING, not from the pre-ictal
# baseline.  Calibrating it on the pre-ictal state is a trap: this seizure
# takes <omega> from 30 to 60 rad/s, so a band of 0.5-1.5x the pre-ictal mean
# rejects the real event as "non-viable", and every genuinely seizing parameter
# cell gets silently discarded before it can be counted.  A criterion that
# would reject the ground truth is not a criterion, so the observed trajectory
# is pushed through it below as a calibration check.
#
OM_BAND = (0.75 * float(np.min(OMEGA.mean(axis=1))),
           1.25 * float(np.max(OMEGA.mean(axis=1))))


def is_viable(s: dict) -> bool:
    """Seizure-free must mean 'still a working oscillator', not 'the model
    died'.  Viable = completed, never sat on a coordinate boundary, stayed
    inside the frequency band the recording itself spans, and did not let the
    amplitude proxy collapse below the 5th percentile of the recording."""
    return bool(
        s["complete"]
        and s["min_sech2"] > 1e-3
        and OM_BAND[0] <= s["omega_mean"] <= OM_BAND[1]
        and s["kappa_mean"] >= AMP_FLOOR
    )


data_stats = {"complete": True, "min_sech2": 1.0,
              "omega_mean": float(mo_ref.mean()), "kappa_mean": float(mk_ref.mean())}
data_viable = is_viable(data_stats)
print(
    f"[VIABLE] band {OM_BAND[0]:.1f}..{OM_BAND[1]:.1f} rad/s, amplitude floor "
    f"{AMP_FLOOR:+.2f}; the MEASURED seizure trajectory scores viable="
    f"{data_viable}"
    + ("" if data_viable else "   [CALIBRATION ERROR: the criterion rejects the "
                              "real event, so no cell can ever count as seizing]"),
    flush=True,
)


# ---------------------------------------------------------------------------
# 8. The (delta, gain) sweep -- where does the seizure stop happening?
# ---------------------------------------------------------------------------
DELTA_GRID = np.linspace(-3.0, 3.0, N_DELTA) * float(np.std(Z))
DELTA_U_GRID = np.linspace(-2.0, 3.0, N_DELTA_U) * float(np.mean(U_R))
GG, DD = np.meshgrid(GAIN_GRID, DELTA_GRID, indexing="ij")
N_PLANE = GG.size
sweep_gain = np.concatenate([GG.ravel(), np.ones(N_PLAST), np.ones(N_DELTA_U)])
sweep_dz = np.concatenate([DD.ravel(), np.zeros(N_PLAST), np.zeros(N_DELTA_U)])
sweep_du = np.concatenate([np.zeros(N_PLANE), np.zeros(N_PLAST), DELTA_U_GRID])
sweep_plast = np.concatenate([np.ones(N_PLANE), PLAST_GRID, np.ones(N_DELTA_U)])

print(
    f"\n[SWEEP] {N_GAIN} coupling gains x {N_DELTA} z-excitability offsets, "
    f"+ {N_PLAST} plasticity gains, + {N_DELTA_U} omega-excitability offsets "
    f"= {len(sweep_gain)} autonomous rollouts from the same pre-ictal state",
    flush=True,
)
sweep = rollout(np.tile(ETA[i0], (len(sweep_gain), 1)), STEPS,
                sweep_gain, sweep_dz, sweep_du, sweep_plast)
sweep_stats = [summarize(sweep, i) for i in range(len(sweep_gain))]

rise_map = np.array([s["omega_rise"] for s in sweep_stats[:N_PLANE]]).reshape(GG.shape)
order_map = np.array([s["order_rise"] for s in sweep_stats[:N_PLANE]]).reshape(GG.shape)
viable_map = np.array([is_viable(s) for s in sweep_stats[:N_PLANE]]).reshape(GG.shape)
free_map = viable_map & (rise_map < EXC_THRESHOLD)
plast_stats = sweep_stats[N_PLANE : N_PLANE + N_PLAST]
du_stats = sweep_stats[N_PLANE + N_PLAST:]

#
# Which question the sweep is answering depends on where the FITTED model sits,
# and that is not knowable in advance.
#
#   fitted seizes      -> ABOLITION: find the smallest move that removes the
#                         event.  This is the question as originally posed.
#   fitted is stable   -> INDUCTION: the fitted model is already seizure-free,
#                         so "which parameters give freedom?" is vacuous and
#                         the informative quantity is the DISTANCE TO THE
#                         BOUNDARY -- the smallest move that produces an event.
#                         Reporting 100% freedom without saying this would be
#                         true and completely misleading.
#
# Either way the search runs over the (delta_z, gain) plane first: those act on
# omega only indirectly, via the slow variable and via synchrony.  delta_u is
# the direct knob on the equation that produces the excursion, so it is a
# labelled fallback rather than a default winner -- "stop omega from rising and
# omega does not rise" is not a finding.
#
fitted_seizes = bool(fit_stats["omega_rise"] >= EXC_THRESHOLD)
MODE = "abolition" if fitted_seizes else "induction"
target_map = free_map if fitted_seizes else (viable_map & (rise_map >= EXC_THRESHOLD))

g_span = max(GAIN_GRID.max() - GAIN_GRID.min(), 1e-9)
d_span = max(DELTA_GRID.max() - DELTA_GRID.min(), 1e-9)
dist = np.sqrt(((GG - 1.0) / g_span) ** 2 + (DD / d_span) ** 2)
dist_masked = np.where(target_map, dist, np.inf)
found_target = bool(np.isfinite(dist_masked).any())
DU_ALT = 0.0
if found_target:
    gi, di = np.unravel_index(np.argmin(dist_masked), dist.shape)
    G_ALT, D_ALT = float(GG[gi, di]), float(DD[gi, di])
    route = "coupling/z-excitability plane"
else:
    du_ok = [(abs(DELTA_U_GRID[i]), i) for i, s in enumerate(du_stats)
             if is_viable(s) and ((s["omega_rise"] < EXC_THRESHOLD) == fitted_seizes)]
    if du_ok:
        i_du = min(du_ok)[1]
        G_ALT, D_ALT, DU_ALT = 1.0, 0.0, float(DELTA_U_GRID[i_du])
        found_target, route = True, "omega-excitability fallback (direct knob)"
    else:
        pick = np.argmin(np.where(viable_map, rise_map, np.inf)) if fitted_seizes else \
            np.argmax(np.where(viable_map, rise_map, -np.inf))
        gi, di = np.unravel_index(pick, rise_map.shape)
        G_ALT, D_ALT = float(GG[gi, di]), float(DD[gi, di])
        route = "no qualifying cell in the swept range - showing the extreme viable cell"

print(
    f"        fitted point (g=1, delta=0): rise={fit_stats['omega_rise']:+.1f} rad/s "
    f"vs seizure threshold {EXC_THRESHOLD:.1f}  ->  fitted model "
    f"{'SEIZES' if fitted_seizes else 'DOES NOT SEIZE'}, mode={MODE}",
    flush=True,
)
print(
    f"        plane: seizure-free & viable {int(free_map.sum())}/{free_map.size}, "
    f"viable {int(viable_map.sum())}/{viable_map.size}, "
    f"seizing & viable {int((viable_map & (rise_map >= EXC_THRESHOLD)).sum())}/{free_map.size}; "
    f"omega-offset scan: {sum(is_viable(s) and s['omega_rise'] >= EXC_THRESHOLD for s in du_stats)}"
    f"/{len(du_stats)} seizing",
    flush=True,
)
print(
    f"        smallest move that {'abolishes' if fitted_seizes else 'induces'} the event, "
    f"via {route} -> gain={G_ALT:.2f}, delta_z={D_ALT:+.3f}, delta_u={DU_ALT:+.3f}",
    flush=True,
)

alt = rollout(ETA[i0], STEPS, G_ALT, D_ALT, DU_ALT, 1.0, keep_full=True)
alt_stats = summarize(alt)
print(
    f"[ALT]   contrast replay: rise={alt_stats['omega_rise']:+.1f} rad/s "
    f"(vs {fit_stats['omega_rise']:+.1f} fitted), order rise="
    f"{alt_stats['order_rise']:+.3f} (vs {fit_stats['order_rise']:+.3f}), "
    f"viable={is_viable(alt_stats)}",
    flush=True,
)

# Backwards-compatible aliases: whichever rollout is the seizure-FREE one.



# ---------------------------------------------------------------------------
# 9. Does freedom hold from other states, and does the fit seize from all of
#    them?  (A model that seizes from everywhere has learned nothing.)
# ---------------------------------------------------------------------------
ic_times = np.linspace(float(T[0]) + 1.0, float(T[-1]) - ROLLOUT_SECONDS - 1.0, N_IC)
ic_idx = np.array([int(np.argmin(np.abs(T - s))) for s in ic_times])
ic_eta = np.vstack([ETA[ic_idx], ETA[ic_idx]])
ic_gain = np.concatenate([np.ones(N_IC), np.full(N_IC, G_ALT)])
ic_dz = np.concatenate([np.zeros(N_IC), np.full(N_IC, D_ALT)])
ic_du = np.concatenate([np.zeros(N_IC), np.full(N_IC, DU_ALT)])
print(f"\n[IC]    {N_IC} start times x 2 parameter sets", flush=True)
ic_res = rollout(ic_eta, STEPS, ic_gain, ic_dz, ic_du, 1.0)
ic_stats = [summarize(ic_res, i) for i in range(2 * N_IC)]
ic_rise_fit = np.array([s["omega_rise"] for s in ic_stats[:N_IC]])
ic_rise_alt = np.array([s["omega_rise"] for s in ic_stats[N_IC:]])
is_pre = ic_times < EVENT_WINDOW[0]
is_post = ic_times > EVENT_WINDOW[1]
rate_pre = float(np.mean(ic_rise_fit[is_pre] > EXC_THRESHOLD)) if is_pre.any() else float("nan")
rate_post = float(np.mean(ic_rise_fit[is_post] > EXC_THRESHOLD)) if is_post.any() else float("nan")
rate_alt = float(np.mean(ic_rise_alt > EXC_THRESHOLD))
print(
    f"        excursion rate at fitted params: pre-ictal={rate_pre:.2f}, "
    f"post-ictal={rate_post:.2f}; at contrast params (all starts)={rate_alt:.2f}",
    flush=True,
)


# ---------------------------------------------------------------------------
# 10. Gates
# ---------------------------------------------------------------------------
gates = {
    "coupling_beats_omega_only": {"required": 0.02, "achieved": theta_gain,
                                  "passed": theta_gain >= 0.02},
    "du_beats_zero_change": {"required": 0.02, "achieved": du_gain, "passed": du_gain >= 0.02},
    "dv_beats_zero_change": {"required": 0.02, "achieved": dv_gain, "passed": dv_gain >= 0.02},
    "dz_beats_zero_change": {"required": 0.02, "achieved": dz_gain, "passed": dz_gain >= 0.02},
    "onset_emerges_from_pre_ictal_IC": {
        "required": f">= 50% of observed rise ({0.5 * peak_ref:.1f} rad/s)",
        "achieved": fit_stats["omega_rise"],
        "passed": bool(fit_stats["omega_rise"] >= 0.5 * peak_ref)},
    "frozen_control_shows_no_event": {
        "required": f"<= 20% of observed rise ({0.2 * peak_ref:.1f} rad/s)",
        "achieved": froz_stats["omega_rise"],
        "passed": bool(froz_stats["omega_rise"] <= 0.2 * peak_ref)},
    "offset_occurs": {"required": "recovery >= 0.5", "achieved": fit_stats["recovery"],
                      "passed": bool(fit_stats["recovery"] >= 0.5)},
    "specific_to_pre_ictal_states": {
        "required": "post-ictal excursion rate <= 0.5 * pre-ictal rate",
        "achieved": [rate_pre, rate_post],
        "passed": bool(rate_post <= 0.5 * rate_pre) if rate_pre > 0 else False},
    "no_clipping_used": {"required": "min sech^2 > 0 (bounded by construction)",
                         "achieved": fit_stats["min_sech2"],
                         "passed": bool(fit_stats["min_sech2"] > 1e-3)},
    "seizure_free_region_exists": {"required": ">= 1 viable cell below threshold",
                                   "achieved": int(free_map.sum()),
                                   "passed": bool(free_map.any())},
    "viability_criterion_admits_the_real_seizure": {
        "required": "the measured trajectory scores viable",
        "achieved": data_viable, "passed": bool(data_viable)},
    "seizure_regime_exists_in_swept_range": {
        "required": ">= 1 viable cell at or above threshold",
        "achieved": int((viable_map & (rise_map >= EXC_THRESHOLD)).sum()),
        "passed": bool((viable_map & (rise_map >= EXC_THRESHOLD)).any())},
    "contrast_replay_separates_the_regimes": {
        "required": "the two replays land on opposite sides of the threshold",
        "achieved": [fit_stats["omega_rise"], alt_stats["omega_rise"], EXC_THRESHOLD],
        "passed": bool(found_target and is_viable(alt_stats)
                       and ((alt_stats["omega_rise"] >= EXC_THRESHOLD)
                            != (fit_stats["omega_rise"] >= EXC_THRESHOLD)))},
}
passed = print_check(
    all(g["passed"] for g in gates.values()),
    "seizure emerges and terminates autonomously, and the sweep locates a "
    "boundary between seizing and seizure-free parameters",
)
for name, g in gates.items():
    print(f"        {'PASS' if g['passed'] else 'FAIL'}  {name}: {g['achieved']}", flush=True)


# ---------------------------------------------------------------------------
# 11. Equation printout, metrics, arrays
# ---------------------------------------------------------------------------
print("\n[EQUATION] fitted system (g=1, delta=0, p=1 reproduces the fit)")
print("           dtheta_i/dt = omega_i(u_i) + g * Phi_i(1, omega, {k*s,k*c,s,c,k})")
print(f"           du_i/dt     = (U_i(1,z,R,R*z,<k>,<k>*z,om,om*z) + d_u - u_i)/{TAU_U:.2f}")
print(f"           dv_p/dt     = p * (V_p(1,z,R,R*z,{{s,c,k,k*z}}) - v_p)/{TAU_V:.2f}")
print(f"           dz/dt       = (h(fast) + d_z - z)/{TAU_Z:.2f}   [h has no z: contraction]")
print(f"           omega_i in [{OM_LO.min():.0f}, {OM_HI.max():.0f}] rad/s by construction")

metrics = {
    "task": "adaptive-Kuramoto seizure model with explicit omega, constant column "
            "and a slow variable; autonomous replay and seizure-freedom sweep",
    "channels": LABELS,
    "note": "seizure is IN the training data by design; this tests emergence, "
            "offset, specificity and parameter dependence, not out-of-sample "
            "seizure prediction. g and delta are model coefficients, not "
            "clinical interventions.",
    "windows": {"event": list(EVENT_WINDOW), "pre_ictal": list(PRE_ICTAL),
                "post_ictal": list(POST_ICTAL), "validation": list(VAL_WINDOW),
                "heldout": list(HELDOUT_WINDOW)},
    "regularization": {
        "grid": LAMB_GRID,
        "selected_on": "validation time block, never the test block",
        "per_block": lamb_choices,
    },
    "structure": {
        "state": "[theta(M) | u(M)->omega | v(P)->kappa | z]",
        "eta_dim": N_ETA,
        "lift_dims": {"theta": THETA_LIFT_DIM, "u": U_LIFT_DIM,
                      "v": V_LIFT_DIM, "h": H_LIFT_DIM},
        "phase_form": "dtheta_i/dt = omega_i + g*Phi_i, Phi fitted to the residual",
        "constant_column": True,
        "explicit_omega_columns": M,
        "increment_lag_seconds": LAG_SECONDS,
        "z_increment_lag_seconds": Z_LAG_SECONDS,
        "omega_box": [[float(a), float(b)] for a, b in zip(OM_LO, OM_HI)],
        "kappa_box": [[float(a), float(b)] for a, b in zip(KP_LO, KP_HI)],
        "z_box": [Z_LO, Z_HI],
        "relaxation": "every slow block is dy/dt = (squashed target - y)/tau",
        "tau_scans": tau_choices,
    },
    "heldout_r2_mean": {k: float(np.nanmean(v)) for k, v in scores.items()},
    "gains_over_baseline": {"coupling": theta_gain, "du": du_gain,
                            "dv": dv_gain, "dz": dz_gain},
    "coupling_magnitude": {"rms_coupling_rad_s": coupling_rms,
                           "rms_omega_rad_s": omega_rms,
                           "ratio": coupling_ratio},
    "replay": {
        "start_time": float(T[i0]), "seconds": ROLLOUT_SECONDS,
        "data_rise": peak_ref, "data_recovery": float(recover_ref),
        "data_order_rise": R_rise_ref,
        "fitted": fit_stats, "frozen_control": froz_stats,
        "contrast": alt_stats, "mode": MODE,
        "fitted_model_seizes": fitted_seizes,
    },
    "sweep": {
        "gain_grid": [float(v) for v in GAIN_GRID],
        "delta_z_grid": [float(v) for v in DELTA_GRID],
        "delta_u_grid": [float(v) for v in DELTA_U_GRID],
        "plasticity_grid": [float(v) for v in PLAST_GRID],
        "excursion_threshold": EXC_THRESHOLD,
        "omega_rise_map": rise_map,
        "order_rise_map": order_map,
        "viable_map": viable_map.astype(int),
        "seizure_free_map": free_map.astype(int),
        "n_seizure_free_viable": int(free_map.sum()),
        "plasticity_scan_rise": [s["omega_rise"] for s in plast_stats],
        "delta_u_scan_rise": [s["omega_rise"] for s in du_stats],
        "delta_u_scan_viable": [is_viable(s) for s in du_stats],
        "mode": MODE,
        "smallest_move": {"gain": G_ALT, "delta_z": D_ALT, "delta_u": DU_ALT,
                          "route": route, "found": found_target,
                          "direction": "abolishes the event" if fitted_seizes
                          else "induces an event (distance to the boundary)"},
    },
    "initial_condition_scan": {
        "times": [float(v) for v in ic_times],
        "rise_fitted": [float(v) for v in ic_rise_fit],
        "rise_contrast": [float(v) for v in ic_rise_alt],
        "rate_pre_ictal": rate_pre, "rate_post_ictal": rate_post,
        "rate_at_contrast_params": rate_alt,
    },
    "viability": {"omega_band": list(OM_BAND), "amplitude_floor": AMP_FLOOR,
                  "measured_seizure_scores_viable": data_viable,
                  "note": "calibrated on the recording; the measured trajectory "
                          "must pass or seizing cells cannot be counted"},
    "gates": gates,
    "passed": passed,
}
save_metrics(RESULT_DIR / "metrics.json", metrics)

torch.save(
    {
        "theta_state_dict": model_theta.model_.state_dict(),
        "u_state_dict": model_u.model_.state_dict(),
        "v_state_dict": model_v.model_.state_dict(),
        "h_state_dict": model_h.model_.state_dict(),
        "scales": {"theta": th_scale, "u": u_scale, "v": v_scale, "h": h_scale},
        "boxes": {"om": [OM_LO, OM_HI], "kp": [KP_LO, KP_HI],
                  "u": [U_LO, U_HI], "v": [V_LO, V_HI], "z": [Z_LO, Z_HI]},
        "taus": {"u": TAU_U, "v": TAU_V, "z": TAU_Z},
        "labels": LABELS,
        "pairs": PAIRS,
        "metadata": metrics["structure"],
    },
    RESULT_DIR / "fitted_seizure_freedom_state.pt",
)

sig_fit = amp_from_kappa(squash(main["eta"][:, 0, V_S], KP_LO, KP_HI)) * \
    np.cos(main["eta"][:, 0, TH_S])
sig_alt = amp_from_kappa(squash(alt["eta"][:, 0, V_S], KP_LO, KP_HI)) * \
    np.cos(alt["eta"][:, 0, TH_S])
sig_ref = SIGNAL[i0 : i0 + len(sig_fit)]

np.savez_compressed(
    RESULT_DIR / "seizure_freedom_arrays.npz",
    t=T, labels=np.asarray(LABELS), signal=SIGNAL, theta=THETA, omega=OMEGA,
    kappa=KAPPA, u=U, v=V, z=Z, dz=DZ, h_data=H_DATA,
    t_rollout=t_roll, mo_ref=mo_ref, R_ref=R_ref, mk_ref=mk_ref, z_ref=z_ref,
    mo_fit=main["mean_omega"][:, 0], R_fit=main["order"][:, 0],
    mk_fit=main["mean_kappa"][:, 0], z_fit=main["z"][:, 0],
    mo_frozen=froz["mean_omega"][:, 0],
    mo_alt=alt["mean_omega"][:, 0], R_alt=alt["order"][:, 0],
    mk_alt=alt["mean_kappa"][:, 0], z_alt=alt["z"][:, 0],
    signal_ref=sig_ref, signal_fit=sig_fit, signal_alt=sig_alt,
    gain_grid=GAIN_GRID, delta_grid=DELTA_GRID, delta_u_grid=DELTA_U_GRID,
    plast_grid=PLAST_GRID,
    rise_map=rise_map, order_map=order_map, viable_map=viable_map,
    free_map=free_map, ic_times=ic_times,
    ic_rise_fit=ic_rise_fit, ic_rise_alt=ic_rise_alt,
    tau_u_grid=TAU_U_GRID, tau_u_scores=tau_u_scores,
    tau_v_grid=TAU_V_GRID, tau_v_scores=tau_v_scores,
    tau_z_grid=TAU_Z_GRID, tau_z_scores=tau_z_scores,
)


# ---------------------------------------------------------------------------
# 12. Figures
# ---------------------------------------------------------------------------
use_pub_style()
n_roll = len(main["mean_omega"])
t_fit = t_roll[:n_roll]
t_alt = t_roll[: len(alt["mean_omega"])]
t_froz = t_roll[: len(froz["mean_omega"])]


def plot_state() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for ax in axes:
        ax.axvspan(*EVENT_WINDOW, color="#f4a261", alpha=0.20, lw=0)
    for i, lab in enumerate(LABELS):
        axes[0].plot(T, OMEGA[:, i], lw=0.8, label=lab)
    axes[0].set_ylabel(r"$\omega_i$ (rad/s)")
    axes[0].legend(ncol=3, fontsize=7)
    axes[0].set_title("Measured state: the seizure is the frequency excursion")
    axes[1].plot(T, np.abs(np.mean(np.exp(1j * THETA), axis=1)), color="#2ca02c", lw=0.9,
                 label="order parameter R")
    axes[1].plot(T, KAPPA.mean(axis=1), color="#9467bd", lw=0.9, label=r"mean $\kappa$")
    axes[1].legend(fontsize=8)
    axes[1].set_ylabel("sync / coupling")
    axes[2].plot(T, Z, color="#1f77b4", lw=1.2, label="z")
    axes[2].plot(T[: len(H_DATA)], H_DATA, color="#2ca02c", lw=0.8, alpha=0.8,
                 label=rf"$h=z+\tau\dot z$ ($\tau$={TAU_Z:.1f}s)")
    axes[2].legend(fontsize=8)
    axes[2].set_ylabel("slow z")
    axes[2].set_xlabel("recording time (s)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"state_variables.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_replay() -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for ax in axes:
        ax.axvspan(*EVENT_WINDOW, color="#f4a261", alpha=0.20, lw=0)
    axes[0].plot(t_roll, mo_ref, color="#1f77b4", lw=1.4, label="data")
    axes[0].plot(t_fit, main["mean_omega"][:, 0], color="#d62728", lw=1.4, ls="--",
                 label="model (autonomous)")
    axes[0].plot(t_froz, froz["mean_omega"][:, 0], color="0.45", lw=1.2, ls=":",
                 label="frozen-parameter control")
    axes[0].set_ylabel(r"$\langle\omega\rangle$ (rad/s)")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"Autonomous replay from a pre-ictal state (t={T[i0]:.1f}s): "
                      "no forcing, no nudging, no clipping")
    axes[1].plot(t_roll, R_ref, color="#1f77b4", lw=1.4)
    axes[1].plot(t_fit, main["order"][:, 0], color="#d62728", lw=1.4, ls="--")
    axes[1].set_ylabel("order R")
    axes[2].plot(t_roll, z_ref, color="#1f77b4", lw=1.4)
    axes[2].plot(t_fit, main["z"][:, 0], color="#d62728", lw=1.4, ls="--")
    axes[2].set_ylabel("slow z")
    axes[3].plot(t_fit, sig_ref[:, 0], color="#1f77b4", lw=0.6, alpha=0.75)
    axes[3].plot(t_fit, sig_fit[:, 0], color="#d62728", lw=0.6, ls="--", alpha=0.9)
    axes[3].set_ylabel(f"{LABELS[0]}")
    axes[3].set_xlabel("recording time (s)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"seizure_replay.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_freedom_map() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    ext = [DELTA_GRID[0], DELTA_GRID[-1], GAIN_GRID[0], GAIN_GRID[-1]]
    im = axes[0].imshow(rise_map, origin="lower", aspect="auto", extent=ext, cmap="magma")
    axes[0].contour(DELTA_GRID, GAIN_GRID, rise_map, levels=[EXC_THRESHOLD],
                    colors="w", linewidths=1.6)
    axes[0].set_title(r"rise in $\langle\omega\rangle$ (rad/s)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    klass = np.where(~viable_map, 0, np.where(free_map, 1, 2))
    im2 = axes[1].imshow(klass, origin="lower", aspect="auto", extent=ext,
                         cmap=matplotlib.colors.ListedColormap(["0.75", "#2ca02c", "#d62728"]),
                         vmin=0, vmax=2)
    axes[1].set_title("non-viable / seizure-free / seizure")
    cb = fig.colorbar(im2, ax=axes[1], fraction=0.046, ticks=[0.33, 1.0, 1.67])
    cb.ax.set_yticklabels(["non-viable", "free", "seizure"], fontsize=7)

    for ax in axes[:2]:
        ax.scatter([0.0], [1.0], marker="*", s=180, c="w", edgecolors="k",
                   zorder=5, label="fitted")
        if found_target and DU_ALT == 0.0:
            ax.scatter([D_ALT], [G_ALT], marker="o", s=70, c="#00d0ff",
                       edgecolors="k", zorder=5,
                       label="nearest " + ("free" if fitted_seizes else "seizing"))
        ax.set_xlabel(r"z-excitability offset $\delta_z$")
        ax.set_ylabel("coupling gain $g$")
        ax.legend(fontsize=7, loc="lower left")

    for ax, grid, stats, xlabel, title, fitted_at in (
        (axes[2], DELTA_U_GRID, du_stats, r"$\omega$-excitability offset $\delta_u$",
         r"direct knob, at $g=1,\ \delta_z=0$", 0.0),
        (axes[3], PLAST_GRID, plast_stats, "plasticity gain $p$",
         r"$d\kappa/dt$ gain, at $g=1,\ \delta_z=0$", 1.0),
    ):
        rise = [s["omega_rise"] for s in stats]
        ok = np.array([is_viable(s) for s in stats])
        ax.plot(grid, rise, "-", color="0.6", zorder=1)
        ax.scatter(np.asarray(grid)[ok], np.asarray(rise)[ok], s=34, color="#1f77b4",
                   zorder=3, label="viable")
        if (~ok).any():
            ax.scatter(np.asarray(grid)[~ok], np.asarray(rise)[~ok], s=34,
                       color="0.75", marker="x", zorder=3, label="non-viable")
        ax.axhline(EXC_THRESHOLD, color="#d62728", ls="--", lw=1.0,
                   label="seizure threshold")
        ax.axvline(fitted_at, color="0.4", ls=":", lw=1.0, label="fitted")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"rise in $\langle\omega\rangle$")
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.suptitle("Seizure freedom in parameter space, same pre-ictal initial condition")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"seizure_freedom_map.{e}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_freedom_replay() -> None:
    nch = min(M, 4)
    fig, axes = plt.subplots(3 + nch, 1, figsize=(11, 2.0 * (3 + nch)), sharex=True)
    for ax in axes:
        ax.axvspan(*EVENT_WINDOW, color="#f4a261", alpha=0.18, lw=0)
    axes[0].plot(t_roll, mo_ref, color="#1f77b4", lw=1.3, label="data")
    axes[0].plot(t_fit, main["mean_omega"][:, 0], color="#d62728", lw=1.5, ls="--",
                 label=r"fitted  $g$=1, $\delta$=0")
    axes[0].plot(t_alt, alt["mean_omega"][:, 0], color="#2ca02c", lw=1.6,
                 label=rf"{'abolished' if fitted_seizes else 'induced'}  "
                       rf"$g$={G_ALT:.2f}, $\delta_z$={D_ALT:+.2f}, "
                       rf"$\delta_u$={DU_ALT:+.2f}")
    axes[0].set_ylabel(r"$\langle\omega\rangle$")
    axes[0].legend(fontsize=8, ncol=3)
    axes[0].set_title(
        "Same initial condition, same integrator: "
        + ("the seizure and its abolition" if fitted_seizes else
           "the fitted model is already seizure-free; green is the nearest "
           "parameter setting that does seize"))
    axes[1].plot(t_roll, R_ref, color="#1f77b4", lw=1.3)
    axes[1].plot(t_fit, main["order"][:, 0], color="#d62728", lw=1.5, ls="--")
    axes[1].plot(t_alt, alt["order"][:, 0], color="#2ca02c", lw=1.6)
    axes[1].set_ylabel("order R")
    axes[2].plot(t_roll, z_ref, color="#1f77b4", lw=1.3)
    axes[2].plot(t_fit, main["z"][:, 0], color="#d62728", lw=1.5, ls="--")
    axes[2].plot(t_alt, alt["z"][:, 0], color="#2ca02c", lw=1.6)
    axes[2].set_ylabel("slow z")
    for ch in range(nch):
        ax = axes[3 + ch]
        ax.plot(t_fit, sig_fit[:, ch], color="#d62728", lw=0.6, alpha=0.85)
        ax.plot(t_alt, sig_alt[:, ch], color="#2ca02c", lw=0.6, alpha=0.85)
        ax.set_ylabel(LABELS[ch], rotation=0, ha="right", va="center", fontsize=8)
    axes[-1].set_xlabel("recording time (s)")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"seizure_freedom_replay.{e}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_ic_scan() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.axvspan(*EVENT_WINDOW, color="#f4a261", alpha=0.22, lw=0)
    ax.axhline(EXC_THRESHOLD, color="0.3", ls="--", lw=1.0, label="seizure threshold")
    ax.plot(ic_times, ic_rise_fit, "o-", color="#d62728", label=r"fitted $g$=1, $\delta$=0")
    ax.plot(ic_times, ic_rise_alt, "s-", color="#2ca02c",
            label=rf"$g$={G_ALT:.2f}, $\delta_z$={D_ALT:+.2f}, $\delta_u$={DU_ALT:+.2f}")
    ax.set_xlabel("rollout start time (s)")
    ax.set_ylabel(r"rise in $\langle\omega\rangle$ (rad/s)")
    ax.set_title("Does it seize from everywhere, and does freedom hold everywhere?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"initial_condition_scan.{e}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_losses() -> None:
    for mdl, tag in ((model_theta, "loss_theta"), (model_u, "loss_omega"),
                     (model_v, "loss_kappa"), (model_h, "loss_h")):
        if getattr(mdl, "train_results_", None):
            fig, _ = plot_loss_curves(mdl.train_results_, save=str(RESULT_DIR / tag))
            plt.close(fig)


plot_state()
plot_replay()
plot_freedom_map()
plot_freedom_replay()
plot_ic_scan()
plot_losses()
print(f"\n[FIGS]  Saved {RESULT_DIR}", flush=True)
