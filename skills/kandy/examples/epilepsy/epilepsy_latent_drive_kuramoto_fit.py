#!/usr/bin/env python3
"""Latent-drive adaptive-Kuramoto KANDy model for the epilepsy recording.

This is the non-circular follow-up to ``epilepsy_adaptive_kuramoto_fit.py``.
That script replayed the seizure by injecting an empirical residual drive
``q_data(t)``.  Here we compress that residual with SVD, keep ten latent drive
coordinates, and fit an augmented autonomous system:

    x = [theta, kappa, z]

    dtheta/dt = F_theta(theta, kappa, z)
    dkappa/dt = F_kappa(theta, kappa, z)
    dz/dt     = F_z(theta, kappa, z)

where ``z`` are the first ten SVD coordinates of the normalized residual drive.
During validation there is no q_data(t) injection and no nudging; the latent
state is initialized from the recording and then rolled forward by its learned
dynamics.

This is still not a fully validated digital twin: the latent coordinates are
identified from this one recording.  It is the right structural experiment,
though: replacing a prescribed forcing trace with internal latent dynamics.
"""

from __future__ import annotations

import contextlib
import json
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

# ---------------------------------------------------------------------------
# 0. Parameters
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = REPO_ROOT / "data" / "Dory_2021_04_12_006_preprocessed.pkl"
RESULT_DIR = ensure_outdir(REPO_ROOT / "results" / "Epilepsy-Latent-Drive")

BANDPASS_HZ = (1.0, 40.0)
FILTER_ORDER = 4
EDGE_TRIM_SECONDS = 2.0
PHASE_SAVGOL_SECONDS = 0.155
KAPPA_SAVGOL_SECONDS = 0.505
Q_SAVGOL_SECONDS = 0.255
Z_SAVGOL_SECONDS = 0.505

LATENT_DIM = 10
GRID = 4
BASELINE_STEPS = 18
AUGMENTED_STEPS = 18
LATENT_STEPS = 100
N_TRAIN_MAX = 2500
N_TEST_MAX = 1500
TEST_STRIDE = 5
SHORT_ROLLOUT_SECONDS = 0.25
EVENT_WINDOW = (-1.0, 4.0)
ROLL_OUT_CLIP_MARGIN = 0.10


# ---------------------------------------------------------------------------
# 1. Data and phase/kappa preprocessing
# ---------------------------------------------------------------------------
def odd_window(seconds: float, fs: float, minimum: int = 5) -> int:
    win = max(minimum, int(round(seconds * fs)))
    if win % 2 == 0:
        win += 1
    return win


def zscore_columns(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-12
    return (X - mean) / std, mean, std


def bandpass(X: np.ndarray, fs: float) -> np.ndarray:
    nyq = 0.5 * fs
    low, high = BANDPASS_HZ
    b, a = butter(FILTER_ORDER, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, X, axis=0)


def quiet_fit(model: KANDy, **kwargs) -> None:
    with contextlib.redirect_stdout(StringIO()):
        model.fit(**kwargs)


with DATA_PATH.open("rb") as fh:
    payload = pickle.load(fh)

X_RAW = np.asarray(payload["X"], dtype=float)
FS = float(payload["fs"])
DT = float(payload["dt"])
T_ALL = np.asarray(payload["t"], dtype=float)
LABELS = list(payload["labels"])
N = X_RAW.shape[1]

PAIR_FEATURES = [(i, j) for i in range(N) for j in range(i + 1, N)]
PAIR_INDEX = {pair: idx for idx, pair in enumerate(PAIR_FEATURES)}
N_PAIR = len(PAIR_FEATURES)
N_STATE = N + N_PAIR
N_AUG = N_STATE + LATENT_DIM

print("[DATA]  Loaded epilepsy recording")
print(f"        samples={len(X_RAW)}, channels={N}, fs={FS:.1f} Hz")
print(f"        labels={LABELS}")

X_FILT = bandpass(X_RAW, FS)
analytic = hilbert(X_FILT, axis=0)
theta_unwrapped = np.unwrap(np.angle(analytic), axis=0)

phase_window = odd_window(PHASE_SAVGOL_SECONDS, FS)
theta = savgol_filter(theta_unwrapped, phase_window, 3, axis=0, mode="interp")
theta_dot = savgol_filter(
    theta_unwrapped,
    phase_window,
    3,
    deriv=1,
    delta=DT,
    axis=0,
    mode="interp",
)

amplitude = np.abs(analytic)
log_amp, amp_mean, amp_std = zscore_columns(np.log(amplitude + 1e-6))
pair_log_amp = np.column_stack(
    [0.5 * (log_amp[:, i] + log_amp[:, j]) for i, j in PAIR_FEATURES]
)
kappa_window = odd_window(KAPPA_SAVGOL_SECONDS, FS)
kappa_raw = savgol_filter(pair_log_amp, kappa_window, 3, axis=0, mode="interp")
kappa_dot_raw = savgol_filter(
    pair_log_amp,
    kappa_window,
    3,
    deriv=1,
    delta=DT,
    axis=0,
    mode="interp",
)
kappa, kappa_mean, kappa_std = zscore_columns(kappa_raw)
kappa_dot = kappa_dot_raw / kappa_std

edge = int(round(EDGE_TRIM_SECONDS * FS))
valid = slice(edge, len(X_RAW) - edge)
T = T_ALL[valid]
SIGNAL = X_FILT[valid]
STATE = np.hstack([theta[valid], kappa[valid]])
STATE_DOT = np.hstack([theta_dot[valid], kappa_dot[valid]])
THETA_DOT = STATE_DOT[:, :N]
KAPPA_DOT = STATE_DOT[:, N:]

phase_scale = THETA_DOT.std(axis=0) + 1e-12
kappa_scale = KAPPA_DOT.std(axis=0) + 1e-12

all_idx = np.arange(len(STATE))
test_idx = all_idx[all_idx % TEST_STRIDE == 0]
train_idx = all_idx[all_idx % TEST_STRIDE != 0]
rng = np.random.default_rng(SEED)
if len(train_idx) > N_TRAIN_MAX:
    train_idx = np.sort(rng.choice(train_idx, N_TRAIN_MAX, replace=False))
if len(test_idx) > N_TEST_MAX:
    test_idx = np.sort(rng.choice(test_idx, N_TEST_MAX, replace=False))

print(f"[PRE]   used samples={len(STATE)}, train={len(train_idx)}, heldout={len(test_idx)}")


# ---------------------------------------------------------------------------
# 2. Shared feature builders
# ---------------------------------------------------------------------------
def split_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return state[:, :N], state[:, N : N + N_PAIR]


def split_aug(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return state[:, :N], state[:, N : N + N_PAIR], state[:, N + N_PAIR :]


def mean_field_np(theta_state: np.ndarray) -> np.ndarray:
    delta = theta_state[:, None, :] - theta_state[:, :, None]
    sin_m = np.mean(np.sin(delta), axis=2)
    cos_m = np.mean(np.cos(delta), axis=2)
    return np.column_stack([sin_m, cos_m])


def mean_field_torch(theta_state: torch.Tensor) -> torch.Tensor:
    delta = theta_state[:, None, :] - theta_state[:, :, None]
    sin_m = torch.mean(torch.sin(delta), dim=2)
    cos_m = torch.mean(torch.cos(delta), dim=2)
    return torch.cat([sin_m, cos_m], dim=1)


def base_phase_lift_np(state: np.ndarray) -> np.ndarray:
    theta_state, kappa_state = split_state(state)
    features = []
    for i, j in PAIR_FEATURES:
        k = kappa_state[:, PAIR_INDEX[(i, j)]]
        delta = theta_state[:, j] - theta_state[:, i]
        s = np.sin(delta)
        c = np.cos(delta)
        features.extend([k * s, k * c, s, c, k])
    return np.column_stack(features)


def base_phase_lift_torch(state: torch.Tensor) -> torch.Tensor:
    theta_state = state[:, :N]
    kappa_state = state[:, N : N + N_PAIR]
    features = []
    for i, j in PAIR_FEATURES:
        k = kappa_state[:, PAIR_INDEX[(i, j)]]
        delta = theta_state[:, j] - theta_state[:, i]
        s = torch.sin(delta)
        c = torch.cos(delta)
        features.extend([k * s, k * c, s, c, k])
    return torch.stack(features, dim=1)


def base_kappa_lift_np(state: np.ndarray) -> np.ndarray:
    theta_state, kappa_state = split_state(state)
    features = []
    for i, j in PAIR_FEATURES:
        delta = theta_state[:, j] - theta_state[:, i]
        features.extend([np.sin(delta), np.cos(delta)])
    features.extend([kappa_state[:, idx] for idx in range(N_PAIR)])
    return np.column_stack(features)


def base_kappa_lift_torch(state: torch.Tensor) -> torch.Tensor:
    theta_state = state[:, :N]
    kappa_state = state[:, N : N + N_PAIR]
    features = []
    for i, j in PAIR_FEATURES:
        delta = theta_state[:, j] - theta_state[:, i]
        features.extend([torch.sin(delta), torch.cos(delta)])
    features.extend([kappa_state[:, idx] for idx in range(N_PAIR)])
    return torch.stack(features, dim=1)


# ---------------------------------------------------------------------------
# 3. Baseline autonomous fit to define q(t), then SVD latents z(t)
# ---------------------------------------------------------------------------
model_base_phase = KANDy(
    lift=CustomLift(
        fn=base_phase_lift_np,
        output_dim=5 * N_PAIR,
        torch_fn=base_phase_lift_torch,
        name="baseline_phase",
    ),
    grid=GRID,
    k=3,
    steps=BASELINE_STEPS,
    seed=SEED,
    base_fun=torch.nn.Identity(),
    device="cpu",
)
model_base_kappa = KANDy(
    lift=CustomLift(
        fn=base_kappa_lift_np,
        output_dim=3 * N_PAIR,
        torch_fn=base_kappa_lift_torch,
        name="baseline_kappa",
    ),
    grid=GRID,
    k=3,
    steps=BASELINE_STEPS,
    seed=SEED,
    base_fun=torch.nn.Identity(),
    device="cpu",
)

print("[BASE]  Fitting baseline models for residual q(t) ...")
quiet_fit(
    model_base_phase,
    X=STATE[train_idx],
    X_dot=THETA_DOT[train_idx] / phase_scale,
    val_frac=0.1,
    test_frac=0.1,
    lamb=0.0,
    opt="LBFGS",
    lr=1.0,
    batch=-1,
    stop_grid_update_step=5,
    patience=0,
    verbose=False,
)
quiet_fit(
    model_base_kappa,
    X=STATE[train_idx],
    X_dot=KAPPA_DOT[train_idx] / kappa_scale,
    val_frac=0.1,
    test_frac=0.1,
    lamb=0.0,
    opt="LBFGS",
    lr=1.0,
    batch=-1,
    stop_grid_update_step=5,
    patience=0,
    verbose=False,
)

baseline_dot = np.hstack(
    [
        model_base_phase.predict(STATE) * phase_scale,
        model_base_kappa.predict(STATE) * kappa_scale,
    ]
)
residual_dot = STATE_DOT - baseline_dot
q_window = odd_window(Q_SAVGOL_SECONDS, FS)
q_dot = savgol_filter(residual_dot, q_window, 3, axis=0, mode="interp")
q_scale = np.concatenate([phase_scale, kappa_scale])
q_normalized = q_dot / q_scale
q_svd = (q_normalized - q_normalized.mean(axis=0)) / (q_normalized.std(axis=0) + 1e-12)

U, S, Vt = np.linalg.svd(q_svd, full_matrices=False)
variance_ratio = (S ** 2) / np.sum(S ** 2)
cumulative_variance = np.cumsum(variance_ratio)
z_raw = U[:, :LATENT_DIM] * S[:LATENT_DIM]
z_raw = savgol_filter(
    z_raw,
    odd_window(Z_SAVGOL_SECONDS, FS),
    3,
    axis=0,
    mode="interp",
)
Z, z_mean, z_std = zscore_columns(z_raw)
z_window = odd_window(Z_SAVGOL_SECONDS, FS)
Z_DOT = savgol_filter(Z, z_window, 3, deriv=1, delta=DT, axis=0, mode="interp")
z_scale = Z_DOT.std(axis=0) + 1e-12

STATE_AUG = np.hstack([STATE, Z])
STATE_AUG_DOT = np.hstack([THETA_DOT, KAPPA_DOT, Z_DOT])

print(
    "[SVD]   q(t) cumulative variance "
    f"rank1={cumulative_variance[0]:.3f}, "
    f"rank3={cumulative_variance[2]:.3f}, "
    f"rank5={cumulative_variance[4]:.3f}, "
    f"rank10={cumulative_variance[9]:.3f}"
)


# ---------------------------------------------------------------------------
# 4. Augmented latent-drive KANDy lifts
# ---------------------------------------------------------------------------
def augmented_phase_lift_np(state: np.ndarray) -> np.ndarray:
    theta_state, kappa_state, z_state = split_aug(state)
    base = base_phase_lift_np(state[:, :N_STATE])
    mf = mean_field_np(theta_state)
    features = [base, z_state]
    z_mf = [
        z_state[:, m] * mf[:, feature_idx]
        for m in range(LATENT_DIM)
        for feature_idx in range(mf.shape[1])
    ]
    features.append(np.column_stack(z_mf))
    return np.column_stack(features)


def augmented_phase_lift_torch(state: torch.Tensor) -> torch.Tensor:
    theta_state = state[:, :N]
    z_state = state[:, N + N_PAIR :]
    base = base_phase_lift_torch(state[:, :N_STATE])
    mf = mean_field_torch(theta_state)
    features = [base, z_state]
    z_mf = [
        z_state[:, m] * mf[:, feature_idx]
        for m in range(LATENT_DIM)
        for feature_idx in range(mf.shape[1])
    ]
    features.append(torch.stack(z_mf, dim=1))
    return torch.cat(features, dim=1)


def augmented_kappa_lift_np(state: np.ndarray) -> np.ndarray:
    _theta_state, kappa_state, z_state = split_aug(state)
    base = base_kappa_lift_np(state[:, :N_STATE])
    zk = [
        z_state[:, m] * kappa_state[:, pair_idx]
        for m in range(LATENT_DIM)
        for pair_idx in range(N_PAIR)
    ]
    return np.column_stack([base, z_state, np.column_stack(zk)])


def augmented_kappa_lift_torch(state: torch.Tensor) -> torch.Tensor:
    kappa_state = state[:, N : N + N_PAIR]
    z_state = state[:, N + N_PAIR :]
    base = base_kappa_lift_torch(state[:, :N_STATE])
    zk = [
        z_state[:, m] * kappa_state[:, pair_idx]
        for m in range(LATENT_DIM)
        for pair_idx in range(N_PAIR)
    ]
    return torch.cat([base, z_state, torch.stack(zk, dim=1)], dim=1)


def latent_lift_np(state: np.ndarray) -> np.ndarray:
    theta_state, kappa_state, z_state = split_aug(state)
    r_sin = np.mean(np.sin(theta_state), axis=1)
    r_cos = np.mean(np.cos(theta_state), axis=1)
    r = np.sqrt(r_sin ** 2 + r_cos ** 2)
    extras = np.column_stack([r, r_sin, r_cos, kappa_state.mean(axis=1)])
    return np.column_stack([z_state, extras])


def latent_lift_torch(state: torch.Tensor) -> torch.Tensor:
    theta_state = state[:, :N]
    kappa_state = state[:, N : N + N_PAIR]
    z_state = state[:, N + N_PAIR :]
    r_sin = torch.mean(torch.sin(theta_state), dim=1)
    r_cos = torch.mean(torch.cos(theta_state), dim=1)
    r = torch.sqrt(r_sin ** 2 + r_cos ** 2 + 1e-12)
    extras = torch.stack([r, r_sin, r_cos, torch.mean(kappa_state, dim=1)], dim=1)
    return torch.cat([z_state, extras], dim=1)


PHASE_LIFT_DIM = 5 * N_PAIR + LATENT_DIM + LATENT_DIM * 2 * N
KAPPA_LIFT_DIM = 3 * N_PAIR + LATENT_DIM + LATENT_DIM * N_PAIR
LATENT_LIFT_DIM = LATENT_DIM + 4

print(
    "[LIFT]  augmented dims "
    f"phase={PHASE_LIFT_DIM}, kappa={KAPPA_LIFT_DIM}, latent={LATENT_LIFT_DIM}"
)


# ---------------------------------------------------------------------------
# 5. Fit augmented autonomous system
# ---------------------------------------------------------------------------
model_phase = KANDy(
    lift=CustomLift(
        fn=augmented_phase_lift_np,
        output_dim=PHASE_LIFT_DIM,
        torch_fn=augmented_phase_lift_torch,
        name="latent_phase",
    ),
    grid=GRID,
    k=3,
    steps=AUGMENTED_STEPS,
    seed=SEED,
    base_fun=torch.nn.Identity(),
    device="cpu",
)
model_kappa = KANDy(
    lift=CustomLift(
        fn=augmented_kappa_lift_np,
        output_dim=KAPPA_LIFT_DIM,
        torch_fn=augmented_kappa_lift_torch,
        name="latent_kappa",
    ),
    grid=GRID,
    k=3,
    steps=AUGMENTED_STEPS,
    seed=SEED,
    base_fun=torch.nn.Identity(),
    device="cpu",
)
model_latent = KANDy(
    lift=CustomLift(
        fn=latent_lift_np,
        output_dim=LATENT_LIFT_DIM,
        torch_fn=latent_lift_torch,
        name="latent_drive",
    ),
    grid=GRID,
    k=3,
    steps=LATENT_STEPS,
    seed=SEED,
    base_fun=torch.nn.Identity(),
    device="cpu",
)

print("[TRAIN] Fitting augmented phase model ...")
quiet_fit(
    model_phase,
    X=STATE_AUG[train_idx],
    X_dot=THETA_DOT[train_idx] / phase_scale,
    val_frac=0.1,
    test_frac=0.1,
    lamb=0.0,
    opt="LBFGS",
    lr=1.0,
    batch=-1,
    stop_grid_update_step=5,
    patience=0,
    verbose=False,
)
print("[TRAIN] Fitting augmented kappa model ...")
quiet_fit(
    model_kappa,
    X=STATE_AUG[train_idx],
    X_dot=KAPPA_DOT[train_idx] / kappa_scale,
    val_frac=0.1,
    test_frac=0.1,
    lamb=0.0,
    opt="LBFGS",
    lr=1.0,
    batch=-1,
    stop_grid_update_step=5,
    patience=0,
    verbose=False,
)
print("[TRAIN] Fitting latent z model ...")
quiet_fit(
    model_latent,
    X=STATE_AUG[train_idx],
    X_dot=Z_DOT[train_idx] / z_scale,
    val_frac=0.1,
    test_frac=0.1,
    lamb=0.0,
    opt="LBFGS",
    lr=1.0,
    batch=-1,
    stop_grid_update_step=5,
    patience=0,
    verbose=False,
)


# ---------------------------------------------------------------------------
# 6. Validation, rollout, and signal reconstruction
# ---------------------------------------------------------------------------
def r2_per_output(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    denom = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    denom = np.where(denom > 0, denom, np.nan)
    return 1.0 - np.sum((y_true - y_pred) ** 2, axis=0) / denom


pred_phase = model_phase.predict(STATE_AUG[test_idx]) * phase_scale
pred_kappa = model_kappa.predict(STATE_AUG[test_idx]) * kappa_scale
pred_z = model_latent.predict(STATE_AUG[test_idx]) * z_scale
phase_r2 = r2_per_output(THETA_DOT[test_idx], pred_phase)
kappa_r2 = r2_per_output(KAPPA_DOT[test_idx], pred_kappa)
z_r2 = r2_per_output(Z_DOT[test_idx], pred_z)
print(
    "[EVAL]  heldout R2 means "
    f"phase={np.nanmean(phase_r2):.3f}, "
    f"kappa={np.nanmean(kappa_r2):.3f}, "
    f"z={np.nanmean(z_r2):.3f}"
)

pair_average_matrix = np.zeros((N_PAIR, N))
for row_idx, (i, j) in enumerate(PAIR_FEATURES):
    pair_average_matrix[row_idx, i] = 0.5
    pair_average_matrix[row_idx, j] = 0.5
pair_to_channel = np.linalg.pinv(pair_average_matrix)


def state_to_signal(state: np.ndarray) -> np.ndarray:
    state_2d = np.atleast_2d(state)
    pair_log_amp = state_2d[:, N:N_STATE] * kappa_std + kappa_mean
    channel_log_amp_z = pair_log_amp @ pair_to_channel.T
    channel_log_amp_z = np.clip(channel_log_amp_z, -5.0, 5.0)
    amp_hat = np.exp(channel_log_amp_z * amp_std + amp_mean)
    return amp_hat * np.cos(state_2d[:, :N])


def rhs_aug(state: np.ndarray) -> np.ndarray:
    state_2d = state[None, :]
    return np.concatenate(
        [
            (model_phase.predict(state_2d) * phase_scale).ravel(),
            (model_kappa.predict(state_2d) * kappa_scale).ravel(),
            (model_latent.predict(state_2d) * z_scale).ravel(),
        ]
    )


kappa_lo = STATE_AUG[:, N:N_STATE].min(axis=0)
kappa_hi = STATE_AUG[:, N:N_STATE].max(axis=0)
z_lo = STATE_AUG[:, N_STATE:].min(axis=0)
z_hi = STATE_AUG[:, N_STATE:].max(axis=0)
kappa_span = kappa_hi - kappa_lo
z_span = z_hi - z_lo
kappa_lo = kappa_lo - ROLL_OUT_CLIP_MARGIN * kappa_span
kappa_hi = kappa_hi + ROLL_OUT_CLIP_MARGIN * kappa_span
z_lo = z_lo - ROLL_OUT_CLIP_MARGIN * z_span
z_hi = z_hi + ROLL_OUT_CLIP_MARGIN * z_span


def rollout_aug(x0: np.ndarray, steps: int) -> np.ndarray:
    x = x0.astype(float).copy()
    out = np.empty((steps + 1, len(x0)))
    out[0] = x
    for step in range(steps):
        x = x + DT * rhs_aug(x)
        x[N:N_STATE] = np.clip(x[N:N_STATE], kappa_lo, kappa_hi)
        x[N_STATE:] = np.clip(x[N_STATE:], z_lo, z_hi)
        if not np.isfinite(x).all():
            return out[: step + 1]
        out[step + 1] = x
    return out


def summarize_rollout(pred: np.ndarray, ref: np.ndarray, start_time: float) -> dict:
    r_pred = np.abs(np.mean(np.exp(1j * pred[:, :N]), axis=1))
    r_ref = np.abs(np.mean(np.exp(1j * ref[:, :N]), axis=1))
    return {
        "start_time": float(start_time),
        "samples": int(len(pred)),
        "theta_rmse_wrapped": float(
            np.sqrt(np.mean(wrap_pi(pred[:, :N] - ref[:, :N]) ** 2))
        ),
        "kappa_rmse": float(np.sqrt(np.mean((pred[:, N:N_STATE] - ref[:, N:N_STATE]) ** 2))),
        "z_rmse": float(np.sqrt(np.mean((pred[:, N_STATE:] - ref[:, N_STATE:]) ** 2))),
        "order_parameter_rmse": float(np.sqrt(np.mean((r_pred - r_ref) ** 2))),
    }


short_steps = int(round(SHORT_ROLLOUT_SECONDS / DT))
short_rows = []
short_traces = []
for start_s in [-8.0, 0.0, 8.0, 18.0]:
    start = int(np.argmin(np.abs(T - start_s)))
    pred = rollout_aug(STATE_AUG[start], short_steps)
    ref = STATE_AUG[start : start + len(pred)]
    row = summarize_rollout(pred, ref, T[start])
    short_rows.append(row)
    short_traces.append(
        {
            "start_time": float(T[start]),
            "time": T[start : start + len(pred)] - T[start],
            "pred": pred,
            "ref": ref,
        }
    )

event_start = max(EVENT_WINDOW[0], float(T[0]))
event_end = min(EVENT_WINDOW[1], float(T[-1]))
event_steps = int(round((event_end - event_start) / DT))
event_idx = int(np.argmin(np.abs(T - event_start)))
event_pred = rollout_aug(STATE_AUG[event_idx], event_steps)
event_ref = STATE_AUG[event_idx : event_idx + len(event_pred)]
event_time = T[event_idx : event_idx + len(event_pred)]
event_signal_pred = state_to_signal(event_pred[:, :N_STATE])
event_signal_ref = SIGNAL[event_idx : event_idx + len(event_pred)]
event_order_pred = np.abs(np.mean(np.exp(1j * event_pred[:, :N]), axis=1))
event_order_ref = np.abs(np.mean(np.exp(1j * event_ref[:, :N]), axis=1))
event_corr = [
    float(np.corrcoef(event_signal_pred[:, i], event_signal_ref[:, i])[0, 1])
    for i in range(N)
]
event_metrics = {
    **summarize_rollout(event_pred, event_ref, event_time[0]),
    "end_time": float(event_time[-1]),
    "signal_rmse": float(np.sqrt(np.mean((event_signal_pred - event_signal_ref) ** 2))),
    "signal_corr_mean": float(np.mean(event_corr)),
    "signal_corr_per_channel": event_corr,
}

print("[ROLLOUT] short autonomous latent rollouts:")
for row in short_rows:
    print(
        "          "
        f"t0={row['start_time']:.2f}s, theta={row['theta_rmse_wrapped']:.3f}, "
        f"kappa={row['kappa_rmse']:.3f}, z={row['z_rmse']:.3f}, "
        f"r={row['order_parameter_rmse']:.3f}"
    )
print(
    "[EVENT] latent autonomous replay "
    f"{event_metrics['start_time']:.2f}s..{event_metrics['end_time']:.2f}s: "
    f"signal_corr={event_metrics['signal_corr_mean']:.3f}, "
    f"signal_rmse={event_metrics['signal_rmse']:.3f}, "
    f"r_rmse={event_metrics['order_parameter_rmse']:.3f}"
)

thresholds = {
    "phase_r2_mean_min": 0.20,
    "kappa_r2_mean_min": 0.10,
    "z_r2_mean_min": 0.05,
    "short_order_rmse_mean_max": 0.35,
}
metrics_passed = print_check(
    float(np.nanmean(phase_r2)) >= thresholds["phase_r2_mean_min"]
    and float(np.nanmean(kappa_r2)) >= thresholds["kappa_r2_mean_min"]
    and float(np.nanmean(z_r2)) >= thresholds["z_r2_mean_min"]
    and float(np.mean([r["order_parameter_rmse"] for r in short_rows]))
    <= thresholds["short_order_rmse_mean_max"],
    "latent-drive augmented autonomous fit has measurable heldout skill",
)

metrics = {
    "task": "latent-drive adaptive Kuramoto KANDy fit",
    "data_path": str(DATA_PATH),
    "labels": LABELS,
    "latent_dim": LATENT_DIM,
    "samples_used": int(len(STATE_AUG)),
    "state_dim": N_AUG,
    "lift_dims": {
        "phase": PHASE_LIFT_DIM,
        "kappa": KAPPA_LIFT_DIM,
        "latent": LATENT_LIFT_DIM,
    },
    "fit": {
        "grid": GRID,
        "baseline_steps": BASELINE_STEPS,
        "augmented_steps": AUGMENTED_STEPS,
        "latent_steps": LATENT_STEPS,
        "n_train": int(len(train_idx)),
        "n_heldout": int(len(test_idx)),
        "rollout_projection": "kappa and z clipped to observed range plus margin",
        "rollout_clip_margin": ROLL_OUT_CLIP_MARGIN,
    },
    "q_svd_variance_ratio": [float(v) for v in variance_ratio],
    "q_svd_cumulative_variance": [float(v) for v in cumulative_variance],
    "phase_r2_heldout": [float(v) for v in phase_r2],
    "phase_r2_heldout_mean": float(np.nanmean(phase_r2)),
    "kappa_r2_heldout": [float(v) for v in kappa_r2],
    "kappa_r2_heldout_mean": float(np.nanmean(kappa_r2)),
    "z_r2_heldout": [float(v) for v in z_r2],
    "z_r2_heldout_mean": float(np.nanmean(z_r2)),
    "short_rollouts": short_rows,
    "short_order_rmse_mean": float(np.mean([r["order_parameter_rmse"] for r in short_rows])),
    "event_replay": event_metrics,
    "thresholds": thresholds,
    "passed": metrics_passed,
}
save_metrics(RESULT_DIR / "metrics.json", metrics)

torch.save(
    {
        "phase_model_state_dict": model_phase.model_.state_dict(),
        "kappa_model_state_dict": model_kappa.model_.state_dict(),
        "latent_model_state_dict": model_latent.model_.state_dict(),
        "phase_scale": torch.tensor(phase_scale),
        "kappa_scale": torch.tensor(kappa_scale),
        "z_scale": torch.tensor(z_scale),
        "labels": LABELS,
        "pair_features": PAIR_FEATURES,
        "latent_dim": LATENT_DIM,
        "metadata": metrics["fit"],
    },
    RESULT_DIR / "fitted_latent_model_state.pt",
)
np.savez_compressed(
    RESULT_DIR / "latent_fit_arrays.npz",
    t=T,
    labels=np.asarray(LABELS),
    state_aug=STATE_AUG,
    state_aug_dot=STATE_AUG_DOT,
    q_dot=q_dot,
    q_normalized=q_normalized,
    q_svd_scores=U * S,
    q_svd_singular_values=S,
    q_svd_components=Vt,
    q_svd_variance_ratio=variance_ratio,
    q_svd_cumulative_variance=cumulative_variance,
    z=Z,
    z_dot=Z_DOT,
    event_time=event_time,
    event_signal_ref=event_signal_ref,
    event_signal_pred=event_signal_pred,
    event_state_ref=event_ref,
    event_state_pred=event_pred,
    event_order_ref=event_order_ref,
    event_order_pred=event_order_pred,
)


# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
def plot_svd() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8))
    modes = np.arange(1, min(12, len(S)) + 1)
    axes[0].bar(modes, variance_ratio[: len(modes)], color="#1f77b4", alpha=0.75)
    axes[0].plot(
        modes,
        cumulative_variance[: len(modes)],
        marker="o",
        color="#d62728",
        label="cumulative",
    )
    axes[0].set_ylabel("variance")
    axes[0].set_title("Residual drive q(t) SVD")
    axes[0].legend(fontsize=8)

    for mode in range(min(5, LATENT_DIM)):
        axes[1].plot(T, Z[:, mode], lw=0.75, label=f"z{mode + 1}")
    axes[1].axvspan(event_time[0], event_time[-1], color="0.9", zorder=-1)
    axes[1].axvline(0.0, color="0.2", lw=0.8, ls=":")
    axes[1].set_ylabel("latent score")
    axes[1].legend(ncol=5, fontsize=7)

    event_mask = (T >= event_time[0]) & (T <= event_time[-1])
    for mode in range(min(5, LATENT_DIM)):
        axes[2].plot(T[event_mask], Z[event_mask, mode], lw=0.9, label=f"z{mode + 1}")
    axes[2].axvline(0.0, color="0.2", lw=0.8, ls=":")
    axes[2].set_xlabel("recording time (s)")
    axes[2].set_ylabel("event latent score")
    axes[2].legend(ncol=5, fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "latent_svd.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "latent_svd.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_short_rollouts() -> None:
    fig, axes = plt.subplots(len(short_traces), 2, figsize=(11, 2.4 * len(short_traces)), sharex="col")
    if len(short_traces) == 1:
        axes = axes[None, :]
    for row_idx, trace in enumerate(short_traces):
        t_local = trace["time"]
        pred = trace["pred"]
        ref = trace["ref"]
        ax_sig, ax_r = axes[row_idx]
        for ch_idx, label in enumerate(LABELS):
            ax_sig.plot(t_local, np.cos(ref[:, ch_idx]), lw=0.7, alpha=0.55, label=label if row_idx == 0 else None)
            ax_sig.plot(t_local, np.cos(pred[:, ch_idx]), lw=0.8, ls="--")
        r_ref = np.abs(np.mean(np.exp(1j * ref[:, :N]), axis=1))
        r_pred = np.abs(np.mean(np.exp(1j * pred[:, :N]), axis=1))
        ax_r.plot(t_local, r_ref, lw=1.0, label="data")
        ax_r.plot(t_local, r_pred, lw=1.0, ls="--", label="latent KANDy")
        ax_sig.set_ylabel(f"t0={trace['start_time']:.1f}s")
        ax_r.set_ylim(0, 1.05)
    axes[0, 0].set_title("cos(theta)")
    axes[0, 1].set_title("order parameter")
    axes[0, 0].legend(ncol=3, fontsize=7)
    axes[0, 1].legend(fontsize=7)
    axes[-1, 0].set_xlabel("time since start (s)")
    axes[-1, 1].set_xlabel("time since start (s)")
    fig.suptitle("Autonomous latent-drive KANDy short rollouts")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "latent_short_rollouts.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "latent_short_rollouts.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_event_replay() -> None:
    fig, axes = plt.subplots(N + 1, 1, figsize=(11, 1.35 * (N + 1)), sharex=True)
    for ch_idx, ax in enumerate(axes[:N]):
        ax.plot(event_time, event_signal_ref[:, ch_idx], lw=0.65, color="#1f77b4")
        ax.plot(event_time, event_signal_pred[:, ch_idx], lw=0.8, ls="--", color="#d62728")
        ax.axvline(0.0, color="0.2", lw=0.8, ls=":")
        ax.set_ylabel(LABELS[ch_idx], rotation=0, ha="right", va="center", fontsize=8)
    axes[-1].plot(event_time, event_order_ref, lw=1.0, color="#1f77b4", label="data")
    axes[-1].plot(event_time, event_order_pred, lw=1.0, ls="--", color="#d62728", label="latent KANDy")
    axes[-1].axvline(0.0, color="0.2", lw=0.8, ls=":")
    axes[-1].set_ylabel("order", rotation=0, ha="right", va="center", fontsize=8)
    axes[-1].set_xlabel("recording time (s)")
    axes[-1].set_ylim(0, 1.05)
    axes[-1].legend(fontsize=8)
    fig.suptitle("Seizure-window autonomous replay with 10 latent variables")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "latent_seizure_replay.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "latent_seizure_replay.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_losses() -> None:
    for name, model in [
        ("loss_phase", model_phase),
        ("loss_kappa", model_kappa),
        ("loss_latent", model_latent),
    ]:
        if getattr(model, "train_results_", None):
            fig, _ax = plot_loss_curves(model.train_results_, save=str(RESULT_DIR / name))
            plt.close(fig)


use_pub_style()
plot_svd()
plot_short_rollouts()
plot_event_replay()
plot_losses()

print(f"[FIGS]  Saved {RESULT_DIR}")
