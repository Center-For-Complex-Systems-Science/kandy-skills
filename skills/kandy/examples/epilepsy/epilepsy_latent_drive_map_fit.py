#!/usr/bin/env python3
"""Discrete-time delayed latent-drive KANDy map for the epilepsy recording.

This follows the Hénon example's map-mode recipe:

    model.fit(X=current_state, X_dot=future_increment)
    future_state = current_state + model.predict(current_state)

The augmented state is

    x_n = [theta_n, kappa_n, z_n, z_{n-L}, z_{n-2L}, ...]

where ``z`` are the first ten SVD coordinates of the empirical residual drive
``q(t)``.  Unlike the non-autonomous replay script, this file does not inject
q_data(t) during rollout and does not nudge to the measured trajectory.  The
latent delay buffer is initialized from the recording, then advanced by a
learned finite-lag KANDy map.  This follows the Mackey-Glass delay-embedding
alignment rule: once delayed rows are built, the target is trimmed to the
future of the most recent delayed row.
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

# ---------------------------------------------------------------------------
# 0. Parameters
# ---------------------------------------------------------------------------
def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = REPO_ROOT / "data" / "Dory_2021_04_12_006_preprocessed.pkl"
RESULT_DIR = ensure_outdir(REPO_ROOT / "results" / "Epilepsy-Latent-Delay-Map")

BANDPASS_HZ = (1.0, 40.0)
FILTER_ORDER = 4
EDGE_TRIM_SECONDS = 2.0
PHASE_SAVGOL_SECONDS = 0.155
KAPPA_SAVGOL_SECONDS = 0.505
Q_SAVGOL_SECONDS = 0.255
Z_SAVGOL_SECONDS = 0.505

LATENT_DIM = 10
GRID = 4
MAP_LAG_SAMPLES = env_int("KANDY_MAP_LAG_SAMPLES", 1)
Z_DELAY_COUNT = env_int("KANDY_Z_DELAY_COUNT", 6)
Z_DELAY_STEP_SAMPLES = env_int("KANDY_Z_DELAY_STEP_SAMPLES", 10)
THETA_DELAY_DIFFS = bool(env_int("KANDY_THETA_DELAY_DIFFS", 1))
BASELINE_STEPS = env_int("KANDY_BASELINE_STEPS", 8)
MAP_STEPS = env_int("KANDY_MAP_STEPS", 24)
Z_MAP_STEPS = env_int("KANDY_Z_MAP_STEPS", 36)
N_TRAIN_MAX = env_int("KANDY_N_TRAIN_MAX", 3000)
N_TEST_MAX = env_int("KANDY_N_TEST_MAX", 1500)
TEST_STRIDE = 5
SHORT_ROLLOUT_SECONDS = 0.25
EVENT_WINDOW = (-1.0, 4.0)
ROLL_OUT_CLIP_MARGIN = 0.10


# ---------------------------------------------------------------------------
# 1. Preprocessing
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
    b, a = butter(FILTER_ORDER, [BANDPASS_HZ[0] / nyq, BANDPASS_HZ[1] / nyq], btype="band")
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
N_CORE = N_STATE + LATENT_DIM
Z_DELAY_DIM = LATENT_DIM * Z_DELAY_COUNT
THETA_HISTORY_DIM = N * (Z_DELAY_COUNT - 1) if THETA_DELAY_DIFFS else 0
N_AUG = N_STATE + Z_DELAY_DIM + THETA_HISTORY_DIM

print("[DATA]  Loaded epilepsy recording", flush=True)
print(f"        samples={len(X_RAW)}, channels={N}, fs={FS:.1f} Hz", flush=True)

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

baseline_all_idx = np.arange(len(STATE) - MAP_LAG_SAMPLES)
baseline_test_idx = baseline_all_idx[baseline_all_idx % TEST_STRIDE == 0]
baseline_train_idx = baseline_all_idx[baseline_all_idx % TEST_STRIDE != 0]
rng = np.random.default_rng(SEED)
if len(baseline_train_idx) > N_TRAIN_MAX:
    baseline_train_idx = np.sort(rng.choice(baseline_train_idx, N_TRAIN_MAX, replace=False))
if len(baseline_test_idx) > N_TEST_MAX:
    baseline_test_idx = np.sort(rng.choice(baseline_test_idx, N_TEST_MAX, replace=False))

print(
    f"[PRE]   used samples={len(STATE)}, map lag={MAP_LAG_SAMPLES} samples "
    f"({MAP_LAG_SAMPLES * DT:.3f}s), baseline train={len(baseline_train_idx)}, "
    f"heldout={len(baseline_test_idx)}",
    flush=True,
)


# ---------------------------------------------------------------------------
# 2. Baseline continuous models for q(t) extraction
# ---------------------------------------------------------------------------
def split_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return state[:, :N], state[:, N : N + N_PAIR]


def split_aug_scaled(state_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state = unscale_aug(state_scaled)
    z_current = state[:, N + N_PAIR : N + N_PAIR + LATENT_DIM]
    return state, state[:, :N], state[:, N : N + N_PAIR], z_current


def delayed_context_scaled(state_scaled: np.ndarray) -> np.ndarray:
    return state_scaled[:, N_STATE + Z_DELAY_DIM :]


def delayed_context_torch(state_scaled: torch.Tensor) -> torch.Tensor:
    return state_scaled[:, N_STATE + Z_DELAY_DIM :]


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


print("[BASE]  Fitting baseline models for residual q(t) ...", flush=True)
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
quiet_fit(
    model_base_phase,
    X=STATE[baseline_train_idx],
    X_dot=THETA_DOT[baseline_train_idx] / phase_scale,
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
    X=STATE[baseline_train_idx],
    X_dot=KAPPA_DOT[baseline_train_idx] / kappa_scale,
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
z_raw = savgol_filter(z_raw, odd_window(Z_SAVGOL_SECONDS, FS), 3, axis=0, mode="interp")
Z, z_mean, z_std = zscore_columns(z_raw)

STATE_AUG = np.hstack([STATE, Z])
core_mean = STATE_AUG.mean(axis=0)
core_std = STATE_AUG.std(axis=0) + 1e-12

delay_offsets = np.arange(Z_DELAY_COUNT, dtype=int) * Z_DELAY_STEP_SAMPLES
max_delay = int(delay_offsets[-1])
start_indices = np.arange(max_delay, len(STATE_AUG) - MAP_LAG_SAMPLES)
z_delay_blocks = [Z[start_indices - offset] for offset in delay_offsets]
theta_delay_blocks = [STATE[start_indices - offset, :N] for offset in delay_offsets]
theta_delay_diffs = [
    theta_delay_blocks[idx] - theta_delay_blocks[idx + 1]
    for idx in range(Z_DELAY_COUNT - 1)
]
history_blocks = theta_delay_diffs if THETA_DELAY_DIFFS else []
INPUT_AUG = np.hstack([STATE[start_indices], *z_delay_blocks, *history_blocks])
aug_mean = INPUT_AUG.mean(axis=0)
aug_std = INPUT_AUG.std(axis=0) + 1e-12
X_MAP = (INPUT_AUG - aug_mean) / aug_std

STATE_AUG_CURRENT = STATE_AUG[start_indices]
STATE_AUG_FUTURE = STATE_AUG[start_indices + MAP_LAG_SAMPLES]
DELTA_AUG = STATE_AUG_FUTURE - STATE_AUG_CURRENT
delta_mean = DELTA_AUG.mean(axis=0)
delta_std = DELTA_AUG.std(axis=0) + 1e-12
Y_MAP = (DELTA_AUG - delta_mean) / delta_std

all_idx = np.arange(len(X_MAP))
test_idx = all_idx[all_idx % TEST_STRIDE == 0]
train_idx = all_idx[all_idx % TEST_STRIDE != 0]
if len(train_idx) > N_TRAIN_MAX:
    train_idx = np.sort(rng.choice(train_idx, N_TRAIN_MAX, replace=False))
if len(test_idx) > N_TEST_MAX:
    test_idx = np.sort(rng.choice(test_idx, N_TEST_MAX, replace=False))

print(
    "[SVD]   q(t) cumulative variance "
    f"rank1={cumulative_variance[0]:.3f}, "
    f"rank3={cumulative_variance[2]:.3f}, "
    f"rank5={cumulative_variance[4]:.3f}, "
    f"rank10={cumulative_variance[9]:.3f}",
    flush=True,
)
print(
    "[DELAY] latent delays "
    f"count={Z_DELAY_COUNT}, step={Z_DELAY_STEP_SAMPLES} samples "
    f"({Z_DELAY_STEP_SAMPLES * DT:.3f}s), max={max_delay} samples "
    f"({max_delay * DT:.3f}s), theta_diffs={THETA_DELAY_DIFFS}, "
    f"map train={len(train_idx)}, heldout={len(test_idx)}",
    flush=True,
)


# ---------------------------------------------------------------------------
# 3. Map-mode lifts: receive scaled augmented state, return features
# ---------------------------------------------------------------------------
def unscale_aug(state_scaled: np.ndarray) -> np.ndarray:
    return state_scaled * aug_std + aug_mean


def scale_aug(state: np.ndarray) -> np.ndarray:
    return (state - aug_mean) / aug_std


def scale_core(state: np.ndarray) -> np.ndarray:
    return (state - core_mean) / core_std


def unnormalize_delta(delta_normalized: np.ndarray) -> np.ndarray:
    return delta_normalized * delta_std + delta_mean


def unscale_aug_torch(state_scaled: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(aug_mean, dtype=state_scaled.dtype, device=state_scaled.device)
    std = torch.tensor(aug_std, dtype=state_scaled.dtype, device=state_scaled.device)
    return state_scaled * std + mean


def mean_field_np(theta_state: np.ndarray) -> np.ndarray:
    delta = theta_state[:, None, :] - theta_state[:, :, None]
    return np.column_stack([np.mean(np.sin(delta), axis=2), np.mean(np.cos(delta), axis=2)])


def mean_field_torch(theta_state: torch.Tensor) -> torch.Tensor:
    delta = theta_state[:, None, :] - theta_state[:, :, None]
    return torch.cat([torch.mean(torch.sin(delta), dim=2), torch.mean(torch.cos(delta), dim=2)], dim=1)


def phase_map_lift_np(state_scaled: np.ndarray) -> np.ndarray:
    physical, theta_state, kappa_state, z_state = split_aug_scaled(state_scaled)
    raw = state_scaled[:, :N]
    base = base_phase_lift_np(physical[:, :N_STATE])
    z_flat = physical[:, N_STATE : N_STATE + Z_DELAY_DIM]
    context = delayed_context_scaled(state_scaled)
    mf = mean_field_np(theta_state)
    z_mf = [
        z_state[:, m] * mf[:, feature_idx]
        for m in range(LATENT_DIM)
        for feature_idx in range(mf.shape[1])
    ]
    return np.column_stack([raw, base, z_flat, context, np.column_stack(z_mf)])


def phase_map_lift_torch(state_scaled: torch.Tensor) -> torch.Tensor:
    physical = unscale_aug_torch(state_scaled)
    theta_state = physical[:, :N]
    z_flat = physical[:, N_STATE : N_STATE + Z_DELAY_DIM]
    context = delayed_context_torch(state_scaled)
    z_state = z_flat[:, :LATENT_DIM]
    raw = state_scaled[:, :N]
    base = base_phase_lift_torch(physical[:, :N_STATE])
    mf = mean_field_torch(theta_state)
    z_mf = [
        z_state[:, m] * mf[:, feature_idx]
        for m in range(LATENT_DIM)
        for feature_idx in range(mf.shape[1])
    ]
    return torch.cat([raw, base, z_flat, context, torch.stack(z_mf, dim=1)], dim=1)


def kappa_map_lift_np(state_scaled: np.ndarray) -> np.ndarray:
    physical, _theta_state, kappa_state, z_state = split_aug_scaled(state_scaled)
    raw = state_scaled[:, N : N + N_PAIR]
    base = base_kappa_lift_np(physical[:, :N_STATE])
    z_flat = physical[:, N_STATE : N_STATE + Z_DELAY_DIM]
    context = delayed_context_scaled(state_scaled)
    zk = [
        z_state[:, m] * kappa_state[:, pair_idx]
        for m in range(LATENT_DIM)
        for pair_idx in range(N_PAIR)
    ]
    return np.column_stack([raw, base, z_flat, context, np.column_stack(zk)])


def kappa_map_lift_torch(state_scaled: torch.Tensor) -> torch.Tensor:
    physical = unscale_aug_torch(state_scaled)
    kappa_state = physical[:, N : N + N_PAIR]
    z_flat = physical[:, N_STATE : N_STATE + Z_DELAY_DIM]
    context = delayed_context_torch(state_scaled)
    z_state = z_flat[:, :LATENT_DIM]
    raw = state_scaled[:, N : N + N_PAIR]
    base = base_kappa_lift_torch(physical[:, :N_STATE])
    zk = [
        z_state[:, m] * kappa_state[:, pair_idx]
        for m in range(LATENT_DIM)
        for pair_idx in range(N_PAIR)
    ]
    return torch.cat([raw, base, z_flat, context, torch.stack(zk, dim=1)], dim=1)


def z_map_lift_np(state_scaled: np.ndarray) -> np.ndarray:
    physical, theta_state, kappa_state, z_state = split_aug_scaled(state_scaled)
    raw = state_scaled[:, N_STATE : N_STATE + Z_DELAY_DIM]
    context = delayed_context_scaled(state_scaled)
    z_flat = physical[:, N_STATE : N_STATE + Z_DELAY_DIM].reshape(len(physical), Z_DELAY_COUNT, LATENT_DIM)
    delay_diffs = [
        z_flat[:, delay_idx, mode] - z_flat[:, delay_idx + 1, mode]
        for delay_idx in range(Z_DELAY_COUNT - 1)
        for mode in range(LATENT_DIM)
    ]
    products = [
        z_state[:, i] * z_state[:, j]
        for i in range(LATENT_DIM)
        for j in range(i + 1, LATENT_DIM)
    ]
    r_sin = np.mean(np.sin(theta_state), axis=1)
    r_cos = np.mean(np.cos(theta_state), axis=1)
    r = np.sqrt(r_sin ** 2 + r_cos ** 2)
    extras = np.column_stack([r, r_sin, r_cos, kappa_state.mean(axis=1)])
    return np.column_stack([raw, context, np.column_stack(delay_diffs), np.column_stack(products), extras])


def z_map_lift_torch(state_scaled: torch.Tensor) -> torch.Tensor:
    physical = unscale_aug_torch(state_scaled)
    theta_state = physical[:, :N]
    kappa_state = physical[:, N : N + N_PAIR]
    z_flat = physical[:, N_STATE : N_STATE + Z_DELAY_DIM].reshape(physical.shape[0], Z_DELAY_COUNT, LATENT_DIM)
    z_state = z_flat[:, 0, :]
    raw = state_scaled[:, N_STATE : N_STATE + Z_DELAY_DIM]
    context = delayed_context_torch(state_scaled)
    delay_diffs = [
        z_flat[:, delay_idx, mode] - z_flat[:, delay_idx + 1, mode]
        for delay_idx in range(Z_DELAY_COUNT - 1)
        for mode in range(LATENT_DIM)
    ]
    products = [
        z_state[:, i] * z_state[:, j]
        for i in range(LATENT_DIM)
        for j in range(i + 1, LATENT_DIM)
    ]
    r_sin = torch.mean(torch.sin(theta_state), dim=1)
    r_cos = torch.mean(torch.cos(theta_state), dim=1)
    r = torch.sqrt(r_sin ** 2 + r_cos ** 2 + 1e-12)
    extras = torch.stack([r, r_sin, r_cos, torch.mean(kappa_state, dim=1)], dim=1)
    return torch.cat(
        [raw, context, torch.stack(delay_diffs, dim=1), torch.stack(products, dim=1), extras],
        dim=1,
    )


PHASE_LIFT_DIM = N + 5 * N_PAIR + Z_DELAY_DIM + THETA_HISTORY_DIM + LATENT_DIM * 2 * N
KAPPA_LIFT_DIM = N_PAIR + 3 * N_PAIR + Z_DELAY_DIM + THETA_HISTORY_DIM + LATENT_DIM * N_PAIR
Z_LIFT_DIM = (
    Z_DELAY_DIM
    + THETA_HISTORY_DIM
    + LATENT_DIM * (Z_DELAY_COUNT - 1)
    + LATENT_DIM * (LATENT_DIM - 1) // 2
    + 4
)
print(
    "[LIFT]  map dims "
    f"phase={PHASE_LIFT_DIM}, kappa={KAPPA_LIFT_DIM}, z={Z_LIFT_DIM}",
    flush=True,
)


# ---------------------------------------------------------------------------
# 4. Fit KANDy finite-lag increment maps
# ---------------------------------------------------------------------------
model_phase = KANDy(
    lift=CustomLift(
        fn=phase_map_lift_np,
        output_dim=PHASE_LIFT_DIM,
        torch_fn=phase_map_lift_torch,
        name="phase_map",
    ),
    grid=GRID,
    k=3,
    steps=MAP_STEPS,
    seed=SEED,
    base_fun=torch.nn.Identity(),
    device="cpu",
)
model_kappa = KANDy(
    lift=CustomLift(
        fn=kappa_map_lift_np,
        output_dim=KAPPA_LIFT_DIM,
        torch_fn=kappa_map_lift_torch,
        name="kappa_map",
    ),
    grid=GRID,
    k=3,
    steps=MAP_STEPS,
    seed=SEED,
    base_fun=torch.nn.Identity(),
    device="cpu",
)
model_z = KANDy(
    lift=CustomLift(
        fn=z_map_lift_np,
        output_dim=Z_LIFT_DIM,
        torch_fn=z_map_lift_torch,
        name="z_map",
    ),
    grid=GRID,
    k=3,
    steps=Z_MAP_STEPS,
    seed=SEED,
    base_fun=torch.nn.Identity(),
    device="cpu",
)

print("[TRAIN] Fitting phase finite-lag increment map ...", flush=True)
quiet_fit(
    model_phase,
    X=X_MAP[train_idx],
    X_dot=Y_MAP[train_idx, :N],
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
print("[TRAIN] Fitting kappa finite-lag increment map ...", flush=True)
quiet_fit(
    model_kappa,
    X=X_MAP[train_idx],
    X_dot=Y_MAP[train_idx, N:N_STATE],
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
print("[TRAIN] Fitting z finite-lag increment map ...", flush=True)
quiet_fit(
    model_z,
    X=X_MAP[train_idx],
    X_dot=Y_MAP[train_idx, N_STATE:],
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
# 5. Validation and autoregressive rollout
# ---------------------------------------------------------------------------
def predict_delta_normalized(x_scaled: np.ndarray) -> np.ndarray:
    x2 = np.atleast_2d(x_scaled)
    return np.column_stack(
        [
            model_phase.predict(x2),
            model_kappa.predict(x2),
            model_z.predict(x2),
        ]
    )


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    denom = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    denom = np.where(denom > 0, denom, np.nan)
    return 1.0 - np.sum((y_true - y_pred) ** 2, axis=0) / denom


pred_delta_norm = predict_delta_normalized(X_MAP[test_idx])
true_delta_norm = Y_MAP[test_idx]
current_test_phys = STATE_AUG_CURRENT[test_idx]
true_future_phys = STATE_AUG_FUTURE[test_idx]
pred_delta_phys = unnormalize_delta(pred_delta_norm)
true_delta_phys = DELTA_AUG[test_idx]
pred_future_phys = current_test_phys + pred_delta_phys
true_future_s = scale_core(true_future_phys)
pred_future_s = scale_core(pred_future_phys)
next_r2_scaled = r2(true_future_s, pred_future_s)
increment_r2_norm = r2(true_delta_norm, pred_delta_norm)
increment_r2_phys = r2(true_delta_phys, pred_delta_phys)

print(
    "[EVAL]  one-step scaled R2 means "
    f"theta={np.nanmean(next_r2_scaled[:N]):.3f}, "
    f"kappa={np.nanmean(next_r2_scaled[N:N_STATE]):.3f}, "
    f"z={np.nanmean(next_r2_scaled[N_STATE:]):.3f}",
    flush=True,
)
print(
    "[EVAL]  normalized increment R2 means "
    f"theta={np.nanmean(increment_r2_norm[:N]):.3f}, "
    f"kappa={np.nanmean(increment_r2_norm[N:N_STATE]):.3f}, "
    f"z={np.nanmean(increment_r2_norm[N_STATE:]):.3f}",
    flush=True,
)
print(
    "[EVAL]  physical increment R2 means "
    f"theta={np.nanmean(increment_r2_phys[:N]):.3f}, "
    f"kappa={np.nanmean(increment_r2_phys[N:N_STATE]):.3f}, "
    f"z={np.nanmean(increment_r2_phys[N_STATE:]):.3f}",
    flush=True,
)

kappa_lo = X_MAP[:, N:N_STATE].min(axis=0)
kappa_hi = X_MAP[:, N:N_STATE].max(axis=0)
z_lo = X_MAP[:, N_STATE:].min(axis=0)
z_hi = X_MAP[:, N_STATE:].max(axis=0)
kappa_span = kappa_hi - kappa_lo
z_span = z_hi - z_lo
kappa_lo = kappa_lo - ROLL_OUT_CLIP_MARGIN * kappa_span
kappa_hi = kappa_hi + ROLL_OUT_CLIP_MARGIN * kappa_span
z_lo = z_lo - ROLL_OUT_CLIP_MARGIN * z_span
z_hi = z_hi + ROLL_OUT_CLIP_MARGIN * z_span


def compose_input(core_phys: np.ndarray, z_queue: np.ndarray, theta_queue: np.ndarray) -> np.ndarray:
    z_blocks = [z_queue[offset] for offset in delay_offsets]
    theta_history = []
    if THETA_DELAY_DIFFS:
        theta_blocks = [theta_queue[offset] for offset in delay_offsets]
        theta_history = [
            theta_blocks[idx] - theta_blocks[idx + 1]
            for idx in range(Z_DELAY_COUNT - 1)
        ]
    return np.concatenate([core_phys[:N_STATE], *z_blocks, *theta_history])


def rollout_map_from_index(start: int, steps: int) -> np.ndarray:
    history_offsets = np.arange(max_delay + 1)
    z_queue = Z[start - history_offsets].copy()
    theta_queue = STATE[start - history_offsets, :N].copy()
    core_phys = STATE_AUG[start].copy()
    input_phys = compose_input(core_phys, z_queue, theta_queue)
    out = np.empty((steps + 1, N_CORE))
    out[0] = core_phys
    for step in range(steps):
        input_scaled = scale_aug(input_phys)
        delta = unnormalize_delta(predict_delta_normalized(input_scaled)[0])
        core_phys = core_phys + delta
        z_queue = np.vstack([core_phys[N_STATE:], z_queue[:-1]])
        theta_queue = np.vstack([core_phys[:N], theta_queue[:-1]])
        input_phys = compose_input(core_phys, z_queue, theta_queue)

        input_scaled = scale_aug(input_phys)
        input_scaled[N:N_STATE] = np.clip(input_scaled[N:N_STATE], kappa_lo, kappa_hi)
        input_scaled[N_STATE:] = np.clip(input_scaled[N_STATE:], z_lo, z_hi)
        input_phys = unscale_aug(input_scaled)
        core_phys = np.concatenate([input_phys[:N_STATE], input_phys[N_STATE : N_STATE + LATENT_DIM]])
        z_queue[0] = core_phys[N_STATE:]
        theta_queue[0] = core_phys[:N]
        input_phys = compose_input(core_phys, z_queue, theta_queue)

        if not np.isfinite(core_phys).all():
            return out[: step + 1]
        out[step + 1] = core_phys
    return out


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


def summarize(pred_phys: np.ndarray, ref_phys: np.ndarray, start_time: float) -> dict:
    r_pred = np.abs(np.mean(np.exp(1j * pred_phys[:, :N]), axis=1))
    r_ref = np.abs(np.mean(np.exp(1j * ref_phys[:, :N]), axis=1))
    return {
        "start_time": float(start_time),
        "samples": int(len(pred_phys)),
        "theta_rmse_wrapped": float(np.sqrt(np.mean(wrap_pi(pred_phys[:, :N] - ref_phys[:, :N]) ** 2))),
        "kappa_rmse": float(np.sqrt(np.mean((pred_phys[:, N:N_STATE] - ref_phys[:, N:N_STATE]) ** 2))),
        "z_rmse": float(np.sqrt(np.mean((pred_phys[:, N_STATE:] - ref_phys[:, N_STATE:]) ** 2))),
        "order_parameter_rmse": float(np.sqrt(np.mean((r_pred - r_ref) ** 2))),
    }


MAP_DT = MAP_LAG_SAMPLES * DT
MAP_ROW_TIME = T[start_indices]
short_steps = int(round(SHORT_ROLLOUT_SECONDS / MAP_DT))
short_rows = []
short_traces = []
for start_s in [-8.0, 0.0, 8.0, 18.0]:
    start_row = int(np.argmin(np.abs(MAP_ROW_TIME - start_s)))
    start = int(start_indices[start_row])
    pred = rollout_map_from_index(start, short_steps)
    trace_idx = start + np.arange(len(pred)) * MAP_LAG_SAMPLES
    valid_len = int(np.sum(trace_idx < len(STATE_AUG)))
    pred = pred[:valid_len]
    trace_idx = trace_idx[:valid_len]
    ref = STATE_AUG[trace_idx]
    row = summarize(pred, ref, T[start])
    short_rows.append(row)
    short_traces.append(
        {
            "start_time": float(T[start]),
            "time": T[trace_idx] - T[start],
            "pred": pred,
            "ref": ref,
        }
    )

event_start = max(EVENT_WINDOW[0], float(T[0]))
event_end = min(EVENT_WINDOW[1], float(T[-1]))
event_steps = int(round((event_end - event_start) / MAP_DT))
event_row = int(np.argmin(np.abs(MAP_ROW_TIME - event_start)))
event_idx = int(start_indices[event_row])
event_pred = rollout_map_from_index(event_idx, event_steps)
event_trace_idx = event_idx + np.arange(len(event_pred)) * MAP_LAG_SAMPLES
event_valid_len = int(np.sum(event_trace_idx < len(STATE_AUG)))
event_pred = event_pred[:event_valid_len]
event_trace_idx = event_trace_idx[:event_valid_len]
event_ref = STATE_AUG[event_trace_idx]
event_time = T[event_trace_idx]
event_signal_pred = state_to_signal(event_pred[:, :N_STATE])
event_signal_ref = SIGNAL[event_trace_idx]
event_order_pred = np.abs(np.mean(np.exp(1j * event_pred[:, :N]), axis=1))
event_order_ref = np.abs(np.mean(np.exp(1j * event_ref[:, :N]), axis=1))
event_corr = [
    float(np.corrcoef(event_signal_pred[:, i], event_signal_ref[:, i])[0, 1])
    for i in range(N)
]
event_metrics = {
    **summarize(event_pred, event_ref, event_time[0]),
    "end_time": float(event_time[-1]),
    "signal_rmse": float(np.sqrt(np.mean((event_signal_pred - event_signal_ref) ** 2))),
    "signal_corr_mean": float(np.mean(event_corr)),
    "signal_corr_per_channel": event_corr,
}

print("[ROLLOUT] short delayed-latent map rollouts:", flush=True)
for row in short_rows:
    print(
        "          "
        f"t0={row['start_time']:.2f}s, theta={row['theta_rmse_wrapped']:.3f}, "
        f"kappa={row['kappa_rmse']:.3f}, z={row['z_rmse']:.3f}, "
        f"r={row['order_parameter_rmse']:.3f}",
        flush=True,
    )
print(
    "[EVENT] delayed-latent map replay "
    f"{event_metrics['start_time']:.2f}s..{event_metrics['end_time']:.2f}s: "
    f"signal_corr={event_metrics['signal_corr_mean']:.3f}, "
    f"signal_rmse={event_metrics['signal_rmse']:.3f}, "
    f"r_rmse={event_metrics['order_parameter_rmse']:.3f}",
    flush=True,
)

thresholds = {
    "increment_r2_mean_min": 0.20,
    "short_order_rmse_mean_max": 0.35,
    "event_signal_corr_mean_min": 0.25,
}
passed = print_check(
    float(np.nanmean(increment_r2_phys)) >= thresholds["increment_r2_mean_min"]
    and float(np.mean([row["order_parameter_rmse"] for row in short_rows]))
    <= thresholds["short_order_rmse_mean_max"]
    and event_metrics["signal_corr_mean"] >= thresholds["event_signal_corr_mean_min"],
    "discrete delayed-latent increment map has measurable rollout skill",
)

metrics = {
    "task": "discrete delayed-latent adaptive Kuramoto map",
    "data_path": str(DATA_PATH),
    "labels": LABELS,
    "latent_dim": LATENT_DIM,
    "core_state_dim": N_CORE,
    "input_state_dim": N_AUG,
    "samples_used": int(len(STATE_AUG)),
    "map_rows": int(len(X_MAP)),
    "lift_dims": {
        "phase": PHASE_LIFT_DIM,
        "kappa": KAPPA_LIFT_DIM,
        "z": Z_LIFT_DIM,
    },
    "fit": {
        "grid": GRID,
        "baseline_steps": BASELINE_STEPS,
        "map_steps": MAP_STEPS,
        "z_map_steps": Z_MAP_STEPS,
        "n_train": int(len(train_idx)),
        "n_heldout": int(len(test_idx)),
        "map_lag_samples": MAP_LAG_SAMPLES,
        "map_lag_seconds": MAP_DT,
        "z_delay_count": Z_DELAY_COUNT,
        "z_delay_step_samples": Z_DELAY_STEP_SAMPLES,
        "z_delay_step_seconds": Z_DELAY_STEP_SAMPLES * DT,
        "z_delay_max_seconds": max_delay * DT,
        "theta_delay_diffs": THETA_DELAY_DIFFS,
        "theta_history_dim": THETA_HISTORY_DIM,
        "map_mode": "X=[theta,kappa,delayed z, theta delay differences] scaled state, X_dot=normalized finite-lag core increment",
        "rollout_projection": "kappa and z clipped to observed scaled range plus margin",
        "rollout_clip_margin": ROLL_OUT_CLIP_MARGIN,
    },
    "q_svd_variance_ratio": [float(v) for v in variance_ratio],
    "q_svd_cumulative_variance": [float(v) for v in cumulative_variance],
    "next_scaled_r2": [float(v) for v in next_r2_scaled],
    "next_scaled_r2_mean": float(np.nanmean(next_r2_scaled)),
    "normalized_increment_r2": [float(v) for v in increment_r2_norm],
    "normalized_increment_r2_mean": float(np.nanmean(increment_r2_norm)),
    "increment_r2": [float(v) for v in increment_r2_phys],
    "increment_r2_mean": float(np.nanmean(increment_r2_phys)),
    "short_rollouts": short_rows,
    "short_order_rmse_mean": float(np.mean([row["order_parameter_rmse"] for row in short_rows])),
    "event_replay": event_metrics,
    "thresholds": thresholds,
    "passed": passed,
}
save_metrics(RESULT_DIR / "metrics.json", metrics)

torch.save(
    {
        "phase_map_state_dict": model_phase.model_.state_dict(),
        "kappa_map_state_dict": model_kappa.model_.state_dict(),
        "z_map_state_dict": model_z.model_.state_dict(),
        "aug_mean": torch.tensor(aug_mean),
        "aug_std": torch.tensor(aug_std),
        "core_mean": torch.tensor(core_mean),
        "core_std": torch.tensor(core_std),
        "delta_mean": torch.tensor(delta_mean),
        "delta_std": torch.tensor(delta_std),
        "labels": LABELS,
        "pair_features": PAIR_FEATURES,
        "latent_dim": LATENT_DIM,
        "metadata": metrics["fit"],
    },
    RESULT_DIR / "fitted_latent_map_state.pt",
)
np.savez_compressed(
    RESULT_DIR / "latent_map_arrays.npz",
    t=T,
    labels=np.asarray(LABELS),
    state_aug=STATE_AUG,
    input_aug=INPUT_AUG,
    input_aug_scaled=X_MAP,
    start_indices=start_indices,
    delay_offsets=delay_offsets,
    theta_delay_diffs=np.asarray(theta_delay_diffs) if THETA_DELAY_DIFFS else np.empty((0,)),
    delta_aug=DELTA_AUG,
    aug_mean=aug_mean,
    aug_std=aug_std,
    core_mean=core_mean,
    core_std=core_std,
    delta_mean=delta_mean,
    delta_std=delta_std,
    q_dot=q_dot,
    q_normalized=q_normalized,
    q_svd_scores=U * S,
    q_svd_singular_values=S,
    q_svd_components=Vt,
    q_svd_variance_ratio=variance_ratio,
    q_svd_cumulative_variance=cumulative_variance,
    z=Z,
    event_time=event_time,
    event_signal_ref=event_signal_ref,
    event_signal_pred=event_signal_pred,
    event_state_ref=event_ref,
    event_state_pred=event_pred,
    event_order_ref=event_order_ref,
    event_order_pred=event_order_pred,
)


# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
def plot_svd() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8))
    modes = np.arange(1, min(12, len(S)) + 1)
    axes[0].bar(modes, variance_ratio[: len(modes)], color="#1f77b4", alpha=0.75)
    axes[0].plot(modes, cumulative_variance[: len(modes)], marker="o", color="#d62728")
    axes[0].set_ylabel("variance")
    axes[0].set_title("Residual drive q(t) SVD for delayed latent map")
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
    fig.savefig(RESULT_DIR / "latent_map_svd.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "latent_map_svd.pdf", dpi=250, bbox_inches="tight")
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
        ax_r.plot(t_local, r_pred, lw=1.0, ls="--", label="latent map")
        ax_sig.set_ylabel(f"t0={trace['start_time']:.1f}s")
        ax_r.set_ylim(0, 1.05)
    axes[0, 0].set_title("cos(theta)")
    axes[0, 1].set_title("order parameter")
    axes[0, 0].legend(ncol=3, fontsize=7)
    axes[0, 1].legend(fontsize=7)
    axes[-1, 0].set_xlabel("time since start (s)")
    axes[-1, 1].set_xlabel("time since start (s)")
    fig.suptitle("Discrete delayed-latent KANDy short rollouts")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "latent_map_short_rollouts.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "latent_map_short_rollouts.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_event_replay() -> None:
    fig, axes = plt.subplots(N + 1, 1, figsize=(11, 1.35 * (N + 1)), sharex=True)
    for ch_idx, ax in enumerate(axes[:N]):
        ax.plot(event_time, event_signal_ref[:, ch_idx], lw=0.65, color="#1f77b4")
        ax.plot(event_time, event_signal_pred[:, ch_idx], lw=0.8, ls="--", color="#d62728")
        ax.axvline(0.0, color="0.2", lw=0.8, ls=":")
        ax.set_ylabel(LABELS[ch_idx], rotation=0, ha="right", va="center", fontsize=8)
    axes[-1].plot(event_time, event_order_ref, lw=1.0, color="#1f77b4", label="data")
    axes[-1].plot(event_time, event_order_pred, lw=1.0, ls="--", color="#d62728", label="latent map")
    axes[-1].axvline(0.0, color="0.2", lw=0.8, ls=":")
    axes[-1].set_ylabel("order", rotation=0, ha="right", va="center", fontsize=8)
    axes[-1].set_xlabel("recording time (s)")
    axes[-1].set_ylim(0, 1.05)
    axes[-1].legend(fontsize=8)
    fig.suptitle("Seizure-window autonomous replay with delayed 10D latent map")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "latent_map_seizure_replay.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "latent_map_seizure_replay.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_losses() -> None:
    for name, model in [
        ("loss_phase_map", model_phase),
        ("loss_kappa_map", model_kappa),
        ("loss_z_map", model_z),
    ]:
        if getattr(model, "train_results_", None):
            fig, _ax = plot_loss_curves(model.train_results_, save=str(RESULT_DIR / name))
            plt.close(fig)


use_pub_style()
plot_svd()
plot_short_rollouts()
plot_event_replay()
plot_losses()

print(f"[FIGS]  Saved {RESULT_DIR}", flush=True)
