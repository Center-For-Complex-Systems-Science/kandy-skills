#!/usr/bin/env python3
"""Fit an adaptive-Kuramoto-style KANDy model to the epilepsy recording.

The data in ``data/Dory_2021_04_12_006_preprocessed.pkl`` contains six
standardized neural time series, not simulated Kuramoto states.  The measured
state is therefore converted into oscillator variables before fitting:

    signal channel -> band-passed analytic signal -> phase theta_i(t)
    analytic amplitude envelopes -> pairwise adaptive coupling proxy kappa_ij(t)

This follows the adaptive Kuramoto example's modeling shape:

    dtheta_i/dt = F_i({kappa_ab * sin(theta_b - theta_a),
                      kappa_ab * cos(theta_b - theta_a),
                      sin(theta_b - theta_a),
                      cos(theta_b - theta_a),
                      kappa_ab})

    dkappa_ab/dt = G_ab({sin(theta_b - theta_a), cos(theta_b - theta_a), kappa_ab})

The important limitation is that ``kappa_ij`` is not observed as a physical
synaptic or network coupling.  Here it is a data-derived, slowly varying
amplitude-envelope proxy.  The model is useful as a compact oscillator
surrogate/diagnostic for this recording, not as a causal connectivity claim.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import pickle
import sys
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
from scipy.signal import butter, filtfilt, hilbert, savgol_filter, welch

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
RESULT_DIR = ensure_outdir(REPO_ROOT / "results" / "Epilepsy-Adaptive-Kuramoto")

BANDPASS_HZ = (1.0, 40.0)
FILTER_ORDER = 4
EDGE_TRIM_SECONDS = 2.0
PHASE_SAVGOL_SECONDS = 0.155
KAPPA_SAVGOL_SECONDS = 0.505

FIT_STEPS = 24
GRID = 4
FORCING_SAVGOL_SECONDS = 0.055
N_TRAIN_MAX = 3000
N_TEST_MAX = 1500
TEST_STRIDE = 5
ROLLOUT_SECONDS = 0.25
ROLLOUT_START_SECONDS = [-8.0, 0.0, 8.0, 18.0]
NONAUTONOMOUS_NUDGE_RATE = 20.0
SEIZURE_REPLAY_WINDOW = (-1.0, 4.0)


# ---------------------------------------------------------------------------
# 1. Data loading and preprocessing
# ---------------------------------------------------------------------------
def odd_window(seconds: float, fs: float, minimum: int = 5) -> int:
    """Convert a time span to an odd Savitzky-Golay window length."""
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


def load_epilepsy_data(path: Path) -> dict:
    with path.open("rb") as fh:
        payload = pickle.load(fh)

    X = np.asarray(payload["X"], dtype=float)
    fs = float(payload["fs"])
    dt = float(payload["dt"])
    t = np.asarray(payload["t"], dtype=float)
    labels = list(payload["labels"])
    return {"X": X, "fs": fs, "dt": dt, "t": t, "labels": labels}


data = load_epilepsy_data(DATA_PATH)
X_RAW = data["X"]
FS = data["fs"]
DT = data["dt"]
T_ALL = data["t"]
LABELS = data["labels"]
N = X_RAW.shape[1]

PAIR_FEATURES = [(i, j) for i in range(N) for j in range(i + 1, N)]
PAIR_INDEX = {pair: idx for idx, pair in enumerate(PAIR_FEATURES)}
OD_PAIRS = [(i, j) for i in range(N) for j in range(N) if i != j]
N_PAIR = len(PAIR_FEATURES)
N_OD = len(OD_PAIRS)
N_STATE = N + N_PAIR

PHASE_FEATURE_NAMES = [
    name
    for i, j in PAIR_FEATURES
    for name in (
        f"k{i}{j}*s{i}{j}",
        f"k{i}{j}*c{i}{j}",
        f"s{i}{j}",
        f"c{i}{j}",
        f"k{i}{j}",
    )
]
KAPPA_FEATURE_NAMES = (
    [f"s{i}{j}" for i, j in PAIR_FEATURES]
    + [f"c{i}{j}" for i, j in PAIR_FEATURES]
    + [f"k{i}{j}" for i, j in PAIR_FEATURES]
)

print("[DATA]  Loaded epilepsy recording")
print(f"        path={DATA_PATH}")
print(f"        samples={len(X_RAW)}, channels={N}, fs={FS:.1f} Hz, dt={DT:.4f}s")
print(f"        labels={LABELS}")

X_FILT = bandpass(X_RAW, FS)
analytic = hilbert(X_FILT, axis=0)
theta_unwrapped = np.unwrap(np.angle(analytic), axis=0)

phase_window = odd_window(PHASE_SAVGOL_SECONDS, FS)
theta_smooth = savgol_filter(
    theta_unwrapped,
    phase_window,
    3,
    axis=0,
    mode="interp",
)
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
log_amp, _amp_mean, _amp_std = zscore_columns(np.log(amplitude + 1e-6))
pair_log_amp = np.column_stack(
    [0.5 * (log_amp[:, i] + log_amp[:, j]) for i, j in PAIR_FEATURES]
)
kappa_window = odd_window(KAPPA_SAVGOL_SECONDS, FS)
kappa_raw = savgol_filter(
    pair_log_amp,
    kappa_window,
    3,
    axis=0,
    mode="interp",
)
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
STATE = np.hstack([theta_smooth[valid], kappa[valid]])
THETA_DOT = theta_dot[valid]
KAPPA_DOT = kappa_dot[valid]
SIGNAL = X_FILT[valid]

phase_scale = THETA_DOT.std(axis=0) + 1e-12
kappa_scale = KAPPA_DOT.std(axis=0) + 1e-12

print(f"[PRE]   bandpass={BANDPASS_HZ[0]:.1f}-{BANDPASS_HZ[1]:.1f} Hz")
print(f"[PRE]   trimmed {EDGE_TRIM_SECONDS:.1f}s per edge -> {len(STATE)} samples")
print(f"[PRE]   phase Savitzky-Golay window={phase_window} samples")
print(f"[PRE]   kappa proxy window={kappa_window} samples")
print(f"[PRE]   state dimension={N_STATE} ({N} phases + {N_PAIR} kappa proxies)")
print(f"[PRE]   phase target std={np.round(phase_scale, 3)}")
print(
    "[PRE]   kappa target std range="
    f"{kappa_scale.min():.3f}..{kappa_scale.max():.3f}"
)


# ---------------------------------------------------------------------------
# 2. Lifts, matching the adaptive Kuramoto example
# ---------------------------------------------------------------------------
def pair_for(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def phase_lift_np(state: np.ndarray) -> np.ndarray:
    """Adaptive products plus fixed coupling terms for unordered phase pairs."""
    theta = state[:, :N]
    kappa_state = state[:, N:]
    features = []
    for i, j in PAIR_FEATURES:
        k = kappa_state[:, PAIR_INDEX[(i, j)]]
        delta = theta[:, j] - theta[:, i]
        sin_delta = np.sin(delta)
        cos_delta = np.cos(delta)
        features.extend([k * sin_delta, k * cos_delta, sin_delta, cos_delta, k])
    return np.column_stack(features)


def phase_lift_torch(state: torch.Tensor) -> torch.Tensor:
    """Torch version of phase_lift_np."""
    theta = state[:, :N]
    kappa_state = state[:, N:]
    features = []
    for i, j in PAIR_FEATURES:
        k = kappa_state[:, PAIR_INDEX[(i, j)]]
        delta = theta[:, j] - theta[:, i]
        sin_delta = torch.sin(delta)
        cos_delta = torch.cos(delta)
        features.extend([k * sin_delta, k * cos_delta, sin_delta, cos_delta, k])
    return torch.stack(features, dim=1)


def kappa_lift_np(state: np.ndarray) -> np.ndarray:
    """[sin(delta), cos(delta), kappa] for each unordered phase pair."""
    theta = state[:, :N]
    kappa_state = state[:, N:]
    sin_features = []
    cos_features = []
    for i, j in PAIR_FEATURES:
        delta = theta[:, j] - theta[:, i]
        sin_features.append(np.sin(delta))
        cos_features.append(np.cos(delta))
    return np.column_stack(
        sin_features
        + cos_features
        + [kappa_state[:, idx] for idx in range(N_PAIR)]
    )


def kappa_lift_torch(state: torch.Tensor) -> torch.Tensor:
    """Torch version of kappa_lift_np."""
    theta = state[:, :N]
    kappa_state = state[:, N:]
    sin_features = []
    cos_features = []
    for i, j in PAIR_FEATURES:
        delta = theta[:, j] - theta[:, i]
        sin_features.append(torch.sin(delta))
        cos_features.append(torch.cos(delta))
    return torch.stack(
        sin_features
        + cos_features
        + [kappa_state[:, idx] for idx in range(N_PAIR)],
        dim=1,
    )


phase_design = np.column_stack([np.ones(len(STATE)), phase_lift_np(STATE)])
kappa_design = np.column_stack([np.ones(len(STATE)), kappa_lift_np(STATE)])
print(f"[LIFT]  phase lift: {len(PHASE_FEATURE_NAMES)} features")
print(f"[LIFT]  kappa lift: {len(KAPPA_FEATURE_NAMES)} features")
print(f"[LIFT]  phase design cond={np.linalg.cond(phase_design):.2e}")
print(f"[LIFT]  kappa design cond={np.linalg.cond(kappa_design):.2e}")


# ---------------------------------------------------------------------------
# 3. Train/test split and model fitting
# ---------------------------------------------------------------------------
all_idx = np.arange(len(STATE))
test_idx = all_idx[all_idx % TEST_STRIDE == 0]
train_idx = all_idx[all_idx % TEST_STRIDE != 0]
rng = np.random.default_rng(SEED)
if len(train_idx) > N_TRAIN_MAX:
    train_idx = np.sort(rng.choice(train_idx, N_TRAIN_MAX, replace=False))
if len(test_idx) > N_TEST_MAX:
    test_idx = np.sort(rng.choice(test_idx, N_TEST_MAX, replace=False))

print(f"[SPLIT] train={len(train_idx)}, heldout={len(test_idx)}")

phase_lift = CustomLift(
    fn=phase_lift_np,
    output_dim=len(PHASE_FEATURE_NAMES),
    torch_fn=phase_lift_torch,
    name="epilepsy_adaptive_phase_products",
)
kappa_lift = CustomLift(
    fn=kappa_lift_np,
    output_dim=len(KAPPA_FEATURE_NAMES),
    torch_fn=kappa_lift_torch,
    name="epilepsy_amplitude_proxy_kappa",
)

model_phase = KANDy(
    lift=phase_lift,
    grid=GRID,
    k=3,
    steps=FIT_STEPS,
    seed=SEED,
    base_fun=torch.nn.Identity(),
    device="cpu",
)
model_kappa = KANDy(
    lift=kappa_lift,
    grid=GRID,
    k=3,
    steps=FIT_STEPS,
    seed=SEED,
    base_fun=torch.nn.Identity(),
    device="cpu",
)


def quiet_fit(model: KANDy, **kwargs) -> None:
    """Fit while suppressing PyKAN checkpoint/progress chatter."""
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(**kwargs)


print("[TRAIN] Fitting phase model ...")
quiet_fit(
    model_phase,
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

print("[TRAIN] Fitting kappa-proxy model ...")
quiet_fit(
    model_kappa,
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

T_MIN = float(T[0])
T_MAX = float(T[-1])


# ---------------------------------------------------------------------------
# 4. Validation and readout
# ---------------------------------------------------------------------------
def r2_per_output(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    denom = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    denom = np.where(denom > 0, denom, np.nan)
    return 1.0 - np.sum((y_true - y_pred) ** 2, axis=0) / denom


def effective_linear_readout(
    model: KANDy,
    lift_fn,
    state: np.ndarray,
    target_scale: np.ndarray,
) -> np.ndarray:
    """Least-squares linear surrogate of physical model predictions."""
    phi = lift_fn(state)
    design = np.column_stack([np.ones(len(phi)), phi])
    coef, *_ = np.linalg.lstsq(
        design,
        model.predict(state) * target_scale,
        rcond=None,
    )
    return coef.T


phase_pred_train = model_phase.predict(STATE[train_idx]) * phase_scale
phase_pred_test = model_phase.predict(STATE[test_idx]) * phase_scale
kappa_pred_train = model_kappa.predict(STATE[train_idx]) * kappa_scale
kappa_pred_test = model_kappa.predict(STATE[test_idx]) * kappa_scale
AUTONOMOUS_DOT = np.hstack(
    [
        model_phase.predict(STATE) * phase_scale,
        model_kappa.predict(STATE) * kappa_scale,
    ]
)
STATE_DOT = np.hstack([THETA_DOT, KAPPA_DOT])

phase_r2_train = r2_per_output(THETA_DOT[train_idx], phase_pred_train)
phase_r2_test = r2_per_output(THETA_DOT[test_idx], phase_pred_test)
kappa_r2_train = r2_per_output(KAPPA_DOT[train_idx], kappa_pred_train)
kappa_r2_test = r2_per_output(KAPPA_DOT[test_idx], kappa_pred_test)

print(f"[EVAL]  phase R2 train={np.round(phase_r2_train, 3)}")
print(f"[EVAL]  phase R2 heldout={np.round(phase_r2_test, 3)}")
print(
    "[EVAL]  kappa R2 heldout "
    f"mean={np.nanmean(kappa_r2_test):.3f}, "
    f"min={np.nanmin(kappa_r2_test):.3f}, max={np.nanmax(kappa_r2_test):.3f}"
)

RESIDUAL_DOT = STATE_DOT - AUTONOMOUS_DOT
residual_scale = RESIDUAL_DOT[train_idx].std(axis=0) + 1e-12
forcing_window = odd_window(FORCING_SAVGOL_SECONDS, FS)
FORCING_DOT = savgol_filter(
    RESIDUAL_DOT,
    forcing_window,
    3,
    axis=0,
    mode="interp",
)
NONAUTONOMOUS_DOT = AUTONOMOUS_DOT + FORCING_DOT
phase_r2_forced_test = r2_per_output(
    THETA_DOT[test_idx],
    NONAUTONOMOUS_DOT[test_idx, :N],
)
kappa_r2_forced_test = r2_per_output(
    KAPPA_DOT[test_idx],
    NONAUTONOMOUS_DOT[test_idx, N:],
)
print(
    "[FORCE] sampled non-autonomous residual q(t) "
    f"window={forcing_window} samples"
)
print(
    "[EVAL]  driven derivative R2 on replay samples "
    f"phase_mean={np.nanmean(phase_r2_forced_test):.3f}, "
    f"kappa_mean={np.nanmean(kappa_r2_forced_test):.3f}"
)

Q_NORMALIZED = FORCING_DOT / np.concatenate([phase_scale, kappa_scale])
q_svd_mean = Q_NORMALIZED.mean(axis=0)
q_svd_std = Q_NORMALIZED.std(axis=0) + 1e-12
Q_SVD = (Q_NORMALIZED - q_svd_mean) / q_svd_std
U_Q, S_Q, Vt_Q = np.linalg.svd(Q_SVD, full_matrices=False)
q_variance_ratio = (S_Q ** 2) / np.sum(S_Q ** 2)
q_cumulative_variance = np.cumsum(q_variance_ratio)
print(
    "[SVD]   q(t) variance explained "
    f"rank1={q_cumulative_variance[0]:.3f}, "
    f"rank3={q_cumulative_variance[min(2, len(q_cumulative_variance)-1)]:.3f}, "
    f"rank5={q_cumulative_variance[min(4, len(q_cumulative_variance)-1)]:.3f}"
)

A_phase = effective_linear_readout(
    model_phase,
    phase_lift_np,
    STATE[train_idx],
    phase_scale,
)
A_kappa = effective_linear_readout(
    model_kappa,
    kappa_lift_np,
    STATE[train_idx],
    kappa_scale,
)

phase_abs_coef = np.abs(A_phase[:, 1:])
directed_strength = np.zeros((N, N))
for pair_idx, (a, b) in enumerate(PAIR_FEATURES):
    start = 5 * pair_idx
    for target, source in ((a, b), (b, a)):
        directed_strength[target, source] = float(
            np.linalg.norm(phase_abs_coef[target, start : start + 5])
        )

top_edges = []
for i, j in sorted(
    OD_PAIRS,
    key=lambda pair: directed_strength[pair[0], pair[1]],
    reverse=True,
)[:8]:
    top_edges.append(
        {
            "source": LABELS[j],
            "target": LABELS[i],
            "source_index": int(j),
            "target_index": int(i),
            "strength": float(directed_strength[i, j]),
        }
    )

print("[READOUT] strongest phase-equation directed terms:")
for edge_info in top_edges[:5]:
    print(
        "          "
        f"{edge_info['source']} -> {edge_info['target']}: "
        f"{edge_info['strength']:.4f}"
    )


def interpolate_columns(values: np.ndarray, time_value: float) -> np.ndarray:
    """Interpolate a matrix whose first axis is aligned to T."""
    return np.array(
        [np.interp(time_value, T, values[:, col]) for col in range(values.shape[1])]
    )


def reference_state_at(time_value: float) -> np.ndarray:
    return interpolate_columns(STATE, time_value)


def learned_rhs(state: np.ndarray) -> np.ndarray:
    state_2d = state[None, :]
    dtheta = (model_phase.predict(state_2d) * phase_scale).ravel()
    dkappa = (model_kappa.predict(state_2d) * kappa_scale).ravel()
    return np.concatenate([dtheta, dkappa])


def nonautonomous_rhs(
    state: np.ndarray,
    time_value: float,
    nudge_rate: float = 0.0,
) -> np.ndarray:
    """Driven replay RHS: f_KANDy(x) + q(t) + lambda*(x_data(t)-x)."""
    rhs = learned_rhs(state) + interpolate_columns(FORCING_DOT, time_value)
    if nudge_rate > 0:
        rhs = rhs + nudge_rate * (reference_state_at(time_value) - state)
    return rhs


def rollout_autonomous(state0: np.ndarray, steps: int, dt: float) -> np.ndarray:
    out = np.empty((steps + 1, len(state0)))
    x = state0.astype(float).copy()
    out[0] = x
    for step in range(steps):
        k1 = learned_rhs(x)
        k2 = learned_rhs(x + 0.5 * dt * k1)
        k3 = learned_rhs(x + 0.5 * dt * k2)
        k4 = learned_rhs(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(x).all():
            return out[: step + 1]
        out[step + 1] = x
    return out


def rollout_nonautonomous(
    state0: np.ndarray,
    start_time: float,
    steps: int,
    dt: float,
    nudge_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Euler replay on the measured sampling grid for the time-driven model."""
    out = np.empty((steps + 1, len(state0)))
    times = start_time + np.arange(steps + 1) * dt
    x = state0.astype(float).copy()
    out[0] = x
    for step in range(steps):
        x = x + dt * nonautonomous_rhs(x, times[step], nudge_rate=nudge_rate)
        if not np.isfinite(x).all():
            return times[: step + 1], out[: step + 1]
        out[step + 1] = x
    return times, out


def rollout_summary(
    pred: np.ndarray,
    ref: np.ndarray,
    start_time: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    r_pred = np.abs(np.mean(np.exp(1j * pred[:, :N]), axis=1))
    r_ref = np.abs(np.mean(np.exp(1j * ref[:, :N]), axis=1))
    row = {
        "start_time": float(start_time),
        "samples": int(len(pred)),
        "theta_rmse_wrapped": float(
            np.sqrt(np.mean(wrap_pi(pred[:, :N] - ref[:, :N]) ** 2))
        ),
        "kappa_rmse": float(np.sqrt(np.mean((pred[:, N:] - ref[:, N:]) ** 2))),
        "order_parameter_rmse": float(np.sqrt(np.mean((r_pred - r_ref) ** 2))),
    }
    return row, r_pred, r_ref


rollout_steps = int(round(ROLLOUT_SECONDS / DT))
autonomous_rollout_rows = []
nonautonomous_rollout_rows = []
rollout_traces = []
for start_s in ROLLOUT_START_SECONDS:
    start = int(np.argmin(np.abs(T - start_s)))
    if start + rollout_steps >= len(STATE):
        continue
    pred_auto = rollout_autonomous(STATE[start], rollout_steps, DT)
    ref_auto = STATE[start : start + len(pred_auto)]
    auto_row, _auto_r_pred, _auto_r_ref = rollout_summary(
        pred_auto,
        ref_auto,
        T[start],
    )
    autonomous_rollout_rows.append(auto_row)

    forced_times, pred = rollout_nonautonomous(
        STATE[start],
        T[start],
        rollout_steps,
        DT,
        nudge_rate=NONAUTONOMOUS_NUDGE_RATE,
    )
    ref = np.vstack([reference_state_at(t_value) for t_value in forced_times])
    forced_row, r_pred, r_ref = rollout_summary(pred, ref, T[start])
    nonautonomous_rollout_rows.append(forced_row)
    rollout_traces.append(
        {
            "start_time": float(T[start]),
            "time": forced_times - T[start],
            "pred": pred,
            "ref": ref,
            "r_pred": r_pred,
            "r_ref": r_ref,
        }
    )

print("[ROLLOUT] short autonomous rollouts:")
for row in autonomous_rollout_rows:
    print(
        "          "
        f"t0={row['start_time']:.2f}s, "
        f"theta_rmse={row['theta_rmse_wrapped']:.3f}, "
        f"kappa_rmse={row['kappa_rmse']:.3f}, "
        f"r_rmse={row['order_parameter_rmse']:.3f}"
    )

print("[ROLLOUT] short non-autonomous replay:")
for row in nonautonomous_rollout_rows:
    print(
        "          "
        f"t0={row['start_time']:.2f}s, "
        f"theta_rmse={row['theta_rmse_wrapped']:.3f}, "
        f"kappa_rmse={row['kappa_rmse']:.3f}, "
        f"r_rmse={row['order_parameter_rmse']:.3f}"
    )

pair_average_matrix = np.zeros((N_PAIR, N))
for row_idx, (i, j) in enumerate(PAIR_FEATURES):
    pair_average_matrix[row_idx, i] = 0.5
    pair_average_matrix[row_idx, j] = 0.5
pair_to_channel = np.linalg.pinv(pair_average_matrix)


def state_to_signal(state: np.ndarray) -> np.ndarray:
    """Reconstruct a band-limited signal proxy from theta and kappa states."""
    state_2d = np.atleast_2d(state)
    pair_log_amp = state_2d[:, N:] * kappa_std + kappa_mean
    channel_log_amp_z = pair_log_amp @ pair_to_channel.T
    channel_log_amp_z = np.clip(channel_log_amp_z, -5.0, 5.0)
    amp_hat = np.exp(channel_log_amp_z * _amp_std + _amp_mean)
    return amp_hat * np.cos(state_2d[:, :N])


def reference_signal_at(time_value: float) -> np.ndarray:
    return interpolate_columns(SIGNAL, time_value)


event_start = max(SEIZURE_REPLAY_WINDOW[0], T_MIN)
event_end = min(SEIZURE_REPLAY_WINDOW[1], T_MAX)
event_steps = int(round((event_end - event_start) / DT))
event_start_idx = int(np.argmin(np.abs(T - event_start)))
event_times, event_pred_state = rollout_nonautonomous(
    STATE[event_start_idx],
    T[event_start_idx],
    event_steps,
    DT,
    nudge_rate=NONAUTONOMOUS_NUDGE_RATE,
)
event_ref_state = np.vstack([reference_state_at(t_value) for t_value in event_times])
event_ref_signal = np.vstack([reference_signal_at(t_value) for t_value in event_times])
event_pred_signal = state_to_signal(event_pred_state)
event_r_pred = np.abs(np.mean(np.exp(1j * event_pred_state[:, :N]), axis=1))
event_r_ref = np.abs(np.mean(np.exp(1j * event_ref_state[:, :N]), axis=1))
event_signal_rmse = float(np.sqrt(np.mean((event_pred_signal - event_ref_signal) ** 2)))
event_signal_corr = [
    float(np.corrcoef(event_pred_signal[:, i], event_ref_signal[:, i])[0, 1])
    for i in range(N)
]
event_theta_rmse = float(
    np.sqrt(np.mean(wrap_pi(event_pred_state[:, :N] - event_ref_state[:, :N]) ** 2))
)
event_kappa_rmse = float(
    np.sqrt(np.mean((event_pred_state[:, N:] - event_ref_state[:, N:]) ** 2))
)
event_order_rmse = float(np.sqrt(np.mean((event_r_pred - event_r_ref) ** 2)))
event_replay = {
    "start_time": float(event_times[0]),
    "end_time": float(event_times[-1]),
    "samples": int(len(event_times)),
    "signal_rmse": event_signal_rmse,
    "signal_corr_mean": float(np.mean(event_signal_corr)),
    "signal_corr_per_channel": event_signal_corr,
    "theta_rmse_wrapped": event_theta_rmse,
    "kappa_rmse": event_kappa_rmse,
    "order_parameter_rmse": event_order_rmse,
}
print(
    "[SEIZURE] non-autonomous replay "
    f"{event_times[0]:.2f}s..{event_times[-1]:.2f}s: "
    f"signal_corr_mean={event_replay['signal_corr_mean']:.3f}, "
    f"signal_rmse={event_signal_rmse:.3f}, "
    f"r_rmse={event_order_rmse:.3f}"
)

thresholds = {
    "phase_r2_mean_min": 0.10,
    "kappa_r2_mean_min": 0.10,
    "driven_phase_r2_replay_mean_min": 0.20,
    "driven_kappa_r2_replay_mean_min": 0.15,
    "rollout_theta_rmse_mean_max": 2.0,
    "rollout_kappa_rmse_mean_max": 5.0,
    "rollout_order_parameter_rmse_mean_max": 0.20,
    "event_signal_corr_mean_min": 0.40,
    "event_order_parameter_rmse_max": 0.20,
}
autonomous_rollout_theta_mean = float(
    np.mean([row["theta_rmse_wrapped"] for row in autonomous_rollout_rows])
)
autonomous_rollout_kappa_mean = float(
    np.mean([row["kappa_rmse"] for row in autonomous_rollout_rows])
)
autonomous_rollout_r_mean = float(
    np.mean([row["order_parameter_rmse"] for row in autonomous_rollout_rows])
)
rollout_theta_mean = float(
    np.mean([row["theta_rmse_wrapped"] for row in nonautonomous_rollout_rows])
)
rollout_kappa_mean = float(
    np.mean([row["kappa_rmse"] for row in nonautonomous_rollout_rows])
)
rollout_r_mean = float(
    np.mean([row["order_parameter_rmse"] for row in nonautonomous_rollout_rows])
)
phase_r2_mean = float(np.nanmean(phase_r2_test))
kappa_r2_mean = float(np.nanmean(kappa_r2_test))
driven_phase_r2_mean = float(np.nanmean(phase_r2_forced_test))
driven_kappa_r2_mean = float(np.nanmean(kappa_r2_forced_test))
passed = print_check(
    phase_r2_mean >= thresholds["phase_r2_mean_min"]
    and kappa_r2_mean >= thresholds["kappa_r2_mean_min"]
    and driven_phase_r2_mean >= thresholds["driven_phase_r2_replay_mean_min"]
    and driven_kappa_r2_mean >= thresholds["driven_kappa_r2_replay_mean_min"]
    and rollout_theta_mean <= thresholds["rollout_theta_rmse_mean_max"]
    and rollout_kappa_mean <= thresholds["rollout_kappa_rmse_mean_max"]
    and rollout_r_mean <= thresholds["rollout_order_parameter_rmse_mean_max"]
    and event_replay["signal_corr_mean"] >= thresholds["event_signal_corr_mean_min"]
    and event_replay["order_parameter_rmse"] <= thresholds["event_order_parameter_rmse_max"],
    "non-autonomous epilepsy replay fitted with measurable heldout skill",
)

metrics = {
    "task": "epilepsy adaptive Kuramoto KANDy fit",
    "data_path": str(DATA_PATH),
    "labels": LABELS,
    "samples_raw": int(len(X_RAW)),
    "samples_used": int(len(STATE)),
    "fs": FS,
    "dt": DT,
    "bandpass_hz": list(BANDPASS_HZ),
    "state": {
        "n_channels": N,
        "n_pair_kappa_proxies": N_PAIR,
        "n_directed_phase_terms": N_OD,
        "dimension": N_STATE,
        "kappa_proxy": "z-scored smoothed pairwise log-amplitude envelope",
    },
    "fit": {
        "grid": GRID,
        "steps": FIT_STEPS,
        "forcing": "sampled Savitzky-Golay-smoothed residual q(t)",
        "forcing_window_samples": int(forcing_window),
        "forcing_window_seconds": FORCING_SAVGOL_SECONDS,
        "n_train": int(len(train_idx)),
        "n_heldout": int(len(test_idx)),
        "target_scaling": "per-output standard deviation",
        "nonautonomous_form": "dx/dt = f_KANDy(x) + q_KANDy(t) + lambda*(x_data(t)-x)",
        "nudge_rate": NONAUTONOMOUS_NUDGE_RATE,
    },
    "phase_r2_train": [float(v) for v in phase_r2_train],
    "phase_r2_heldout": [float(v) for v in phase_r2_test],
    "phase_r2_heldout_mean": phase_r2_mean,
    "kappa_r2_train": [float(v) for v in kappa_r2_train],
    "kappa_r2_heldout": [float(v) for v in kappa_r2_test],
    "kappa_r2_heldout_mean": kappa_r2_mean,
    "driven_phase_r2_replay": [float(v) for v in phase_r2_forced_test],
    "driven_phase_r2_replay_mean": driven_phase_r2_mean,
    "driven_kappa_r2_replay": [float(v) for v in kappa_r2_forced_test],
    "driven_kappa_r2_replay_mean": driven_kappa_r2_mean,
    "phase_target_scale": [float(v) for v in phase_scale],
    "kappa_target_scale": [float(v) for v in kappa_scale],
    "residual_target_scale": [float(v) for v in residual_scale],
    "q_svd_variance_ratio": [float(v) for v in q_variance_ratio],
    "q_svd_cumulative_variance": [float(v) for v in q_cumulative_variance],
    "directed_strength": directed_strength,
    "top_directed_terms": top_edges,
    "autonomous_rollouts": autonomous_rollout_rows,
    "autonomous_rollout_theta_rmse_mean": autonomous_rollout_theta_mean,
    "autonomous_rollout_kappa_rmse_mean": autonomous_rollout_kappa_mean,
    "autonomous_rollout_order_parameter_rmse_mean": autonomous_rollout_r_mean,
    "nonautonomous_rollouts": nonautonomous_rollout_rows,
    "rollout_theta_rmse_mean": rollout_theta_mean,
    "rollout_kappa_rmse_mean": rollout_kappa_mean,
    "rollout_order_parameter_rmse_mean": rollout_r_mean,
    "seizure_replay": event_replay,
    "thresholds": thresholds,
    "passed": passed,
}
save_metrics(RESULT_DIR / "metrics.json", metrics)

model_state_path = RESULT_DIR / "fitted_model_state.pt"
torch.save(
    {
        "phase_model_state_dict": model_phase.model_.state_dict(),
        "kappa_model_state_dict": model_kappa.model_.state_dict(),
        "phase_scale": torch.tensor(phase_scale),
        "kappa_scale": torch.tensor(kappa_scale),
        "residual_scale": torch.tensor(residual_scale),
        "labels": LABELS,
        "pair_features": PAIR_FEATURES,
        "directed_pairs": OD_PAIRS,
        "phase_feature_names": PHASE_FEATURE_NAMES,
        "kappa_feature_names": KAPPA_FEATURE_NAMES,
        "metadata": {
            "data_path": str(DATA_PATH),
            "bandpass_hz": BANDPASS_HZ,
            "grid": GRID,
            "k": 3,
            "fit_steps": FIT_STEPS,
            "forcing": "sampled Savitzky-Golay-smoothed residual q(t)",
            "forcing_window_samples": int(forcing_window),
            "forcing_window_seconds": FORCING_SAVGOL_SECONDS,
            "nudge_rate": NONAUTONOMOUS_NUDGE_RATE,
            "kappa_proxy": "z-scored smoothed pairwise log-amplitude envelope",
        },
    },
    model_state_path,
)
print(f"[MODEL] Saved {model_state_path}")

np.savez_compressed(
    RESULT_DIR / "fit_arrays.npz",
    t=T,
    labels=np.asarray(LABELS),
    signal=SIGNAL,
    theta=STATE[:, :N],
    kappa=STATE[:, N:],
    theta_dot=THETA_DOT,
    kappa_dot=KAPPA_DOT,
    theta_dot_pred=model_phase.predict(STATE) * phase_scale,
    kappa_dot_pred=model_kappa.predict(STATE) * kappa_scale,
    forcing_dot=FORCING_DOT,
    nonautonomous_dot=NONAUTONOMOUS_DOT,
    q_normalized=Q_NORMALIZED,
    q_svd_scores=U_Q * S_Q,
    q_svd_singular_values=S_Q,
    q_svd_components=Vt_Q,
    q_svd_variance_ratio=q_variance_ratio,
    q_svd_cumulative_variance=q_cumulative_variance,
    directed_strength=directed_strength,
    phase_linear_readout=A_phase,
    kappa_linear_readout=A_kappa,
    event_time=event_times,
    event_signal_ref=event_ref_signal,
    event_signal_pred=event_pred_signal,
    event_state_ref=event_ref_state,
    event_state_pred=event_pred_state,
    event_order_ref=event_r_ref,
    event_order_pred=event_r_pred,
)
with (RESULT_DIR / "readout.json").open("w", encoding="utf-8") as fh:
    json.dump(
        {
            "phase_feature_names": PHASE_FEATURE_NAMES,
            "kappa_feature_names": KAPPA_FEATURE_NAMES,
            "phase_linear_readout_shape": list(A_phase.shape),
            "kappa_linear_readout_shape": list(A_kappa.shape),
            "top_directed_terms": top_edges,
        },
        fh,
        indent=2,
    )
    fh.write("\n")


# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------
def plot_signals() -> None:
    fig, axes = plt.subplots(N, 1, figsize=(10, 1.45 * N), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(T, SIGNAL[:, i], lw=0.45)
        ax.set_ylabel(LABELS[i], rotation=0, ha="right", va="center", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Band-passed epilepsy recording")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "signals.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_phase_and_order() -> None:
    r = np.abs(np.mean(np.exp(1j * STATE[:, :N]), axis=1))
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for i, label in enumerate(LABELS):
        axes[0].plot(T, wrap_pi(STATE[:, i]), lw=0.45, label=label)
    axes[0].set_ylabel("phase")
    axes[0].legend(ncol=3, fontsize=7, loc="upper right")
    axes[1].plot(T, r, lw=1.0, color="#d62728")
    axes[1].set_ylabel("order parameter")
    axes[1].set_xlabel("time (s)")
    fig.suptitle("Hilbert phases and synchrony")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "phases_order_parameter.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_scatter() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    pred = model_phase.predict(STATE[test_idx]) * phase_scale
    for i, ax in enumerate(axes.ravel()):
        ax.scatter(THETA_DOT[test_idx, i], pred[:, i], s=4, alpha=0.35)
        lo = min(THETA_DOT[test_idx, i].min(), pred[:, i].min())
        hi = max(THETA_DOT[test_idx, i].max(), pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_title(f"{LABELS[i]}  R2={phase_r2_test[i]:.2f}", fontsize=9)
    fig.supxlabel("true phase velocity")
    fig.supylabel("KANDy prediction")
    fig.suptitle("Heldout phase-velocity fit")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "phase_velocity_fit.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_strength_heatmap() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(directed_strength, cmap="magma")
    ax.set_xticks(range(N))
    ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(N))
    ax.set_yticklabels(LABELS, fontsize=8)
    ax.set_xlabel("source channel j")
    ax.set_ylabel("target equation i")
    ax.set_title("Effective directed phase-term strength")
    fig.colorbar(im, ax=ax, label="coefficient magnitude")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "directed_phase_strength.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_spectrum() -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, label in enumerate(LABELS):
        freq, psd = welch(X_RAW[:, i], fs=FS, nperseg=min(2048, len(X_RAW)))
        mask = (freq >= 0.5) & (freq <= 60.0)
        ax.semilogy(freq[mask], psd[mask], lw=0.9, label=label)
    ax.axvspan(BANDPASS_HZ[0], BANDPASS_HZ[1], color="0.85", zorder=-1)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Recording spectra and fitted band")
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "spectra.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_vs_data() -> None:
    """Compare short non-autonomous KANDy replays against measured states."""
    if not rollout_traces:
        return

    top_kappa_indices = []
    for edge_info in top_edges:
        pair = pair_for(edge_info["target_index"], edge_info["source_index"])
        pair_idx = PAIR_INDEX[pair]
        if pair_idx not in top_kappa_indices:
            top_kappa_indices.append(pair_idx)
        if len(top_kappa_indices) == 3:
            break

    colors = plt.cm.tab10.colors
    n_rows = len(rollout_traces)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(14, max(3.0, 2.4 * n_rows)),
        sharex="col",
    )
    if n_rows == 1:
        axes = axes[None, :]

    for row_idx, trace in enumerate(rollout_traces):
        t_local = trace["time"]
        ref = trace["ref"]
        pred = trace["pred"]

        ax_sig, ax_r, ax_k = axes[row_idx]
        for ch_idx, label in enumerate(LABELS):
            color = colors[ch_idx % len(colors)]
            ax_sig.plot(
                t_local,
                np.cos(ref[:, ch_idx]),
                color=color,
                lw=0.9,
                alpha=0.65,
                label=label if row_idx == 0 else None,
            )
            ax_sig.plot(
                t_local,
                np.cos(pred[:, ch_idx]),
                color=color,
                lw=0.9,
                ls="--",
                alpha=0.95,
            )
        ax_sig.set_ylabel(f"t0={trace['start_time']:.1f}s")
        ax_sig.set_ylim(-1.1, 1.1)

        ax_r.plot(t_local, trace["r_ref"], color="#1f77b4", lw=1.2, label="data")
        ax_r.plot(
            t_local,
            trace["r_pred"],
            color="#d62728",
            lw=1.2,
            ls="--",
            label="KANDy",
        )
        ax_r.set_ylim(0, 1.05)

        for k_idx in top_kappa_indices:
            pair = PAIR_FEATURES[k_idx]
            label = f"k{pair[0]}{pair[1]}"
            color = colors[k_idx % len(colors)]
            ax_k.plot(
                t_local,
                ref[:, N + k_idx],
                color=color,
                lw=0.9,
                alpha=0.65,
                label=label if row_idx == 0 else None,
            )
            ax_k.plot(
                t_local,
                pred[:, N + k_idx],
                color=color,
                lw=0.9,
                ls="--",
                alpha=0.95,
            )

    axes[0, 0].set_title("oscillator signals  cos(theta)")
    axes[0, 1].set_title("order parameter")
    axes[0, 2].set_title("top kappa proxies")
    axes[-1, 0].set_xlabel("time since rollout start (s)")
    axes[-1, 1].set_xlabel("time since rollout start (s)")
    axes[-1, 2].set_xlabel("time since rollout start (s)")
    axes[0, 0].legend(ncol=2, fontsize=7, loc="upper right")
    axes[0, 1].legend(fontsize=7, loc="upper right")
    axes[0, 2].legend(fontsize=7, loc="upper right")
    fig.suptitle("Short non-autonomous KANDy replay vs measured epilepsy state")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "rollout_vs_data.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "rollout_vs_data.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_seizure_replay() -> None:
    fig, axes = plt.subplots(N + 1, 1, figsize=(11, 1.35 * (N + 1)), sharex=True)
    for ch_idx, ax in enumerate(axes[:N]):
        ax.plot(event_times, event_ref_signal[:, ch_idx], lw=0.65, color="#1f77b4")
        ax.plot(
            event_times,
            event_pred_signal[:, ch_idx],
            lw=0.8,
            ls="--",
            color="#d62728",
        )
        ax.axvline(0.0, color="0.2", lw=0.8, ls=":")
        ax.set_ylabel(LABELS[ch_idx], rotation=0, ha="right", va="center", fontsize=8)

    axes[-1].plot(event_times, event_r_ref, lw=1.0, color="#1f77b4", label="data")
    axes[-1].plot(
        event_times,
        event_r_pred,
        lw=1.1,
        ls="--",
        color="#d62728",
        label="non-aut KANDy",
    )
    axes[-1].axvline(0.0, color="0.2", lw=0.8, ls=":")
    axes[-1].set_ylabel("order", rotation=0, ha="right", va="center", fontsize=8)
    axes[-1].set_ylim(0, 1.05)
    axes[-1].set_xlabel("recording time (s)")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Seizure-window replay: data vs non-autonomous KANDy",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "seizure_replay.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "seizure_replay.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_forcing_drive() -> None:
    phase_norm = FORCING_DOT[:, :N] / phase_scale
    kappa_norm = FORCING_DOT[:, N:] / kappa_scale

    pair_norm = np.linalg.norm(kappa_norm, axis=0)
    top_pairs = np.argsort(pair_norm)[-4:][::-1]

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for i, label in enumerate(LABELS):
        axes[0].plot(T, phase_norm[:, i], lw=0.7, label=label)
    axes[0].axvspan(event_times[0], event_times[-1], color="0.9", zorder=-1)
    axes[0].axvline(0.0, color="0.2", lw=0.8, ls=":")
    axes[0].set_ylabel(r"$q_\theta / \sigma_{\dot\theta}$")
    axes[0].set_title("Empirical non-autonomous residual drive q(t)")
    axes[0].legend(ncol=3, fontsize=7, loc="upper right")

    for pair_idx in top_pairs:
        i, j = PAIR_FEATURES[pair_idx]
        axes[1].plot(T, kappa_norm[:, pair_idx], lw=0.8, label=f"k{i}{j}")
    axes[1].axvspan(event_times[0], event_times[-1], color="0.9", zorder=-1)
    axes[1].axvline(0.0, color="0.2", lw=0.8, ls=":")
    axes[1].set_ylabel(r"$q_\kappa / \sigma_{\dot\kappa}$")
    axes[1].legend(ncol=4, fontsize=7, loc="upper right")

    drive_norm = np.sqrt(np.mean(np.column_stack([phase_norm, kappa_norm]) ** 2, axis=1))
    axes[2].plot(T, drive_norm, lw=0.9, color="#d62728")
    axes[2].axvspan(event_times[0], event_times[-1], color="0.9", zorder=-1)
    axes[2].axvline(0.0, color="0.2", lw=0.8, ls=":")
    axes[2].set_ylabel("RMS normalized drive")
    axes[2].set_xlabel("recording time (s)")

    fig.tight_layout()
    fig.savefig(RESULT_DIR / "forcing_drive.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "forcing_drive.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    event_mask = (T >= event_times[0]) & (T <= event_times[-1])
    for i, label in enumerate(LABELS):
        axes[0].plot(T[event_mask], phase_norm[event_mask, i], lw=0.9, label=label)
    axes[0].axvline(0.0, color="0.2", lw=0.8, ls=":")
    axes[0].set_ylabel(r"$q_\theta / \sigma_{\dot\theta}$")
    axes[0].set_title("q(t) over the seizure replay window")
    axes[0].legend(ncol=3, fontsize=7, loc="upper right")

    for pair_idx in top_pairs:
        i, j = PAIR_FEATURES[pair_idx]
        axes[1].plot(
            T[event_mask],
            kappa_norm[event_mask, pair_idx],
            lw=0.9,
            label=f"k{i}{j}",
        )
    axes[1].axvline(0.0, color="0.2", lw=0.8, ls=":")
    axes[1].set_ylabel(r"$q_\kappa / \sigma_{\dot\kappa}$")
    axes[1].set_xlabel("recording time (s)")
    axes[1].legend(ncol=4, fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(RESULT_DIR / "forcing_drive_event.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "forcing_drive_event.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_forcing_svd() -> None:
    n_modes = min(5, len(S_Q))
    scores = U_Q[:, :n_modes] * S_Q[:n_modes]
    event_mask = (T >= event_times[0]) & (T <= event_times[-1])
    feature_labels = [f"qtheta_{label}" for label in LABELS] + [
        f"qk{i}{j}" for i, j in PAIR_FEATURES
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11, 8))
    modes = np.arange(1, min(12, len(S_Q)) + 1)
    axes[0].bar(modes, q_variance_ratio[: len(modes)], color="#1f77b4", alpha=0.75)
    axes[0].plot(
        modes,
        q_cumulative_variance[: len(modes)],
        color="#d62728",
        marker="o",
        lw=1.0,
        label="cumulative",
    )
    axes[0].set_ylabel("variance")
    axes[0].set_title("SVD of normalized q(t)")
    axes[0].legend(fontsize=8)

    for mode_idx in range(n_modes):
        axes[1].plot(
            T,
            scores[:, mode_idx] / np.std(scores[:, mode_idx]),
            lw=0.8,
            label=f"mode {mode_idx + 1}",
        )
    axes[1].axvspan(event_times[0], event_times[-1], color="0.9", zorder=-1)
    axes[1].axvline(0.0, color="0.2", lw=0.8, ls=":")
    axes[1].set_ylabel("score / std")
    axes[1].legend(ncol=n_modes, fontsize=7, loc="upper right")

    for mode_idx in range(min(3, n_modes)):
        axes[2].plot(
            T[event_mask],
            scores[event_mask, mode_idx] / np.std(scores[:, mode_idx]),
            lw=1.0,
            label=f"mode {mode_idx + 1}",
        )
    axes[2].axvline(0.0, color="0.2", lw=0.8, ls=":")
    axes[2].set_xlabel("recording time (s)")
    axes[2].set_ylabel("event score / std")
    axes[2].legend(ncol=3, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(RESULT_DIR / "forcing_svd.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "forcing_svd.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    x = np.arange(len(feature_labels))
    for mode_idx, ax in enumerate(axes):
        loading = Vt_Q[mode_idx]
        colors = np.array(["#1f77b4"] * N + ["#d62728"] * N_PAIR)
        ax.bar(x, loading, color=colors, alpha=0.8)
        ax.axvline(N - 0.5, color="0.2", lw=0.8)
        ax.set_ylabel(f"mode {mode_idx + 1}")
        ax.set_title(
            f"loading, var={q_variance_ratio[mode_idx]:.3f}",
            fontsize=9,
        )
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(feature_labels, rotation=70, ha="right", fontsize=7)
    fig.suptitle("q(t) SVD loadings: phase drive then kappa drive")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "forcing_svd_loadings.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "forcing_svd_loadings.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_losses() -> None:
    if getattr(model_phase, "train_results_", None):
        fig, _ax = plot_loss_curves(
            model_phase.train_results_,
            save=str(RESULT_DIR / "loss_phase"),
        )
        plt.close(fig)
    if getattr(model_kappa, "train_results_", None):
        fig, _ax = plot_loss_curves(
            model_kappa.train_results_,
            save=str(RESULT_DIR / "loss_kappa"),
        )
        plt.close(fig)


use_pub_style()
plot_signals()
plot_phase_and_order()
plot_prediction_scatter()
plot_strength_heatmap()
plot_spectrum()
plot_rollout_vs_data()
plot_seizure_replay()
plot_forcing_drive()
plot_forcing_svd()
plot_losses()

print(f"[FIGS]  Saved {RESULT_DIR}")
