#!/usr/bin/env python3
"""Delay-embedded observable KANDy map for the epilepsy recording.

This is the direct execution of the Mackey-Glass idea on the measured signal:

    filtered EEG -> PCA scores -> delay coordinates -> KANDy map

The learned state is an observable delay vector, not a Kuramoto phase/amplitude
surrogate.  KANDy first learns the one-step map, then the same KAN is refined
with an explicit multi-step rollout loss using the discrete-map convention

    dynamics_fn(x) = F_KANDy(x) - x

so Euler integration with dt=1 is exactly map iteration.  During rollout the
model receives only the initial delay history; no q(t), replay forcing, or
trajectory nudging is injected.
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
from scipy.signal import butter, filtfilt

from _utils import ensure_outdir, print_check, save_metrics
from kandy import CustomLift, KANDy
from kandy.plotting import plot_loss_curves, use_pub_style
from kandy.training import fit_kan


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


# ---------------------------------------------------------------------------
# 0. Parameters
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = REPO_ROOT / "data" / "Dory_2021_04_12_006_preprocessed.pkl"
RESULT_DIR = ensure_outdir(REPO_ROOT / "results" / "Epilepsy-Observable-Delay-Map")

BANDPASS_HZ = (1.0, 40.0)
FILTER_ORDER = 4
EDGE_TRIM_SECONDS = 2.0

PCA_DIM = env_int("KANDY_OBS_PCA_DIM", 6)
DELAY_COUNT = env_int("KANDY_OBS_DELAY_COUNT", 8)
GRID = env_int("KANDY_OBS_GRID", 4)
ONE_STEP_STEPS = env_int("KANDY_OBS_ONE_STEP_STEPS", 80)
ONE_STEP_OPT = env_str("KANDY_OBS_ONE_STEP_OPT", "LBFGS")
ROLLOUT_STEPS = env_int("KANDY_OBS_ROLLOUT_STEPS", 120)
ROLLOUT_HORIZON = env_int("KANDY_OBS_ROLLOUT_HORIZON", 16)
ROLLOUT_WINDOWS = env_int("KANDY_OBS_ROLLOUT_WINDOWS", 192)
EVENT_ROLLOUT_WINDOWS = env_int("KANDY_OBS_EVENT_ROLLOUT_WINDOWS", 192)
ROLLOUT_TEST_WINDOWS = env_int("KANDY_OBS_ROLLOUT_TEST_WINDOWS", 64)
ROLLOUT_WEIGHT = env_float("KANDY_OBS_ROLLOUT_WEIGHT", 0.2)
LR_ONE_STEP = env_float("KANDY_OBS_LR_ONE_STEP", 1.0)
LR_ROLLOUT = env_float("KANDY_OBS_LR_ROLLOUT", 1.0e-4)
BATCH = env_int("KANDY_OBS_BATCH", 512)
TRAJ_BATCH = env_int("KANDY_OBS_TRAJ_BATCH", 48)
EVENT_WINDOW = (-1.0, 4.0)
SHORT_ROLLOUT_SECONDS = 0.5
ROLL_OUT_CLIP_MARGIN = 0.15
VAL_FRAC = 0.10
TEST_FRAC = 0.10


# ---------------------------------------------------------------------------
# 1. Data and observable delay coordinates
# ---------------------------------------------------------------------------
def bandpass(X: np.ndarray, fs: float) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(FILTER_ORDER, [BANDPASS_HZ[0] / nyq, BANDPASS_HZ[1] / nyq], btype="band")
    return filtfilt(b, a, X, axis=0)


def zscore(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-12
    return (X - mean) / std, mean, std


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
N_CH = X_RAW.shape[1]

if PCA_DIM > N_CH:
    raise ValueError(f"PCA_DIM={PCA_DIM} exceeds number of channels {N_CH}.")

print("[DATA]  Loaded epilepsy recording", flush=True)
print(f"        samples={len(X_RAW)}, channels={N_CH}, fs={FS:.1f} Hz", flush=True)

X_FILT = bandpass(X_RAW, FS)
edge = int(round(EDGE_TRIM_SECONDS * FS))
SIGNAL = X_FILT[edge : len(X_FILT) - edge]
T = T_ALL[edge : len(X_FILT) - edge]

SIGNAL_Z, signal_mean, signal_std = zscore(SIGNAL)
U, S, Vt = np.linalg.svd(SIGNAL_Z, full_matrices=False)
pca_variance = (S**2) / np.sum(S**2)
pca_cumulative = np.cumsum(pca_variance)
COMPONENTS = Vt[:PCA_DIM]
SCORES = U[:, :PCA_DIM] * S[:PCA_DIM]
SCORES_Z, score_mean, score_std = zscore(SCORES)

delay_rows = len(SCORES_Z) - DELAY_COUNT + 1
DELAY_STATE = np.hstack(
    [SCORES_Z[DELAY_COUNT - 1 - lag : len(SCORES_Z) - lag] for lag in range(DELAY_COUNT)]
)
state_mean = DELAY_STATE.mean(axis=0)
state_std = DELAY_STATE.std(axis=0) + 1e-12
STATE_S = (DELAY_STATE - state_mean) / state_std
STATE_DIM = STATE_S.shape[1]
X_MAP = STATE_S[:-1]
Y_MAP = STATE_S[1:]
Y_CURRENT = Y_MAP[:, :PCA_DIM]
ROW_TIME = T[DELAY_COUNT - 1 :]
N_MAP = len(X_MAP)

n_train = int(N_MAP * (1.0 - VAL_FRAC - TEST_FRAC))
n_val = int(N_MAP * VAL_FRAC)
n_test = N_MAP - n_train - n_val

print(
    f"[PRE]   PCA dim={PCA_DIM}, explained={pca_cumulative[PCA_DIM - 1]:.3f}, "
    f"delay_count={DELAY_COUNT}, state_dim={STATE_DIM}",
    flush=True,
)
print(f"[PRE]   map rows={N_MAP}, train={n_train}, val={n_val}, test={n_test}", flush=True)


# ---------------------------------------------------------------------------
# 2. KANDy map and rollout-loss refinement
# ---------------------------------------------------------------------------
def identity_np(X: np.ndarray) -> np.ndarray:
    return X


def identity_torch(X: torch.Tensor) -> torch.Tensor:
    return X


identity_lift = CustomLift(
    fn=identity_np,
    output_dim=STATE_DIM,
    torch_fn=identity_torch,
    name="observable_delay_identity",
)

model = KANDy(
    lift=identity_lift,
    grid=GRID,
    k=3,
    steps=ONE_STEP_STEPS,
    seed=SEED,
    base_fun=torch.nn.Identity(),
    device="cpu",
)

print("[TRAIN] Fitting one-step observable delay map ...", flush=True)
quiet_fit(
    model,
    X=X_MAP,
    X_dot=Y_CURRENT,
    val_frac=VAL_FRAC,
    test_frac=TEST_FRAC,
    lamb=0.0,
    opt=ONE_STEP_OPT,
    lr=LR_ONE_STEP,
    batch=BATCH,
    stop_grid_update_step=5,
    patience=0,
    verbose=False,
)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    denom = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    denom = np.where(denom > 0, denom, np.nan)
    return 1.0 - np.sum((y_true - y_pred) ** 2, axis=0) / denom


def sample_windows(series: np.ndarray, start: int, stop: int, horizon: int, count: int) -> np.ndarray:
    max_start = stop - horizon
    if max_start <= start:
        raise ValueError("Not enough samples for rollout windows.")
    candidates = np.arange(start, max_start)
    if len(candidates) > count:
        candidates = np.sort(np.random.default_rng(SEED).choice(candidates, count, replace=False))
    return np.stack([series[i : i + horizon + 1] for i in candidates])


def sample_event_windows(series: np.ndarray, horizon: int, count: int) -> np.ndarray:
    lo_t = EVENT_WINDOW[0] - 0.5
    hi_t = EVENT_WINDOW[1] - horizon * DT
    candidates = np.where((ROW_TIME >= lo_t) & (ROW_TIME <= hi_t))[0]
    candidates = candidates[candidates + horizon < n_train + 1]
    if len(candidates) == 0:
        return np.empty((0, horizon + 1, series.shape[1]))
    if len(candidates) > count:
        candidates = np.sort(np.random.default_rng(SEED + 1).choice(candidates, count, replace=False))
    return np.stack([series[i : i + horizon + 1] for i in candidates])


background_windows = sample_windows(STATE_S, 0, n_train + 1, ROLLOUT_HORIZON, ROLLOUT_WINDOWS)
event_windows = sample_event_windows(STATE_S, ROLLOUT_HORIZON, EVENT_ROLLOUT_WINDOWS)
train_windows = np.concatenate([background_windows, event_windows], axis=0)
test_windows = sample_windows(STATE_S, n_train + n_val, len(STATE_S), ROLLOUT_HORIZON, ROLLOUT_TEST_WINDOWS)
train_t = torch.arange(ROLLOUT_HORIZON + 1, dtype=torch.float32)

theta = X_MAP.astype(np.float32)
labels = Y_MAP.astype(np.float32)
current_labels = Y_CURRENT.astype(np.float32)
dataset = {
    "train_input": torch.tensor(theta[:n_train], dtype=torch.float32),
    "train_label": torch.tensor(current_labels[:n_train], dtype=torch.float32),
    "val_input": torch.tensor(theta[n_train : n_train + n_val], dtype=torch.float32),
    "val_label": torch.tensor(current_labels[n_train : n_train + n_val], dtype=torch.float32),
    "test_input": torch.tensor(theta[n_train + n_val :], dtype=torch.float32),
    "test_label": torch.tensor(current_labels[n_train + n_val :], dtype=torch.float32),
    "train_traj": torch.tensor(train_windows, dtype=torch.float32),
    "train_t": train_t,
    "test_traj": torch.tensor(test_windows, dtype=torch.float32),
    "test_t": train_t,
}


def discrete_dynamics(state: torch.Tensor) -> torch.Tensor:
    next_current = model.model_(identity_torch(state))
    next_state = torch.cat([next_current, state[:, :-PCA_DIM]], dim=1)
    return next_state - state


def current_weighted_mse(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    diff = pred - true
    diff = diff.clone()
    diff[:, :, :PCA_DIM] = 2.0 * diff[:, :, :PCA_DIM]
    return torch.mean(diff**2)


print(
    f"[TRAIN] Refining with rollout loss: horizon={ROLLOUT_HORIZON}, "
    f"windows={len(train_windows)} ({len(event_windows)} event), "
    f"weight={ROLLOUT_WEIGHT:g}",
    flush=True,
)
model.train_results_ = fit_kan(
    model.model_,
    dataset,
    opt="Adam",
    lr=LR_ROLLOUT,
    batch=BATCH,
    steps=ROLLOUT_STEPS,
    lamb=0.0,
    rollout_weight=ROLLOUT_WEIGHT,
    rollout_horizon=ROLLOUT_HORIZON,
    rollout_loss_fn=current_weighted_mse,
    dynamics_fn=discrete_dynamics,
    integrator="euler",
    traj_batch=TRAJ_BATCH,
    stop_grid_update_step=0,
    patience=0,
    log=max(1, ROLLOUT_STEPS // 8),
)


# ---------------------------------------------------------------------------
# 3. Evaluation and reconstruction
# ---------------------------------------------------------------------------
def predict_next(x: np.ndarray) -> np.ndarray:
    x2 = np.atleast_2d(x)
    next_current = model.predict(x2)
    return np.column_stack([next_current, x2[:, :-PCA_DIM]])


def rollout_map(x0: np.ndarray, steps: int) -> np.ndarray:
    lo = STATE_S.min(axis=0)
    hi = STATE_S.max(axis=0)
    span = hi - lo
    lo = lo - ROLL_OUT_CLIP_MARGIN * span
    hi = hi + ROLL_OUT_CLIP_MARGIN * span

    x = x0.copy()
    out = np.empty((steps + 1, len(x)))
    out[0] = x
    for step in range(steps):
        x = predict_next(x)[0]
        x = np.clip(x, lo, hi)
        if not np.isfinite(x).all():
            return out[: step + 1]
        out[step + 1] = x
    return out


def state_to_scores(state_s: np.ndarray) -> np.ndarray:
    state = np.atleast_2d(state_s) * state_std + state_mean
    scores_z = state[:, :PCA_DIM]
    return scores_z * score_std + score_mean


def state_to_signal(state_s: np.ndarray) -> np.ndarray:
    scores = state_to_scores(state_s)
    signal_z = scores @ COMPONENTS
    return signal_z * signal_std + signal_mean


pred_test = predict_next(X_MAP[n_train + n_val :])
true_test = Y_MAP[n_train + n_val :]
one_step_r2 = r2(true_test, pred_test)
current_one_step_r2 = r2(true_test[:, :PCA_DIM], pred_test[:, :PCA_DIM])
current_r2_mean = float(np.nanmean(current_one_step_r2))
state_r2_mean = float(np.nanmean(one_step_r2))
print(
    f"[EVAL]  held-out one-step R2 current={current_r2_mean:.3f}, "
    f"state_with_exact_shift={state_r2_mean:.3f}",
    flush=True,
)

short_steps = int(round(SHORT_ROLLOUT_SECONDS / DT))
short_rows = []
short_traces = []
for start_s in [-8.0, 0.0, 8.0, 18.0]:
    row = int(np.argmin(np.abs(ROW_TIME - start_s)))
    pred = rollout_map(STATE_S[row], short_steps)
    ref = STATE_S[row : row + len(pred)]
    sig_pred = state_to_signal(pred)
    sig_ref = SIGNAL[DELAY_COUNT - 1 + row : DELAY_COUNT - 1 + row + len(pred)]
    corr = [
        float(np.corrcoef(sig_pred[:, ch], sig_ref[:, ch])[0, 1])
        for ch in range(N_CH)
    ]
    row_metrics = {
        "start_time": float(ROW_TIME[row]),
        "samples": int(len(pred)),
        "signal_corr_mean": float(np.mean(corr)),
        "signal_rmse": float(np.sqrt(np.mean((sig_pred - sig_ref) ** 2))),
        "state_rmse": float(np.sqrt(np.mean((pred - ref) ** 2))),
    }
    short_rows.append(row_metrics)
    short_traces.append(
        {
            "time": ROW_TIME[row : row + len(pred)] - ROW_TIME[row],
            "pred": sig_pred,
            "ref": sig_ref,
            "start_time": float(ROW_TIME[row]),
        }
    )

event_start = max(EVENT_WINDOW[0], float(ROW_TIME[0]))
event_end = min(EVENT_WINDOW[1], float(ROW_TIME[-1]))
event_row = int(np.argmin(np.abs(ROW_TIME - event_start)))
event_steps = int(round((event_end - event_start) / DT))
event_pred_state = rollout_map(STATE_S[event_row], event_steps)
event_time = ROW_TIME[event_row : event_row + len(event_pred_state)]
event_signal_pred = state_to_signal(event_pred_state)
event_signal_ref = SIGNAL[
    DELAY_COUNT - 1 + event_row : DELAY_COUNT - 1 + event_row + len(event_pred_state)
]
event_corr = [
    float(np.corrcoef(event_signal_pred[:, ch], event_signal_ref[:, ch])[0, 1])
    for ch in range(N_CH)
]
event_metrics = {
    "start_time": float(event_time[0]),
    "end_time": float(event_time[-1]),
    "samples": int(len(event_time)),
    "signal_corr_mean": float(np.mean(event_corr)),
    "signal_corr_per_channel": event_corr,
    "signal_rmse": float(np.sqrt(np.mean((event_signal_pred - event_signal_ref) ** 2))),
}

print("[ROLLOUT] short observable-delay rollouts:", flush=True)
for row in short_rows:
    print(
        f"          t0={row['start_time']:.2f}s, corr={row['signal_corr_mean']:.3f}, "
        f"rmse={row['signal_rmse']:.3f}",
        flush=True,
    )
print(
    "[EVENT] observable-delay seizure replay "
    f"{event_metrics['start_time']:.2f}s..{event_metrics['end_time']:.2f}s: "
    f"corr={event_metrics['signal_corr_mean']:.3f}, "
    f"rmse={event_metrics['signal_rmse']:.3f}",
    flush=True,
)

thresholds = {
    "current_one_step_r2_min": 0.85,
    "event_signal_corr_mean_min": 0.25,
}
passed = print_check(
    current_r2_mean >= thresholds["current_one_step_r2_min"]
    and event_metrics["signal_corr_mean"] >= thresholds["event_signal_corr_mean_min"],
    "observable delay map has one-step and seizure-window rollout skill",
)

metrics = {
    "task": "observable delay KANDy map with rollout loss",
    "data_path": str(DATA_PATH),
    "labels": LABELS,
    "fit": {
        "pca_dim": PCA_DIM,
        "pca_cumulative_variance": float(pca_cumulative[PCA_DIM - 1]),
        "delay_count": DELAY_COUNT,
        "state_dim": STATE_DIM,
        "grid": GRID,
        "one_step_steps": ONE_STEP_STEPS,
        "one_step_opt": ONE_STEP_OPT,
        "rollout_steps": ROLLOUT_STEPS,
        "rollout_horizon": ROLLOUT_HORIZON,
        "rollout_windows": int(len(train_windows)),
        "event_rollout_windows": int(len(event_windows)),
        "rollout_weight": ROLLOUT_WEIGHT,
        "map_mode": "X=current observable delay state, X_dot=next current PCA block; delay shift is exact",
        "rollout_loss": "Euler map rollout via dynamics_fn that predicts current block and shifts delay buffer",
        "rollout_projection": "delay state clipped to observed scaled range plus margin",
        "rollout_clip_margin": ROLL_OUT_CLIP_MARGIN,
    },
    "one_step_r2": [float(v) for v in one_step_r2],
    "current_one_step_r2": [float(v) for v in current_one_step_r2],
    "current_one_step_r2_mean": current_r2_mean,
    "state_one_step_r2_mean": state_r2_mean,
    "short_rollouts": short_rows,
    "event_replay": event_metrics,
    "thresholds": thresholds,
    "passed": passed,
}
save_metrics(RESULT_DIR / "metrics.json", metrics)

torch.save(
    {
        "model_state_dict": model.model_.state_dict(),
        "state_mean": torch.tensor(state_mean),
        "state_std": torch.tensor(state_std),
        "score_mean": torch.tensor(score_mean),
        "score_std": torch.tensor(score_std),
        "signal_mean": torch.tensor(signal_mean),
        "signal_std": torch.tensor(signal_std),
        "components": torch.tensor(COMPONENTS),
        "labels": LABELS,
        "metadata": metrics["fit"],
    },
    RESULT_DIR / "fitted_observable_delay_map_state.pt",
)
np.savez_compressed(
    RESULT_DIR / "observable_delay_arrays.npz",
    t=T,
    row_time=ROW_TIME,
    labels=np.asarray(LABELS),
    signal=SIGNAL,
    scores=SCORES,
    scores_z=SCORES_Z,
    components=COMPONENTS,
    pca_variance=pca_variance,
    pca_cumulative=pca_cumulative,
    state_s=STATE_S,
    event_time=event_time,
    event_signal_ref=event_signal_ref,
    event_signal_pred=event_signal_pred,
)


# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
def plot_pca() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    modes = np.arange(1, N_CH + 1)
    axes[0].bar(modes, pca_variance[:N_CH], color="#4c78a8", alpha=0.8)
    axes[0].plot(modes, pca_cumulative[:N_CH], marker="o", color="#f58518")
    axes[0].axvline(PCA_DIM, color="0.2", lw=0.8, ls=":")
    axes[0].set_ylabel("variance")
    axes[0].set_title("Filtered EEG PCA")
    for mode in range(PCA_DIM):
        axes[1].plot(T, SCORES_Z[:, mode], lw=0.7, label=f"PC{mode + 1}")
    axes[1].axvspan(event_time[0], event_time[-1], color="0.9", zorder=-1)
    axes[1].axvline(0.0, color="0.2", lw=0.8, ls=":")
    axes[1].set_xlabel("recording time (s)")
    axes[1].set_ylabel("score z")
    axes[1].legend(ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "observable_pca.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "observable_pca.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_short() -> None:
    fig, axes = plt.subplots(len(short_traces), 1, figsize=(11, 2.2 * len(short_traces)), sharex=True)
    if len(short_traces) == 1:
        axes = [axes]
    for ax, trace in zip(axes, short_traces):
        t_local = trace["time"]
        for ch, label in enumerate(LABELS):
            ax.plot(t_local, trace["ref"][:, ch], lw=0.65, alpha=0.55, label=label if ax is axes[0] else None)
            ax.plot(t_local, trace["pred"][:, ch], lw=0.75, ls="--", alpha=0.8)
        ax.set_ylabel(f"t0={trace['start_time']:.1f}s")
    axes[0].legend(ncol=3, fontsize=7)
    axes[-1].set_xlabel("time since start (s)")
    fig.suptitle("Observable-delay KANDy short rollouts")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "observable_delay_short_rollouts.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "observable_delay_short_rollouts.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_event() -> None:
    fig, axes = plt.subplots(N_CH, 1, figsize=(11, 1.45 * N_CH), sharex=True)
    for ch, ax in enumerate(axes):
        ax.plot(event_time, event_signal_ref[:, ch], lw=0.65, color="#1f77b4")
        ax.plot(event_time, event_signal_pred[:, ch], lw=0.75, ls="--", color="#d62728")
        ax.axvline(0.0, color="0.2", lw=0.8, ls=":")
        ax.set_ylabel(LABELS[ch], rotation=0, ha="right", va="center", fontsize=8)
    axes[-1].set_xlabel("recording time (s)")
    fig.suptitle("Seizure-window autonomous replay with observable-delay KANDy map")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "observable_delay_seizure_replay.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "observable_delay_seizure_replay.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_losses() -> None:
    if getattr(model, "train_results_", None) and model.train_results_.get("train_loss"):
        fig, _ax = plot_loss_curves(
            model.train_results_,
            save=str(RESULT_DIR / "loss_observable_delay_map"),
        )
        plt.close(fig)


use_pub_style()
plot_pca()
plot_short()
plot_event()
plot_losses()

print(f"[FIGS]  Saved {RESULT_DIR}", flush=True)
