#!/usr/bin/env python3
"""KANDy Jacobian diagnostics for the epilepsy observable-delay map.

The observable-delay map learns only the next current PCA block,

    y_{n+1} = F_KANDy([y_n, y_{n-1}, ...]),

and the delay-buffer shift is exact.  Its full map Jacobian is therefore a
companion-style matrix: the learned KANDy Jacobian occupies the first block
row and the lower block rows are known identity shifts.

This script evaluates that structure along the recording.  It does not try to
force a pointwise replay; it asks whether the learned local linearization
exposes the seizure as a change in local growth, transient amplification, or
delay-block attribution.
"""

from __future__ import annotations

import os
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
from kan import KAN

from _utils import ensure_outdir, print_check, save_metrics
from kandy.plotting import use_pub_style


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

SOURCE_DIR = REPO_ROOT / "results" / "Epilepsy-Observable-Delay-Map"
RESULT_DIR = ensure_outdir(REPO_ROOT / "results" / "Epilepsy-Observable-Jacobian")
MODEL_PATH = SOURCE_DIR / "fitted_observable_delay_map_state.pt"
ARRAY_PATH = SOURCE_DIR / "observable_delay_arrays.npz"

JAC_STRIDE = env_int("KANDY_JAC_STRIDE", 5)
EVENT_WINDOW = (-1.0, 4.0)
PRE_WINDOW = (-10.0, -2.0)
POST_WINDOW = (8.0, 20.0)

if not MODEL_PATH.exists() or not ARRAY_PATH.exists():
    raise FileNotFoundError(
        "Observable-delay map artifacts are missing. Run "
        "skills/kandy/examples/oscillators/epilepsy_observable_delay_map_fit.py first."
    )

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
metadata = checkpoint["metadata"]
arrays = np.load(ARRAY_PATH, allow_pickle=True)

PCA_DIM = int(metadata["pca_dim"])
DELAY_COUNT = int(metadata["delay_count"])
STATE_DIM = int(metadata["state_dim"])
GRID = int(metadata["grid"])
DT = float(np.median(np.diff(arrays["t"])))
LABELS = [str(x) for x in arrays["labels"]]
N_CH = len(LABELS)

STATE_S = np.asarray(arrays["state_s"], dtype=float)
ROW_TIME = np.asarray(arrays["row_time"], dtype=float)
SIGNAL = np.asarray(arrays["signal"], dtype=float)

print("[LOAD]  observable-delay KANDy map", flush=True)
print(
    f"        state_dim={STATE_DIM}, pca_dim={PCA_DIM}, delay_count={DELAY_COUNT}, "
    f"jac_stride={JAC_STRIDE}",
    flush=True,
)

model = KAN(
    width=[STATE_DIM, PCA_DIM],
    grid=GRID,
    k=3,
    seed=SEED,
    base_fun=torch.nn.Identity(),
    device="cpu",
    auto_save=False,
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def learned_block_jacobian(x: np.ndarray) -> np.ndarray:
    """Return dF_KANDy/dx for the learned current-block map."""
    xt = torch.tensor(x[None, :], dtype=torch.float32, requires_grad=True)
    yt = model(xt)
    rows = []
    for out_idx in range(PCA_DIM):
        grad = torch.autograd.grad(yt[0, out_idx], xt, retain_graph=True)[0]
        rows.append(grad.detach().cpu().numpy()[0])
    return np.vstack(rows)


def full_delay_jacobian(block: np.ndarray) -> np.ndarray:
    """Compose learned top block with the exact delay-shift rows."""
    J = np.zeros((STATE_DIM, STATE_DIM), dtype=float)
    J[:PCA_DIM] = block
    J[PCA_DIM:, :-PCA_DIM] = np.eye(STATE_DIM - PCA_DIM)
    return J


def summarize_window(name: str, mask: np.ndarray, values: dict[str, np.ndarray]) -> dict:
    return {
        "name": name,
        "samples": int(mask.sum()),
        "spectral_radius_mean": float(np.mean(values["spectral_radius"][mask])),
        "spectral_radius_max": float(np.max(values["spectral_radius"][mask])),
        "log_growth_per_second_mean": float(np.mean(values["log_growth_per_second"][mask])),
        "sigma_max_mean": float(np.mean(values["sigma_max"][mask])),
        "unstable_count_mean": float(np.mean(values["unstable_count"][mask])),
        "signal_rms_mean": float(np.mean(values["signal_rms"][mask])),
    }


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUROC; returns nan if one class is absent."""
    score = np.asarray(score)
    label = np.asarray(label).astype(bool)
    n_pos = int(label.sum())
    n_neg = int((~label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos_rank_sum = float(ranks[label].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


row_indices = np.arange(0, len(STATE_S), JAC_STRIDE)
if row_indices[-1] != len(STATE_S) - 1:
    row_indices = np.append(row_indices, len(STATE_S) - 1)

times = ROW_TIME[row_indices]
signal_offset = DELAY_COUNT - 1
signal_rows = signal_offset + row_indices
signal_rms = np.sqrt(np.mean(SIGNAL[signal_rows] ** 2, axis=1))

block_jacs = np.empty((len(row_indices), PCA_DIM, STATE_DIM), dtype=float)
spectral_radius = np.empty(len(row_indices), dtype=float)
sigma_max = np.empty(len(row_indices), dtype=float)
unstable_count = np.empty(len(row_indices), dtype=float)
det_abs = np.empty(len(row_indices), dtype=float)
eig_samples: dict[str, np.ndarray] = {}

print(f"[JAC]   evaluating {len(row_indices)} Jacobians ...", flush=True)
for out_idx, row in enumerate(row_indices):
    block = learned_block_jacobian(STATE_S[row])
    J = full_delay_jacobian(block)
    eigvals = np.linalg.eigvals(J)
    svals = np.linalg.svd(J, compute_uv=False)
    block_jacs[out_idx] = block
    spectral_radius[out_idx] = float(np.max(np.abs(eigvals)))
    sigma_max[out_idx] = float(svals[0])
    unstable_count[out_idx] = float(np.sum(np.abs(eigvals) > 1.0))
    sign, logdet = np.linalg.slogdet(J)
    det_abs[out_idx] = float(np.exp(logdet)) if sign != 0 else 0.0
    if out_idx % max(1, len(row_indices) // 8) == 0:
        print(f"        {out_idx:4d}/{len(row_indices)}", flush=True)

log_growth_per_step = np.log(np.maximum(spectral_radius, 1e-12))
log_growth_per_second = log_growth_per_step / DT

values = {
    "spectral_radius": spectral_radius,
    "sigma_max": sigma_max,
    "unstable_count": unstable_count,
    "signal_rms": signal_rms,
    "log_growth_per_second": log_growth_per_second,
}

pre_mask = (times >= PRE_WINDOW[0]) & (times <= PRE_WINDOW[1])
event_mask = (times >= EVENT_WINDOW[0]) & (times <= EVENT_WINDOW[1])
post_mask = (times >= POST_WINDOW[0]) & (times <= POST_WINDOW[1])
comparison_mask = pre_mask | event_mask | post_mask
window_summaries = [
    summarize_window("pre", pre_mask, values),
    summarize_window("event", event_mask, values),
    summarize_window("post", post_mask, values),
]

event_labels = event_mask[comparison_mask]
event_auroc = {
    "spectral_radius": float(auroc(spectral_radius[comparison_mask], event_labels)),
    "log_growth_per_second": float(auroc(log_growth_per_second[comparison_mask], event_labels)),
    "sigma_max": float(auroc(sigma_max[comparison_mask], event_labels)),
    "unstable_count": float(auroc(unstable_count[comparison_mask], event_labels)),
    "signal_rms": float(auroc(signal_rms[comparison_mask], event_labels)),
}

delay_block_norms = np.empty((len(row_indices), PCA_DIM, DELAY_COUNT), dtype=float)
for lag in range(DELAY_COUNT):
    cols = slice(lag * PCA_DIM, (lag + 1) * PCA_DIM)
    delay_block_norms[:, :, lag] = np.linalg.norm(block_jacs[:, :, cols], axis=2)

window_delay_audit = {}
for name, mask in [("pre", pre_mask), ("event", event_mask), ("post", post_mask)]:
    window_delay_audit[name] = np.mean(delay_block_norms[mask], axis=0) if mask.any() else np.nan

event_minus_pre = (
    window_delay_audit["event"] - window_delay_audit["pre"]
    if np.ndim(window_delay_audit["pre"]) == 2 and np.ndim(window_delay_audit["event"]) == 2
    else np.full((PCA_DIM, DELAY_COUNT), np.nan)
)

thresholds = {
    "event_spectral_radius_lift_min": 0.05,
    "event_jacobian_auroc_min": 0.65,
}
event_lift = (
    window_summaries[1]["spectral_radius_mean"]
    - window_summaries[0]["spectral_radius_mean"]
)
best_jacobian_auroc = max(
    event_auroc["spectral_radius"],
    event_auroc["log_growth_per_second"],
    event_auroc["sigma_max"],
    event_auroc["unstable_count"],
)
passed = print_check(
    event_lift >= thresholds["event_spectral_radius_lift_min"]
    or best_jacobian_auroc >= thresholds["event_jacobian_auroc_min"],
    "KANDy Jacobian exposes a seizure-window local stability change",
)

print(
    "[SUMMARY] spectral radius mean "
    f"pre={window_summaries[0]['spectral_radius_mean']:.3f}, "
    f"event={window_summaries[1]['spectral_radius_mean']:.3f}, "
    f"post={window_summaries[2]['spectral_radius_mean']:.3f}",
    flush=True,
)
print(
    "[SUMMARY] AUROC "
    + ", ".join(f"{k}={v:.3f}" for k, v in event_auroc.items()),
    flush=True,
)

metrics = {
    "task": "observable-delay KANDy Jacobian diagnostic",
    "source_model": str(MODEL_PATH),
    "source_arrays": str(ARRAY_PATH),
    "metadata": metadata,
    "jac_stride": JAC_STRIDE,
    "n_jacobians": int(len(row_indices)),
    "windows": window_summaries,
    "event_minus_pre_spectral_radius": float(event_lift),
    "event_auroc": event_auroc,
    "delay_block_norms_mean": {
        name: np.asarray(audit).tolist() for name, audit in window_delay_audit.items()
    },
    "event_minus_pre_delay_block_norms": event_minus_pre.tolist(),
    "thresholds": thresholds,
    "passed": passed,
}
save_metrics(RESULT_DIR / "metrics.json", metrics)

np.savez_compressed(
    RESULT_DIR / "jacobian_arrays.npz",
    row_indices=row_indices,
    times=times,
    block_jacs=block_jacs,
    spectral_radius=spectral_radius,
    log_growth_per_second=log_growth_per_second,
    sigma_max=sigma_max,
    unstable_count=unstable_count,
    det_abs=det_abs,
    signal_rms=signal_rms,
    delay_block_norms=delay_block_norms,
    labels=np.asarray(LABELS),
)


def plot_timeseries() -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(times, signal_rms, color="#1f77b4", lw=0.9)
    axes[0].set_ylabel("signal RMS")
    axes[1].plot(times, spectral_radius, color="#d62728", lw=0.9)
    axes[1].axhline(1.0, color="0.25", lw=0.8, ls=":")
    axes[1].set_ylabel("spectral radius")
    axes[2].plot(times, log_growth_per_second, color="#9467bd", lw=0.9)
    axes[2].axhline(0.0, color="0.25", lw=0.8, ls=":")
    axes[2].set_ylabel("log growth / s")
    axes[3].plot(times, sigma_max, color="#2ca02c", lw=0.9, label=r"$\sigma_{\max}$")
    axes[3].plot(times, unstable_count, color="#ff7f0e", lw=0.8, label="unstable count")
    axes[3].set_ylabel("transient / count")
    axes[3].set_xlabel("recording time (s)")
    axes[3].legend(fontsize=8)
    for ax in axes:
        ax.axvspan(EVENT_WINDOW[0], EVENT_WINDOW[1], color="0.9", zorder=-1)
        ax.axvline(0.0, color="0.2", lw=0.8, ls=":")
    fig.suptitle("Observable-delay KANDy full-map Jacobian diagnostics")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "jacobian_timeseries.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "jacobian_timeseries.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_delay_audit() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
    matrices = [
        window_delay_audit["pre"],
        window_delay_audit["event"],
        window_delay_audit["post"],
        event_minus_pre,
    ]
    titles = ["pre", "event", "post", "event - pre"]
    vmax = np.nanmax(np.abs(matrices[:3]))
    diff_v = np.nanmax(np.abs(event_minus_pre))
    for ax, matrix, title in zip(axes, matrices, titles):
        if title == "event - pre":
            im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-diff_v, vmax=diff_v)
        else:
            im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("input delay block")
        ax.set_xticks(np.arange(DELAY_COUNT))
        ax.set_xticklabels([str(i) for i in range(DELAY_COUNT)], fontsize=7)
        ax.set_yticks(np.arange(PCA_DIM))
        ax.set_yticklabels([f"PC{i + 1}" for i in range(PCA_DIM)], fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("output current block")
    fig.suptitle("KANDy learned-block Jacobian norm by delay term")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "jacobian_delay_audit.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "jacobian_delay_audit.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_eigen_cloud() -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="0.4", lw=0.8, ls=":")
    window_defs = [("pre", pre_mask, "#1f77b4"), ("event", event_mask, "#d62728"), ("post", post_mask, "#2ca02c")]
    for name, mask, color in window_defs:
        selected = np.where(mask)[0]
        if len(selected) == 0:
            continue
        if len(selected) > 12:
            selected = selected[np.linspace(0, len(selected) - 1, 12).astype(int)]
        eig_list = []
        for idx in selected:
            J = full_delay_jacobian(block_jacs[idx])
            eig_list.append(np.linalg.eigvals(J))
        eigvals = np.concatenate(eig_list)
        ax.scatter(eigvals.real, eigvals.imag, s=8, alpha=0.45, color=color, label=name)
    ax.axhline(0.0, color="0.5", lw=0.6)
    ax.axvline(0.0, color="0.5", lw=0.6)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Re eigenvalue")
    ax.set_ylabel("Im eigenvalue")
    ax.legend(fontsize=8)
    fig.suptitle("Full delay-map Jacobian eigenvalues")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "jacobian_eigen_cloud.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "jacobian_eigen_cloud.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


use_pub_style()
plot_timeseries()
plot_delay_audit()
plot_eigen_cloud()
print(f"[FIGS]  Saved {RESULT_DIR}", flush=True)
