#!/usr/bin/env python3
"""Forecasting benchmark on hippocampus_R: does an LSTM/GRU beat a linear model?

Straight question, straight test.  Given the last W samples of the channel,
predict the next H samples.  Four models on identical windows, identical
splits, identical scaling:

    persistence   repeat the last observed value
    linear        optimal linear predictor from the same W inputs (this is the
                  strongest classical baseline -- a direct multi-horizon AR,
                  not a recursive one, so it cannot be beaten by tuning AR order)
    LSTM          2 layers, hidden 64
    GRU           2 layers, hidden 64

Two design choices that decide whether the benchmark means anything
-------------------------------------------------------------------
**Scored across horizons, not at one step.**  At 200 Hz on a band-limited
signal, predicting the very next sample is nearly free: consecutive-sample
delay coordinates reach R^2 = 1.0000 on this channel, which is the local Taylor
expansion, not dynamics.  A one-step benchmark would rank every model at ~0.99
and tell you nothing.  Skill is therefore reported as a curve over horizons
from 5 ms to 1 s, and the number that matters is where each model crosses
persistence and where it crosses zero.

**Direct multi-horizon, not recursive.**  Each model predicts all H future
samples at once.  Recursive iteration would confound "how much information is
in the past" with "how fast do rollout errors compound", and the second effect
would dominate.  Direct prediction isolates the first, which is the question
being asked.

Split is chronological with a guard band wider than any filter support, and
scaling is fitted on TRAIN only.  The seizure sits inside the training region
and the test block is post-ictal, so this measures ordinary forecasting skill
on this recording, not seizure prediction.
"""

from __future__ import annotations

import os
import pickle
import sys
import time
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
import torch.nn as nn
from scipy.signal import butter, filtfilt

from _utils import ensure_outdir, print_check, save_metrics
from kandy.plotting import use_pub_style


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_float(name, default):
    return float(os.environ.get(name, default))


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = REPO_ROOT / "data" / "Dory_2021_04_12_006_preprocessed.pkl"
RESULT_DIR = ensure_outdir(REPO_ROOT / "results" / "Epilepsy-Forecast-Benchmark")

CHANNEL = os.environ.get("KANDY_FC_CHANNEL", "hippocampus_R")
BANDPASS_HZ = (1.0, 40.0)
WINDOW = env_int("KANDY_FC_WINDOW", 100)        # 0.5 s of history
HORIZON = env_int("KANDY_FC_HORIZON", 200)      # predict up to 1.0 s ahead
HIDDEN = env_int("KANDY_FC_HIDDEN", 64)
LAYERS = env_int("KANDY_FC_LAYERS", 2)
EPOCHS = env_int("KANDY_FC_EPOCHS", 60)
BATCH = env_int("KANDY_FC_BATCH", 256)
LR = env_float("KANDY_FC_LR", 1e-3)
PATIENCE = env_int("KANDY_FC_PATIENCE", 8)
REPORT_H = [1, 2, 5, 10, 20, 40, 70, 100, 150, 200]

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
with DATA_PATH.open("rb") as fh:
    payload = pickle.load(fh)

X_RAW = np.asarray(payload["X"], dtype=float)
FS = float(payload["fs"])
T_ALL = np.asarray(payload["t"], dtype=float)
LABELS = list(payload["labels"])
ch = LABELS.index(CHANNEL)
nyq = 0.5 * FS
bb, aa = butter(4, [BANDPASS_HZ[0] / nyq, BANDPASS_HZ[1] / nyq], btype="band")
edge = int(round(2.0 * FS))
sl = slice(edge, len(X_RAW) - edge)
T = T_ALL[sl]
X = filtfilt(bb, aa, X_RAW[:, ch])[sl]
N = len(X)

idx = np.arange(0, N - WINDOW - HORIZON)
n_tr = int(len(idx) * 0.60)
n_va = int(len(idx) * 0.15)
guard = WINDOW + HORIZON
tr_i = idx[: n_tr - guard]
va_i = idx[n_tr: n_tr + n_va - guard]
te_i = idx[n_tr + n_va + guard:]

mu, sd = X[: n_tr + WINDOW].mean(), X[: n_tr + WINDOW].std()   # TRAIN stats only
Z = (X - mu) / sd

print(f"[DATA]  {CHANNEL}: {N} samples at {FS:.0f} Hz, {T[0]:.1f}..{T[-1]:.1f}s",
      flush=True)
print(f"[TASK]  history {WINDOW} samples ({WINDOW / FS:.2f}s) -> next {HORIZON} "
      f"({HORIZON / FS:.2f}s), direct multi-horizon", flush=True)
print(f"[SPLIT] train={len(tr_i)}, val={len(va_i)}, test={len(te_i)} windows; "
      f"test starts t={T[te_i[0] + WINDOW]:.1f}s (seizure is in TRAIN)", flush=True)


def make(ii):
    Xs = np.stack([Z[i: i + WINDOW] for i in ii]).astype(np.float32)
    Ys = np.stack([Z[i + WINDOW: i + WINDOW + HORIZON] for i in ii]).astype(np.float32)
    return Xs, Ys


Xtr, Ytr = make(tr_i)
Xva, Yva = make(va_i)
Xte, Yte = make(te_i)


def r2_per_h(pred, true):
    """R^2 at each horizon, against the variance of the test target itself."""
    ss = np.sum((true - pred) ** 2, axis=0)
    tot = np.sum((true - true.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss / np.where(tot > 0, tot, np.nan)


scores, preds = {}, {}

# ---------------------------------------------------------------------------
# 2. Persistence and the optimal linear predictor
# ---------------------------------------------------------------------------
scores["persistence"] = r2_per_h(np.repeat(Xte[:, -1:], HORIZON, axis=1), Yte)
preds["persistence"] = np.repeat(Xte[:, -1:], HORIZON, axis=1)

D_tr = np.column_stack([Xtr, np.ones(len(Xtr))])
D_te = np.column_stack([Xte, np.ones(len(Xte))])
W_lin, *_ = np.linalg.lstsq(D_tr, Ytr, rcond=None)
preds["linear"] = D_te @ W_lin
scores["linear"] = r2_per_h(preds["linear"], Yte)
print(f"[FIT]   linear: {D_tr.shape[1]} inputs -> {HORIZON} outputs", flush=True)


# ---------------------------------------------------------------------------
# 3. Recurrent nets
# ---------------------------------------------------------------------------
class Recurrent(nn.Module):
    def __init__(self, kind):
        super().__init__()
        cls = nn.LSTM if kind == "LSTM" else nn.GRU
        self.rnn = cls(1, HIDDEN, LAYERS, batch_first=True)
        self.head = nn.Linear(HIDDEN, HORIZON)

    def forward(self, x):
        out, _ = self.rnn(x.unsqueeze(-1))
        return self.head(out[:, -1])


def train(kind):
    torch.manual_seed(SEED)
    net = Recurrent(kind)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    xt, yt = torch.tensor(Xtr), torch.tensor(Ytr)
    xv, yv = torch.tensor(Xva), torch.tensor(Yva)
    best, best_state, bad = np.inf, None, 0
    t0 = time.time()
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(xt))
        for k in range(0, len(xt), BATCH):
            b = perm[k: k + BATCH]
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(xt[b]), yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(nn.functional.mse_loss(net(xv), yv))
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        p = net(torch.tensor(Xte)).numpy()
    print(f"[FIT]   {kind}: {ep + 1} epochs, best val MSE {best:.5f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return p


for kind in ("LSTM", "GRU"):
    preds[kind] = train(kind)
    scores[kind] = r2_per_h(preds[kind], Yte)


# ---------------------------------------------------------------------------
# 4. Results
# ---------------------------------------------------------------------------
order = ["persistence", "linear", "LSTM", "GRU"]
print(f"\n[EVAL]  held-out R2 by forecast horizon (test block, post-ictal)")
print("        " + "horizon".ljust(12) + "".join(f"{m:>13}" for m in order))
for h in REPORT_H:
    if h > HORIZON:
        continue
    row = "".join(f"{scores[m][h - 1]:>13.4f}" for m in order)
    print(f"        {h:>4} ({h / FS * 1000:>5.0f} ms){row}", flush=True)


def crossing(m, thresh=0.0):
    s = scores[m]
    below = np.where(s < thresh)[0]
    return int(below[0] + 1) if len(below) else None


summary = {}
for m in order:
    z = crossing(m, 0.0)
    p05 = crossing(m, 0.05)
    summary[m] = {
        "r2_by_horizon": [float(v) for v in scores[m]],
        "horizon_r2_below_0.05": p05, "horizon_r2_below_0": z,
        "r2_at_10ms": float(scores[m][1]), "r2_at_100ms": float(scores[m][19]),
        "r2_at_500ms": float(scores[m][99]),
    }
    print(f"        {m:>12}: R2 drops under 0.05 at horizon "
          f"{'never' if p05 is None else f'{p05} ({p05 / FS * 1000:.0f} ms)'}",
          flush=True)

best_net = max(("LSTM", "GRU"), key=lambda m: np.nanmean(scores[m]))
gain = scores[best_net] - scores["linear"]
print(f"\n[VERDICT] best net ({best_net}) minus linear, by horizon: "
      f"10ms={gain[1]:+.4f}, 50ms={gain[9]:+.4f}, 100ms={gain[19]:+.4f}, "
      f"500ms={gain[99]:+.4f}", flush=True)
print(f"[VERDICT] mean gain over all horizons: {float(np.nanmean(gain)):+.4f}",
      flush=True)

net_helps = bool(np.nanmean(gain) >= 0.02)
useful_h = summary[best_net]["horizon_r2_below_0.05"]
passed = print_check(
    net_helps,
    f"a recurrent net beats the optimal linear predictor on {CHANNEL}",
)
print(f"        mean R2 gain of {best_net} over linear = {np.nanmean(gain):+.4f}",
      flush=True)
print(f"        useful horizon of {best_net} (R2>0.05) = "
      f"{'all' if useful_h is None else f'{useful_h / FS * 1000:.0f} ms'}", flush=True)

save_metrics(RESULT_DIR / "metrics.json", {
    "task": f"{CHANNEL} direct multi-horizon forecasting benchmark",
    "channel": CHANNEL, "fs": FS, "window": WINDOW, "horizon": HORIZON,
    "n_train_windows": int(len(tr_i)), "n_test_windows": int(len(te_i)),
    "test_start_time": float(T[te_i[0] + WINDOW]),
    "seizure_in_train": True,
    "models": summary,
    "best_net": best_net,
    "net_minus_linear_mean": float(np.nanmean(gain)),
    "net_beats_linear": net_helps,
    "passed": passed,
})
np.savez_compressed(RESULT_DIR / "forecast_arrays.npz",
                    horizons=np.arange(1, HORIZON + 1) / FS,
                    **{f"r2_{m}": scores[m] for m in order},
                    Yte=Yte[:200], **{f"pred_{m}": preds[m][:200] for m in order})


# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------
use_pub_style()
COL = {"persistence": "0.55", "linear": "#1f77b4", "LSTM": "#d62728", "GRU": "#2ca02c"}


def plot_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    hs = np.arange(1, HORIZON + 1) / FS * 1000
    for m in order:
        axes[0].plot(hs, scores[m], lw=1.6, color=COL[m], label=m)
        axes[1].plot(hs, scores[m] - scores["linear"], lw=1.6, color=COL[m], label=m)
    axes[0].axhline(0.0, color="k", lw=0.8)
    axes[0].axhline(0.05, color="0.6", ls=":", lw=1.0)
    axes[0].set_ylabel(r"held-out $R^2$")
    axes[0].set_title("forecast skill vs horizon")
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_ylabel(r"$R^2$ minus linear")
    axes[1].set_title("does the net add anything?")
    for ax in axes:
        ax.set_xlabel("forecast horizon (ms)")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
    fig.suptitle(f"{CHANNEL}: direct multi-horizon forecasting, post-ictal test block")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"forecast_skill.{e}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_examples() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    tt = np.arange(-WINDOW, HORIZON) / FS * 1000
    for ax, k in zip(axes, np.linspace(0, len(Xte) - 1, 3).astype(int)):
        ax.plot(tt, np.concatenate([Xte[k], Yte[k]]), color="k", lw=1.0, label="data")
        for m in order:
            ax.plot(tt[WINDOW:], preds[m][k], lw=1.1, color=COL[m], ls="--", label=m)
        ax.axvline(0.0, color="0.3", lw=0.8, ls=":")
        ax.set_ylabel("z")
    axes[0].legend(fontsize=7, ncol=5)
    axes[-1].set_xlabel("time relative to forecast origin (ms)")
    fig.suptitle(f"{CHANNEL}: example forecasts (left of 0 is the given history)")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"examples.{e}", dpi=250, bbox_inches="tight")
    plt.close(fig)


plot_curves()
plot_examples()
print(f"\n[FIGS]  Saved {RESULT_DIR}", flush=True)
