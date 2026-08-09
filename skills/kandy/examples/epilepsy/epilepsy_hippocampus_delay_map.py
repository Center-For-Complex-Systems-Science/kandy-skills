#!/usr/bin/env python3
"""Mackey-Glass recipe applied to one epilepsy channel: hippocampus_R.

    scalar x(t)  ->  DelayEmbedding  ->  pick the informative lag  ->  KANDy

Everything else in this directory models six channels at once and asks the
data to support a coupling structure it does not support.  This file drops all
of that.  One channel -- the seizure-onset zone -- one scalar series, two delay
coordinates, one equation.  The point is to see whether a readable equation
exists at all for this recording when nothing is assumed about network
structure, oscillators, or coupling.

Differences from ``odes/mackey_glass_delay_example.py``
------------------------------------------------------
* **Map mode, not a derivative.**  Mackey-Glass has an exact analytic dx/dt.
  Here a derivative would have to be estimated by finite differences on a noisy
  recording, so the target is the one-step increment instead:

      x_{n+1} - x_n = F(x_n, x_{n-L})

  and rollout is iteration of that map (the Henon recipe).

* **Regime-split fits.**  The seizure is a regime change, so the same two-
  coordinate model is fitted separately on pre-ictal, ictal and post-ictal
  windows.  If a seizure means anything at the level of a scalar equation, the
  spline edges should differ between those fits, and the attractors should
  differ visibly.  That comparison is the actual experiment; the global fit is
  reported first as a reference.

Finding 1 from the Mackey-Glass file applies here with force
------------------------------------------------------------
In delay coordinates a high one-step R^2 is cheap and means nothing -- a rich
enough delay space linearises almost any smooth dynamics.  At fs=200 Hz on a
1-40 Hz band-limited signal, consecutive samples are strongly correlated, so
predicting x_{n+1} from x_n is close to trivial.  A plain linear fit is
therefore reported alongside every KANDy number, and the verdict is taken from
the ROLLOUT ATTRACTOR, never from one-step R^2.  Related: the six-channel
delay map in this directory reaches one-step R^2 = 0.994 and still has an
autonomous seizure-window correlation of 0.04.

No ground truth
---------------
Mackey-Glass validates its edges against the true Hill term.  There is no such
reference here, so the edges are reported as shapes and the symbolic snap is
reported with its own R^2 -- an equation that only fits after a bad snap is
worse than no equation.  Chaotic or noise-driven trajectories decorrelate, so
the rollout is judged on the attractor, not pointwise.
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
from kandy.plotting import get_edge_activation, plot_loss_curves, use_pub_style


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_float(name, default):
    return float(os.environ.get(name, default))


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = REPO_ROOT / "data" / "Dory_2021_04_12_006_preprocessed.pkl"
RESULT_DIR = ensure_outdir(REPO_ROOT / "results" / "Epilepsy-Hippocampus-Delay-Map")

CHANNEL = os.environ.get("KANDY_HD_CHANNEL", "hippocampus_R")
BANDPASS_HZ = (1.0, 40.0)
FILTER_ORDER = 4
EDGE_TRIM_SECONDS = 2.0

MAX_LAG = env_int("KANDY_HD_MAX_LAG", 120)        # 0.6 s at 200 Hz
GRID = env_int("KANDY_HD_GRID", 8)
STEPS = env_int("KANDY_HD_STEPS", 120)
N_FIT_MAX = env_int("KANDY_HD_N_FIT_MAX", 5000)
ROLL_SECONDS = env_float("KANDY_HD_ROLL_SECONDS", 6.0)
FORMULA_R2 = env_float("KANDY_HD_FORMULA_R2", 0.95)

PRE = (-13.0, -2.0)
ICTAL = (-1.0, 4.0)
POST = (8.0, 20.0)


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(StringIO()):
        return fn(*a, **k)


# ---------------------------------------------------------------------------
# 1. One scalar channel
# ---------------------------------------------------------------------------
with DATA_PATH.open("rb") as fh:
    payload = pickle.load(fh)

X_RAW = np.asarray(payload["X"], dtype=float)
FS = float(payload["fs"])
DT = float(payload["dt"])
T_ALL = np.asarray(payload["t"], dtype=float)
LABELS = list(payload["labels"])
if CHANNEL not in LABELS:
    raise ValueError(f"{CHANNEL!r} not in {LABELS}")
ch = LABELS.index(CHANNEL)

nyq = 0.5 * FS
bb, aa = butter(FILTER_ORDER, [BANDPASS_HZ[0] / nyq, BANDPASS_HZ[1] / nyq], btype="band")
x_filt = filtfilt(bb, aa, X_RAW[:, ch])
edge = int(round(EDGE_TRIM_SECONDS * FS))
sl = slice(edge, len(X_RAW) - edge)
T = T_ALL[sl]
X = x_filt[sl]
X = (X - X.mean()) / (X.std() + 1e-12)
N = len(X)

print(f"[DATA]  {CHANNEL}: {N} samples at fs={FS:.0f} Hz, "
      f"{T[0]:.1f}..{T[-1]:.1f}s, band {BANDPASS_HZ[0]:.0f}-{BANDPASS_HZ[1]:.0f} Hz",
      flush=True)


# ---------------------------------------------------------------------------
# 2. Lag scan -- model-free, exactly the Mackey-Glass binned-mean predictor
# ---------------------------------------------------------------------------
def binned_r2(a, b, y, n_bins=26):
    ia = np.clip(((a - a.min()) / (np.ptp(a) + 1e-12) * n_bins).astype(int), 0, n_bins - 1)
    ib = np.clip(((b - b.min()) / (np.ptp(b) + 1e-12) * n_bins).astype(int), 0, n_bins - 1)
    key = ia * n_bins + ib
    train = np.arange(len(y)) % 2 == 0
    test = ~train
    tot = np.bincount(key[train], weights=y[train], minlength=n_bins ** 2)
    cnt = np.bincount(key[train], minlength=n_bins ** 2)
    means = np.where(cnt > 0, tot / np.maximum(cnt, 1), y[train].mean())
    p = means[key[test]]
    return 1.0 - np.sum((y[test] - p) ** 2) / np.sum((y[test] - y[test].mean()) ** 2)


lags = np.arange(1, MAX_LAG + 1)
scan = np.empty(len(lags))
for k, L in enumerate(lags):
    a = X[MAX_LAG:-1]
    b = X[MAX_LAG - L: -1 - L]
    y = X[MAX_LAG + 1:] - X[MAX_LAG:-1]
    scan[k] = binned_r2(a, b, y)
pred_lag = int(lags[int(np.argmax(scan))])


def autocorr_first_zero(sig, max_lag):
    v = sig - sig.mean()
    ac = np.correlate(v, v, mode="full")[len(v) - 1: len(v) + max_lag]
    ac = ac / ac[0]
    neg = np.where(ac <= 0.0)[0]
    return int(neg[0]) if len(neg) else max_lag


def ami(sig, lag, n_bins=32):
    """Average mutual information between x_t and x_{t-lag}."""
    a, b = sig[lag:], sig[:-lag] if lag else sig
    ea = np.linspace(a.min(), a.max() + 1e-12, n_bins + 1)
    eb = np.linspace(b.min(), b.max() + 1e-12, n_bins + 1)
    h, _, _ = np.histogram2d(a, b, bins=[ea, eb])
    p = h / h.sum()
    pa, pb = p.sum(axis=1, keepdims=True), p.sum(axis=0, keepdims=True)
    nz = p > 0
    return float(np.sum(p[nz] * np.log(p[nz] / (pa @ pb)[nz])))


ami_curve = np.array([ami(X, int(L)) for L in lags])
first_min = next((i for i in range(1, len(ami_curve) - 1)
                  if ami_curve[i] <= ami_curve[i - 1]
                  and ami_curve[i] <= ami_curve[i + 1]), None)
acf_zero = autocorr_first_zero(X, MAX_LAG)
ami_lag = int(lags[first_min]) if first_min is not None else acf_zero
BEST_LAG = env_int("KANDY_HD_LAG", ami_lag)

#
# Why the lag is NOT taken from the predictive scan.
#
# Mackey-Glass is sampled so that tau is 17 samples and the scan peaks there.
# This recording is oversampled relative to its own dynamics -- 200 Hz on a
# 1-40 Hz band, so ~33 samples per cycle of the dominant rhythm -- and the best
# single predictor of the next increment is then always the immediately
# preceding sample.  The scan duly peaks at L=1, which is smoothness, not a
# delay: x_t and x_{t-1} are nearly identical, the embedding collapses onto the
# diagonal, and the learned map contracts to a fixed point (measured: rollout
# std 0.155 against a data std of 1.0).
#
# So the embedding lag comes from the standard Takens criteria instead -- first
# minimum of the average mutual information, cross-checked against the first
# zero crossing of the autocorrelation -- which pick a lag where the two
# coordinates are informative about each other but not redundant.  The
# predictive scan is still computed and plotted, as the diagnostic that shows
# why it cannot be used here.
#
print(f"[SCAN]  predictive scan peaks at L={pred_lag} "
      f"({pred_lag * DT * 1000:.0f} ms, binned R2={scan.max():.4f}) "
      "-- oversampling artefact, not used", flush=True)
print(f"[LAG]   AMI first minimum L={ami_lag} ({ami_lag * DT * 1000:.0f} ms); "
      f"autocorrelation first zero L={acf_zero} ({acf_zero * DT * 1000:.0f} ms)",
      flush=True)
print(f"[LAG]   using L={BEST_LAG} ({BEST_LAG * DT * 1000:.0f} ms)", flush=True)


# ---------------------------------------------------------------------------
# 3. Build the map, fit, and always quote the linear baseline next to it
# ---------------------------------------------------------------------------
def build(rows_mask=None):
    idx = np.arange(MAX_LAG, N - 1)
    if rows_mask is not None:
        idx = idx[rows_mask[idx]]
    state = np.column_stack([X[idx], X[idx - BEST_LAG]])
    target = (X[idx + 1] - X[idx])[:, None]
    return idx, state, target


def r2(y, p):
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def fit_map(state, target, tag):
    n = len(state)
    n_tr = int(n * 0.7)
    guard = BEST_LAG + 2
    tr = np.arange(0, max(n_tr - guard, 1))
    te = np.arange(min(n_tr + guard, n - 1), n)
    if len(tr) > N_FIT_MAX:
        tr = np.sort(np.random.default_rng(SEED).choice(tr, N_FIT_MAX, replace=False))

    lin = np.column_stack([state, np.ones(len(state))])
    cl, *_ = np.linalg.lstsq(lin[tr], target[tr], rcond=None)
    r2_lin = r2(target[te], lin[te] @ cl)

    m = KANDy(lift=CustomLift(fn=lambda A: A, output_dim=2, name="delay_pair"),
              grid=GRID, k=3, steps=STEPS, seed=SEED)
    quiet(m.fit, X=state[tr], X_dot=target[tr], val_frac=0.15, test_frac=0.15,
          lamb=0.0, patience=40, verbose=False)
    r2_kan = r2(target[te], m.predict(state[te]))
    print(f"[FIT]   {tag:>6}: n={n:6d}  held-out one-step R2  "
          f"KANDy={r2_kan:.5f}  linear={r2_lin:.5f}  "
          f"(gain {r2_kan - r2_lin:+.5f})", flush=True)
    return m, {"n": int(n), "r2_kan": r2_kan, "r2_linear": r2_lin,
               "gain": r2_kan - r2_lin}


idx_all, STATE, TARGET = build()
model, stats_global = fit_map(STATE, TARGET, "global")
print("[FIT]   one-step R2 in delay coordinates is cheap (Mackey-Glass finding 1);"
      " the attractor below is the real test", flush=True)


# ---------------------------------------------------------------------------
# 4. Autonomous rollout of the learned map, and its attractor
# ---------------------------------------------------------------------------
net = model.model_


def rollout(start_idx, steps):
    """Iterate x_{n+1} = x_n + F(x_n, x_{n-L}) from measured history."""
    buf = np.empty(steps + BEST_LAG + 1)
    buf[: BEST_LAG + 1] = X[start_idx - BEST_LAG: start_idx + 1]
    with torch.no_grad():
        for i in range(BEST_LAG, BEST_LAG + steps):
            inp = torch.tensor([[buf[i], buf[i - BEST_LAG]]], dtype=torch.float32)
            buf[i + 1] = buf[i] + float(net(inp)[0, 0])
            if not np.isfinite(buf[i + 1]) or abs(buf[i + 1]) > 50.0:
                return buf[BEST_LAG: i + 1]
    return buf[BEST_LAG:]


roll_steps = int(round(ROLL_SECONDS / DT))
start = int(np.argmin(np.abs(T - PRE[0]))) + BEST_LAG + 1
ROLL = rollout(start, roll_steps)
print(f"\n[ROLL]  autonomous map iteration, {len(ROLL)} of {roll_steps + 1} steps "
      f"survived ({len(ROLL) * DT:.2f}s)", flush=True)
print(f"[ROLL]  attractor range   learned [{ROLL.min():+.2f}, {ROLL.max():+.2f}]  "
      f"data [{X.min():+.2f}, {X.max():+.2f}]", flush=True)
print(f"[ROLL]  attractor std      learned {ROLL.std():.3f}  data {X.std():.3f}",
      flush=True)
roll_alive = len(ROLL) >= roll_steps
roll_std_err = abs(ROLL.std() - 1.0)


# ---------------------------------------------------------------------------
# 5. The equation: spline edges, then a symbolic snap reported with its R^2
# ---------------------------------------------------------------------------
state_t = torch.tensor(STATE, dtype=torch.float32)
edges = {}
for i, nm in enumerate(["x_t", f"x_(t-{BEST_LAG})"]):
    xs, ys = get_edge_activation(model.model_, 0, i, 0, X=state_t)
    xs, ys = np.asarray(xs).ravel(), np.asarray(ys).ravel()
    edges[nm] = (xs, ys - ys.mean())

formulas = []
try:
    exprs = quiet(model.get_formula, var_names=["x_t", f"x_tm{BEST_LAG}"],
                  r2_threshold=FORMULA_R2)
    formulas = [str(e) for e in exprs]
    print(f"\n[EQUATION] get_formula(r2_threshold={FORMULA_R2}) -> "
          f"{len(formulas)} expression(s)", flush=True)
    for e in formulas:
        print(f"           x_(n+1) - x_n = {e if len(e) < 300 else e[:300] + ' ...'}",
              flush=True)
except Exception as exc:  # noqa: BLE001 - symbolic snapping is best-effort
    print(f"\n[EQUATION] get_formula failed: {type(exc).__name__}: {exc}", flush=True)


# ---------------------------------------------------------------------------
# 6. Regime split -- does the equation change at the seizure?
# ---------------------------------------------------------------------------
regimes = {}
for name, win in (("pre", PRE), ("ictal", ICTAL), ("post", POST)):
    mask = (T >= win[0]) & (T <= win[1])
    idx_r, st_r, tg_r = build(mask)
    if len(st_r) < 400:
        print(f"[REGIME] {name}: too few rows ({len(st_r)}), skipped", flush=True)
        continue
    m_r, s_r = fit_map(st_r, tg_r, name)
    st_rt = torch.tensor(st_r, dtype=torch.float32)
    ed = {}
    for i, nm in enumerate(["x_t", f"x_(t-{BEST_LAG})"]):
        xs, ys = get_edge_activation(m_r.model_, 0, i, 0, X=st_rt)
        xs, ys = np.asarray(xs).ravel(), np.asarray(ys).ravel()
        ed[nm] = (xs, ys - ys.mean())
    regimes[name] = {"stats": s_r, "edges": ed, "state": st_r, "window": win}

# Compare the ictal equation with the pre-ictal one on a common grid.
edge_shift = {}
if "pre" in regimes and "ictal" in regimes:
    for nm in regimes["pre"]["edges"]:
        xs_p, ys_p = regimes["pre"]["edges"][nm]
        xs_i, ys_i = regimes["ictal"]["edges"][nm]
        lo = max(xs_p.min(), xs_i.min())
        hi = min(xs_p.max(), xs_i.max())
        gx = np.linspace(lo, hi, 200)
        gp = np.interp(gx, xs_p, ys_p)
        gi = np.interp(gx, xs_i, ys_i)
        denom = np.sqrt(np.mean(gp ** 2)) + 1e-12
        edge_shift[nm] = float(np.sqrt(np.mean((gi - gp) ** 2)) / denom)
    print("\n[REGIME] ictal-vs-pre edge change (RMS difference / RMS pre-ictal edge):",
          flush=True)
    for nm, v in edge_shift.items():
        print(f"         {nm:>14}: {v:.3f}", flush=True)


# ---------------------------------------------------------------------------
# 7. Checks
# ---------------------------------------------------------------------------
THRESHOLDS = {"kandy_gain_over_linear_min": 0.001,
              "rollout_must_survive": True,
              "attractor_std_rel_err_max": 0.35}
passed = print_check(
    stats_global["gain"] >= THRESHOLDS["kandy_gain_over_linear_min"]
    and roll_alive and roll_std_err <= THRESHOLDS["attractor_std_rel_err_max"],
    f"{CHANNEL} two-coordinate delay map: nonlinear gain, surviving rollout, "
    "attractor scale reproduced",
)
print(f"        KANDy gain over linear = {stats_global['gain']:+.5f}", flush=True)
print(f"        rollout survived       = {roll_alive}", flush=True)
print(f"        attractor std rel err  = {roll_std_err:.3f}", flush=True)

metrics = {
    "task": f"{CHANNEL} scalar delay-embedding map (Mackey-Glass recipe)",
    "channel": CHANNEL,
    "fs": FS, "n_samples": int(N),
    "bandpass_hz": list(BANDPASS_HZ),
    "lag_selection": {
        "lags": [int(v) for v in lags],
        "predictive_binned_r2": [float(v) for v in scan],
        "predictive_peak_lag": pred_lag,
        "ami_curve": [float(v) for v in ami_curve],
        "ami_first_min_lag": ami_lag,
        "acf_first_zero_lag": acf_zero,
        "used_lag_samples": BEST_LAG,
        "used_lag_ms": BEST_LAG * DT * 1000.0,
        "note": "predictive scan degenerates to L=1 on oversampled data; "
                "lag taken from AMI first minimum (Takens standard)"},
    "global_fit": stats_global,
    "note_one_step": "one-step R2 in delay coordinates is cheap; judged on attractor",
    "rollout": {"seconds_requested": ROLL_SECONDS,
                "steps_survived": int(len(ROLL)),
                "survived": bool(roll_alive),
                "range_learned": [float(ROLL.min()), float(ROLL.max())],
                "range_data": [float(X.min()), float(X.max())],
                "std_learned": float(ROLL.std()), "std_data": float(X.std())},
    "symbolic_formulas": formulas,
    "regimes": {k: v["stats"] for k, v in regimes.items()},
    "ictal_vs_pre_edge_change": edge_shift,
    "thresholds": THRESHOLDS,
    "passed": passed,
}
save_metrics(RESULT_DIR / "metrics.json", metrics)
np.savez_compressed(
    RESULT_DIR / "hippocampus_delay_arrays.npz",
    t=T, x=X, lags=lags, scan=scan, best_lag=BEST_LAG,
    state=STATE, target=TARGET, rollout=ROLL,
    **{f"edge_{i}": np.vstack(edges[nm]) for i, nm in enumerate(edges)},
)


# ---------------------------------------------------------------------------
# 8. Figures -- the attractor is the deliverable
# ---------------------------------------------------------------------------
use_pub_style()


def plot_attractor() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    wins = [("pre-ictal", PRE, "#1f77b4"), ("ictal", ICTAL, "#d62728"),
            ("post-ictal", POST, "#2ca02c")]
    for ax, (name, win, c) in zip(axes[:3], wins):
        m = (T >= win[0]) & (T <= win[1])
        xa = X[m][BEST_LAG:]
        xb = X[m][:-BEST_LAG] if BEST_LAG else X[m]
        ax.plot(xa, xb, lw=0.35, color=c, alpha=0.8)
        ax.set_title(f"{name}  ({win[0]:.0f}..{win[1]:.0f}s)")
        ax.set_xlabel(r"$x_t$")
        ax.set_ylabel(rf"$x_{{t-{BEST_LAG}}}$")
        ax.set_aspect("equal", adjustable="box")
    lim = max(abs(X).max(), abs(ROLL).max()) * 1.05
    axes[3].plot(X[BEST_LAG:], X[:-BEST_LAG], lw=0.3, color="0.6", alpha=0.6,
                 label="data (all)")
    if len(ROLL) > BEST_LAG:
        axes[3].plot(ROLL[BEST_LAG:], ROLL[:-BEST_LAG], lw=0.5, color="#d62728",
                     label="learned map, autonomous")
    axes[3].set_title("learned vs measured attractor")
    axes[3].set_xlabel(r"$x_t$")
    axes[3].set_ylabel(rf"$x_{{t-{BEST_LAG}}}$")
    axes[3].set_aspect("equal", adjustable="box")
    axes[3].legend(fontsize=7)
    for ax in axes:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    fig.suptitle(f"{CHANNEL} delay attractor, lag {BEST_LAG} samples "
                 f"({BEST_LAG * DT * 1000:.0f} ms)")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"attractor.{e}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_scan_and_edges() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ms = lags * DT * 1000
    axes[0].plot(ms, scan, lw=1.0, color="#1f77b4", label="predictive (binned $R^2$)")
    ax2 = axes[0].twinx()
    ax2.plot(ms, ami_curve, lw=1.2, color="#9467bd", label="AMI")
    ax2.set_ylabel("AMI (nats)", color="#9467bd")
    axes[0].axvline(BEST_LAG * DT * 1000, color="#d62728", ls="--", lw=1.4,
                    label=f"used L={BEST_LAG}")
    axes[0].axvline(pred_lag * DT * 1000, color="0.5", ls=":", lw=1.2,
                    label=f"predictive peak L={pred_lag}")
    axes[0].set_xlabel("candidate lag (ms)")
    axes[0].set_ylabel(r"binned-mean $R^2$", color="#1f77b4")
    axes[0].set_title("lag selection (AMI, not prediction)")
    axes[0].legend(fontsize=6, loc="center right")
    for k, nm in enumerate(edges):
        ax = axes[1 + k]
        xs, ys = edges[nm]
        ax.plot(xs, ys, lw=1.6, color="0.2", label="global fit")
        for rn, col in (("pre", "#1f77b4"), ("ictal", "#d62728"), ("post", "#2ca02c")):
            if rn in regimes and nm in regimes[rn]["edges"]:
                exs, eys = regimes[rn]["edges"][nm]
                ax.plot(exs, eys, lw=1.1, color=col, alpha=0.85, label=rn)
        ax.set_xlabel(nm)
        ax.set_ylabel(r"edge contribution to $x_{n+1}-x_n$")
        ax.set_title(f"spline edge on {nm}")
        ax.legend(fontsize=7)
    fig.suptitle(f"{CHANNEL}: the learned equation, by regime")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"equation_edges.{e}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_rollout() -> None:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    t_roll = T[start] + np.arange(len(ROLL)) * DT
    ax.plot(T, X, lw=0.4, color="0.6", label="data")
    ax.plot(t_roll, ROLL, lw=0.7, color="#d62728", label="learned map, autonomous")
    ax.axvspan(*ICTAL, color="#f4a261", alpha=0.2, lw=0)
    ax.set_xlim(t_roll[0] - 1, t_roll[-1] + 1)
    ax.set_xlabel("recording time (s)")
    ax.set_ylabel(f"{CHANNEL} (z)")
    ax.legend(fontsize=8)
    ax.set_title(f"autonomous iteration of the two-coordinate map "
                 f"({len(ROLL) * DT:.1f}s survived)")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"rollout.{e}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_losses() -> None:
    if getattr(model, "train_results_", None):
        fig, _ = plot_loss_curves(model.train_results_, save=str(RESULT_DIR / "loss"))
        plt.close(fig)


plot_attractor()
plot_scan_and_edges()
plot_rollout()
plot_losses()
print(f"\n[FIGS]  Saved {RESULT_DIR}", flush=True)
