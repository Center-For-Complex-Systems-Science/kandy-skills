#!/usr/bin/env python3
"""Is this recording a noise-driven damped oscillator with drifting parameters?

Everything else tried in this directory assumes the signal is a deterministic
trajectory and fails on rollout.  The measured reason: at the dynamical
timescale (lag 7 samples = 35 ms, the AMI first minimum) only ~13% of the
one-step increment variance is predictable from the channel's own past, and
that number is FLAT from embedding dimension 2 to 24.  No deterministic vector
field beats that bound, which is why every autonomous model collapsed.

This file tests the model that treats the other ~87% as the point rather than
as failure:

    x_ddot + 2*zeta(t)*omega(t)*x_dot + omega(t)^2 * x = sigma(t) * xi(t)

a lightly damped oscillator driven by noise, whose three parameters drift
slowly.  Discretely that is an autoregressive model whose dominant complex pole
pair carries (omega, damping) and whose residual scale carries sigma.  So the
parameters are estimated in short sliding windows and the SEIZURE IS READ AS A
PATH IN (omega, zeta, sigma) SPACE, not as a trajectory in x.

Two claims, both falsifiable here
---------------------------------
1. **zeta -> 0 at onset.**  Damping collapsing toward the stability boundary is
   critical slowing down, the standard dynamical account of seizure
   initiation.  If zeta does not move, the model is wrong.  Reported as the
   pole radius r = exp(-zeta*omega*dt): r -> 1 is the boundary.  The
   independent classical CSD indicators (lag-1 autocorrelation, variance) are
   computed on the same windows as a cross-check.

2. **The residual is white.**  If x really is AR-plus-white-noise then after
   removing the fitted AR structure nothing is left.  A structured residual
   means something deterministic is still being missed and the 13% ceiling
   deserves another look.  Ljung-Box per window, plus the fraction of residual
   autocorrelations outside the 95% band.

Why the filter choice matters, and why both are run
---------------------------------------------------
A band-pass imposes its own spectral shape, and an AR(2) cannot represent it,
so residuals would be coloured by the FILTER rather than by the brain and
claim 2 would fail for a reason that has nothing to do with the model.  The
primary analysis therefore uses a 1 Hz high-pass only (drift removed, passband
left broad).  The 1-40 Hz band used everywhere else in this directory is run
alongside as a sensitivity check, and the two are reported side by side.

For the same reason the AR order is not fixed at 2: extra poles are spent on
whatever shaping the front end imposes.  The order is selected by BIC per
window, the DOMINANT complex pole pair is what gets interpreted as the
oscillator, and plain AR(2) is reported next to it so the difference is visible.

This is a description of one recording, not a mechanism claim and not a
prediction of anything.  With a single seizure there is no out-of-sample event,
so the pre/ictal contrast below is descriptive statistics on one instance.
"""

from __future__ import annotations

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
from scipy.signal import butter, filtfilt
from scipy.stats import chi2

from _utils import ensure_outdir, print_check, save_metrics
from kandy.plotting import use_pub_style


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_float(name, default):
    return float(os.environ.get(name, default))


SEED = 42
np.random.seed(SEED)

DATA_PATH = REPO_ROOT / "data" / "Dory_2021_04_12_006_preprocessed.pkl"
RESULT_DIR = ensure_outdir(REPO_ROOT / "results" / "Epilepsy-Stochastic-Oscillator")

CHANNEL = os.environ.get("KANDY_SO_CHANNEL", "hippocampus_R")
WIN_SECONDS = env_float("KANDY_SO_WIN_SECONDS", 1.0)
STEP_SECONDS = env_float("KANDY_SO_STEP_SECONDS", 0.05)
MAX_ORDER = env_int("KANDY_SO_MAX_ORDER", 12)
LB_LAGS = env_int("KANDY_SO_LB_LAGS", 20)

PRE = (-13.0, -2.0)
ICTAL = (-1.0, 4.0)
POST = (8.0, 20.0)


# ---------------------------------------------------------------------------
# 1. Data, two front ends
# ---------------------------------------------------------------------------
with DATA_PATH.open("rb") as fh:
    payload = pickle.load(fh)

X_RAW = np.asarray(payload["X"], dtype=float)
FS = float(payload["fs"])
DT = float(payload["dt"])
T_ALL = np.asarray(payload["t"], dtype=float)
LABELS = list(payload["labels"])
ch = LABELS.index(CHANNEL)
nyq = 0.5 * FS

bh, ah = butter(2, 1.0 / nyq, btype="high")
bb, ab = butter(4, [1.0 / nyq, 40.0 / nyq], btype="band")
edge = int(round(2.0 * FS))
sl = slice(edge, len(X_RAW) - edge)
T = T_ALL[sl]
FRONT_ENDS = {
    "highpass_1Hz": filtfilt(bh, ah, X_RAW[:, ch])[sl],
    "bandpass_1_40Hz": filtfilt(bb, ab, X_RAW[:, ch])[sl],
}
for k in FRONT_ENDS:
    v = FRONT_ENDS[k]
    FRONT_ENDS[k] = (v - v.mean()) / (v.std() + 1e-12)

WIN = int(round(WIN_SECONDS * FS))
STEP = int(round(STEP_SECONDS * FS))
starts = np.arange(0, len(T) - WIN, STEP)
centres = T[starts + WIN // 2]
print(f"[DATA]  {CHANNEL}, fs={FS:.0f} Hz, {len(T)} samples "
      f"{T[0]:.1f}..{T[-1]:.1f}s", flush=True)
print(f"[WIN]   {len(starts)} windows of {WIN} samples ({WIN_SECONDS:.2f}s), "
      f"step {STEP} ({STEP_SECONDS * 1000:.0f} ms)", flush=True)


# ---------------------------------------------------------------------------
# 2. Per-window AR fit -> dominant pole -> (omega, damping, sigma)
# ---------------------------------------------------------------------------
def ar_fit(seg, p):
    """Least-squares AR(p); returns coefficients and residuals."""
    n = len(seg) - p
    D = np.column_stack([seg[p - k - 1: p - k - 1 + n] for k in range(p)])
    y = seg[p:]
    c, *_ = np.linalg.lstsq(D, y, rcond=None)
    return c, y - D @ c


def bic(resid, p, n):
    s2 = float(np.mean(resid ** 2)) + 1e-30
    return n * np.log(s2) + p * np.log(n)


def dominant_pole(coefs):
    """Largest-|z| complex-conjugate root of the AR characteristic polynomial."""
    roots = np.roots(np.concatenate([[1.0], -np.asarray(coefs)]))
    cx = roots[np.abs(roots.imag) > 1e-9]
    if len(cx) == 0:
        return np.nan, np.nan
    z = cx[np.argmax(np.abs(cx))]
    return float(np.abs(z)), float(abs(np.angle(z)))


def ljung_box(resid, h, p):
    n = len(resid)
    r = resid - resid.mean()
    denom = np.sum(r * r) + 1e-30
    acf = np.array([np.sum(r[k:] * r[:-k]) / denom for k in range(1, h + 1)])
    q = n * (n + 2) * np.sum(acf ** 2 / (n - np.arange(1, h + 1)))
    dof = max(h - p, 1)
    return float(q), float(chi2.sf(q, dof)), acf


results = {}
for tag, sig in FRONT_ENDS.items():
    rec = {k: np.full(len(starts), np.nan) for k in
           ("order", "r", "omega", "decay", "zeta", "sigma", "lb_p", "acf_out",
            "r2_ar2", "r_ar2", "omega_ar2", "lag1", "var")}
    for i, s0 in enumerate(starts):
        seg = sig[s0: s0 + WIN]
        seg = seg - seg.mean()
        best = None
        for p in range(2, MAX_ORDER + 1):
            c, e = ar_fit(seg, p)
            b = bic(e, p, len(e))
            if best is None or b < best[0]:
                best = (b, p, c, e)
        _, p, c, e = best
        r, th = dominant_pole(c)
        rec["order"][i] = p
        rec["r"][i] = r
        rec["omega"][i] = th / DT
        rec["decay"][i] = -np.log(max(r, 1e-12)) / DT
        rec["zeta"][i] = (-np.log(max(r, 1e-12)) / DT) / max(th / DT, 1e-12)
        rec["sigma"][i] = float(np.std(e))
        q, pv, acf = ljung_box(e, LB_LAGS, p)
        rec["lb_p"][i] = pv
        rec["acf_out"][i] = float(np.mean(np.abs(acf) > 1.96 / np.sqrt(len(e))))
        c2, e2 = ar_fit(seg, 2)
        r2p, th2 = dominant_pole(c2)
        rec["r_ar2"][i] = r2p
        rec["omega_ar2"][i] = th2 / DT
        rec["r2_ar2"][i] = 1.0 - np.var(e2) / (np.var(seg[2:]) + 1e-30)
        v = seg - seg.mean()
        rec["lag1"][i] = float(np.sum(v[1:] * v[:-1]) / (np.sum(v * v) + 1e-30))
        rec["var"][i] = float(np.var(seg))
    results[tag] = rec
    print(f"[AR]    {tag}: BIC order median={np.nanmedian(rec['order']):.0f}, "
          f"AR(2) within-window R2 median={np.nanmedian(rec['r2_ar2']):.3f}",
          flush=True)


def wmask(win):
    return (centres >= win[0]) & (centres <= win[1])


def auroc(score, label):
    score, label = np.asarray(score), np.asarray(label).astype(bool)
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    npos, nneg = int(label.sum()), int((~label).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


summary = {}
for tag, rec in results.items():
    pre_m, ic_m, post_m = wmask(PRE), wmask(ICTAL), wmask(POST)
    cmp_m = pre_m | ic_m | post_m
    s = {}
    for key in ("r", "omega", "zeta", "sigma", "lag1", "var"):
        s[key] = {w: float(np.nanmean(rec[key][m]))
                  for w, m in (("pre", pre_m), ("ictal", ic_m), ("post", post_m))}
        s[key]["auroc_ictal"] = auroc(rec[key][cmp_m], ic_m[cmp_m])
    s["whiteness"] = {
        "ljung_box_p_median": float(np.nanmedian(rec["lb_p"])),
        "frac_windows_white_p>0.05": float(np.nanmean(rec["lb_p"] > 0.05)),
        "acf_outside_95pct_band_median": float(np.nanmedian(rec["acf_out"])),
        "bic_order_median": float(np.nanmedian(rec["order"])),
    }
    summary[tag] = s

    print(f"\n[{tag}]", flush=True)
    print(f"  {'quantity':>10}  {'pre':>9} {'ictal':>9} {'post':>9}   AUROC(ictal)",
          flush=True)
    for key, lab in (("omega", "omega"), ("r", "pole r"), ("zeta", "zeta"),
                     ("sigma", "sigma"), ("lag1", "lag-1 ac"), ("var", "variance")):
        d = s[key]
        print(f"  {lab:>10}  {d['pre']:>9.4f} {d['ictal']:>9.4f} {d['post']:>9.4f}"
              f"   {d['auroc_ictal']:.3f}", flush=True)
    w = s["whiteness"]
    print(f"  whiteness: Ljung-Box p median={w['ljung_box_p_median']:.3g}, "
          f"windows with p>0.05: {w['frac_windows_white_p>0.05']:.1%}, "
          f"residual ACF outside 95% band: "
          f"{w['acf_outside_95pct_band_median']:.1%} of lags", flush=True)


# ---------------------------------------------------------------------------
# 3. The two claims
# ---------------------------------------------------------------------------
PRIMARY = "highpass_1Hz"
p_rec, p_sum = results[PRIMARY], summary[PRIMARY]
zeta_pre, zeta_ict = p_sum["zeta"]["pre"], p_sum["zeta"]["ictal"]
zeta_drop = (zeta_pre - zeta_ict) / max(abs(zeta_pre), 1e-12)
r_pre, r_ict = p_sum["r"]["pre"], p_sum["r"]["ictal"]
white_frac = p_sum["whiteness"]["frac_windows_white_p>0.05"]

print(f"\n[CLAIM 1] zeta pre={zeta_pre:.4f} -> ictal={zeta_ict:.4f} "
      f"({zeta_drop:+.1%}); pole radius {r_pre:.4f} -> {r_ict:.4f} "
      f"(1.0 = stability boundary); AUROC={p_sum['zeta']['auroc_ictal']:.3f}",
      flush=True)
print(f"[CLAIM 2] residual white in {white_frac:.1%} of windows "
      f"(Ljung-Box p>0.05)", flush=True)

claim1 = bool(zeta_drop >= 0.10 and r_ict > r_pre)
claim2 = bool(white_frac >= 0.5)
passed = print_check(
    claim1 and claim2,
    "noise-driven damped oscillator: damping collapses at onset AND the "
    "residual is white",
)
print(f"        claim 1 (zeta falls >=10%, pole moves toward 1): {claim1}", flush=True)
print(f"        claim 2 (majority of windows white):             {claim2}", flush=True)
if not claim2:
    print("        -> a coloured residual means AR-plus-white-noise is not the "
          "whole story; structure remains that this model does not capture",
          flush=True)

save_metrics(RESULT_DIR / "metrics.json", {
    "task": f"{CHANNEL} sliding-window stochastic damped oscillator",
    "channel": CHANNEL,
    "window_seconds": WIN_SECONDS, "step_seconds": STEP_SECONDS,
    "max_ar_order": MAX_ORDER, "n_windows": int(len(starts)),
    "windows": {"pre": list(PRE), "ictal": list(ICTAL), "post": list(POST)},
    "front_ends": summary,
    "primary_front_end": PRIMARY,
    "claim_1_damping_collapse": {"zeta_pre": zeta_pre, "zeta_ictal": zeta_ict,
                                 "relative_drop": zeta_drop,
                                 "pole_r_pre": r_pre, "pole_r_ictal": r_ict,
                                 "passed": claim1},
    "claim_2_white_residual": {"fraction_white": white_frac, "passed": claim2},
    "note": "one recording, one seizure: descriptive, not predictive",
    "passed": passed,
})
np.savez_compressed(RESULT_DIR / "stochastic_oscillator_arrays.npz",
                    t=T, centres=centres,
                    **{f"{tag}__{k}": v for tag, rec in results.items()
                       for k, v in rec.items()},
                    **{f"signal__{tag}": v for tag, v in FRONT_ENDS.items()})


# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
use_pub_style()


def plot_parameters() -> None:
    fig, axes = plt.subplots(5, 1, figsize=(11, 11), sharex=True)
    for ax in axes:
        ax.axvspan(*ICTAL, color="#f4a261", alpha=0.22, lw=0)
    axes[0].plot(T, FRONT_ENDS[PRIMARY], lw=0.3, color="0.5")
    axes[0].set_ylabel(f"{CHANNEL}")
    axes[0].set_title(f"{CHANNEL}: seizure as a path in "
                      r"$(\omega,\ \zeta,\ \sigma)$")
    for tag, col, ls in ((PRIMARY, "#1f77b4", "-"),
                         ("bandpass_1_40Hz", "#9467bd", "--")):
        axes[1].plot(centres, results[tag]["omega"] / (2 * np.pi), lw=1.1,
                     color=col, ls=ls, label=tag)
        axes[2].plot(centres, results[tag]["zeta"], lw=1.1, color=col, ls=ls)
        axes[3].plot(centres, results[tag]["sigma"], lw=1.1, color=col, ls=ls)
        axes[4].plot(centres, results[tag]["r"], lw=1.1, color=col, ls=ls)
    axes[1].set_ylabel(r"$\omega/2\pi$ (Hz)")
    axes[1].legend(fontsize=7)
    axes[2].set_ylabel(r"damping $\zeta$")
    axes[3].set_ylabel(r"noise $\sigma$")
    axes[4].axhline(1.0, color="#d62728", ls=":", lw=1.2, label="stability boundary")
    axes[4].set_ylabel("pole radius $r$")
    axes[4].legend(fontsize=7)
    axes[4].set_xlabel("recording time (s)")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"parameters.{e}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    rec = results[PRIMARY]
    axes[0].axvspan(*ICTAL, color="#f4a261", alpha=0.22, lw=0)
    axes[0].plot(centres, rec["lag1"], lw=1.1, color="#1f77b4", label="lag-1 autocorr")
    ax2 = axes[0].twinx()
    ax2.plot(centres, rec["var"], lw=1.1, color="#2ca02c", label="variance")
    ax2.set_ylabel("variance", color="#2ca02c")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("lag-1 autocorrelation", color="#1f77b4")
    axes[0].set_title("classical critical-slowing-down indicators")
    axes[1].axvspan(*ICTAL, color="#f4a261", alpha=0.22, lw=0)
    axes[1].plot(centres, rec["lb_p"], lw=0.9, color="#1f77b4")
    axes[1].axhline(0.05, color="#d62728", ls="--", lw=1.0, label="p=0.05")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("Ljung-Box p")
    axes[1].set_title(f"residual whiteness ({white_frac:.0%} of windows white)")
    axes[1].legend(fontsize=7)
    sc = axes[2].scatter(rec["zeta"], rec["sigma"], c=centres, s=10, cmap="coolwarm")
    axes[2].set_xlabel(r"damping $\zeta$")
    axes[2].set_ylabel(r"noise $\sigma$")
    axes[2].set_title("the seizure as a loop in parameter space")
    fig.colorbar(sc, ax=axes[2], label="time (s)")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"diagnostics.{e}", dpi=250, bbox_inches="tight")
    plt.close(fig)


plot_parameters()
plot_diagnostics()
print(f"\n[FIGS]  Saved {RESULT_DIR}", flush=True)
