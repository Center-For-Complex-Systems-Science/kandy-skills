#!/usr/bin/env python3
"""Epileptor-style Heaviside recruitment coupling between bands and channels.

The Epileptor couples its fast and slow subsystems through a hard switch, not a
ramp.  Everything in this directory so far used ReLU(x - th), and an ablation
showed those gates are INERT: removing them costs ~0.01 held-out, and shuffling
which threshold belongs to which band changes nothing.  A ramp carries
magnitude and no event; a Heaviside carries an event and no magnitude.  That is
a genuinely different hypothesis and it is what this script tests.

The coupling variable
---------------------
Not "channel c' band b' influences channel c band b" as 5 separate columns --
alpha, beta and gamma carry only 6-8 INDEPENDENT samples (539 rows at a ~5 s
autocorrelation time), so 5 extra predictors there is guaranteed overfitting.
Instead one column, the RECRUITMENT FRACTION:

    R_{c,b'}(t) = (1/(NC-1)) * sum_{c' != c} H_k( x[c',b'](t) - theta_{b'} )

"what fraction of the OTHER channels currently have band b' above threshold".
That is a field variable in the Epileptor sense, it is one predictor rather
than five, and it is directly interpretable as spread.

H_k is a sigmoid of sharpness k.  k -> infinity is the true Heaviside; a hard
step is not differentiable and is brutal for RK4, so k is SCANNED from soft to
near-hard and the winner is reported.  If soft wins, the step hypothesis is not
supported and that is the finding.

The search, and its multiple-comparison control
-----------------------------------------------
25 source->target band pairs x 3 thresholds x 4 sharpnesses = 300 combinations.
Searching 300 things and reporting the best is how you find noise, so:

  * every candidate is scored HELD OUT on the transition, by a model trained
    without it (set B)
  * the winner is retested against circular-shift surrogates of the source
  * Benjamini-Hochberg FDR is applied across the whole grid
  * a ReLU version of the identical column is run alongside, so "Heaviside
    beats ReLU" is a measured comparison and not a substitution

Only delta and theta targets are treated as decisive (102 and 43 independent
samples); alpha, beta and gamma are reported but flagged as unresolvable.

Run:
    cd skills/kandy/examples && python epilepsy/epilepsy_heaviside_recruitment.py
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, filtfilt, hilbert

import kandy.core as kcore
import kandy.training as ktrain
from _utils import ensure_outdir, print_check, save_metrics
from kandy import CustomLift, KANDy

torch.set_num_threads(1)


def _patch_curve2coef() -> int:
    import sys as _sys

    try:
        from kan import spline as _spline
    except ImportError:
        return 0
    _B, original = _spline.B_batch, _spline.curve2coef

    def curve2coef(x_eval, y_eval, grid, k):
        batch, in_dim = x_eval.shape[0], x_eval.shape[1]
        out_dim, n_coef = y_eval.shape[2], grid.shape[1] - k - 1
        mat = _B(x_eval, grid, k).permute(1, 0, 2)[:, None, :, :]
        mat = mat.expand(in_dim, out_dim, batch, n_coef)
        yy = y_eval.permute(1, 2, 0).unsqueeze(dim=3)
        try:
            return torch.linalg.lstsq(mat, yy).solution[:, :, :, 0]
        except Exception:                                        # noqa: BLE001
            XtX = torch.einsum("ijmn,ijnp->ijmp", mat.permute(0, 1, 3, 2), mat)
            Xty = torch.einsum("ijmn,ijnp->ijmp", mat.permute(0, 1, 3, 2), yy)
            eye = torch.eye(XtX.shape[2], device=mat.device)[None, None]
            return torch.linalg.solve(XtX + 1e-8 * eye, Xty)[:, :, :, 0]

    n = 0
    for _nm, _m in list(_sys.modules.items()):
        if _nm.split(".")[0] == "kan" and _m is not None:
            if getattr(_m, "curve2coef", None) is original:
                _m.curve2coef = curve2coef
                n += 1
    return n


_patch_curve2coef()

SUB = {"lamb_l1": 0.2, "lamb_entropy": 0.0, "lamb_coef": 0.0, "lamb_coefdiff": 1.0}
_orig = ktrain.fit_kan


def _fk(*a, **kw):
    kw.update(SUB)
    return _orig(*a, **kw)


ktrain.fit_kan = _fk
kcore.fit_kan = _fk

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
RESULT_DIR = Path(ensure_outdir(REPO_ROOT / "results" / "Epilepsy-Heaviside-Recruitment"))

BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 50.0)}
BAND = list(BANDS)
NB = len(BAND)
PAIRS = [(a, b) for a in range(NB) for b in range(a + 1, NB)]
SMOOTH_TAU, DS, EDGE_S, LAG_S, GUARD_S = 0.15, 10, 2.0, 2.0, 2.0
BASIN = [(-13.0, -9.0), (22.0, 28.0)]
TRANSITION = (-2.0, 9.0)
THETA_PCT = [50, 70, 85]
SHARPNESS = [1.0, 4.0, 16.0, 64.0]          # 64 is effectively a step
RESOLVED = ["delta", "theta"]               # the bands with enough samples
N_SURR = int(os.environ.get("KANDY_HV_SURR", 300))
RIDGE, FDR_Q = 1e-2, 0.05
TAU = np.array([12.6, 15.9, 20.0, 20.0, 20.0])

payload = pickle.load((REPO_ROOT / "data" / "Dory_2021_04_12_006_preprocessed.pkl").open("rb"))
XR = np.asarray(payload["X"], float)
FS = float(payload["fs"])
T_ALL = np.asarray(payload["t"], float)
LAB = list(payload["labels"])
NC = len(LAB)


def causal_ewma(y, tau_s):
    al = 1.0 - np.exp(-1.0 / (tau_s * FS))
    out = np.empty_like(y)
    acc = y[0].copy()
    for i in range(len(y)):
        acc = acc + al * (y[i] - acc)
        out[i] = acc
    return out


def env_of(lo, hi):
    bb, aa = butter(4, [lo / (0.5 * FS), hi / (0.5 * FS)], btype="band")
    return causal_ewma(np.log(np.abs(hilbert(filtfilt(bb, aa, XR, axis=0), axis=0))
                              + 1e-9), SMOOTH_TAU)


e0 = int(round(EDGE_S * FS))
sl = slice(e0, len(XR) - e0, DS)
T = T_ALL[sl]
N = len(T)
DT = float(T[1] - T[0])
X = np.zeros((N, NC, NB))
for b, nm in enumerate(BAND):
    v = env_of(*BANDS[nm])[sl]
    X[:, :, b] = (v - v.mean(0)) / (v.std(0) + 1e-12)
LAGN = max(1, int(round(LAG_S / DT)))
DX = (X[LAGN:] - X[:-LAGN]) / LAG_S
rows = np.arange(N - LAGN)
t_rows = T[:N - LAGN]
nb = np.zeros(len(rows), bool)
for w in BASIN:
    nb |= (t_rows >= w[0] - GUARD_S) & (t_rows <= w[1] + GUARD_S)
nt = (t_rows >= TRANSITION[0] - GUARD_S) & (t_rows <= TRANSITION[1] + GUARD_S)
trans = rows[(t_rows >= TRANSITION[0]) & (t_rows <= TRANSITION[1])]
trainA, trainB = rows[~nb], rows[~(nb | nt)]

print("=" * 84)
print("HEAVISIDE RECRUITMENT COUPLING")
print("=" * 84)
print(f"{NC} channels x {NB} bands, {N} rows @ {1/DT:.0f} Hz; "
      f"train(B) {len(trainB)}, transition {len(trans)}\n")


def recruit(bsrc, pct, k, src=None):
    """Fraction of the OTHER channels with band bsrc above threshold."""
    S = X[:, :, bsrc] if src is None else src
    th = np.percentile(S, pct)
    H = 1.0 / (1.0 + np.exp(-k * (S - th)))          # k -> inf is Heaviside
    tot = H.sum(axis=1, keepdims=True)
    return (tot - H) / (NC - 1)                       # exclude self, per channel


def relu_recruit(bsrc, pct, src=None):
    S = X[:, :, bsrc] if src is None else src
    th = np.percentile(S, pct)
    Rl = np.maximum(S - th, 0.0)
    tot = Rl.sum(axis=1, keepdims=True)
    return (tot - Rl) / (NC - 1)


def r2(y, q):
    d = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - q) ** 2) / d) if d > 0 else np.nan


def gain(col, c, btgt):
    """Held-out gain on the transition from adding one recruitment column."""
    own = np.column_stack([X[:, c, :], np.ones(N)])
    des = np.column_stack([own, col[:, c]])

    def sc(A):
        G = A[trainB].T @ A[trainB] + RIDGE * np.eye(A.shape[1])
        w = np.linalg.solve(G, A[trainB].T @ DX[trainB, c, btgt])
        return r2(DX[trans, c, btgt], A[trans] @ w)

    return sc(des) - sc(own)


# ---------------------------------------------------------------------------
# 1. The search
# ---------------------------------------------------------------------------
print("[SEARCH] 25 band pairs x 3 thresholds x 4 sharpnesses, "
      "held out on the transition")
grid = []
for si, snm in enumerate(BAND):
    for ti, tnm in enumerate(BAND):
        for pct in THETA_PCT:
            for k in SHARPNESS:
                col = recruit(si, pct, k)
                g = float(np.mean([gain(col, c, ti) for c in range(NC)]))
                grid.append({"source": snm, "target": tnm, "pct": pct,
                             "sharpness": k, "gain": g,
                             "resolved": tnm in RESOLVED})
res_grid = [g for g in grid if g["resolved"]]
res_grid.sort(key=lambda r: -r["gain"])
print(f"\n  top 10 among RESOLVED targets ({', '.join(RESOLVED)}):")
print(f"  {'source -> target':>22} {'th pct':>7} {'sharp':>7} {'gain':>8}")
for r in res_grid[:10]:
    print(f"  {r['source'] + ' -> ' + r['target']:>22} {r['pct']:7d} "
          f"{r['sharpness']:7.0f} {r['gain']:+8.4f}")

best = res_grid[0]
print(f"\n  best: {best['source']} -> {best['target']}, threshold p{best['pct']}, "
      f"sharpness k={best['sharpness']:.0f}")

# does sharpness matter?  average gain by k, over resolved targets
print(f"\n[SHARPNESS] mean gain by k (resolved targets only) -- is a STEP better "
      "than a soft sigmoid?")
by_k = {}
for k in SHARPNESS:
    v = [r["gain"] for r in res_grid if r["sharpness"] == k]
    by_k[k] = float(np.mean(v))
    print(f"            k={k:5.0f}  mean gain {by_k[k]:+.4f}"
          + ("   <- effectively a Heaviside" if k >= 64 else ""))
hard_better = by_k[max(SHARPNESS)] > by_k[min(SHARPNESS)]

# Heaviside vs ReLU at the winner's position
col_h = recruit(BAND.index(best["source"]), best["pct"], best["sharpness"])
col_r = relu_recruit(BAND.index(best["source"]), best["pct"])
ti = BAND.index(best["target"])
g_h = float(np.mean([gain(col_h, c, ti) for c in range(NC)]))
g_r = float(np.mean([gain(col_r, c, ti) for c in range(NC)]))
print(f"\n[VS RELU] at the winning position, Heaviside {g_h:+.4f} vs "
      f"ReLU {g_r:+.4f}  -> "
      + ("Heaviside better" if g_h > g_r else "ReLU better or equal"))

# ---------------------------------------------------------------------------
# 2. Is the winner real?  Surrogates, then FDR over the whole grid.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
print(f"\n[SURROGATE] {N_SURR} circular shifts of the source band at the winner")
null = np.empty(N_SURR)
for s in range(N_SURR):
    shifted = np.roll(X[:, :, BAND.index(best["source"])],
                      int(rng.integers(int(2 / DT), N - int(2 / DT))), axis=0)
    cs = recruit(None, best["pct"], best["sharpness"], src=shifted)
    null[s] = float(np.mean([gain(cs, c, ti) for c in range(NC)]))
p_best = float((np.sum(null >= best["gain"]) + 1) / (N_SURR + 1))
print(f"            observed {best['gain']:+.4f}, null p95 "
      f"{np.percentile(null, 95):+.4f}, p = {p_best:.4f}")

n_grid = len(res_grid)
bonf = 0.05 / n_grid
print(f"            searched {n_grid} resolved combinations, so a Bonferroni "
      f"threshold is p < {bonf:.5f}")
survives = p_best < bonf
print(f"            winner {'SURVIVES' if survives else 'does NOT survive'} "
      "correction for the search")

# ---------------------------------------------------------------------------
# 3. Fit the network with the winning Heaviside coupling
# ---------------------------------------------------------------------------
REC = {b: recruit(b, best["pct"], best["sharpness"]) for b in range(NB)}
LO = X.reshape(N, -1).min(0) - 0.35 * np.ptp(X.reshape(N, -1), 0)
HI = X.reshape(N, -1).max(0) + 0.35 * np.ptp(X.reshape(N, -1), 0)
BC, BR = 0.5 * (LO + HI), 0.5 * (HI - LO)
STATE = X.reshape(N, NC * NB)
TGT = np.zeros((len(DX), NC * NB))
for c in range(NC):
    for b in range(NB):
        j = c * NB + b
        t_ = STATE[:len(DX), j] + TAU[b] * DX[:, c, b]
        TGT[:, j] = BR[j] * np.arctanh(np.clip((t_ - BC[j]) / BR[j], -0.999, 0.999))
TSC = TGT[trainA].std(0) + 1e-12
AUG = np.column_stack([STATE] + [REC[b] for b in range(NB)])
NREC = NB


def make_lift(c, with_rec=True):
    def npf(s):
        s = np.atleast_2d(s)
        xc = s[:, c * NB:(c + 1) * NB]
        f = [np.ones(len(s))] + [xc[:, b] for b in range(NB)]
        f += [xc[:, a] * xc[:, b] for a, b in PAIRS]
        if with_rec:
            f += [s[:, NC * NB + b * NC + c] for b in range(NREC)]
        return np.column_stack(f)

    def tf(s):
        xc = s[:, c * NB:(c + 1) * NB]
        f = [torch.ones_like(s[:, 0])] + [xc[:, b] for b in range(NB)]
        f += [xc[:, a] * xc[:, b] for a, b in PAIRS]
        if with_rec:
            f += [s[:, NC * NB + b * NC + c] for b in range(NREC)]
        return torch.stack(f, 1)

    return npf, tf, 1 + NB + len(PAIRS) + (NREC if with_rec else 0)


AUGF = np.column_stack([STATE] + [REC[b] for b in range(NB)])
print(f"\n[FIT]   six channel models with the recruitment columns "
      f"({1+NB+len(PAIRS)+NREC} lift columns each)")


def fit_c(c, rws, with_rec, lamb=0.03):
    npf, tf, d = make_lift(c, with_rec)
    m = KANDy(lift=CustomLift(npf, d, tf, f"h{c}"), grid=5, k=3, steps=80, seed=SEED,
              base_fun=torch.nn.Identity(), device="cpu")
    with contextlib.redirect_stdout(StringIO()):
        m.fit(X=AUGF[rws], X_dot=TGT[rws, c * NB:(c + 1) * NB] / TSC[c * NB:(c + 1) * NB],
              val_frac=0.15, test_frac=0.15, lamb=lamb, opt="LBFGS", lr=1.0,
              batch=-1, stop_grid_update_step=10, patience=0, verbose=False)
    return m


def score(models, with_rec):
    out = {}
    for c, lab in enumerate(LAB):
        inner = models[c].predict(AUGF[trans]) * TSC[c * NB:(c + 1) * NB]
        sl_ = slice(c * NB, (c + 1) * NB)
        tgt = BC[sl_] + BR[sl_] * np.tanh(inner / BR[sl_])
        rate = (tgt - STATE[trans][:, sl_]) / TAU
        out[lab] = {BAND[b]: float(r2(DX[trans, c, b], rate[:, b])) for b in range(NB)}
    return out


mods_rec = [fit_c(c, trainB, True) for c in range(NC)]
mods_no = [fit_c(c, trainB, False) for c in range(NC)]
s_rec, s_no = score(mods_rec, True), score(mods_no, False)
print(f"\n{'channel':>14} | " + " ".join(f"{b[:3]:>14}" for b in BAND))
print(f"{'':>14} | " + " ".join(f"{'recr / none':>14}" for _ in BAND))
gain_res, gain_all = [], []
for lab in LAB:
    cells = []
    for b in BAND:
        d_ = s_rec[lab][b] - s_no[lab][b]
        gain_all.append(d_)
        if b in RESOLVED:
            gain_res.append(d_)
        cells.append(f"{s_rec[lab][b]:+.2f}/{s_no[lab][b]:+.2f}")
    print(f"{lab:>14} | " + " ".join(f"{c_:>14}" for c_ in cells))
mg_res, mg_all = float(np.mean(gain_res)), float(np.mean(gain_all))
print(f"\n  recruitment gain, resolved bands only : {mg_res:+.4f}")
print(f"  recruitment gain, all bands           : {mg_all:+.4f}")

# ---------------------------------------------------------------------------
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
M = np.full((NB, NB), np.nan)
for r in grid:
    if r["pct"] == best["pct"] and r["sharpness"] == best["sharpness"]:
        M[BAND.index(r["target"]), BAND.index(r["source"])] = r["gain"]
im = axes[0].imshow(M, cmap="RdBu_r", vmin=-abs(np.nanmax(M)), vmax=abs(np.nanmax(M)))
axes[0].set_xticks(range(NB), BAND, rotation=40, ha="right", fontsize=7)
axes[0].set_yticks(range(NB), BAND, fontsize=7)
axes[0].set_xlabel("source band (recruitment)")
axes[0].set_ylabel("target band")
for i in range(NB):
    for j in range(NB):
        axes[0].text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=6)
axes[0].set_title(f"Held-out gain, p{best['pct']}, k={best['sharpness']:.0f}",
                  fontsize=9, fontweight="bold")
fig.colorbar(im, ax=axes[0], shrink=0.8)
axes[1].plot(SHARPNESS, [by_k[k] for k in SHARPNESS], "o-", color="#D55E00")
axes[1].set_xscale("log")
axes[1].set_xlabel("sharpness k  (large = Heaviside)")
axes[1].set_ylabel("mean held-out gain")
axes[1].set_title("Does hardness help?", fontsize=9, fontweight="bold", loc="left")
axes[2].hist(null, bins=30, color="0.75")
axes[2].axvline(best["gain"], color="#D55E00", lw=2,
                label=f"observed (p={p_best:.3f})")
axes[2].set_xlabel("gain")
axes[2].set_ylabel("surrogates")
axes[2].set_title("Winner vs circular shifts", fontsize=9, fontweight="bold",
                  loc="left")
axes[2].legend(frameon=False, fontsize=7)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(RESULT_DIR / f"heaviside_search.{ext}", dpi=250, bbox_inches="tight")
plt.close(fig)

print("\n" + "=" * 84)
ok = print_check(survives, f"the best coupling survives correction for the "
                           f"{n_grid}-combination search (p={p_best:.4f} vs "
                           f"{bonf:.5f})")
print_check(hard_better, "a hard step beats a soft sigmoid "
                         f"(k={max(SHARPNESS):.0f}: {by_k[max(SHARPNESS)]:+.4f} vs "
                         f"k={min(SHARPNESS):.0f}: {by_k[min(SHARPNESS)]:+.4f})")
print_check(g_h > g_r, f"Heaviside beats ReLU at the same position "
                       f"({g_h:+.4f} vs {g_r:+.4f})")
print_check(mg_res > 0.02, f"recruitment helps the fitted network on resolved "
                           f"bands ({mg_res:+.4f})")

save_metrics(RESULT_DIR / "metrics.json", {
    "task": "Epileptor-style Heaviside recruitment coupling: band-pair search",
    "coupling": "R_{c,b'} = mean over other channels of sigmoid(k*(x[c',b'] - th))",
    "resolved_targets": RESOLVED,
    "grid_size": len(grid), "resolved_grid_size": n_grid,
    "best": best, "p_best": p_best, "bonferroni_threshold": bonf,
    "survives_search_correction": bool(survives),
    "gain_by_sharpness": {str(k): v for k, v in by_k.items()},
    "heaviside_vs_relu": {"heaviside": g_h, "relu": g_r},
    "network_scores": {"with_recruitment": s_rec, "without": s_no,
                       "gain_resolved": mg_res, "gain_all": mg_all},
    "top_20": res_grid[:20],
    "note": "alpha/beta/gamma targets carry 6-8 independent samples and are "
            "reported but not decisive.",
})
print(f"[DONE] {RESULT_DIR}")
