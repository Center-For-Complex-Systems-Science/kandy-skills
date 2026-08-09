#!/usr/bin/env python3
"""Refit T_gamma across lambda: are the alpha->gamma edges real or overfitting?

Per-edge extraction found that gamma's three largest edges are alpha (14.7),
alpha*beta (23.2) and alpha*gamma (21.7) -- pathways ``get_formula`` omitted
entirely.  But gamma was fitted at lamb = 0, and its edges sum to 144 against an
output range of 5.9: a CANCELLATION FACTOR of 25.  Twenty large terms almost
annihilating each other is what an unregularized 21-column fit on 539 rows looks
like, and any "finding" read off it could be an artefact of that.

The test is simple and it is a falsification test, not a demonstration.  Refit
gamma alone across a lambda grid and watch three things together:

    cancellation      does the equation become well-conditioned?
    held-out skill    does it predict the transition it never saw?
    alpha share       do the alpha pathways SURVIVE as the fit is sparsified?

If the alpha edges hold their share of the model as lambda rises and
cancellation falls, they are structure.  If they collapse together with the
cancellation, they were the overfitting.

Run:
    cd skills/kandy/examples && python epilepsy/epilepsy_gamma_lambda_scan.py
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, filtfilt, hilbert

from _utils import ensure_outdir, print_check, save_metrics
from kandy import CustomLift, KANDy
from kandy.plotting import get_edge_activation

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

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
RESULT_DIR = Path(ensure_outdir(REPO_ROOT / "results" / "Epilepsy-Cross-Frequency"))

BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 50.0)}
BAND_NAMES = list(BANDS)
NB, GB = 5, 4                                   # GB = index of gamma
PAIRS = [(a, b) for a in range(NB) for b in range(a + 1, NB)]
SMOOTH_TAU, DS, EDGE_S, LAG_S = 0.15, 10, 2.0, 2.0
GRID, STEPS, BOX_MARGIN, GUARD_S = 5, 80, 0.35, 2.0
BASIN_TEST = {"pre_ictal": (-13.0, -9.0), "post_ictal": (22.0, 28.0)}
TRANSITION = (-2.0, 9.0)
LAMB_GRID = [float(v) for v in os.environ.get(
    "KANDY_GS_LAMB", "0.0,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1.0").split(",")]

met = json.loads((RESULT_DIR / "metrics.json").read_text())
TAU = np.array(met["model_definition"]["tau_s"])

payload = pickle.load((REPO_ROOT / "data" / "Dory_2021_04_12_006_preprocessed.pkl").open("rb"))
Xr = np.asarray(payload["X"], float)
FS = float(payload["fs"])
T_ALL = np.asarray(payload["t"], float)

LOG_ENV = {}
for n, (lo, hi) in BANDS.items():
    bb, aa = butter(4, [lo / (0.5 * FS), hi / (0.5 * FS)], btype="band")
    LOG_ENV[n] = np.log(np.abs(hilbert(filtfilt(bb, aa, Xr, axis=0), axis=0)) + 1e-9)


def causal_ewma(y, tau_s):
    al = 1.0 - np.exp(-1.0 / (tau_s * FS))
    out = np.empty_like(y)
    acc = y[0].copy()
    for i in range(len(y)):
        acc = acc + al * (y[i] - acc)
        out[i] = acc
    return out


S = np.column_stack([causal_ewma(LOG_ENV[n], SMOOTH_TAU).mean(axis=1) for n in BAND_NAMES])
S = (S - S.mean(0)) / (S.std(0) + 1e-12)
e0 = int(round(EDGE_S * FS))
STATE, T = S[e0:len(S) - e0:DS], T_ALL[e0:len(S) - e0:DS]
N, DT = len(STATE), float(T[1] - T[0])
LAG = max(1, int(round(LAG_S / DT)))
DX = (STATE[LAG:] - STATE[:-LAG]) / LAG_S
X_LO = STATE.min(0) - BOX_MARGIN * np.ptp(STATE, 0)
X_HI = STATE.max(0) + BOX_MARGIN * np.ptp(STATE, 0)
X_C, X_R = 0.5 * (X_LO + X_HI), 0.5 * (X_HI - X_LO)
THRESH = STATE[(T >= -13.0) & (T <= -2.0)].mean(0)
T_INNER = X_R * np.arctanh(np.clip(((STATE[:len(DX)] + TAU * DX) - X_C) / X_R, -0.999, 0.999))

rows = np.arange(N - LAG)
t_rows = T[:N - LAG]
near_basin = np.zeros(len(rows), bool)
for w in BASIN_TEST.values():
    near_basin |= (t_rows >= w[0] - GUARD_S) & (t_rows <= w[1] + GUARD_S)
near_trans = (t_rows >= TRANSITION[0] - GUARD_S) & (t_rows <= TRANSITION[1] + GUARD_S)
trans_rows = rows[(t_rows >= TRANSITION[0]) & (t_rows <= TRANSITION[1])]
trainA = rows[~near_basin]                       # transition IN train
trainB = rows[~(near_basin | near_trans)]        # transition WITHHELD
T_SCALE = T_INNER[trainA].std(0) + 1e-12

COL = ["1"] + BAND_NAMES + [f"ReLU({n})" for n in BAND_NAMES] \
    + [f"{BAND_NAMES[a]}*{BAND_NAMES[b]}" for a, b in PAIRS]
D = len(COL)
ALPHA_COLS = [i for i, c in enumerate(COL) if "alpha" in c]


def lift_np(x):
    x = np.atleast_2d(x)
    f = [np.ones(len(x))] + [x[:, b] for b in range(NB)]
    f += [np.maximum(x[:, b] - THRESH[b], 0.0) for b in range(NB)]
    f += [x[:, a] * x[:, b] for a, b in PAIRS]
    return np.column_stack(f)


def lift_t(x):
    th = torch.as_tensor(THRESH, dtype=x.dtype)
    f = [torch.ones_like(x[:, 0])] + [x[:, b] for b in range(NB)]
    f += [torch.clamp(x[:, b] - th[b], min=0.0) for b in range(NB)]
    f += [x[:, a] * x[:, b] for a, b in PAIRS]
    return torch.stack(f, 1)


LIFTV = lift_np(STATE)


def r2(y, p):
    den = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / den) if den > 0 else np.nan


def rate_of(model, idx):
    inner = model.predict(STATE[idx]).ravel() * T_SCALE[GB]
    return (X_C[GB] + X_R[GB] * np.tanh(inner / X_R[GB]) - STATE[idx, GB]) / TAU[GB]


def fit_gamma(lamb, rws):
    m = KANDy(lift=CustomLift(lift_np, D, lift_t, "g"), grid=GRID, k=3, steps=STEPS,
              seed=SEED, base_fun=torch.nn.Identity(), device="cpu")
    with contextlib.redirect_stdout(StringIO()):
        m.fit(X=STATE[rws], X_dot=T_INNER[rws, GB:GB + 1] / T_SCALE[GB],
              val_frac=0.15, test_frac=0.15, lamb=lamb, opt="LBFGS", lr=1.0,
              batch=-1, stop_grid_update_step=10, patience=0, verbose=False)
    return m


print("=" * 92)
print("GAMMA LAMBDA SCAN -- do the alpha->gamma pathways survive sparsification?")
print("=" * 92)
print(f"{'lambda':>8} {'cancel':>7} {'active':>7} {'heldout R2':>11} "
      f"{'zero base':>10} | {'alpha share':>12} {'top edge':>14}")
print("-" * 92)

scan = []
for lamb in LAMB_GRID:
    # SET B (transition withheld) for the honest held-out number on the event
    mB = fit_gamma(lamb, trainB)
    hr2 = r2(DX[trans_rows, GB], rate_of(mB, trans_rows))
    zr2 = r2(DX[trans_rows, GB], np.zeros(len(trans_rows)))

    # SET A for the structure we actually report
    mA = fit_gamma(lamb, trainA)
    Xin = torch.tensor(lift_np(STATE[trainA]), dtype=torch.float32)
    mA.model_.save_act = True
    with torch.no_grad():
        mA.model_(Xin)
    rng = np.zeros(D)
    for i in range(D):
        xa, ya = get_edge_activation(mA.model_, l=0, i=i, j=0)
        o = np.argsort(np.asarray(xa))
        rng[i] = float(np.ptp(np.interp(LIFTV[trainA, i],
                                        np.asarray(xa)[o], np.asarray(ya)[o])))
    net = mA.predict(STATE[trainA]).ravel()
    cancel = float(rng.sum() / max(np.ptp(net), 1e-9))
    n_act = int((rng > 0.02).sum())
    a_share = float(rng[ALPHA_COLS].sum() / max(rng.sum(), 1e-9))
    top = COL[int(np.argmax(rng))]
    scan.append({"lamb": lamb, "cancellation": cancel, "n_active": n_act,
                 "heldout_r2_transition": hr2, "zero_baseline": zr2,
                 "alpha_share": a_share, "top_edge": top,
                 "ranges": rng.tolist(),
                 "alpha_ranges": {COL[i]: float(rng[i]) for i in ALPHA_COLS}})
    print(f"{lamb:8.4g} {cancel:7.1f} {n_act:7d} {hr2:+11.3f} {zr2:+10.3f} | "
          f"{100*a_share:11.1f}% {top:>14}")

# ---------------------------------------------------------------------------
best = max(scan, key=lambda r: r["heldout_r2_transition"])
low = [r for r in scan if r["cancellation"] < 5.0]
print(f"\nbest held-out on the transition: lamb={best['lamb']:g} "
      f"(R2 {best['heldout_r2_transition']:+.3f} vs zero {best['zero_baseline']:+.3f})")
if low:
    wc = min(low, key=lambda r: r["cancellation"])
    print(f"first well-conditioned fit (cancellation < 5): lamb={wc['lamb']:g}, "
          f"{wc['n_active']} active edges, alpha share {100*wc['alpha_share']:.1f}%, "
          f"held-out {wc['heldout_r2_transition']:+.3f}")
else:
    wc = None
    print("no lambda on the grid produced a well-conditioned gamma "
          "(cancellation < 5)")

alpha_survives = bool(wc is not None and wc["alpha_share"] > 0.25
                      and wc["n_active"] > 0)
beats_zero = bool(best["heldout_r2_transition"] > best["zero_baseline"])

print("\n[VERDICT]")
if wc is None:
    print("  gamma never becomes well-conditioned on this grid, so its edge")
    print("  magnitudes -- including the alpha pathways -- cannot be separated")
    print("  from overfitting.  Treat alpha->gamma as UNRESOLVED, not as a finding.")
elif alpha_survives:
    print(f"  alpha keeps {100*wc['alpha_share']:.0f}% of the model's edge mass at "
          f"lamb={wc['lamb']:g} where\n  cancellation is only {wc['cancellation']:.1f}x. "
          "The alpha->gamma pathways SURVIVE sparsification.")
else:
    print(f"  at lamb={wc['lamb']:g} the fit is well-conditioned "
          f"({wc['cancellation']:.1f}x) but alpha holds only\n  "
          f"{100*wc['alpha_share']:.0f}% of the edge mass: the alpha->gamma "
          "pathways were largely the overfitting.")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
lam = [max(r["lamb"], 1e-4) for r in scan]
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8})
axes[0].semilogx(lam, [r["cancellation"] for r in scan], "o-", color="#D55E00")
axes[0].axhline(5, color="0.6", ls=":")
axes[0].set_xlabel("lambda")
axes[0].set_ylabel("cancellation factor")
axes[0].set_title("Conditioning", fontweight="bold", loc="left")
axes[1].semilogx(lam, [r["heldout_r2_transition"] for r in scan], "o-",
                 color="#0072B2", label="cross-band")
axes[1].semilogx(lam, [r["zero_baseline"] for r in scan], "--", color="0.6",
                 label="persistence")
axes[1].set_xlabel("lambda")
axes[1].set_ylabel("held-out $R^2$ on the transition")
axes[1].legend(frameon=False)
axes[1].set_title("Generalization", fontweight="bold", loc="left")
axes[2].semilogx(lam, [100 * r["alpha_share"] for r in scan], "o-", color="#009E73")
axes[2].set_xlabel("lambda")
axes[2].set_ylabel("alpha share of edge mass (%)")
axes[2].set_title(r"Do the $\alpha\rightarrow\gamma$ edges survive?",
                  fontweight="bold", loc="left")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(RESULT_DIR / f"gamma_lambda_scan.{ext}", dpi=250, bbox_inches="tight")
plt.close(fig)

ok = print_check(beats_zero, "some lambda lets gamma beat persistence on the "
                             "transition it never saw")
print_check(wc is not None, "gamma can be made well-conditioned (cancellation < 5)")
print_check(alpha_survives, "alpha->gamma survives sparsification")

save_metrics(RESULT_DIR / "gamma_lambda_scan.json", {
    "task": "lambda scan for T_gamma: are the alpha pathways real or overfitting?",
    "passed": bool(ok),
    "alpha_columns": [COL[i] for i in ALPHA_COLS],
    "scan": scan,
    "best_heldout": best,
    "first_well_conditioned": wc,
    "alpha_survives": alpha_survives,
})
print(f"[DONE] {RESULT_DIR}/gamma_lambda_scan.png")
