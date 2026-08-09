#!/usr/bin/env python3
"""Does penalising spline ROUGHNESS rescue T_gamma, where shrinking it did not?

The lambda scan showed gamma has no usable setting under KANDy's default
penalty: as it becomes well-conditioned its edges vanish (20 -> 1) and it never
beats persistence on the transition.  But that default penalises the wrong
thing for this failure mode.  PyKAN's reg is

    reg = lamb_l1 * sum|edge activation|
        + lamb_entropy * (entropy_row + entropy_col)
        + lamb_coef * sum|spline coefficients|
        + lamb_coefdiff * sum|difference of adjacent coefficients|

and the whole thing is scaled by ``lamb``.  Defaults are l1 = 1, entropy = 2,
coef = 0, coefdiff = 0 -- so the size and the SPREAD of edges are penalised and
their WIGGLINESS is not penalised at all.  Gamma's problem is twenty large,
strongly oscillatory edges that cancel 24-fold, which is a roughness problem.

So this compares three penalty shapes at matched strength:

    default     l1=1, entropy=2, coefdiff=0     what has been used so far
    smooth      l1=0, entropy=0, coefdiff=1     roughness only
    mixed       l1=0.2, entropy=0, coefdiff=1   roughness plus light shrinkage

KANDy.fit does not expose the sub-penalties, so fit_kan is wrapped to inject
them.  Note lamb gates everything: with lamb = 0 no sub-penalty has any effect.

The honest prior, before running: gamma has ~6 INDEPENDENT samples (539 rows at
a 4.9 s autocorrelation time) against 168 parameters.  No penalty shape fixes a
28:1 parameter-to-data ratio; the most that can be hoped for is that roughness
control buys a usable equation where shrinkage bought an empty one.

Run:
    cd skills/kandy/examples && python epilepsy/epilepsy_gamma_coefdiff.py
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

import kandy.core as kcore
import kandy.training as ktrain
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

# --- inject the sub-penalties that KANDy.fit does not expose ----------------
SUB = {"lamb_l1": 1.0, "lamb_entropy": 2.0, "lamb_coef": 0.0, "lamb_coefdiff": 0.0}
_orig_fit_kan = ktrain.fit_kan


def _fit_kan(*a, **kw):
    kw.update(SUB)
    return _orig_fit_kan(*a, **kw)


ktrain.fit_kan = _fit_kan
kcore.fit_kan = _fit_kan

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
RESULT_DIR = Path(ensure_outdir(REPO_ROOT / "results" / "Epilepsy-Cross-Frequency"))

BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 50.0)}
BAND_NAMES = list(BANDS)
NB, GB = 5, 4
PAIRS = [(a, b) for a in range(NB) for b in range(a + 1, NB)]
SMOOTH_TAU, DS, EDGE_S, LAG_S = 0.15, 10, 2.0, 2.0
GRID, STEPS, BOX_MARGIN, GUARD_S = 5, 80, 0.35, 2.0
BASIN_TEST = {"pre": (-13.0, -9.0), "post": (22.0, 28.0)}
TRANSITION = (-2.0, 9.0)
LAMB_GRID = [float(v) for v in os.environ.get(
    "KANDY_GC_LAMB", "0.001,0.003,0.01,0.03,0.1,0.3").split(",")]
SCHEMES = {
    "default  (l1=1, ent=2, cdiff=0)": {"lamb_l1": 1.0, "lamb_entropy": 2.0,
                                        "lamb_coef": 0.0, "lamb_coefdiff": 0.0},
    "smooth   (l1=0, ent=0, cdiff=1)": {"lamb_l1": 0.0, "lamb_entropy": 0.0,
                                        "lamb_coef": 0.0, "lamb_coefdiff": 1.0},
    "mixed    (l1=.2, ent=0, cdiff=1)": {"lamb_l1": 0.2, "lamb_entropy": 0.0,
                                         "lamb_coef": 0.0, "lamb_coefdiff": 1.0},
}

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
nb = np.zeros(len(rows), bool)
for w in BASIN_TEST.values():
    nb |= (t_rows >= w[0] - GUARD_S) & (t_rows <= w[1] + GUARD_S)
nt = (t_rows >= TRANSITION[0] - GUARD_S) & (t_rows <= TRANSITION[1] + GUARD_S)
trans_rows = rows[(t_rows >= TRANSITION[0]) & (t_rows <= TRANSITION[1])]
trainA, trainB = rows[~nb], rows[~(nb | nt)]
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


def rate_of(m, idx):
    inner = m.predict(STATE[idx]).ravel() * T_SCALE[GB]
    return (X_C[GB] + X_R[GB] * np.tanh(inner / X_R[GB]) - STATE[idx, GB]) / TAU[GB]


def fit_gamma(lamb, rws):
    m = KANDy(lift=CustomLift(lift_np, D, lift_t, "g"), grid=GRID, k=3, steps=STEPS,
              seed=SEED, base_fun=torch.nn.Identity(), device="cpu")
    with contextlib.redirect_stdout(StringIO()):
        m.fit(X=STATE[rws], X_dot=T_INNER[rws, GB:GB + 1] / T_SCALE[GB],
              val_frac=0.15, test_frac=0.15, lamb=lamb, opt="LBFGS", lr=1.0,
              batch=-1, stop_grid_update_step=10, patience=0, verbose=False)
    return m


def edge_ranges(m, rws):
    Xin = torch.tensor(lift_np(STATE[rws]), dtype=torch.float32)
    m.model_.save_act = True
    with torch.no_grad():
        m.model_(Xin)
    rng = np.zeros(D)
    for i in range(D):
        xa, ya = get_edge_activation(m.model_, l=0, i=i, j=0)
        xa, ya = np.asarray(xa, float), np.asarray(ya, float)
        o = np.argsort(xa)
        rng[i] = float(np.ptp(np.interp(LIFTV[rws, i], xa[o], ya[o])))
    return rng


zero_r2 = r2(DX[trans_rows, GB], np.zeros(len(trans_rows)))
print("=" * 96)
print("T_gamma: does a ROUGHNESS penalty do better than shrinkage?")
print("=" * 96)
print(f"persistence baseline on the transition: R2 = {zero_r2:+.3f}\n")
print(f"{'penalty shape':>34} {'lambda':>7} {'cancel':>7} {'active':>7} "
      f"{'heldout R2':>11} {'alpha share':>12}")
print("-" * 96)

results = []
for sname, sub in SCHEMES.items():
    for lamb in LAMB_GRID:
        SUB.update(sub)
        mB = fit_gamma(lamb, trainB)
        hr2 = r2(DX[trans_rows, GB], rate_of(mB, trans_rows))
        mA = fit_gamma(lamb, trainA)
        rng = edge_ranges(mA, trainA)
        net = mA.predict(STATE[trainA]).ravel()
        cancel = float(rng.sum() / max(np.ptp(net), 1e-9))
        n_act = int((rng > 0.02).sum())
        ash = float(rng[ALPHA_COLS].sum() / max(rng.sum(), 1e-9))
        results.append({"scheme": sname, **sub, "lamb": lamb,
                        "cancellation": cancel, "n_active": n_act,
                        "heldout_r2": hr2, "alpha_share": ash,
                        "top_edge": COL[int(np.argmax(rng))]})
        print(f"{sname:>34} {lamb:7.3g} {cancel:7.1f} {n_act:7d} {hr2:+11.3f} "
              f"{100*ash:11.1f}%")

print("\n[SUMMARY] a usable gamma needs BOTH: cancellation < 5 (well conditioned)")
print("          AND held-out R2 above the persistence baseline.")
usable = [r for r in results
          if r["cancellation"] < 5.0 and r["heldout_r2"] > zero_r2 and r["n_active"] >= 2]
if usable:
    b = max(usable, key=lambda r: r["heldout_r2"])
    print(f"  FOUND: {b['scheme']} at lamb={b['lamb']:g} -> "
          f"cancellation {b['cancellation']:.1f}x, {b['n_active']} edges, "
          f"held-out {b['heldout_r2']:+.3f} (vs {zero_r2:+.3f}), "
          f"alpha share {100*b['alpha_share']:.0f}%")
else:
    print("  NONE.  No penalty shape on this grid gives a gamma equation that is")
    print("  both well-conditioned and better than predicting no change.")
    best_h = max(results, key=lambda r: r["heldout_r2"])
    print(f"  best held-out anywhere: {best_h['scheme']} lamb={best_h['lamb']:g} "
          f"-> {best_h['heldout_r2']:+.3f} at cancellation "
          f"{best_h['cancellation']:.1f}x with {best_h['n_active']} edges")

by_scheme = {}
for s in SCHEMES:
    rs = [r for r in results if r["scheme"] == s]
    by_scheme[s] = {"best_heldout": max(r["heldout_r2"] for r in rs),
                    "min_cancellation": min(r["cancellation"] for r in rs)}
    print(f"    {s:>34}: best held-out {by_scheme[s]['best_heldout']:+.3f}, "
          f"min cancellation {by_scheme[s]['min_cancellation']:.1f}x")

mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8})
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
cols = ["#D55E00", "#0072B2", "#009E73"]
for c, (s, _sub) in zip(cols, SCHEMES.items()):
    rs = [r for r in results if r["scheme"] == s]
    lam = [r["lamb"] for r in rs]
    axes[0].semilogx(lam, [r["cancellation"] for r in rs], "o-", color=c, label=s)
    axes[1].semilogx(lam, [r["heldout_r2"] for r in rs], "o-", color=c, label=s)
    axes[2].semilogx(lam, [r["n_active"] for r in rs], "o-", color=c, label=s)
axes[0].axhline(5, ls=":", color="0.6")
axes[0].set_ylabel("cancellation factor")
axes[1].axhline(zero_r2, ls="--", color="0.5", label="persistence")
axes[1].set_ylabel("held-out $R^2$ on the transition")
axes[2].set_ylabel("active edges")
for ax in axes:
    ax.set_xlabel("lambda")
    ax.spines[["top", "right"]].set_visible(False)
axes[1].legend(frameon=False, fontsize=6)
fig.suptitle("T_gamma under three penalty shapes", fontweight="bold")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(RESULT_DIR / f"gamma_coefdiff.{ext}", dpi=250, bbox_inches="tight")
plt.close(fig)

ok = print_check(bool(usable), "a penalty shape yields a usable gamma equation")
save_metrics(RESULT_DIR / "gamma_coefdiff.json", {
    "task": "T_gamma under l1/entropy vs coefficient-smoothness regularization",
    "passed": bool(ok),
    "persistence_baseline": zero_r2,
    "note": "gamma has ~6 independent samples (539 rows, 4.9 s autocorrelation) "
            "against 168 parameters; no penalty shape fixes 28:1.",
    "schemes": SCHEMES, "grid": LAMB_GRID, "results": results,
    "by_scheme": by_scheme, "usable": usable,
})
print(f"[DONE] {RESULT_DIR}/gamma_coefdiff.png")
