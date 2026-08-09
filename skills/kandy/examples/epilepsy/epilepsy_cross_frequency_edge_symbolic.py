#!/usr/bin/env python3
"""Per-edge symbolic extraction: fit EVERY spline edge, then sum them.

Why not get_formula
-------------------
A single-layer KAN is exactly additive over its edges,

    T_j(x) = sum_i  phi_ij( lift_i(x) )                       (+ a constant)

so the faithful way to read an equation off it is to fit each phi_ij on its own
and add them up.  ``get_formula`` instead snaps globally with an r2_threshold
and a topk_edges budget, and on real data that goes badly: for T_gamma it kept
two terms out of twenty live edges and scored R^2 = -3.85 against the splines
it replaced.  Worse, the terms it dropped include gamma's three LARGEST edges --
alpha (13.9), alpha*beta (21.8) and alpha*gamma (20.5) -- so the printed
equation omitted every alpha->gamma pathway in the model.

This script fits each edge against a library of interpretable 1-D forms,
records the fit and its R^2, and assembles the sum.  Nothing is pruned for
being inconvenient; edges are dropped only when their activation range is
negligible, and how many were dropped is reported.

The additivity assumption is CHECKED, not assumed: the sum of the edge
activations is compared against the network's own output before any symbolic
work begins.  If that check fails the decomposition is invalid and everything
downstream would be meaningless.

Library (each fitted by least squares with bounds):
    const, linear, quadratic, cubic, hinge (ReLU with a free knee),
    tanh, gaussian, exp, abs
Selection is by held-out R^2 on the edge's own samples with a small complexity
penalty, so a cubic has to earn its two extra parameters against a line.

Run (after epilepsy_cross_frequency_seizure_freedom.py):
    cd skills/kandy/examples && python epilepsy/epilepsy_cross_frequency_edge_symbolic.py
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
from scipy.optimize import curve_fit
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
NB = 5
PAIRS = [(a, b) for a in range(NB) for b in range(a + 1, NB)]
SMOOTH_TAU, DS, EDGE_S, LAG_S = 0.15, 10, 2.0, 2.0
GRID, STEPS, BOX_MARGIN, GUARD_S = 5, 80, 0.35, 2.0
TEST_W = {"pre_ictal": (-13.0, -9.0), "post_ictal": (22.0, 28.0)}
RANGE_FLOOR = float(os.environ.get("KANDY_CF_EDGE_FLOOR", 0.02))

met = json.loads((RESULT_DIR / "metrics.json").read_text())
LAMB = {b: v["cross"]["selected"] for b, v in met["regularization"]["per_band"].items()}

# ---------------------------------------------------------------------------
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
TAU = np.array(met["model_definition"]["tau_s"])
T_INNER = X_R * np.arctanh(np.clip(((STATE[:len(DX)] + TAU * DX) - X_C) / X_R, -0.999, 0.999))
rows = np.arange(N - LAG)
t_rows = T[:N - LAG]
near = np.zeros(len(rows), bool)
for w in TEST_W.values():
    near |= (t_rows >= w[0] - GUARD_S) & (t_rows <= w[1] + GUARD_S)
train_rows = rows[~near]
T_SCALE = T_INNER[train_rows].std(0) + 1e-12

COL = ["1"] + BAND_NAMES + [f"ReLU({n})" for n in BAND_NAMES] \
    + [f"{BAND_NAMES[a]}*{BAND_NAMES[b]}" for a, b in PAIRS]
D = len(COL)


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


print("=" * 78)
print("PER-EDGE SYMBOLIC EXTRACTION")
print("=" * 78)
models = []
for b, name in enumerate(BAND_NAMES):
    m = KANDy(lift=CustomLift(lift_np, D, lift_t, f"s{b}"), grid=GRID, k=3,
              steps=STEPS, seed=SEED, base_fun=torch.nn.Identity(), device="cpu")
    with contextlib.redirect_stdout(StringIO()):
        m.fit(X=STATE[train_rows], X_dot=T_INNER[train_rows, b:b + 1] / T_SCALE[b],
              val_frac=0.15, test_frac=0.15, lamb=LAMB[name], opt="LBFGS", lr=1.0,
              batch=-1, stop_grid_update_step=10, patience=0, verbose=False)
    models.append(m)
    print(f"[FIT]   {name} (lamb={LAMB[name]:g})", flush=True)

LIFTV = lift_np(STATE)
Xin = torch.tensor(lift_np(STATE[train_rows]), dtype=torch.float32)
for m in models:
    m.model_.save_act = True
    with torch.no_grad():
        m.model_(Xin)

# ---------------------------------------------------------------------------
# 0. Is the network actually additive over its edges?  Check before trusting it.
# ---------------------------------------------------------------------------
# get_edge_activation returns each edge sorted by its OWN input, so the arrays
# are not row-aligned and cannot be summed element-wise -- doing that gives a
# near-zero residual MEAN with a wildly negative R2, which is exactly what the
# first version of this check reported.  Each curve is therefore interpolated
# back onto the actual row values before summing.
print("\n[ADDITIVITY] sum of edge activations vs the network's own output")
EDGES, EDGE_ROW = {}, {}
add_ok = True
for b, name in enumerate(BAND_NAMES):
    tot = np.zeros(len(train_rows))
    for i in range(D):
        xa, ya = get_edge_activation(models[b].model_, l=0, i=i, j=0)
        xa, ya = np.asarray(xa, float), np.asarray(ya, float)
        EDGES[(i, b)] = (xa, ya)
        o = np.argsort(xa)
        row = np.interp(LIFTV[train_rows, i], xa[o], ya[o])
        EDGE_ROW[(i, b)] = row
        tot += row
    net = models[b].predict(STATE[train_rows]).ravel()
    st = np.sum((net - net.mean()) ** 2)
    r2 = 1 - np.sum((net - tot) ** 2) / st if st > 0 else np.nan
    add_ok &= bool(r2 > 0.99)
    # Cancellation: how big are the terms compared with the answer?  A sum of
    # huge, mutually-cancelling edges is ill-conditioned -- a 1% error on each
    # term becomes a large error on the total -- and is what overfitting at
    # lamb=0 looks like structurally.
    tot_amp = float(np.sum([np.ptp(EDGE_ROW[(i, b)]) for i in range(D)]))
    out_amp = float(np.ptp(net))
    print(f"  {name:>6}: R2(sum of edges, network) = {r2:+.6f} | "
          f"sum of edge ranges {tot_amp:8.2f} vs output range {out_amp:6.2f} "
          f"-> cancellation factor {tot_amp / max(out_amp, 1e-9):6.1f}x")
print_check(add_ok, "single-layer KAN is additive over its edges, so per-edge "
                    "extraction is exact up to each edge's own fit error")


# ---------------------------------------------------------------------------
# 1. Library of 1-D forms
# ---------------------------------------------------------------------------
def _lin(x, a, c):
    return a * x + c


def _quad(x, a, b_, c):
    return a * x ** 2 + b_ * x + c


def _cub(x, a, b_, c, d_):
    return a * x ** 3 + b_ * x ** 2 + c * x + d_


def _hinge(x, a, k, c):
    return a * np.maximum(x - k, 0.0) + c


def _tanh(x, a, b_, k, c):
    return a * np.tanh(b_ * (x - k)) + c


def _gauss(x, a, k, w, c):
    return a * np.exp(-(((x - k) / np.maximum(np.abs(w), 1e-6)) ** 2)) + c


def _expo(x, a, b_, c):
    return a * np.exp(np.clip(b_ * x, -30, 30)) + c


def _absv(x, a, k, c):
    return a * np.abs(x - k) + c


def _sine(x, a, w, k, c):
    return a * np.sin(w * x + k) + c


def _fourier2(x, a1, b1, a2, b2, w, c):
    return (a1 * np.sin(w * x) + b1 * np.cos(w * x)
            + a2 * np.sin(2 * w * x) + b2 * np.cos(2 * w * x) + c)


LIB = [
    ("const", lambda x, c: np.full_like(x, c), 1, lambda x, y: [y.mean()]),
    ("linear", _lin, 2, lambda x, y: [np.polyfit(x, y, 1)[0], y.mean()]),
    ("quadratic", _quad, 3, lambda x, y: list(np.polyfit(x, y, 2))),
    ("cubic", _cub, 4, lambda x, y: list(np.polyfit(x, y, 3))),
    ("hinge", _hinge, 3, lambda x, y: [np.polyfit(x, y, 1)[0], np.median(x), y.mean()]),
    ("tanh", _tanh, 4, lambda x, y: [0.5 * np.ptp(y), 1.0, np.median(x), y.mean()]),
    ("gaussian", _gauss, 4, lambda x, y: [np.ptp(y), x[np.argmax(np.abs(y - y.mean()))],
                                          0.5 * np.std(x) + 1e-3, y.mean()]),
    ("exp", _expo, 3, lambda x, y: [np.sign(np.ptp(y)) * 0.1, 0.3, y.mean()]),
    ("abs", _absv, 3, lambda x, y: [np.polyfit(x, y, 1)[0], np.median(x), y.mean()]),
    # Trig is included so "are there Fourier modes?" is answered by the fit
    # rather than by omission -- but see period_fraction below.  A sine with a
    # free phase can imitate ANY single bend using a fragment of one period,
    # which is how the earlier get_formula run produced a spurious sine in
    # T_gamma that a quadratic reproduced at R^2 = 0.999.  So every trig fit is
    # reported with the fraction of a full period the data actually spans, and
    # anything under half a period is labelled a bent segment, not a rhythm.
    ("sine", _sine, 4, lambda x, y: [0.5 * np.ptp(y), 2 * np.pi / max(np.ptp(x), 1e-6),
                                     0.0, y.mean()]),
    ("fourier2", _fourier2, 6, lambda x, y: [0.5 * np.ptp(y), 0.0, 0.0, 0.0,
                                             2 * np.pi / max(np.ptp(x), 1e-6),
                                             y.mean()]),
]
COMPLEXITY_PENALTY = float(os.environ.get("KANDY_CF_COMPLEXITY", 0.004))

# ---------------------------------------------------------------------------
# MIXTURE fitting.  A single named form is the wrong model for these edges:
# they are not "a cubic" or "a sine", they are a polynomial trend PLUS
# oscillation.  Winner-take-all forces a choice and then reports the loser's
# structure as absent -- alpha*gamma scores 0.988 as a cubic and 0.997 as a
# 2-term Fourier series, which really means it is both.
#
# So each edge is fitted as a sparse linear combination over a joint
# dictionary of polynomials and harmonics, selected by forward stepwise
# regression with a held-out stop.  The fundamental omega is 2*pi/span(x), so
# harmonic k completes exactly k cycles across the observed range and a "half
# cycle" cannot masquerade as a rhythm: any selected harmonic is a genuine
# oscillation over the data, by construction of the basis.
#
POLY_MAX = int(os.environ.get("KANDY_CF_POLY_MAX", 5))
HARM_MAX = int(os.environ.get("KANDY_CF_HARM_MAX", 4))


def mixture_basis(x, lo, hi):
    """Dictionary columns and their names, for x over [lo, hi]."""
    span = max(hi - lo, 1e-9)
    u = (x - lo) / span                       # 0..1 across the observed range
    w = 2.0 * np.pi
    cols, names, kind = [np.ones_like(x)], ["1"], ["poly"]
    for pdeg in range(1, POLY_MAX + 1):
        cols.append(u ** pdeg)
        names.append(f"u^{pdeg}" if pdeg > 1 else "u")
        kind.append("poly")
    for k in range(1, HARM_MAX + 1):
        cols.append(np.sin(k * w * u))
        names.append(f"sin({k}*2pi*u)")
        kind.append("trig")
        cols.append(np.cos(k * w * u))
        names.append(f"cos({k}*2pi*u)")
        kind.append("trig")
    return np.column_stack(cols), names, kind


def fit_mixture(xa, ya, max_terms=8):
    """Forward stepwise selection over the joint poly+Fourier dictionary."""
    o = np.argsort(xa)
    x, y = xa[o], ya[o]
    lo, hi = float(x.min()), float(x.max())
    A, names, kind = mixture_basis(x, lo, hi)
    te = np.zeros(len(x), bool)
    te[::4] = True
    tr = ~te
    chosen, best_hist = [], []
    for _ in range(max_terms):
        best_j, best_r2 = None, -np.inf
        for j in range(A.shape[1]):
            if j in chosen:
                continue
            cand = chosen + [j]
            coef, *_ = np.linalg.lstsq(A[np.ix_(tr, cand)], y[tr], rcond=None)
            pred = A[np.ix_(te, cand)] @ coef
            den = np.sum((y[te] - y[te].mean()) ** 2)
            r2 = 1 - np.sum((y[te] - pred) ** 2) / den if den > 0 else 1.0
            if r2 > best_r2:
                best_j, best_r2 = j, r2
        if best_j is None:
            break
        if best_hist and best_r2 < best_hist[-1] + 1e-4:
            break                                        # held-out stop
        chosen.append(best_j)
        best_hist.append(best_r2)
    if not chosen:
        return None
    coef, *_ = np.linalg.lstsq(A[np.ix_(tr, chosen)], y[tr], rcond=None)
    full = A[:, chosen] @ coef
    den = np.sum((y - y.mean()) ** 2)
    # how much of the fitted variation is polynomial vs oscillatory?
    contrib = {"poly": 0.0, "trig": 0.0}
    for c, j in zip(coef, chosen):
        if names[j] == "1":
            continue
        contrib[kind[j]] += float(np.ptp(c * A[:, j]))
    tot = contrib["poly"] + contrib["trig"]
    terms = [(names[j], float(c), kind[j]) for c, j in zip(coef, chosen)]
    expr = " + ".join(f"{c:.4g}*{n}" if n != "1" else f"{c:.4g}"
                      for n, c, _k in terms)
    return {"name": "mixture", "terms": terms, "expr": expr,
            "r2_heldout": float(best_hist[-1]),
            "r2_full": float(1 - np.sum((y - full) ** 2) / den) if den > 0 else 1.0,
            "n_terms": len(chosen), "lo": lo, "hi": hi,
            "poly_share": float(contrib["poly"] / tot) if tot > 0 else 0.0,
            "trig_share": float(contrib["trig"] / tot) if tot > 0 else 0.0,
            "n_harmonics": sum(1 for _n, _c, k in terms if k == "trig")}


def eval_mixture(fit, xv):
    A, names, _k = mixture_basis(np.asarray(xv, float), fit["lo"], fit["hi"])
    idx = {n: i for i, n in enumerate(names)}
    out = np.zeros(len(np.atleast_1d(xv)))
    for n, c, _kk in fit["terms"]:
        out += c * A[:, idx[n]]
    return out


def fit_edge(xa, ya):
    """Fit every library form; select by held-out R2 minus a complexity cost."""
    o = np.argsort(xa)
    x, y = xa[o], ya[o]
    tr = np.ones(len(x), bool)
    tr[::4] = False                                # every 4th point held out
    best = None
    for nm, fn, npar, p0f in LIB:
        try:
            p0 = p0f(x[tr], y[tr])
            with np.errstate(all="ignore"):
                popt, _ = curve_fit(fn, x[tr], y[tr], p0=p0, maxfev=20000)
                yh_te = fn(x[~tr], *popt)
                yh_all = fn(x, *popt)
            if not np.isfinite(yh_all).all():
                continue
            den = np.sum((y[~tr] - y[~tr].mean()) ** 2)
            r2 = 1 - np.sum((y[~tr] - yh_te) ** 2) / den if den > 0 else 1.0
            score = r2 - COMPLEXITY_PENALTY * npar
            if best is None or score > best["score"]:
                den_a = np.sum((y - y.mean()) ** 2)
                best = {"name": nm, "params": [float(v) for v in popt],
                        "r2_heldout": float(r2), "score": float(score),
                        "r2_full": float(1 - np.sum((y - yh_all) ** 2) / den_a)
                        if den_a > 0 else 1.0,
                        "fn": fn}
        except Exception:                                        # noqa: BLE001
            continue
    return best


def render(nm, p, col):
    f = lambda v: f"{v:.4g}"                                     # noqa: E731
    if nm == "const":
        return f(p[0])
    if nm == "linear":
        return f"{f(p[0])}*{col} + {f(p[1])}"
    if nm == "quadratic":
        return f"{f(p[0])}*{col}^2 + {f(p[1])}*{col} + {f(p[2])}"
    if nm == "cubic":
        return f"{f(p[0])}*{col}^3 + {f(p[1])}*{col}^2 + {f(p[2])}*{col} + {f(p[3])}"
    if nm == "hinge":
        return f"{f(p[0])}*ReLU({col} - {f(p[1])}) + {f(p[2])}"
    if nm == "tanh":
        return f"{f(p[0])}*tanh({f(p[1])}*({col} - {f(p[2])})) + {f(p[3])}"
    if nm == "gaussian":
        return f"{f(p[0])}*exp(-(({col} - {f(p[1])})/{f(p[2])})^2) + {f(p[3])}"
    if nm == "exp":
        return f"{f(p[0])}*exp({f(p[1])}*{col}) + {f(p[2])}"
    if nm == "abs":
        return f"{f(p[0])}*|{col} - {f(p[1])}| + {f(p[2])}"
    if nm == "sine":
        return f"{f(p[0])}*sin({f(p[1])}*{col} + {f(p[2])}) + {f(p[3])}"
    if nm == "fourier2":
        return (f"{f(p[0])}*sin(w*{col}) + {f(p[1])}*cos(w*{col}) + "
                f"{f(p[2])}*sin(2w*{col}) + {f(p[3])}*cos(2w*{col}) + {f(p[5])}"
                f"   [w={f(p[4])}]")
    return nm


def period_fraction(nm, params, xa):
    """How much of a full period does the data span?  < 0.5 means the 'sine' is
    a fragment of one arc -- a bend, not an oscillation."""
    if nm == "sine":
        w = abs(params[1])
    elif nm == "fourier2":
        w = abs(params[4])
    else:
        return None
    return float(abs(np.ptp(xa)) * w / (2 * np.pi)) if w > 0 else None


# ---------------------------------------------------------------------------
# 2. Fit every edge, assemble, and score the assembly against the splines
# ---------------------------------------------------------------------------
print(f"\n[EXTRACT] fitting all {D} edges for each of {NB} equations "
      f"(range floor {RANGE_FLOOR})")
results, assembled = {}, {}
for b, name in enumerate(BAND_NAMES):
    terms, dropped = [], 0
    approx = np.zeros(len(train_rows))
    for i in range(D):
        xa, ya = EDGES[(i, b)]
        rng = float(np.ptp(ya))
        if rng < RANGE_FLOOR:
            dropped += 1
            approx += ya.mean()
            continue
        single = fit_edge(xa, ya)
        mix = fit_mixture(xa, ya)
        use_mix = mix is not None and (single is None
                                       or mix["r2_heldout"] > single["r2_heldout"] + 1e-3)
        if use_mix:
            fit = dict(mix)
            fit["params"] = []
            approx += eval_mixture(mix, LIFTV[train_rows, i])
        elif single is not None:
            fit = single
            approx += fit["fn"](LIFTV[train_rows, i], *fit["params"])
        else:
            dropped += 1
            continue
        entry = {"column": COL[i], "col_index": i, "range": rng,
                 "form": fit["name"], "params": fit.get("params", []),
                 "r2_heldout": fit["r2_heldout"], "r2_full": fit["r2_full"],
                 "single_form_name": (single["name"] if single else None),
                 "single_form_r2": (single["r2_heldout"] if single else None)}
        if fit["name"] == "mixture":
            entry.update({"expr": fit["expr"], "n_terms": fit["n_terms"],
                          "poly_share": fit["poly_share"],
                          "trig_share": fit["trig_share"],
                          "n_harmonics": fit["n_harmonics"],
                          "u_range": [fit["lo"], fit["hi"]],
                          "period_fraction": None})
        else:
            entry["expr"] = render(fit["name"], fit["params"], COL[i])
            entry["period_fraction"] = period_fraction(fit["name"],
                                                       fit["params"], xa)
        entry["is_real_oscillation"] = bool(
            entry["form"] == "mixture" and entry.get("n_harmonics", 0) > 0)
        terms.append(entry)
    net = models[b].predict(STATE[train_rows]).ravel()
    den = np.sum((net - net.mean()) ** 2)
    tot_r2 = float(1 - np.sum((net - approx) ** 2) / den) if den > 0 else np.nan
    cancel = float(np.sum([np.ptp(EDGE_ROW[(i, b)]) for i in range(D)])
                   / max(np.ptp(net), 1e-9))
    terms.sort(key=lambda t: -t["range"])
    results[name] = {"n_terms": len(terms), "n_dropped": dropped,
                     "assembled_r2_vs_spline": tot_r2,
                     "cancellation_factor": cancel, "terms": terms}
    assembled[name] = approx
    prev = met["equations"].get(name, {})
    prev_r2 = prev.get("r2_vs_spline") if isinstance(prev, dict) else None
    print(f"\n  T_{name}:  {len(terms)} terms kept, {dropped} flat edges dropped")
    print(f"    assembled vs spline R2 = {tot_r2:+.4f}"
          + (f"   (get_formula was {prev_r2:+.4f})" if prev_r2 is not None else "")
          + f"   | cancellation {cancel:.1f}x")
    if cancel > 10:
        print(f"    -> the edges sum to {cancel:.0f}x the output range, so they "
              "largely cancel.  A 1% error per\n       term is a large error on "
              "the total: this equation is ill-conditioned, not just long.")
    for t in terms[:10]:
        if t["form"] == "mixture":
            print(f"      {t['column']:>14}  range {t['range']:7.3f}  "
                  f"mixture({t['n_terms']}) R2 {t['r2_heldout']:+.3f}  "
                  f"[{100*t['poly_share']:.0f}% poly / {100*t['trig_share']:.0f}% "
                  f"Fourier, {t['n_harmonics']} harmonics; best single form was "
                  f"{t['single_form_name']} at {t['single_form_r2']:+.3f}]")
            print(f"      {'':>14}  u = ({t['column']} - {t['u_range'][0]:.3g})"
                  f"/{t['u_range'][1]-t['u_range'][0]:.3g}")
            print(f"      {'':>14}  {t['expr']}")
        else:
            print(f"      {t['column']:>14}  range {t['range']:7.3f}  "
                  f"{t['form']:<9} R2 {t['r2_heldout']:+.3f}   {t['expr']}")
    if len(terms) > 10:
        print(f"      ... and {len(terms) - 10} more")

# ---------------------------------------------------------------------------
# 3. Which connections appear that get_formula missed?
# ---------------------------------------------------------------------------
print("\n[FOURIER] how much of each edge is polynomial and how much is "
      "oscillatory?")
print(f"{'edge':>26} {'terms':>6} {'poly':>6} {'Fourier':>8} {'harm':>5} "
      f"{'mixture R2':>11} {'best single':>13}")
n_trig = n_real = 0
for name, r in results.items():
    for t in r["terms"]:
        if t["form"] != "mixture":
            continue
        n_trig += int(t["n_harmonics"] > 0)
        n_real += int(t["n_harmonics"] > 0)
        print(f"{t['column'] + ' -> ' + name:>26} {t['n_terms']:6d} "
              f"{100*t['poly_share']:5.0f}% {100*t['trig_share']:7.0f}% "
              f"{t['n_harmonics']:5d} {t['r2_heldout']:+11.3f} "
              f"{str(t['single_form_name']):>8}"
              f"{t['single_form_r2']:+.3f}" if t["single_form_r2"] is not None
              else "")
print(f"  {n_trig} edge(s) carry at least one harmonic.  The fundamental is "
      "2*pi/span(x),\n  so harmonic k completes k full cycles across the "
      "observed range -- a fragment\n  of one arc cannot be selected as a "
      "rhythm here, unlike a free-phase sine.")

print("\n[NEW] terms recovered here that get_formula's equation never mentioned")
prev_txt = {k: (v.get("display", "") if isinstance(v, dict) else "")
            for k, v in met["equations"].items()}
new_terms = {}
for name, r in results.items():
    miss = [t for t in r["terms"] if t["column"] not in prev_txt.get(name, "")]
    new_terms[name] = [t["column"] for t in miss]
    if miss:
        print(f"  T_{name}: " + ", ".join(
            f"{t['column']}(range {t['range']:.1f})" for t in miss[:8])
            + (f" ... +{len(miss)-8}" if len(miss) > 8 else ""))

# alpha's involvement specifically
print("\n[ALPHA] every surviving edge whose column involves alpha")
for name, r in results.items():
    al = [t for t in r["terms"] if "alpha" in t["column"]]
    if al:
        for t in al:
            print(f"  {t['column']:>14} -> T_{name:<6} range {t['range']:7.3f}  "
                  f"{t['form']:<9} : {t['expr']}")

# ---------------------------------------------------------------------------
# 4. Figure
# ---------------------------------------------------------------------------
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
for b, name in enumerate(BAND_NAMES):
    terms = results[name]["terms"][:10]
    if not terms:
        continue
    nc = min(5, len(terms))
    nr = int(np.ceil(len(terms) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.3 * nr), squeeze=False)
    for k, t in enumerate(terms):
        ax = axes.flat[k]
        xa, ya = EDGES[(t["col_index"], b)]
        o = np.argsort(xa)
        fn = dict((n_, f_) for n_, f_, _p, _q in LIB)[t["form"]]
        ax.plot(xa[o], ya[o], color="0.35", lw=2.0, label="spline")
        ax.plot(xa[o], fn(xa[o], *t["params"]), color="#D55E00", lw=1.4, ls="--",
                label=t["form"])
        ax.set_title(f"{t['column']}  (R2 {t['r2_heldout']:.3f})", fontsize=8)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.legend(fontsize=6, frameon=False)
    for k in range(len(terms), nr * nc):
        axes.flat[k].set_axis_off()
    fig.suptitle(f"$T_{{{name}}}$: per-edge symbolic fits  "
                 f"(assembled R$^2$ = {results[name]['assembled_r2_vs_spline']:.3f})",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RESULT_DIR / "edges" / f"symbolic_{name}.{ext}", dpi=150,
                    bbox_inches="tight")
    plt.close(fig)

mean_r2 = float(np.mean([r["assembled_r2_vs_spline"] for r in results.values()]))
prev_vals = [v.get("r2_vs_spline") for v in met["equations"].values()
             if isinstance(v, dict) and v.get("r2_vs_spline") is not None]
print("\n" + "=" * 78)
ok = print_check(add_ok, "additivity verified")
ok &= print_check(mean_r2 > 0.9, f"assembled per-edge model is faithful "
                                 f"(mean R2 {mean_r2:+.4f})")
if prev_vals:
    print_check(mean_r2 > float(np.mean(prev_vals)),
                f"per-edge extraction beats get_formula "
                f"({mean_r2:+.4f} vs {float(np.mean(prev_vals)):+.4f})")

save_metrics(RESULT_DIR / "edge_symbolic_metrics.json", {
    "task": "per-edge symbolic extraction of the cross-frequency model",
    "passed": bool(ok),
    "additivity_verified": add_ok,
    "range_floor": RANGE_FLOOR,
    "complexity_penalty": COMPLEXITY_PENALTY,
    "library": [n for n, _f, _p, _q in LIB],
    "mean_assembled_r2": mean_r2,
    "get_formula_r2": {k: v.get("r2_vs_spline") for k, v in met["equations"].items()
                       if isinstance(v, dict)},
    "per_band": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
    "terms_missed_by_get_formula": new_terms,
    "fourier": {"n_trig_fits": n_trig, "n_real_oscillations": n_real,
                "note": "a trig fit spanning < 0.5 of a period is a bend with a "
                        "free phase, not a rhythm"},
})
print(f"[DONE] {RESULT_DIR}/edge_symbolic_metrics.json and edges/symbolic_*.png")
