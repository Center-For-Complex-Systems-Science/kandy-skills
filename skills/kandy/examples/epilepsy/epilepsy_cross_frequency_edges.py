#!/usr/bin/env python3
"""Plot the fitted spline edges of the cross-frequency envelope model.

Each band's equation is a KAN of shape [21, 1]: one spline edge per lift
column.  ``get_formula`` collapses those 21 curves into a single symbolic line,
and for gamma that collapse scores R^2 = -3.85 against the splines it replaced.
This script draws the curves themselves so the collapse can be inspected rather
than trusted.

Three views:

  1. per band, all 21 edges (kandy.plotting.plot_all_edges)
  2. an ACTIVATION-RANGE heat map -- which lift columns each band actually uses.
     A column whose edge is flat contributes nothing regardless of what the
     symbolic line says about it.
  3. for the columns that carry real range, the spline overlaid with the
     symbolic snap, so a faithful equation (delta, R^2 0.95) and an unfaithful
     one (gamma, R^2 -3.85) can be told apart by eye.

Only the FINAL models are refitted, at the lambda already chosen by blocked CV
in the main script, so this costs five fits rather than the full 250-fit sweep.

Run (after epilepsy_cross_frequency_seizure_freedom.py):
    cd skills/kandy/examples && python epilepsy/epilepsy_cross_frequency_edges.py
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

from _utils import ensure_outdir, save_metrics
from kandy import CustomLift, KANDy
from kandy.plotting import get_edge_activation, plot_all_edges

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
RESULT_DIR = ensure_outdir(REPO_ROOT / "results" / "Epilepsy-Cross-Frequency")
EDGE_DIR = ensure_outdir(Path(RESULT_DIR) / "edges")

BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 50.0)}
BAND_NAMES = list(BANDS)
NB = 5
PAIRS = [(a, b) for a in range(NB) for b in range(a + 1, NB)]
SMOOTH_TAU, DS, EDGE_S, LAG_S = 0.15, 10, 2.0, 2.0
GRID, STEPS, BOX_MARGIN = 5, 80, 0.35
GUARD_S = 2.0
TEST_W = {"pre_ictal": (-13.0, -9.0), "post_ictal": (22.0, 28.0)}

met = json.loads((Path(RESULT_DIR) / "metrics.json").read_text())
LAMB = {b: v["cross"]["selected"] for b, v in met["regularization"]["per_band"].items()}
SYMB = met.get("equations", {})
print("[SETUP] refitting final models at the CV-selected lambda: "
      + ", ".join(f"{b}={LAMB[b]:g}" for b in BAND_NAMES), flush=True)

# ---------------------------------------------------------------------------
payload = pickle.load((REPO_ROOT / "data" / "Dory_2021_04_12_006_preprocessed.pkl").open("rb"))
X = np.asarray(payload["X"], float)
FS = float(payload["fs"])
T_ALL = np.asarray(payload["t"], float)

LOG_ENV = {}
for n, (lo, hi) in BANDS.items():
    bb, aa = butter(4, [lo / (0.5 * FS), hi / (0.5 * FS)], btype="band")
    LOG_ENV[n] = np.log(np.abs(hilbert(filtfilt(bb, aa, X, axis=0), axis=0)) + 1e-9)


def causal_ewma(y, tau_s):
    al = 1.0 - np.exp(-1.0 / (tau_s * FS))
    out = np.empty_like(y)
    acc = y[0].copy()
    for i in range(len(y)):
        acc = acc + al * (y[i] - acc)
        out[i] = acc
    return out


S = np.column_stack([causal_ewma(LOG_ENV[n], SMOOTH_TAU).mean(axis=1)
                     for n in BAND_NAMES])
S = (S - S.mean(0)) / (S.std(0) + 1e-12)
edge = int(round(EDGE_S * FS))
STATE = S[edge:len(S) - edge:DS]
T = T_ALL[edge:len(S) - edge:DS]
N, DT = len(STATE), float(T[1] - T[0])
LAG = max(1, int(round(LAG_S / DT)))
DX = (STATE[LAG:] - STATE[:-LAG]) / LAG_S

X_LO = STATE.min(0) - BOX_MARGIN * np.ptp(STATE, 0)
X_HI = STATE.max(0) + BOX_MARGIN * np.ptp(STATE, 0)
X_C, X_R = 0.5 * (X_LO + X_HI), 0.5 * (X_HI - X_LO)
PRE = (T >= -13.0) & (T <= -2.0)
THRESH = STATE[PRE].mean(0)
TAU = np.array(met["model_definition"]["tau_s"])
T_TARGET = STATE[:len(DX)] + TAU * DX
T_INNER = X_R * np.arctanh(np.clip((T_TARGET - X_C) / X_R, -0.999, 0.999))

rows = np.arange(N - LAG)
t_rows = T[:N - LAG]
near = np.zeros(len(rows), bool)
for w in TEST_W.values():
    near |= (t_rows >= w[0] - GUARD_S) & (t_rows <= w[1] + GUARD_S)
train_rows = rows[~near]
T_SCALE = T_INNER[train_rows].std(0) + 1e-12

COL_NAMES = ["1"] + BAND_NAMES + [f"ReLU({n})" for n in BAND_NAMES] \
    + [f"{BAND_NAMES[a]}*{BAND_NAMES[b]}" for a, b in PAIRS]
LIFT_DIM = len(COL_NAMES)


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


models = []
for b, name in enumerate(BAND_NAMES):
    m = KANDy(lift=CustomLift(lift_np, LIFT_DIM, lift_t, f"e{b}"), grid=GRID, k=3,
              steps=STEPS, seed=SEED, base_fun=torch.nn.Identity(), device="cpu")
    with contextlib.redirect_stdout(StringIO()):
        m.fit(X=STATE[train_rows], X_dot=T_INNER[train_rows, b:b + 1] / T_SCALE[b],
              val_frac=0.15, test_frac=0.15, lamb=LAMB[name], opt="LBFGS", lr=1.0,
              batch=-1, stop_grid_update_step=10, patience=0, verbose=False)
    models.append(m)
    print(f"[FIT]   {name} done", flush=True)

# model_ is the bare KAN, which sits AFTER the lift: it expects the 21 lift
# columns, not the 5 raw state variables.  KANDy.predict applies the lift first.
Xin = torch.tensor(lift_np(STATE[train_rows]), dtype=torch.float32)
for m in models:
    m.model_.save_act = True
    with torch.no_grad():
        m.model_(Xin)

# ---------------------------------------------------------------------------
# 1. every edge, per band
# ---------------------------------------------------------------------------
# kandy.plotting.plot_all_edges puts every edge in ONE ROW when the model has a
# single output -- 21 panels, ~19000 px wide, and each panel autoscaled, so an
# edge with range 0.0001 looks exactly as structured as one with range 3.0.
# Both problems matter here, because the whole question is which edges are flat.
# So: a grid, and a SHARED y-scale per band.  The autoscaled version is still
# written alongside for anyone who wants to see the shape of a dead edge.
def plot_band_edges(b, name, curves, ranges, shared=True):
    ncol, nrow = 5, 5
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.0 * nrow),
                             sharex=False, sharey=shared)
    hi = max(ranges[:, b].max(), 1e-9)
    active_thr = max(0.05 * hi, 1e-3)
    for i in range(nrow * ncol):
        ax = axes.flat[i]
        if i >= LIFT_DIM:
            ax.set_axis_off()
            continue
        xa, ya = curves[(i, b)]
        o = np.argsort(xa)
        on = ranges[i, b] > active_thr
        ax.plot(xa[o], ya[o], color="#D55E00" if on else "#9E9E9E",
                lw=1.8 if on else 1.0)
        if shared:
            mid = 0.5 * (np.max(ya) + np.min(ya))
            ax.set_ylim(mid - 0.6 * hi, mid + 0.6 * hi)
        ax.set_title(f"{COL_NAMES[i]}   ({ranges[i, b]:.3f})",
                     fontsize=8, fontweight="bold" if on else "normal",
                     color="black" if on else "0.45")
        ax.tick_params(labelsize=6)
    n_on = int((ranges[:, b] > active_thr).sum())
    fig.suptitle(f"$T_{{{name}}}$: all {LIFT_DIM} spline edges  --  "
                 f"{n_on} carry range above 5% of the maximum"
                 + ("   (shared y-scale: flat means flat)" if shared else
                    "   (each panel autoscaled)"),
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout()
    tag = "" if shared else "_autoscaled"
    for ext in ("png", "pdf"):
        fig.savefig(EDGE_DIR / f"edges_{name}{tag}.{ext}", dpi=140,
                    bbox_inches="tight")
    plt.close(fig)


PLOT_AFTER_RANGES = True

# ---------------------------------------------------------------------------
# 2. activation range: which columns each band actually uses
# ---------------------------------------------------------------------------
RANGE = np.zeros((LIFT_DIM, NB))
CURVES = {}
for b in range(NB):
    for i in range(LIFT_DIM):
        xa, ya = get_edge_activation(models[b].model_, l=0, i=i, j=0)
        RANGE[i, b] = float(np.max(ya) - np.min(ya))
        CURVES[(i, b)] = (xa, ya)

mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})

for b, name in enumerate(BAND_NAMES):
    plot_band_edges(b, name, CURVES, RANGE, shared=True)
    plot_band_edges(b, name, CURVES, RANGE, shared=False)
    print(f"[PLOT]  edges_{name} (grid, shared + autoscaled)", flush=True)

fig, ax = plt.subplots(figsize=(6.2, 8.0))
im = ax.imshow(RANGE, aspect="auto", cmap="magma")
ax.set_xticks(range(NB), BAND_NAMES, rotation=30, ha="right")
ax.set_yticks(range(LIFT_DIM), COL_NAMES, fontsize=6.5)
ax.set_xlabel("equation (target band)")
ax.set_title("Spline activation range per lift column\n"
             "(flat edge = the column contributes nothing)", fontsize=9,
             fontweight="bold")
for i in range(LIFT_DIM):
    for b in range(NB):
        if RANGE[i, b] > 0.02 * RANGE.max():
            ax.text(b, i, f"{RANGE[i, b]:.2f}", ha="center", va="center",
                    fontsize=5.5,
                    color="white" if RANGE[i, b] < 0.6 * RANGE.max() else "black")
fig.colorbar(im, ax=ax, shrink=0.6, label="max - min of the edge")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(Path(RESULT_DIR) / f"edge_activation_map.{ext}", dpi=250,
                bbox_inches="tight")
plt.close(fig)

print(f"\n[ACTIVE] columns with range > 5% of the per-band max")
active = {}
for b, name in enumerate(BAND_NAMES):
    # relative threshold alone is meaningless when every edge is ~0 (alpha and
    # beta at lamb=1), so an absolute floor is required as well.
    thr = max(0.05 * RANGE[:, b].max(), 1e-3)
    idx = [i for i in range(LIFT_DIM) if RANGE[i, b] > thr]
    active[name] = [COL_NAMES[i] for i in idx]
    print(f"  {name:>6} ({len(idx):2d}/{LIFT_DIM}): "
          + ", ".join(f"{COL_NAMES[i]}({RANGE[i, b]:.2f})" for i in idx[:8])
          + (" ..." if len(idx) > 8 else ""), flush=True)

# ---------------------------------------------------------------------------
# 3. the edges that matter, drawn large
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(NB, 4, figsize=(12, 2.5 * NB))
for b, name in enumerate(BAND_NAMES):
    order = np.argsort(RANGE[:, b])[::-1][:4]
    for k, i in enumerate(order):
        ax = axes[b, k]
        xa, ya = CURVES[(i, b)]
        o = np.argsort(xa)
        ax.plot(xa[o], ya[o], color="#0072B2", lw=1.6)
        ax.set_title(f"{COL_NAMES[i]} $\\rightarrow$ T$_{{{name}}}$  "
                     f"(range {RANGE[i, b]:.2f})", fontsize=7.5)
        ax.tick_params(labelsize=6)
        if k == 0:
            ax.set_ylabel(name, fontweight="bold")
fig.suptitle("The four highest-range spline edges per equation", fontsize=11,
             fontweight="bold", y=0.995)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(Path(RESULT_DIR) / f"edge_top4.{ext}", dpi=250, bbox_inches="tight")
plt.close(fig)

save_metrics(Path(RESULT_DIR) / "edge_metrics.json", {
    "task": "spline edge activations of the cross-frequency envelope model",
    "lambda_used": LAMB,
    "lift_columns": COL_NAMES,
    "activation_range": {n: RANGE[:, b].tolist() for b, n in enumerate(BAND_NAMES)},
    "active_columns": active,
    "symbolic_fidelity": {k: v.get("r2_vs_spline")
                          for k, v in SYMB.items() if isinstance(v, dict)},
    "note": "range = max-min of the edge over the training inputs. A flat edge "
            "contributes nothing no matter what the symbolic line says.",
})
print(f"\n[DONE] {EDGE_DIR} and {RESULT_DIR}/edge_*.png")
