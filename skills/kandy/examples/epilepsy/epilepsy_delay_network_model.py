#!/usr/bin/env python3
"""Delay-coupled network of cross-frequency oscillators: the governing equations.

This is the model the rest of the directory was working toward.  Every design
choice below is a lesson that was paid for, so they are stated with the
measurement that forced them.

WHAT WENT INTO IT
-----------------
1. PER CHANNEL, not the channel mean.  The averaged state agreed with
   hippocampus_R at r = 0.66 and motor_R at r = 0.01; splitting it gained
   +0.223 held-out and rescued both motor channels.

2. THE COUPLING IS DELAYED.  This is the piece that was missing.  Every earlier
   lift read the source at time t, and at lag 0 the hippocampal beta -> posterior
   delta switch is worth +0.014.  At 1.5 s it is +0.156 (p = 0.005), and that
   lag agrees with the independently measured cross-correlation lead (0.9 s).
   A switch that acts with a delay is invisible to an instantaneous test, and
   that is why four separate searches found nothing.

3. HEAVISIDE, NOT ReLU.  At the same position: +0.088 vs -0.021.  The ReLU
   gates were inert (removing them cost 0.01; shuffling their thresholds
   changed nothing).  A ramp carries magnitude; a switch carries an event.
   Sharpness is scanned anyway -- k = 1 won over k = 64, so what matters is
   being a BOUNDED SATURATING switch, not literally a step.

4. NO SYMBOLIC ATOM IS ASSUMED.  The splines on real data are not clean atoms:
   get_formula collapsed T_gamma to two terms out of twenty and scored R^2 =
   -3.85 against its own splines.  So the KAN fits freely and the equations are
   read off PER EDGE afterwards as sparse polynomial + Fourier mixtures, which
   reached R^2 0.999 where single atoms reached 0.95.  Fourier modes are in the
   dictionary for every coupling edge, as they should be.

5. SMOOTHNESS REGULARIZATION.  KANDy's lamb defaults to l1 + entropy, which on
   this data forces a choice between well-conditioned-and-empty and populated-
   and-ill-conditioned.  lamb_coefdiff keeps the edges and fixes the
   conditioning (cancellation 24x -> 4x, held-out x4).

6. TAU IS RE-SCANNED per channel and band.  Inheriting tau = 20 s from the
   channel-mean fit is what made every previous rollout too slow to have an
   onset.

7. THE SAMPLE BUDGET IS THE BINDING CONSTRAINT.  delta carries ~102 independent
   samples, theta 43, and alpha/beta/gamma 6-8.  So the lift is deliberately
   small and links must survive an FDR-corrected surrogate screen before they
   are allowed into the model at all.

THE MODEL
---------
    dx[c,b]/dt = ( T[c,b](own bands, delayed couplings) - x[c,b] ) / tau[c,b]

with couplings of the form  x[c',b'](t - d)  and  H_k(x[c',b'](t - d) - theta),
which makes the integration a DELAY differential equation, initialised from the
recording's own history.

WHAT IT IS FOR
--------------
Perturbation.  A model is only worth perturbing if it reproduces the event
unperturbed, so that is gated first and the perturbation section does not run
otherwise.  Then: node resection, per-band modulation, and periodic (DBS-like)
drive, each scored on whether it yields a VIABLE seizure-free trajectory rather
than a dead one.

Run:
    cd skills/kandy/examples && python epilepsy/epilepsy_delay_network_model.py
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
SUB = {"lamb_l1": 0.2, "lamb_entropy": 0.0, "lamb_coef": 0.0, "lamb_coefdiff": 1.0}
_orig = ktrain.fit_kan


def _fk(*a, **kw):
    kw.update(SUB)
    return _orig(*a, **kw)


ktrain.fit_kan = _fk
kcore.fit_kan = _fk


def ei(n, d):
    return int(os.environ.get(n, d))


def ef(n, d):
    return float(os.environ.get(n, d))


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
RESULT_DIR = Path(ensure_outdir(REPO_ROOT / "results" / "Epilepsy-Delay-Network"))

BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 50.0)}
BAND = list(BANDS)
NB = len(BAND)
SMOOTH_TAU, DS, EDGE_S, LAG_S, GUARD_S = 0.15, 10, 2.0, 2.0, 2.0
BASIN = [(-13.0, -9.0), (22.0, 28.0)]
TRANSITION = (-2.0, 9.0)
DELAYS_S = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
THETA_PCT, SHARP = 50, 1.0
MAX_LINKS = ei("KANDY_DN_MAX_LINKS", 4)         # per target, sample budget
FDR_Q = 0.10                                     # screening, not a final claim
N_SURR = ei("KANDY_DN_SURR", 200)
TAU_GRID = np.logspace(np.log10(0.3), np.log10(20.0), 16)
LAMB_GRID = [0.01, 0.03, 0.1]
CV_FOLDS, GRID, STEPS, RIDGE = 3, 5, 80, 1e-2
BOX_MARGIN = 0.35
ROLL_START, ROLL_SEC = ef("KANDY_DN_START", -8.0), ef("KANDY_DN_SEC", 25.0)

payload = pickle.load((REPO_ROOT / "data" / "Dory_2021_04_12_006_preprocessed.pkl").open("rb"))
XR = np.asarray(payload["X"], float)
FS = float(payload["fs"])
T_ALL = np.asarray(payload["t"], float)
LAB = list(payload["labels"])
NC = len(LAB)
HIPP = LAB.index("hippocampus_R")


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
STATE = X.reshape(N, NC * NB)
NS = NC * NB
LAGN = max(1, int(round(LAG_S / DT)))
DX = (X[LAGN:] - X[:-LAGN]) / LAG_S
rows = np.arange(N - LAGN)
t_rows = T[:N - LAGN]
nbm = np.zeros(len(rows), bool)
for w in BASIN:
    nbm |= (t_rows >= w[0] - GUARD_S) & (t_rows <= w[1] + GUARD_S)
ntm = (t_rows >= TRANSITION[0] - GUARD_S) & (t_rows <= TRANSITION[1] + GUARD_S)
# The transition is split in two.  Link selection scores on the FIRST half and
# every reported number is computed on the SECOND.  Without this the links are
# chosen by their performance on the same block used to evaluate them, which is
# what the first version of this script did: the models never saw the
# transition, but the LINK SET did, and that is enough to invalidate the
# headline number.
trans_all = rows[(t_rows >= TRANSITION[0]) & (t_rows <= TRANSITION[1])]
_half = len(trans_all) // 2
trans_sel = trans_all[:_half]          # choose links here
trans = trans_all[_half:]              # report everything here
trainA, trainB = rows[~nbm], rows[~(nbm | ntm)]
gap = int(round(GUARD_S / DT))
cv = [(trainA[(trainA < np.asarray(f)[0] - gap) | (trainA > np.asarray(f)[-1] + gap)],
       np.asarray(f)) for f in np.array_split(trainA, CV_FOLDS)]

print("=" * 92)
print("DELAY-COUPLED NETWORK OF CROSS-FREQUENCY OSCILLATORS")
print("=" * 92)
print(f"{NC} channels x {NB} bands = {NS} states, {N} rows @ {1/DT:.0f} Hz; "
      f"focus = {LAB[HIPP]}")
print(f"transition split: {len(trans_sel)} rows to SELECT links, "
      f"{len(trans)} rows to REPORT on -- disjoint")

TH = {b: np.percentile(X[:, :, b], THETA_PCT) for b in range(NB)}


def H(v, b):
    return 1.0 / (1.0 + np.exp(-SHARP * (v - TH[b])))


def r2(y, q):
    d = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - q) ** 2) / d) if d > 0 else np.nan


# ---------------------------------------------------------------------------
# 1. LINK DISCOVERY -- delayed, surrogate-tested, FDR screened
# ---------------------------------------------------------------------------
print(f"\n[LINKS] screening {NC*NB} sources x {len(DELAYS_S)} delays for each of "
      f"{NS} targets, held out on the transition")


def own_design(c):
    return np.column_stack([X[:, c, :], np.ones(N)])


def link_gain(c, b, sc, sb, d):
    lag = int(round(d / DT))
    u = np.roll(X[:, sc, sb], lag)
    col = np.column_stack([u, H(u, sb)])
    own = own_design(c)
    des = np.column_stack([own, col])

    def s(A):
        G = A[trainB].T @ A[trainB] + RIDGE * np.eye(A.shape[1])
        w = np.linalg.solve(G, A[trainB].T @ DX[trainB, c, b])
        return r2(DX[trans_sel, c, b], A[trans_sel] @ w)

    return s(des) - s(own)


rng = np.random.default_rng(SEED)
cands = []
for c in range(NC):
    for b in range(NB):
        for sc in range(NC):
            for sb in range(NB):
                if sc == c and sb == b:
                    continue
                for d in DELAYS_S:
                    if sc == c and d == 0.0:
                        continue          # own band at lag 0 is already in `own`
                    cands.append((c, b, sc, sb, d, link_gain(c, b, sc, sb, d)))
cands.sort(key=lambda r: -r[5])
print(f"        {len(cands)} candidates; surrogate-testing the top "
      f"{min(200, len(cands))}")
tested = []
for (c, b, sc, sb, d, g) in cands[:200]:
    lag = int(round(d / DT))
    null = np.empty(N_SURR)
    for s_ in range(N_SURR):
        sh = np.roll(X[:, sc, sb], int(rng.integers(int(2 / DT), N - int(2 / DT))))
        u = np.roll(sh, lag)
        col = np.column_stack([u, H(u, sb)])
        own = own_design(c)
        des = np.column_stack([own, col])

        def s(A):
            G = A[trainB].T @ A[trainB] + RIDGE * np.eye(A.shape[1])
            w = np.linalg.solve(G, A[trainB].T @ DX[trainB, c, b])
            return r2(DX[trans_sel, c, b], A[trans_sel] @ w)

        null[s_] = s(des) - s(own)
    tested.append({"c": c, "b": b, "sc": sc, "sb": sb, "d": d, "gain": g,
                   "p": float((np.sum(null >= g) + 1) / (N_SURR + 1))})
# The top 200 were chosen by gain on the selection half, so the effective
# multiple-comparison burden is the FULL candidate count, not 200.  FDR is
# applied at a level scaled accordingly rather than pretending only 200 were
# looked at.
FDR_EFF = FDR_Q * len(tested) / len(cands)
print(f"        FDR is applied at {FDR_EFF:.4f}, not {FDR_Q:.2f}: the top "
      f"{len(tested)} were pre-selected by gain from {len(cands)} candidates")
ps = np.array([t["p"] for t in tested])
order = np.argsort(ps)
thr = (np.arange(1, len(ps) + 1) / len(ps)) * FDR_EFF
keep_n = int(np.max(np.where(ps[order] <= thr)[0]) + 1) if (ps[order] <= thr).any() else 0
survivors = [tested[i] for i in order[:keep_n]]
print(f"        {keep_n} of {len(tested)} survive BH FDR at the "
      f"search-corrected level {FDR_EFF:.4f}")

LINKS = {(c, b): [] for c in range(NC) for b in range(NB)}
for s_ in sorted(survivors, key=lambda r: -r["gain"]):
    k = (s_["c"], s_["b"])
    if len(LINKS[k]) < MAX_LINKS:
        LINKS[k].append(s_)
n_used = sum(len(v) for v in LINKS.values())
print(f"        {n_used} links kept (<= {MAX_LINKS} per target)\n")
print(f"        {'target':>22} {'<- source':>22} {'delay':>7} {'gain':>8} {'p':>7}")
for s_ in sorted(survivors, key=lambda r: -r["gain"])[:14]:
    tgt = f"{LAB[s_['c']]}.{BAND[s_['b']]}"
    src = f"{LAB[s_['sc']]}.{BAND[s_['sb']]}"
    print(f"        {tgt:>22} {src:>22} {s_['d']:6.1f}s {s_['gain']:+8.3f} "
          f"{s_['p']:7.3f}")
n_from_hipp = sum(1 for s_ in survivors if s_["sc"] == HIPP)
print(f"\n        links originating in the focus ({LAB[HIPP]}): "
      f"{n_from_hipp}/{len(survivors)}")
d_used = [s_["d"] for s_ in survivors]
print(f"        delays used: " + ", ".join(
    f"{d}s x{d_used.count(d)}" for d in DELAYS_S if d_used.count(d)))

# ---------------------------------------------------------------------------
# 2. TAU per (channel, band)
# ---------------------------------------------------------------------------
print("\n[TAU]   re-scanned per channel and band (blocked CV on the RATE)")
TAU = np.zeros((NC, NB))
for c in range(NC):
    for b in range(NB):
        A = np.column_stack([X[:, c, :], np.ones(N)])
        best = (None, -np.inf)
        for tau in TAU_GRID:
            tgt = X[:len(DX), c, b] + tau * DX[:, c, b]
            sc_ = []
            for tr, te in cv:
                G = A[tr].T @ A[tr] + RIDGE * np.eye(A.shape[1])
                w = np.linalg.solve(G, A[tr].T @ tgt[tr])
                rate = (A[te] @ w - X[te, c, b]) / tau
                sc_.append(r2(DX[te, c, b], rate))
            m_ = float(np.mean(sc_))
            if m_ > best[1]:
                best = (tau, m_)
        TAU[c, b] = best[0]
print("        " + "  ".join(f"{b[:3]}" for b in BAND))
for c, lab in enumerate(LAB):
    print(f"        {lab:>14} " + " ".join(f"{TAU[c,b]:5.1f}s" for b in range(NB)))

# ---------------------------------------------------------------------------
# 3. Lift and fit
# ---------------------------------------------------------------------------
LO = STATE.min(0) - BOX_MARGIN * np.ptp(STATE, 0)
HIb = STATE.max(0) + BOX_MARGIN * np.ptp(STATE, 0)
BC, BR = 0.5 * (LO + HIb), 0.5 * (HIb - LO)
TGT = np.zeros((len(DX), NS))
for c in range(NC):
    for b in range(NB):
        j = c * NB + b
        t_ = STATE[:len(DX), j] + TAU[c, b] * DX[:, c, b]
        TGT[:, j] = BR[j] * np.arctanh(np.clip((t_ - BC[j]) / BR[j], -0.999, 0.999))
TSC = TGT[trainA].std(0) + 1e-12

# Augmented design: 30 states + every delayed coupling column any target needs.
COUPLE = []
for k, lst in LINKS.items():
    for s_ in lst:
        key = (s_["sc"], s_["sb"], s_["d"])
        if key not in COUPLE:
            COUPLE.append(key)
CMAP = {k: i for i, k in enumerate(COUPLE)}
CCOL = np.zeros((N, 2 * len(COUPLE)))
for k, i in CMAP.items():
    sc, sb, d = k
    u = np.roll(X[:, sc, sb], int(round(d / DT)))
    CCOL[:, 2 * i] = u
    CCOL[:, 2 * i + 1] = H(u, sb)
AUG = np.column_stack([STATE, CCOL])
print(f"\n[LIFT]  {len(COUPLE)} distinct delayed coupling variables "
      f"(raw + Heaviside = {2*len(COUPLE)} columns available)")


def lift_for(c, gains=None):
    """gains: dict (sc,sb,d) -> multiplier, for lesions.  1.0 = intact."""
    idx = []
    for b in range(NB):
        for s_ in LINKS[(c, b)]:
            i = CMAP[(s_["sc"], s_["sb"], s_["d"])]
            if i not in idx:
                idx.append(i)
    ncol = 1 + NB + 2 * len(idx)
    names = ["1"] + BAND[:] + sum(
        [[f"{LAB[COUPLE[i][0]]}.{BAND[COUPLE[i][1]]}(t-{COUPLE[i][2]:.1f})",
          f"H[{LAB[COUPLE[i][0]]}.{BAND[COUPLE[i][1]]}(t-{COUPLE[i][2]:.1f})]"]
         for i in idx], [])

    def npf(s):
        s = np.atleast_2d(s)
        f = [np.ones(len(s))] + [s[:, c * NB + b] for b in range(NB)]
        for i in idx:
            g = 1.0 if gains is None else gains.get(COUPLE[i], 1.0)
            base = TH[COUPLE[i][1]]
            u = base + g * (s[:, NS + 2 * i] - base)
            f += [u, 1.0 / (1.0 + np.exp(-SHARP * (u - base)))]
        return np.column_stack(f)

    def tf(s):
        f = [torch.ones_like(s[:, 0])] + [s[:, c * NB + b] for b in range(NB)]
        for i in idx:
            g = 1.0 if gains is None else gains.get(COUPLE[i], 1.0)
            base = float(TH[COUPLE[i][1]])
            u = base + g * (s[:, NS + 2 * i] - base)
            f += [u, torch.sigmoid(SHARP * (u - base))]
        return torch.stack(f, 1)

    return npf, tf, ncol, names, idx


def fit_c(c, rws, lamb):
    npf, tf, d_, _n, _i = lift_for(c)
    m = KANDy(lift=CustomLift(npf, d_, tf, f"d{c}"), grid=GRID, k=3, steps=STEPS,
              seed=SEED, base_fun=torch.nn.Identity(), device="cpu")
    with contextlib.redirect_stdout(StringIO()):
        m.fit(X=AUG[rws], X_dot=TGT[rws, c * NB:(c + 1) * NB] / TSC[c * NB:(c + 1) * NB],
              val_frac=0.15, test_frac=0.15, lamb=lamb, opt="LBFGS", lr=1.0,
              batch=-1, stop_grid_update_step=10, patience=0, verbose=False)
    return m


print("\n[TRAIN] six channel models")
models, modelsB, lambs = [], [], []
for c, lab in enumerate(LAB):
    best = (None, -np.inf)
    for lamb in LAMB_GRID:
        sc_ = []
        for tr, te in cv:
            m = fit_c(c, tr, lamb)
            sc_.append(float(np.nanmean([
                r2(TGT[te, c * NB + b] / TSC[c * NB + b],
                   m.predict(AUG[te])[:, b]) for b in range(NB)])))
        v = float(np.mean(sc_))
        if v > best[1]:
            best = (lamb, v)
    models.append(fit_c(c, trainA, best[0]))
    modelsB.append(fit_c(c, trainB, best[0]))
    lambs.append(best[0])
    nl = sum(len(LINKS[(c, b)]) for b in range(NB))
    print(f"        {lab:>14}: lamb={best[0]:g}, CV {best[1]:+.3f}, {nl} links")

# ---------------------------------------------------------------------------
# 4. Held-out on the transition
# ---------------------------------------------------------------------------
def rate_of(m, c, idx):
    inner = m.predict(AUG[idx]) * TSC[c * NB:(c + 1) * NB]
    sl_ = slice(c * NB, (c + 1) * NB)
    tgt = BC[sl_] + BR[sl_] * np.tanh(inner / BR[sl_])
    return (tgt - STATE[idx][:, sl_]) / TAU[c]


print("\n[EVAL]  held-out R2 of dx/dt on the transition (set B never saw it)")
print(f"{'channel':>14} | " + " ".join(f"{b[:3]:>7}" for b in BAND) + " |   mean  persist")
gains_v = []
heldout = {}
for c, lab in enumerate(LAB):
    pr = rate_of(modelsB[c], c, trans)
    rr = [r2(DX[trans, c, b], pr[:, b]) for b in range(NB)]
    zz = [r2(DX[trans, c, b], np.zeros(len(trans))) for b in range(NB)]
    heldout[lab] = dict(zip(BAND, rr))
    gains_v.append(np.nanmean(rr) - np.nanmean(zz))
    print(f"{lab:>14} | " + " ".join(f"{v:+7.3f}" for v in rr)
          + f" | {np.nanmean(rr):+6.3f} {np.nanmean(zz):+7.3f}")
mean_gain = float(np.mean(gains_v))
print(f"{'mean gain over persistence':>44}  {mean_gain:+.3f}")

# ---------------------------------------------------------------------------
# 5. DELAY-DIFFERENTIAL ROLLOUT, initialised from real history
# ---------------------------------------------------------------------------
MAXD = int(round(max(DELAYS_S) / DT)) + 2
i0 = int(np.argmin(np.abs(T - ROLL_START)))
nst = int(round(ROLL_SEC / DT))
LIFT_CACHE: dict = {}


def lifts(gains_key, gains):
    if gains_key not in LIFT_CACHE:
        LIFT_CACHE[gains_key] = [lift_for(c, gains)[0] for c in range(NC)]
    return LIFT_CACHE[gains_key]


def rollout(gains=None, gkey="base", resect=None, drive=None, band_mod=None):
    """DDE: history buffer holds the last MAXD states; delays index into it."""
    hist = [STATE[i0 - MAXD + k].copy() for k in range(MAXD)]
    s = STATE[i0].copy()
    traj = np.empty((nst + 1, NS))
    traj[0] = s
    fns = lifts(gkey, gains)
    for it in range(nst):
        def deriv(st, buf):
            aug = np.zeros(NS + 2 * len(COUPLE))
            aug[:NS] = st
            for k_, i_ in CMAP.items():
                sc, sb, d = k_
                back = buf[max(0, len(buf) - 1 - int(round(d / DT)))]
                u = back[sc * NB + sb]
                if resect is not None and sc in resect:
                    u = TH[sb]                    # a resected node emits nothing
                aug[NS + 2 * i_] = u
                aug[NS + 2 * i_ + 1] = 1.0 / (1.0 + np.exp(-SHARP * (u - TH[sb])))
            out = np.zeros(NS)
            for c in range(NC):
                if resect is not None and c in resect:
                    continue                      # resected node has no dynamics
                with torch.no_grad():
                    lv = torch.tensor(fns[c](aug[None]), dtype=torch.float32)
                    inner = models[c].model_(lv).numpy()[0] * TSC[c*NB:(c+1)*NB]
                sl_ = slice(c * NB, (c + 1) * NB)
                tg = BC[sl_] + BR[sl_] * np.tanh(inner / BR[sl_])
                if band_mod is not None:
                    for b, f_ in band_mod.items():
                        tg[b] = st[sl_][b] + f_ * (tg[b] - st[sl_][b])
                out[sl_] = (tg - st[sl_]) / TAU[c]
                if drive is not None and c in drive[0]:
                    out[sl_] += drive[1] * np.sin(2 * np.pi * drive[2] * it * DT)
            return out

        k1 = deriv(s, hist)
        k2 = deriv(s + 0.5 * DT * k1, hist)
        k3 = deriv(s + 0.5 * DT * k2, hist)
        k4 = deriv(s + DT * k3, hist)
        s = s + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(s).all():
            return traj[:it + 1]
        hist.append(s.copy())
        hist.pop(0)
        traj[it + 1] = s
    return traj


print(f"\n[ROLLOUT] delay-differential RK4 from t={T[i0]:.1f}s for {ROLL_SEC:.0f}s, "
      f"history {MAXD} steps")
traj = rollout()
nr = len(traj)
t_r = T[i0:i0 + nr]
ref = STATE[i0:i0 + nr]
PRE = (T >= -13) & (T <= -2)
POST = (T >= 18) & (T <= 28)
pc, poc = STATE[PRE].mean(0), STATE[POST].mean(0)
ax_ = poc - pc
an = float(np.linalg.norm(ax_))
u_ = ax_ / an


def tscore(a):
    return (np.atleast_2d(a) - pc) @ u_ / an


s_data, s_mod = tscore(ref).ravel(), tscore(traj).ravel()
per_ch = {}
print(f"{'channel':>14} | " + " ".join(f"{b[:3]:>7}" for b in BAND) + " |   mean r")
for c, lab in enumerate(LAB):
    rr = [float(np.corrcoef(traj[:, c*NB+b], ref[:, c*NB+b])[0, 1]) for b in range(NB)]
    per_ch[lab] = {"per_band_r": rr, "mean_r": float(np.mean(rr))}
    print(f"{lab:>14} | " + " ".join(f"{v:+7.3f}" for v in rr)
          + f" | {np.mean(rr):+7.3f}")
mean_r = float(np.mean([v["mean_r"] for v in per_ch.values()]))
rise_d = float(s_data.max() - s_data[:int(1/DT)].mean())
rise_m = float(s_mod.max() - s_mod[:int(1/DT)].mean())


def half(s_):
    b_ = s_[:int(1/DT)].mean()
    h = b_ + 0.5 * (s_.max() - b_)
    return float(t_r[int(np.argmax(s_ >= h))]) if np.any(s_ >= h) else float(t_r[-1])


print(f"\n          mean per-band r across all six channels: {mean_r:+.3f}")
print(f"          transition rise  data {rise_d:+.2f}  model {rise_m:+.2f}")
print(f"          half-rise        data {half(s_data):+.1f}s  model {half(s_mod):+.1f}s "
      f"(lag {half(s_mod)-half(s_data):+.1f}s)")

REPRODUCES = bool(mean_r > 0.5 and rise_m > 0.5 * rise_d
                  and abs(half(s_mod) - half(s_data)) < 4.0)
print(f"\n[GATE]    does the unperturbed model reproduce the seizure?  "
      f"{'YES' if REPRODUCES else 'NO'}")
if not REPRODUCES:
    print("          Perturbation analysis is SKIPPED.  Asking which intervention")
    print("          abolishes an event in a model that does not produce the event")
    print("          is vacuous -- every knob would look like a cure.")

# ---------------------------------------------------------------------------
# 6. PERTURBATIONS -- only if the model earned them
# ---------------------------------------------------------------------------
pert = {}
if REPRODUCES:
    lo_ok = STATE.min(0) - 0.15 * np.ptp(STATE, 0)
    hi_ok = STATE.max(0) + 0.15 * np.ptp(STATE, 0)

    def summarise(tr):
        if len(tr) < nst + 1:
            return {"complete": False, "rise": np.nan, "viable": False}
        s_ = tscore(tr).ravel()
        inside = np.all((tr >= lo_ok) & (tr <= hi_ok), axis=1).mean()
        r_ = float(s_.max() - s_[:int(1/DT)].mean())
        return {"complete": True, "rise": r_, "viable": bool(inside > 0.9),
                "in_range": float(inside),
                "seizure_free": bool(r_ < 0.5 * rise_d and inside > 0.9)}

    print("\n[PERTURB] each intervention scored on: does it abolish the event AND")
    print("          leave a viable trajectory (inside the observed range)?\n")
    print(f"          {'intervention':>34} {'rise':>7} {'in range':>9} {'seizure-free':>13}")
    print(f"          {'unperturbed':>34} {rise_m:+7.2f} {1.0:9.2f} {'--':>13}")

    for c in range(NC):
        tr = rollout(gkey=f"res{c}", resect={c})
        r_ = summarise(tr)
        pert[f"resect {LAB[c]}"] = r_
        print(f"          {'resect ' + LAB[c]:>34} {r_['rise']:+7.2f} "
              f"{r_.get('in_range', np.nan):9.2f} {str(r_['seizure_free']):>13}")

    for b, nm in enumerate(BAND):
        for f_ in (0.0, 0.5):
            tr = rollout(gkey=f"bm{b}{f_}", band_mod={b: f_})
            r_ = summarise(tr)
            pert[f"modulate {nm} x{f_}"] = r_
            print(f"          {f'modulate {nm} drive x{f_}':>34} {r_['rise']:+7.2f} "
                  f"{r_.get('in_range', np.nan):9.2f} {str(r_['seizure_free']):>13}")

    for amp in (0.05, 0.15):
        for fr in (0.5, 2.0):
            tr = rollout(gkey=f"dbs{amp}{fr}", drive=({HIPP}, amp, fr))
            r_ = summarise(tr)
            pert[f"DBS hipp {fr}Hz a={amp}"] = r_
            print(f"          {f'DBS {LAB[HIPP]} {fr}Hz amp {amp}':>34} "
                  f"{r_['rise']:+7.2f} {r_.get('in_range', np.nan):9.2f} "
                  f"{str(r_['seizure_free']):>13}")

    free = [k for k, v in pert.items() if v.get("seizure_free")]
    print(f"\n          {len(free)} intervention(s) give a VIABLE seizure-free "
          f"trajectory: {', '.join(free) if free else 'none'}")

# ---------------------------------------------------------------------------
# 7. Figure
# ---------------------------------------------------------------------------
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(NC + 1, 1, figsize=(11, 1.6 * (NC + 1)), sharex=True)
for c, lab in enumerate(LAB):
    a = axes[c]
    a.axvspan(1, 8, color="#F0E442", alpha=0.2, lw=0)
    for b in range(NB):
        a.plot(t_r, ref[:, c*NB+b], lw=0.6, alpha=0.4, color="0.45")
        a.plot(t_r, traj[:, c*NB+b], lw=1.3, label=BAND[b] if c == 0 else None)
    a.axvline(0, color="#D55E00", ls=":", lw=0.8)
    a.set_ylabel(f"{lab}\nr={per_ch[lab]['mean_r']:.2f}", fontsize=7)
axes[0].legend(ncol=5, frameon=False, fontsize=6)
axes[-1].plot(t_r, s_data, color="0.2", lw=1.5, label="recording")
axes[-1].plot(t_r, s_mod, color="#D55E00", lw=2.0, label="model")
axes[-1].axvline(0, color="#D55E00", ls=":", lw=0.8)
axes[-1].set_ylabel("transition\nscore")
axes[-1].set_xlabel("recording time (s), onset at 0")
axes[-1].legend(frameon=False, ncol=2)
axes[0].set_title(f"Delay-coupled network: {n_used} lagged links, "
                  f"mean r={mean_r:.2f}", loc="left", fontweight="bold")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(RESULT_DIR / f"delay_network_rollout.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)

gates = {
    "links_survive_fdr": {"achieved": keep_n, "required": ">=1", "passed": keep_n >= 1},
    "beats_persistence": {"achieved": mean_gain, "required": 0.05,
                          "passed": bool(mean_gain > 0.05)},
    "rollout_tracks_all_six": {"achieved": mean_r, "required": 0.5,
                               "passed": bool(mean_r > 0.5)},
    "reproduces_the_event": {"achieved": [rise_m, rise_d], "required": ">=50% of rise",
                             "passed": bool(rise_m > 0.5 * rise_d)},
    "onset_timing": {"achieved": half(s_mod) - half(s_data), "required": "<4s",
                     "passed": bool(abs(half(s_mod) - half(s_data)) < 4.0)},
    "perturbation_analysis_is_meaningful": {
        "achieved": REPRODUCES, "required": "model reproduces the event first",
        "passed": REPRODUCES},
}
print("\n" + "=" * 92)
ok = True
for k, v in gates.items():
    ok &= print_check(v["passed"], f"{k}: achieved={v['achieved']}, required={v['required']}")

save_metrics(RESULT_DIR / "metrics.json", {
    "task": "delay-coupled network of cross-frequency oscillators",
    "passed": bool(ok), "channels": LAB, "bands": {k: list(v) for k, v in BANDS.items()},
    "links_screened": len(cands), "links_surrogate_tested": len(tested),
    "fdr_level_used": FDR_EFF,
    "selection_block": [float(t_rows[trans_sel[0]]), float(t_rows[trans_sel[-1]])],
    "evaluation_block": [float(t_rows[trans[0]]), float(t_rows[trans[-1]])],
    "links_surviving_fdr": keep_n, "links_used": n_used,
    "links": [{**s_, "target": f"{LAB[s_['c']]}.{BAND[s_['b']]}",
               "source": f"{LAB[s_['sc']]}.{BAND[s_['sb']]}"} for s_ in survivors],
    "links_from_focus": n_from_hipp,
    "tau": TAU.tolist(), "lamb": dict(zip(LAB, lambs)),
    "heldout_transition": heldout, "mean_gain_over_persistence": mean_gain,
    "rollout": {"mean_r": mean_r, "per_channel": per_ch,
                "rise_data": rise_d, "rise_model": rise_m,
                "half_data": half(s_data), "half_model": half(s_mod)},
    "reproduces_event": REPRODUCES, "perturbations": pert,
    "gates": gates,
})
print(f"[DONE] {RESULT_DIR}")
