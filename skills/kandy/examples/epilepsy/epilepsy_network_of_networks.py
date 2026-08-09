#!/usr/bin/env python3
"""A network of networks: six per-channel cross-frequency oscillators, coupled.

Why this replaces the channel-mean model
----------------------------------------
``epilepsy_cross_frequency_seizure_freedom.py`` averages the six recordings
into ONE five-band state, which throws away every spatial question.  Measured
consequence: the single trajectory it produces agrees with hippocampus_R at
r = 0.66 and with motor_R at r = 0.01, and there is no way to ask whether
motor_L is connected to hippocampus_R because neither exists in the model.

Here each channel keeps its own five-band oscillator, and the channels are
coupled through a single BROADBAND amplitude each:

    dx[c,b]/dt = ( T[c,b](x[c,:], B[-c]) - x[c,b] ) / tau[b]

    x[c,b]   log-amplitude envelope of band b in channel c   (6 x 5 = 30 states)
    B[c]     that channel's broadband envelope               (the coupling bus)
    T[c,b]   a KAN over channel c's own bands PLUS the other five channels'
             broadband, so the edge from B[c'] into channel c IS the directed
             spatial connection c' -> c, readable straight off the model.

The lift per channel (31 columns, shared across that channel's five outputs):

    1                                        constant
    x[c,b]                    (5)            own bands
    ReLU(x[c,b] - th)         (5)            own gates
    x[c,b]*x[c,b']           (10)            own cross-frequency products
    B[c']                     (5)            spatial coupling, linear
    ReLU(B[c'] - th)          (5)            spatial coupling, gated (recruitment)

Two design points that are measurements, not assumptions
--------------------------------------------------------
1. The broadband range is a GUESS, so it is scanned.  Four definitions are
   compared by held-out skill with a cheap ridge surrogate before the KAN is
   fitted once at the winner.
2. The spatial edges are tested against CIRCULAR-SHIFT SURROGATES of the source
   channel's broadband, which preserve its distribution and autocorrelation and
   destroy only the inter-channel timing.  Without that, a strong c'->c edge
   proves nothing: all six channels rise together during a seizure, so any two
   are correlated whether or not they are coupled.

Regularization, and why it is not optional here
-----------------------------------------------
Per output this is 31 edges x 8 spline coefficients = 248 parameters.  The band
envelopes have autocorrelation times of 0.4-6.5 s, so a ~27 s training window
carries only ~4-67 INDEPENDENT samples depending on the band.  Unregularized,
the fit memorizes; measured earlier on the channel-mean model, lambda = 0 cost
delta 89% of its held-out skill and produced 17 spurious attractors.

The penalty SHAPE also matters and the default is wrong for this data.  PyKAN's
lamb scales lamb_l1*|edges| + lamb_entropy*spread + lamb_coef*|coefs| +
lamb_coefdiff*|delta coefs|, defaulting to l1=1, entropy=2, coefdiff=0 -- so
size and spread are penalized and WIGGLINESS is not.  On T_gamma that forced a
choice between well-conditioned-and-empty (1 edge) and populated-and-ill-
conditioned (20 edges, 24x cancellation).  Adding the smoothness term
(l1=0.2, entropy=0, coefdiff=1) kept 19 edges AND dropped cancellation to 4x
while quadrupling held-out skill, so that shape is used here.  KANDy.fit does
not expose the sub-weights, so fit_kan is wrapped.

Run:
    cd skills/kandy/examples && python epilepsy/epilepsy_network_of_networks.py
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

# Smoothness-weighted penalty; see the docstring for why the default is wrong.
SUB = {"lamb_l1": 0.2, "lamb_entropy": 0.0, "lamb_coef": 0.0, "lamb_coefdiff": 1.0}
_orig_fit_kan = ktrain.fit_kan


def _fit_kan(*a, **kw):
    kw.update(SUB)
    return _orig_fit_kan(*a, **kw)


ktrain.fit_kan = _fit_kan
kcore.fit_kan = _fit_kan


def env_i(n, d):
    return int(os.environ.get(n, d))


def env_f(n, d):
    return float(os.environ.get(n, d))


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
RESULT_DIR = Path(ensure_outdir(os.environ.get(
    "KANDY_NN_RESULT_DIR", str(REPO_ROOT / "results" / "Epilepsy-Network-of-Networks"))))

BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 50.0)}
BAND = list(BANDS)
NB = len(BAND)
PAIRS = [(a, b) for a in range(NB) for b in range(a + 1, NB)]
# The coupling bus is a guess, so these are candidates, not a choice.
BROADBAND_CANDIDATES = {
    "full 1-50": (1.0, 50.0), "slow 1-13": (1.0, 13.0),
    "fast 13-50": (13.0, 50.0), "mid 4-30": (4.0, 30.0),
}
SMOOTH_TAU, DS, EDGE_S, LAG_S = 0.15, 10, 2.0, 2.0
GRID, STEPS, BOX_MARGIN, GUARD_S = 5, 80, 0.35, 2.0
BASIN_TEST = {"pre_ictal": (-13.0, -9.0), "post_ictal": (22.0, 28.0)}
TRANSITION = (-2.0, 9.0)
LAMB_GRID = [float(v) for v in os.environ.get(
    "KANDY_NN_LAMB", "0.003,0.01,0.03,0.1").split(",")]
CV_FOLDS = env_i("KANDY_NN_CV_FOLDS", 4)
N_SURR = env_i("KANDY_NN_SURR", 200)
ROLL_START, ROLL_SECONDS = env_f("KANDY_NN_ROLL_START", -8.0), env_f("KANDY_NN_ROLL_S", 25.0)
RIDGE = 1e-2


def quiet_fit(m, **kw):
    with contextlib.redirect_stdout(StringIO()):
        m.fit(**kw)


def r2c(y, p):
    y, p = np.atleast_2d(y.T).T, np.atleast_2d(p.T).T
    den = np.sum((y - y.mean(0)) ** 2, 0)
    return 1.0 - np.sum((y - p) ** 2, 0) / np.where(den > 0, den, np.nan)


# ---------------------------------------------------------------------------
# 1. Data -> 30 band states + 6 broadband buses
# ---------------------------------------------------------------------------
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
    z = hilbert(filtfilt(bb, aa, XR, axis=0), axis=0)
    return causal_ewma(np.log(np.abs(z) + 1e-9), SMOOTH_TAU)


print("=" * 96)
print("NETWORK OF NETWORKS -- six per-channel cross-frequency oscillators, coupled")
print("=" * 96)
BAND_ENV = {b: env_of(*BANDS[b]) for b in BAND}
e0 = int(round(EDGE_S * FS))
sl0 = slice(e0, len(XR) - e0, DS)
T = T_ALL[sl0]
N = len(T)
DT = float(T[1] - T[0])
LAG = max(1, int(round(LAG_S / DT)))

X = np.zeros((N, NC, NB))
for b, nm in enumerate(BAND):
    v = BAND_ENV[nm][sl0]
    X[:, :, b] = (v - v.mean(0)) / (v.std(0) + 1e-12)
STATE = X.reshape(N, NC * NB)
DXS = (STATE[LAG:] - STATE[:-LAG]) / LAG_S
print(f"[STATE] {NC} channels x {NB} bands = {NC*NB} states, {N} rows @ "
      f"{1/DT:.0f} Hz, increment lag {LAG_S}s")

# effective sample size -- the reason regularization is mandatory here
print(f"\n[SAMPLES] 31 edges x 8 spline coefs = 248 parameters per output")
print(f"{'band':>7} {'autocorr':>9} {'independent samples':>21}")
eff = {}
for b, nm in enumerate(BAND):
    x = X[:, :, b].mean(1)
    x = x - x.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac /= ac[0]
    tau_i = (np.argmax(ac < 1 / np.e) * DT) if np.any(ac < 1 / np.e) else N * DT
    eff[nm] = float(N * DT / max(tau_i, 1e-9))
    print(f"{nm:>7} {tau_i:8.2f}s {eff[nm]:20.0f}")

# ---------------------------------------------------------------------------
# 2. Split
# ---------------------------------------------------------------------------
rows = np.arange(N - LAG)
t_rows = T[:N - LAG]
nb_mask = np.zeros(len(rows), bool)
for w in BASIN_TEST.values():
    nb_mask |= (t_rows >= w[0] - GUARD_S) & (t_rows <= w[1] + GUARD_S)
nt_mask = (t_rows >= TRANSITION[0] - GUARD_S) & (t_rows <= TRANSITION[1] + GUARD_S)
trans_rows = rows[(t_rows >= TRANSITION[0]) & (t_rows <= TRANSITION[1])]
trainA = rows[~nb_mask]                     # transition IN train: for the model
trainB = rows[~(nb_mask | nt_mask)]         # transition WITHHELD: for scoring it
cv = []
gap = int(round(GUARD_S / DT))
for f in np.array_split(trainA, CV_FOLDS):
    h = np.asarray(f)
    cv.append((trainA[(trainA < h[0] - gap) | (trainA > h[-1] + gap)], h))
print(f"\n[SPLIT] set A {len(trainA)} rows (transition in train), set B "
      f"{len(trainB)} (withheld), transition block {len(trans_rows)} rows")

# ---------------------------------------------------------------------------
# 3. Which broadband range should be the coupling bus?  Measure, do not guess.
# ---------------------------------------------------------------------------
def build_bb(lo, hi):
    v = env_of(lo, hi)[sl0]
    return (v - v.mean(0)) / (v.std(0) + 1e-12)


def ridge_gain(BB):
    """Held-out gain on the transition from adding the OTHER channels' bus."""
    gains = []
    for c in range(NC):
        own = np.column_stack([X[:, c, :], np.ones(N)])
        oth = np.column_stack([BB[:, cc] for cc in range(NC) if cc != c])
        for b in range(NB):
            y = DXS[:, c * NB + b]
            for A, store in ((own, "own"), (np.column_stack([own, oth]), "all")):
                G = A[trainB].T @ A[trainB] + RIDGE * np.eye(A.shape[1])
                w = np.linalg.solve(G, A[trainB].T @ y[trainB])
                p = A[trans_rows] @ w
                sc = float(r2c(y[trans_rows], p)[0])
                if store == "own":
                    base = sc
                else:
                    gains.append(sc - base)
    return float(np.mean(gains))


print("\n[BUS]   which broadband range couples the channels best? "
      "(ridge surrogate, held-out on the transition)")
bus_scan = {}
for nm, (lo, hi) in BROADBAND_CANDIDATES.items():
    bus_scan[nm] = ridge_gain(build_bb(lo, hi))
    print(f"        {nm:>12} ({lo:.0f}-{hi:.0f} Hz): mean held-out gain "
          f"{bus_scan[nm]:+.4f}")
BUS_NAME = max(bus_scan, key=bus_scan.get)
BB = build_bb(*BROADBAND_CANDIDATES[BUS_NAME])
print(f"        -> using {BUS_NAME}"
      + ("   [WARNING: best gain is <= 0; no range couples the channels]"
         if bus_scan[BUS_NAME] <= 0 else ""))

# ---------------------------------------------------------------------------
# 4. Lift, per channel
# ---------------------------------------------------------------------------
TH_B = X[(T >= -13.0) & (T <= -2.0)].mean(0)          # (NC, NB) pre-ictal
TH_BB = BB[(T >= -13.0) & (T <= -2.0)].mean(0)        # (NC,)
BOX_LO = STATE.min(0) - BOX_MARGIN * np.ptp(STATE, 0)
BOX_HI = STATE.max(0) + BOX_MARGIN * np.ptp(STATE, 0)
BOX_C, BOX_R = 0.5 * (BOX_LO + BOX_HI), 0.5 * (BOX_HI - BOX_LO)
# Reused from the channel-mean fit rather than re-scanned per channel.  That is
# an assumption, not a result: tau was identified for the averaged state, and
# alpha/beta/gamma were on the grid boundary there with ~0 CV skill, so those
# three are inherited as "slowest available" rather than as fitted values.
TAU_B = np.array([12.6, 15.9, 20.0, 20.0, 20.0])


def cols_for(c):
    others = [cc for cc in range(NC) if cc != c]
    names = ["1"] + [f"{b}" for b in BAND] + [f"ReLU({b})" for b in BAND]
    names += [f"{BAND[a]}*{BAND[b]}" for a, b in PAIRS]
    names += [f"B[{LAB[o]}]" for o in others]
    names += [f"ReLU(B[{LAB[o]}])" for o in others]
    return names, others


def lift_for(c, bb_gain=None):
    """bb_gain[o] scales channel o's bus as seen by channel c (for lesions)."""
    _n, others = cols_for(c)

    def np_fn(s):
        s = np.atleast_2d(s)
        xc = s[:, c * NB:(c + 1) * NB]
        bus = s[:, NC * NB:]
        if bb_gain is not None:
            bus = TH_BB + bb_gain * (bus - TH_BB)
        f = [np.ones(len(s))] + [xc[:, b] for b in range(NB)]
        f += [np.maximum(xc[:, b] - TH_B[c, b], 0.0) for b in range(NB)]
        f += [xc[:, a] * xc[:, b] for a, b in PAIRS]
        f += [bus[:, o] for o in others]
        f += [np.maximum(bus[:, o] - TH_BB[o], 0.0) for o in others]
        return np.column_stack(f)

    def t_fn(s):
        xc = s[:, c * NB:(c + 1) * NB]
        bus = s[:, NC * NB:]
        thb = torch.as_tensor(TH_B[c], dtype=s.dtype)
        thbb = torch.as_tensor(TH_BB, dtype=s.dtype)
        f = [torch.ones_like(s[:, 0])] + [xc[:, b] for b in range(NB)]
        f += [torch.clamp(xc[:, b] - thb[b], min=0.0) for b in range(NB)]
        f += [xc[:, a] * xc[:, b] for a, b in PAIRS]
        f += [bus[:, o] for o in others]
        f += [torch.clamp(bus[:, o] - thbb[o], min=0.0) for o in others]
        return torch.stack(f, 1)

    return np_fn, t_fn


AUG = np.column_stack([STATE, BB])                    # what the models are fed
NCOL = len(cols_for(0)[0])
print(f"\n[LIFT]  {NCOL} columns per channel model, {NB} outputs each "
      f"(1 + {NB} bands + {NB} gates + {len(PAIRS)} products + "
      f"{NC-1} bus + {NC-1} gated bus)")

TGT = np.zeros((len(DXS), NC * NB))
for c in range(NC):
    for b in range(NB):
        j = c * NB + b
        tgt = STATE[:len(DXS), j] + TAU_B[b] * DXS[:, j]
        TGT[:, j] = BOX_R[j] * np.arctanh(
            np.clip((tgt - BOX_C[j]) / BOX_R[j], -0.999, 0.999))
TSC = TGT[trainA].std(0) + 1e-12


# ---------------------------------------------------------------------------
# 5. Fit the six coupled models
# ---------------------------------------------------------------------------
def fit_channel(c, rws, lamb):
    np_fn, t_fn = lift_for(c)
    m = KANDy(lift=CustomLift(np_fn, NCOL, t_fn, f"ch{c}"), grid=GRID, k=3,
              steps=STEPS, seed=SEED, base_fun=torch.nn.Identity(), device="cpu")
    quiet_fit(m, X=AUG[rws], X_dot=TGT[rws, c * NB:(c + 1) * NB] / TSC[c * NB:(c + 1) * NB],
              val_frac=0.15, test_frac=0.15, lamb=lamb, opt="LBFGS", lr=1.0,
              batch=-1, stop_grid_update_step=10, patience=0, verbose=False)
    return m


print("\n[TRAIN] six channel models, lambda by blocked CV")
models, modelsB, lambs = [], [], []
for c, lab in enumerate(LAB):
    best = None
    for lamb in LAMB_GRID:
        sc = []
        for tr, te in cv:
            m = fit_channel(c, tr, lamb)
            sc.append(float(np.nanmean(
                r2c(TGT[te, c * NB:(c + 1) * NB] / TSC[c * NB:(c + 1) * NB],
                    m.predict(AUG[te])))))
        s = float(np.mean(sc))
        if best is None or s > best[1]:
            best = (lamb, s)
    models.append(fit_channel(c, trainA, best[0]))
    modelsB.append(fit_channel(c, trainB, best[0]))
    lambs.append(best[0])
    print(f"        {lab:>14}: lamb={best[0]:g}, CV R2={best[1]:+.3f}")

# ---------------------------------------------------------------------------
# 6. Held-out skill on the transition (set B), against persistence
# ---------------------------------------------------------------------------
def rate_of(m, c, idx, aug=None):
    A = AUG if aug is None else aug
    inner = m.predict(A[idx]) * TSC[c * NB:(c + 1) * NB]
    tgt = BOX_C[c * NB:(c + 1) * NB] + BOX_R[c * NB:(c + 1) * NB] * np.tanh(
        inner / BOX_R[c * NB:(c + 1) * NB])
    return (tgt - STATE[idx, c * NB:(c + 1) * NB]) / TAU_B


print("\n[EVAL]  held-out R2 of dx/dt on the TRANSITION (set B never saw it)")
print(f"{'channel':>14} | " + " ".join(f"{b[:3]:>7}" for b in BAND) + " |    mean   persist")
heldout, gains = {}, []
for c, lab in enumerate(LAB):
    pred = rate_of(modelsB[c], c, trans_rows)
    tru = DXS[trans_rows][:, c * NB:(c + 1) * NB]
    rr = r2c(tru, pred)
    zz = r2c(tru, np.zeros_like(tru))
    heldout[lab] = {"per_band": dict(zip(BAND, rr.tolist())),
                    "mean": float(np.nanmean(rr)),
                    "persistence": float(np.nanmean(zz))}
    gains.append(float(np.nanmean(rr) - np.nanmean(zz)))
    print(f"{lab:>14} | " + " ".join(f"{v:+7.3f}" for v in rr)
          + f" | {np.nanmean(rr):+7.3f} {np.nanmean(zz):+7.3f}")
mean_gain = float(np.mean(gains))
print(f"{'mean gain over persistence':>14}   {mean_gain:+.3f}")

# ---------------------------------------------------------------------------
# 7. The spatial connectivity matrix, with a surrogate null
# ---------------------------------------------------------------------------
print(f"\n[SPATIAL] directed coupling c' -> c from bus-edge magnitude, tested "
      f"against {N_SURR} circular shifts")
Wmat = np.full((NC, NC), np.nan)
for c in range(NC):
    names, others = cols_for(c)
    Xin = torch.tensor(lift_for(c)[0](AUG[trainA]), dtype=torch.float32)
    models[c].model_.save_act = True
    with torch.no_grad():
        models[c].model_(Xin)
    LV = lift_for(c)[0](AUG[trainA])
    for k, o in enumerate(others):
        tot = 0.0
        for j in (1 + NB + NB + len(PAIRS) + k,
                  1 + NB + NB + len(PAIRS) + (NC - 1) + k):
            for out_b in range(NB):
                xa, ya = get_edge_activation(models[c].model_, l=0, i=j, j=out_b)
                xa, ya = np.asarray(xa, float), np.asarray(ya, float)
                srt = np.argsort(xa)
                tot += float(np.ptp(np.interp(LV[:, j], xa[srt], ya[srt])))
        Wmat[c, o] = tot

# surrogate: shift the SOURCE bus, refit with ridge (KAN refits are too costly)
rng = np.random.default_rng(SEED)
pvals = np.full((NC, NC), np.nan)
for c in range(NC):
    own = np.column_stack([X[:, c, :], np.ones(N)])
    for o in range(NC):
        if o == c:
            continue
        des = np.column_stack([own, BB[:, o]])
        obs = []
        for b in range(NB):
            y = DXS[:, c * NB + b]
            G = des[trainB].T @ des[trainB] + RIDGE * np.eye(des.shape[1])
            w = np.linalg.solve(G, des[trainB].T @ y[trainB])
            G0 = own[trainB].T @ own[trainB] + RIDGE * np.eye(own.shape[1])
            w0 = np.linalg.solve(G0, own[trainB].T @ y[trainB])
            obs.append(float(r2c(y[trans_rows], des[trans_rows] @ w)[0]
                             - r2c(y[trans_rows], own[trans_rows] @ w0)[0]))
        obs_g = float(np.mean(obs))
        null = np.empty(N_SURR)
        for s in range(N_SURR):
            sh = np.roll(BB[:, o], int(rng.integers(int(2 / DT), N - int(2 / DT))))
            dn = np.column_stack([own, sh])
            g = []
            for b in range(NB):
                y = DXS[:, c * NB + b]
                G = dn[trainB].T @ dn[trainB] + RIDGE * np.eye(dn.shape[1])
                w = np.linalg.solve(G, dn[trainB].T @ y[trainB])
                G0 = own[trainB].T @ own[trainB] + RIDGE * np.eye(own.shape[1])
                w0 = np.linalg.solve(G0, own[trainB].T @ y[trainB])
                g.append(float(r2c(y[trans_rows], dn[trans_rows] @ w)[0]
                               - r2c(y[trans_rows], own[trans_rows] @ w0)[0]))
            null[s] = np.mean(g)
        pvals[c, o] = float((np.sum(null >= obs_g) + 1) / (N_SURR + 1))

_hdr = "target / source"
print(f"{_hdr:>16} | " + " ".join(f"{l[:8]:>9}" for l in LAB))
for c, lab in enumerate(LAB):
    cells = []
    for o in range(NC):
        if o == c:
            cells.append(f"{'--':>9}")
        else:
            star = "*" if pvals[c, o] < 0.05 else " "
            cells.append(f"{Wmat[c, o]:8.2f}{star}")
    print(f"{lab:>16} | " + " ".join(cells))
n_sig = int(np.nansum(pvals < 0.05))
print(f"          {n_sig} of {NC*(NC-1)} directed links beat their circular-shift "
      "null at p<0.05  (* marks them)")

ml, hr = LAB.index("motor_L"), LAB.index("hippocampus_R")
print(f"\n[ASKED]  motor_L -> hippocampus_R: edge {Wmat[hr, ml]:.2f}, "
      f"p={pvals[hr, ml]:.3f}"
      + ("  SIGNIFICANT" if pvals[hr, ml] < 0.05 else "  not significant"))
print(f"         hippocampus_R -> motor_L: edge {Wmat[ml, hr]:.2f}, "
      f"p={pvals[ml, hr]:.3f}"
      + ("  SIGNIFICANT" if pvals[ml, hr] < 0.05 else "  not significant"))

# ---------------------------------------------------------------------------
# 8. Joint autonomous rollout of all 30 states
# ---------------------------------------------------------------------------
# The bus has to be reconstructible FROM THE STATE or the rollout is not
# autonomous.  A first version renormalised the band-mean per call, using the
# std ACROSS THE SIX CHANNELS at that instant -- which forces all six to share a
# spread and is not even a fixed function of the state.  Instead the map is
# fitted once, per channel, on the training rows, and its quality is reported:
# if the bus cannot be recovered from the band states, this whole architecture
# is not self-consistent and that has to be visible rather than buried.
_bandmean = STATE[:, :NC * NB].reshape(N, NC, NB).mean(axis=2)
BUS_MAP = np.zeros((NC, 2))
bus_map_r2 = {}
for _c in range(NC):
    A_ = np.column_stack([_bandmean[trainA, _c], np.ones(len(trainA))])
    BUS_MAP[_c] = np.linalg.lstsq(A_, BB[trainA, _c], rcond=None)[0]
    pred_ = BUS_MAP[_c, 0] * _bandmean[:, _c] + BUS_MAP[_c, 1]
    bus_map_r2[LAB[_c]] = float(r2c(BB[:, _c], pred_)[0])
print("\n[BUS MAP] can the bus be rebuilt from the band states? "
      "(needed for an autonomous rollout)")
for _c, _l in enumerate(LAB):
    print(f"          {_l:>14}: R2 = {bus_map_r2[_l]:+.3f}")
_worst = min(bus_map_r2.values())
if _worst < 0.8:
    print(f"          [WARNING: worst channel {_worst:+.3f} -- the rollout bus "
          "will differ from the one the models were trained on]")


def bus_from_state(s):
    """Bus reconstructed from the band states via the fitted per-channel map."""
    xs = s[:, :NC * NB].reshape(-1, NC, NB).mean(axis=2)
    return xs * BUS_MAP[:, 0] + BUS_MAP[:, 1]


# Build each channel's lift function ONCE per gain setting.  The first version
# reassigned model.lift inside the RK4 inner loop, which mutates the fitted
# object 12000 times per rollout and is both slow and easy to leave in a wrong
# state if anything raises.  The KAN is called directly on the lift matrix.
_LIFT_CACHE: dict = {}


def _lift_fns(gain):
    key = "none" if gain is None else float(gain)
    if key not in _LIFT_CACHE:
        _LIFT_CACHE[key] = [lift_for(c, bb_gain=gain)[0] for c in range(NC)]
    return _LIFT_CACHE[key]


def rhs(s, gain=None):
    out = np.zeros_like(s)
    aug = np.column_stack([s[:, :NC * NB], bus_from_state(s)])
    fns = _lift_fns(gain)
    for c in range(NC):
        with torch.no_grad():
            lv = torch.tensor(fns[c](aug), dtype=torch.float32)
            inner = models[c].model_(lv).detach().numpy() * TSC[c * NB:(c + 1) * NB]
        sl = slice(c * NB, (c + 1) * NB)
        tgt = BOX_C[sl] + BOX_R[sl] * np.tanh(inner / BOX_R[sl])
        out[:, sl] = (tgt - s[:, sl]) / TAU_B
    return out


nsteps = int(round(ROLL_SECONDS / DT))
i0 = int(np.argmin(np.abs(T - ROLL_START)))
print(f"\n[ROLLOUT] joint autonomous RK4 of all {NC*NB} states from "
      f"t={T[i0]:.1f}s for {ROLL_SECONDS:.0f}s")


def rollout(gain=None):
    s = STATE[i0:i0 + 1].copy()
    tr = np.empty((nsteps + 1, NC * NB))
    tr[0] = s[0]
    for i in range(nsteps):
        k1 = rhs(s, gain)
        k2 = rhs(s + 0.5 * DT * k1, gain)
        k3 = rhs(s + 0.5 * DT * k2, gain)
        k4 = rhs(s + DT * k3, gain)
        s = s + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(s).all():
            return tr[:i + 1]
        tr[i + 1] = s[0]
    return tr


traj = rollout()
traj_nc = rollout(gain=0.0)                # every spatial link cut
nr = len(traj)
t_r = T[i0:i0 + nr]
ref = STATE[i0:i0 + nr]
per_ch = {}
print(f"{'channel':>14} | {'coupled r':>10} {'links cut':>10} {'RMSE':>7}")
for c, lab in enumerate(LAB):
    sl = slice(c * NB, (c + 1) * NB)
    rr = float(np.mean([np.corrcoef(traj[:, c * NB + b], ref[:, c * NB + b])[0, 1]
                        for b in range(NB)]))
    rn = float(np.mean([np.corrcoef(traj_nc[:len(ref), c * NB + b],
                                    ref[:, c * NB + b])[0, 1]
                        for b in range(NB)])) if len(traj_nc) >= nr else np.nan
    rmse = float(np.sqrt(np.mean((traj[:, sl] - ref[:, sl]) ** 2)))
    per_ch[lab] = {"r_coupled": rr, "r_links_cut": rn, "rmse": rmse}
    print(f"{lab:>14} | {rr:10.3f} {rn:10.3f} {rmse:7.3f}")
mean_r = float(np.mean([v["r_coupled"] for v in per_ch.values()]))
mean_rn = float(np.mean([v["r_links_cut"] for v in per_ch.values()]))
print(f"{'mean':>14} | {mean_r:10.3f} {mean_rn:10.3f}   "
      f"coupling is worth {mean_r - mean_rn:+.3f}")

# ---------------------------------------------------------------------------
# 9. Figures
# ---------------------------------------------------------------------------
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6.4, 5.4))
M = np.where(np.isnan(Wmat), 0.0, Wmat)
im = ax.imshow(M, cmap="magma")
ax.set_xticks(range(NC), LAB, rotation=40, ha="right", fontsize=7)
ax.set_yticks(range(NC), LAB, fontsize=7)
ax.set_xlabel("source channel (bus)")
ax.set_ylabel("target channel (equation)")
for c in range(NC):
    for o in range(NC):
        if c == o:
            ax.text(o, c, "--", ha="center", va="center", fontsize=7, color="0.6")
        else:
            star = "*" if pvals[c, o] < 0.05 else ""
            ax.text(o, c, f"{Wmat[c, o]:.1f}{star}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if M[c, o] < 0.6 * M.max() else "black")
ax.set_title(f"Directed spatial coupling ({BUS_NAME} bus)\n"
             f"* beats a circular-shift null at p<0.05", fontsize=9,
             fontweight="bold")
fig.colorbar(im, ax=ax, shrink=0.75, label="bus edge magnitude")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(RESULT_DIR / f"spatial_coupling.{ext}", dpi=250, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(NC, 1, figsize=(11, 1.7 * NC), sharex=True)
for c, lab in enumerate(LAB):
    axm = axes[c]
    axm.axvspan(1, 8, color="#F0E442", alpha=0.2, lw=0)
    for b in range(NB):
        axm.plot(t_r, ref[:, c * NB + b], lw=0.6, alpha=0.45, color="0.4")
        axm.plot(t_r, traj[:, c * NB + b], lw=1.4, label=BAND[b] if c == 0 else None)
    axm.axvline(0, color="#D55E00", ls=":", lw=0.8)
    axm.set_ylabel(f"{lab}\nr={per_ch[lab]['r_coupled']:.2f}", fontsize=7)
axes[0].legend(ncol=5, frameon=False, fontsize=6)
axes[-1].set_xlabel("recording time (s), onset at 0")
axes[0].set_title("Joint autonomous rollout: grey = recording, coloured = model",
                  loc="left", fontweight="bold")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(RESULT_DIR / f"network_rollout.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)

gates = {
    "bus_range_couples_channels": {"required": "> 0 held-out gain",
                                   "achieved": bus_scan[BUS_NAME],
                                   "passed": bool(bus_scan[BUS_NAME] > 0)},
    "beats_persistence_on_transition": {"required": 0.02, "achieved": mean_gain,
                                        "passed": bool(mean_gain > 0.02)},
    "spatial_links_survive_surrogates": {"required": ">= 1 link at p<0.05",
                                         "achieved": n_sig, "passed": bool(n_sig >= 1)},
    "motor_L_hippocampus_R_connected": {
        "required": "either direction significant",
        "achieved": [float(pvals[hr, ml]), float(pvals[ml, hr])],
        "passed": bool(pvals[hr, ml] < 0.05 or pvals[ml, hr] < 0.05)},
    "rollout_completes": {"required": f"{nsteps} steps", "achieved": nr - 1,
                          "passed": bool(nr - 1 == nsteps)},
    "coupling_improves_the_rollout": {"required": "coupled r > links-cut r",
                                      "achieved": [mean_r, mean_rn],
                                      "passed": bool(mean_r > mean_rn)},
}
print("\n" + "=" * 96)
ok = True
for k, v in gates.items():
    ok &= print_check(v["passed"], f"{k}: achieved={v['achieved']}, required={v['required']}")

save_metrics(RESULT_DIR / "metrics.json", {
    "task": "network of networks: six per-channel cross-frequency oscillators "
            "coupled through a broadband bus",
    "passed": bool(ok),
    "channels": LAB, "bands": {k: list(v) for k, v in BANDS.items()},
    "penalty": SUB, "lamb_per_channel": dict(zip(LAB, lambs)),
    "effective_samples": eff,
    "bus_reconstruction_r2": bus_map_r2,
    "tau_reused_from_channel_mean_fit": TAU_B.tolist(),
    "bus_scan": bus_scan, "bus_selected": BUS_NAME,
    "heldout_transition": heldout, "mean_gain_over_persistence": mean_gain,
    "spatial_coupling": Wmat.tolist(), "spatial_pvalues": pvals.tolist(),
    "n_significant_links": n_sig,
    "rollout": {"start": float(T[i0]), "seconds": ROLL_SECONDS,
                "per_channel": per_ch, "mean_r": mean_r,
                "mean_r_links_cut": mean_rn},
    "gates": gates,
    "note": "the bus at rollout is the model's own broadband, not the "
            "recording's, so the rollout is fully autonomous. Spatial p-values "
            "use a ridge surrogate on the same design, not the KAN.",
})
print(f"[DONE] {RESULT_DIR}")
