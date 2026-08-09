#!/usr/bin/env python3
"""Abstract beta->delta intervention on the autonomous cross-frequency model.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
The base is Epilepsy-Cross-Frequency's fitted autonomous envelope ODE.  A
CONTROL TERM is added to the delta equation only:

    dx_delta/dt = fitted_delta_rhs(x)  +  u(t) * a * H_k(x_beta(t - lag) - th)

u(t) is a stimulation window, a its signed amplitude.  `a` is a DESIGN
parameter, not a fitted coefficient: the delayed beta->delta pathway did NOT
survive held-out validation (adding it cost delta 0.137 of held-out R2 once
selection and reporting were separated).  So this asks "if that pathway could
be modulated, what would the fitted system do", which is an intervention study
on a fitted autonomous field -- not evidence that the pathway exists.

It CANNOT support "DBS terminates the seizure".  The base field is monostable:
200 random initial conditions integrated 400 s converge to 1-2 attractors, all
on the ictal side, none near pre-ictal.  A system with one basin can be held
away from it while forced and must return when released.  So every condition is
scored TWICE -- at the end of stimulation and at the end of the rollout -- and
the gap between them is the whole point.  Reporting only the during-stimulation
number would turn "suppressed while forced" into "terminated", which is the
claim this model cannot make.

SWEEP
-----
start x duration x sign x amplitude x lag, all conditions integrated as one
batch so the sweep costs about the same as a single rollout.

Scored per condition:
    suppression   transition score at stimulation end, vs the unstimulated run
    rebound       transition score at rollout end, vs the unstimulated run
    viable        stayed inside the range the recording actually spans
    terminated    suppressed AND still suppressed at the end AND viable

Run:
    cd skills/kandy/examples && python epilepsy/epilepsy_beta_delta_control.py
"""

from __future__ import annotations

import contextlib
import os
import pickle
import sys
from io import StringIO
from itertools import product
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
RESULT_DIR = Path(ensure_outdir(REPO_ROOT / "results" / "Epilepsy-Beta-Delta-Control"))

BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 50.0)}
BAND = list(BANDS)
NB = 5
PAIRS = [(a, b) for a in range(NB) for b in range(a + 1, NB)]
DI, BIx = 0, 3
SMOOTH_TAU, DS, EDGE_S, LAG_S, GUARD_S = 0.15, 10, 2.0, 2.0, 2.0
TAU = np.array([12.6, 15.9, 20.0, 20.0, 20.0])
LAMB, GRID, STEPS, BOX = 0.1, 5, 80, 0.35
ROLL_START, ROLL_SEC = -8.0, 25.0
SHARP = 1.0

# The ACTUATOR BAND is now swept.  Putting the control on delta was the wrong
# target: delta moves 0.22 -> -0.05 across the event while theta/alpha/beta/gamma
# each move ~1.5 sd, so the transition score is almost blind to delta and the
# strongest possible stimulation shifted it by 2%.  The actuator has to sit on a
# state that carries the quantity being scored.
ACTUATORS = list(range(NB))
STARTS = [-4.0, 0.0, 2.0]
DURS = [6.0, 33.0]                     # 33 s = on for the whole rollout
AMPS = [0.30, 1.00]                    # 0.30 alone did nothing; go harder
SIGNS = [+1.0, -1.0]
LAGS = [0.0, 1.5]
# Separately: the GAIN form on a FITTED edge.  T_delta contains -0.25*gamma, so
# G(t) = 1 - a*u(t) turning that input down is a modulation of a coefficient the
# data supports, which is a stronger construction than an imposed additive term.
GAIN_AMPS = [0.5, 1.0, 1.5, 2.0]

payload = pickle.load((REPO_ROOT / "data" / "Dory_2021_04_12_006_preprocessed.pkl").open("rb"))
XR = np.asarray(payload["X"], float)
FS = float(payload["fs"])
T_ALL = np.asarray(payload["t"], float)


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
S = np.column_stack([env_of(*BANDS[b]).mean(1)[sl] for b in BAND])
S = (S - S.mean(0)) / (S.std(0) + 1e-12)
LAGN = max(1, int(round(LAG_S / DT)))
DX = (S[LAGN:] - S[:-LAGN]) / LAG_S
LO = S.min(0) - BOX * np.ptp(S, 0)
HI = S.max(0) + BOX * np.ptp(S, 0)
C, R = 0.5 * (LO + HI), 0.5 * (HI - LO)
TH = S[(T >= -13) & (T <= -2)].mean(0)
THB = float(np.percentile(S[:, BIx], 50))
rows = np.arange(N - LAGN)
t_rows = T[:N - LAGN]
nb = np.zeros(len(rows), bool)
for w in [(-13.0, -9.0), (22.0, 28.0)]:
    nb |= (t_rows >= w[0] - GUARD_S) & (t_rows <= w[1] + GUARD_S)
trainA = rows[~nb]
TGT = np.zeros((len(DX), NB))
for b in range(NB):
    t_ = S[:len(DX), b] + TAU[b] * DX[:, b]
    TGT[:, b] = R[b] * np.arctanh(np.clip((t_ - C[b]) / R[b], -0.999, 0.999))
TSC = TGT[trainA].std(0) + 1e-12


def lift_np(x):
    x = np.atleast_2d(x)
    f = [np.ones(len(x))] + [x[:, b] for b in range(NB)]
    f += [np.maximum(x[:, b] - TH[b], 0.0) for b in range(NB)]
    f += [x[:, a] * x[:, b] for a, b in PAIRS]
    return np.column_stack(f)


def lift_t(x):
    th = torch.as_tensor(TH, dtype=x.dtype)
    f = [torch.ones_like(x[:, 0])] + [x[:, b] for b in range(NB)]
    f += [torch.clamp(x[:, b] - th[b], min=0.0) for b in range(NB)]
    f += [x[:, a] * x[:, b] for a, b in PAIRS]
    return torch.stack(f, 1)


print("=" * 88)
print("ABSTRACT beta->delta INTERVENTION on the autonomous cross-frequency model")
print("=" * 88)
models = []
for b in range(NB):
    m = KANDy(lift=CustomLift(lift_np, 21, lift_t, f"b{b}"), grid=GRID, k=3,
              steps=STEPS, seed=SEED, base_fun=torch.nn.Identity(), device="cpu")
    with contextlib.redirect_stdout(StringIO()):
        m.fit(X=S[trainA], X_dot=TGT[trainA, b:b + 1] / TSC[b], val_frac=0.15,
              test_frac=0.15, lamb=LAMB, opt="LBFGS", lr=1.0, batch=-1,
              stop_grid_update_step=10, patience=0, verbose=False)
    models.append(m)
print(f"base model refitted ({NB} bands, lamb={LAMB})")

PRE = (T >= -13) & (T <= -2)
POST = (T >= 18) & (T <= 28)
pc, poc = S[PRE].mean(0), S[POST].mean(0)
axv = poc - pc
an = float(np.linalg.norm(axv))
uv = axv / an


def tscore(a):
    return (np.atleast_2d(a) - pc) @ uv / an


i0 = int(np.argmin(np.abs(T - ROLL_START)))
nst = int(round(ROLL_SEC / DT))
MAXLAG = int(round(max(LAGS) / DT)) + 2
LOK = S.min(0) - 0.15 * np.ptp(S, 0)
HOK = S.max(0) + 0.15 * np.ptp(S, 0)

CONDS = [{"band": bnd, "start": s_, "dur": d_, "amp": a_, "sign": g_, "lag": l_,
          "kind": "additive"}
         for bnd, s_, d_, a_, g_, l_ in product(ACTUATORS, STARTS, DURS, AMPS,
                                                SIGNS, LAGS)]
CONDS += [{"band": DI, "start": s_, "dur": d_, "amp": a_, "sign": +1.0,
           "lag": 0.0, "kind": "gain"}
          for s_, d_, a_ in product(STARTS, DURS, GAIN_AMPS)]
NCOND = len(CONDS)
n_add = sum(1 for c in CONDS if c["kind"] == "additive")
print(f"sweep: {n_add} additive conditions ({len(ACTUATORS)} actuator bands x "
      f"{len(STARTS)} starts x {len(DURS)} durations x {len(AMPS)} amps x "
      f"{len(SIGNS)} signs x {len(LAGS)} lags)")
print(f"       + {NCOND - n_add} GAIN conditions on the fitted gamma->delta edge")
print("all integrated as ONE batch alongside an unstimulated control\n")

# batch = the unstimulated control (index 0) plus every condition
NB_ROWS = NCOND + 1
start_a = np.array([0.0] + [c["start"] for c in CONDS])
end_a = np.array([0.0] + [c["start"] + c["dur"] for c in CONDS])
amp_a = np.array([0.0] + [c["amp"] * c["sign"] for c in CONDS])
lag_a = np.array([0] + [int(round(c["lag"] / DT)) for c in CONDS])
on_a = np.array([False] + [True] * NCOND)
band_a = np.array([DI] + [c["band"] for c in CONDS])
is_gain = np.array([False] + [c["kind"] == "gain" for c in CONDS])
THG = float(TH[4])                       # gamma's pre-ictal level

hist = np.stack([np.tile(S[i0 - MAXLAG + k], (NB_ROWS, 1)) for k in range(MAXLAG)])
x = np.tile(S[i0], (NB_ROWS, 1))
traj = np.empty((nst + 1, NB_ROWS, NB))
traj[0] = x


def rhs(st, buf, tnow):
    out = np.zeros_like(st)
    u = on_a & (tnow >= start_a) & (tnow < end_a)

    # GAIN form: turn down the gamma input to delta's own fitted equation.
    # G = 1 - a*u(t), applied by rescaling gamma about its pre-ictal level
    # before delta's model sees it.  Only the delta model is affected.
    st_delta = st.copy()
    gmask = is_gain & u
    if gmask.any():
        G = 1.0 - amp_a * gmask
        st_delta[:, 4] = THG + G * (st[:, 4] - THG)

    for b in range(NB):
        src = st_delta if b == DI else st
        inner = models[b].predict(src).ravel() * TSC[b]
        out[:, b] = (C[b] + R[b] * np.tanh(inner / R[b]) - st[:, b]) / TAU[b]

    # ADDITIVE form: signed, gated by the delayed beta Heaviside, on the
    # actuator band chosen per condition.
    beta_del = np.array([buf[max(0, len(buf) - 1 - lag_a[i]), i, BIx]
                         for i in range(NB_ROWS)])
    Hb = 1.0 / (1.0 + np.exp(-SHARP * (beta_del - THB)))
    add = u & ~is_gain
    out[np.arange(NB_ROWS), band_a] += add * amp_a * Hb
    return out


for it in range(nst):
    tn = T[i0] + it * DT
    k1 = rhs(x, hist, tn)
    k2 = rhs(x + 0.5 * DT * k1, hist, tn + 0.5 * DT)
    k3 = rhs(x + 0.5 * DT * k2, hist, tn + 0.5 * DT)
    k4 = rhs(x + DT * k3, hist, tn + DT)
    x = x + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    x = np.where(np.isfinite(x), x, 0.0)
    hist = np.concatenate([hist[1:], x[None]], axis=0)
    traj[it + 1] = x

t_r = T[i0:i0 + nst + 1]
ref = S[i0:i0 + nst + 1]
sc = np.stack([tscore(traj[:, i, :]).ravel() for i in range(NB_ROWS)])
base = sc[0]
s_data = tscore(ref).ravel()
rise_d = float(s_data.max() - s_data[:int(1 / DT)].mean())
print(f"unstimulated: rise {base.max()-base[:int(1/DT)].mean():+.2f} "
      f"(recording {rise_d:+.2f}); end score {base[-1]:+.2f}")

res = []
for i, c in enumerate(CONDS, start=1):
    k_end = min(nst, max(0, int(round((c["start"] + c["dur"] - T[i0]) / DT))))
    inside = np.all((traj[:, i, :] >= LOK) & (traj[:, i, :] <= HOK), axis=1).mean()
    supp = float(base[k_end] - sc[i][k_end])          # >0 means suppressed
    reb = float(base[-1] - sc[i][-1])                 # >0 means still suppressed
    # A condition can only be TESTED for termination if the stimulation stops
    # before the rollout does.  dur = 33 s runs to the end, so its "suppression
    # at rollout end" is still measuring the forced state -- calling that
    # termination would be exactly the error this script exists to avoid.
    stops = bool(c["start"] + c["dur"] < T[i0] + ROLL_SEC - 2.0)
    res.append({**c, "band_name": BAND[c["band"]],
                "suppression_at_stim_end": supp, "suppression_at_rollout_end": reb,
                "in_range": float(inside), "viable": bool(inside > 0.9),
                "score_end": float(sc[i][-1]), "stimulation_stops": stops,
                "suppresses": bool(supp > 0.3 * rise_d and inside > 0.9),
                "terminated": bool(stops and supp > 0.3 * rise_d
                                   and reb > 0.3 * rise_d and inside > 0.9)})

res.sort(key=lambda r: -r["suppression_at_stim_end"])
print(f"\n{'kind':>9} {'band':>7} {'start':>6} {'dur':>5} {'amp':>6} | "
      f"{'supp @ stim end':>15} {'still supp @ end':>17} {'viable':>7}")
for r in res[:14]:
    print(f"{r['kind']:>9} {r['band_name']:>7} {r['start']:+6.1f} {r['dur']:5.0f} "
          f"{r['sign']*r['amp']:+6.2f} | {r['suppression_at_stim_end']:+15.3f} "
          f"{r['suppression_at_rollout_end']:+17.3f} {str(r['viable']):>7}")

print(f"\nbest suppression by ACTUATOR BAND (additive form):")
for b in range(NB):
    sub = [r for r in res if r["kind"] == "additive" and r["band"] == b]
    bb_ = max(sub, key=lambda r: r["suppression_at_stim_end"])
    print(f"        {BAND[b]:>7}: {bb_['suppression_at_stim_end']:+.3f} "
          f"(amp {bb_['sign']*bb_['amp']:+.2f}, viable={bb_['viable']})")
gn = [r for r in res if r["kind"] == "gain"]
if gn:
    bg = max(gn, key=lambda r: r["suppression_at_stim_end"])
    print(f"        gain on the fitted gamma->delta edge: "
          f"{bg['suppression_at_stim_end']:+.3f} (a={bg['amp']:.1f}, "
          f"viable={bg['viable']})")

n_supp = sum(1 for r in res if r["suppresses"])
n_term = sum(1 for r in res if r["terminated"])
n_testable = sum(1 for r in res if r["suppresses"] and r["stimulation_stops"])
print(f"\n  of the suppressing conditions, {n_testable} switch OFF inside the "
      "rollout and can\n  therefore be tested for termination at all; the rest "
      "stimulate continuously.")
if n_testable:
    off = [r for r in res if r["suppresses"] and r["stimulation_stops"]]
    print(f"  {'band':>7} {'amp':>6} {'off at':>7} {'supp @ off':>11} "
          f"{'supp @ end':>11}")
    for r in sorted(off, key=lambda q: -q["suppression_at_stim_end"])[:8]:
        print(f"  {r['band_name']:>7} {r['sign']*r['amp']:+6.2f} "
              f"{r['start']+r['dur']:6.1f}s {r['suppression_at_stim_end']:+11.3f} "
              f"{r['suppression_at_rollout_end']:+11.3f}")
best = res[0]
print(f"\nconditions that SUPPRESS during stimulation (>30% of the observed "
      f"rise, viable): {n_supp}/{NCOND}")
print(f"conditions that remain suppressed AFTER stimulation ends: {n_term}/{NCOND}")
if n_supp and not n_term:
    keep = [r for r in res if r["suppression_at_stim_end"] > 0.3 * rise_d
            and r["viable"]]
    decay = float(np.mean([r["suppression_at_stim_end"]
                           - r["suppression_at_rollout_end"] for r in keep]))
    print(f"\n  Every suppressing condition REBOUNDS: mean loss of suppression "
          f"after\n  stimulation ends is {decay:+.3f}.  That is what a MONOSTABLE "
          "field does --\n  it can be held away from its only attractor and must "
          "return when released.\n  This is suppression under forcing, NOT "
          "termination, and the model cannot\n  distinguish the two because it "
          "has no second basin to be terminated into.")

# also: does permanent stimulation hold it?
perm = [r for r in res if r["dur"] >= 33.0 and r["viable"]]
if perm:
    bp = max(perm, key=lambda r: r["suppression_at_rollout_end"])
    print(f"\n  continuous stimulation for the whole rollout: best holds the score "
          f"{bp['suppression_at_rollout_end']:+.3f} below baseline "
          f"(amp {bp['sign']*bp['amp']:+.2f}, lag {bp['lag']:.1f}s)")

mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].plot(t_r, s_data, color="0.25", lw=1.4, label="recording")
axes[0].plot(t_r, base, color="#0072B2", lw=2.0, label="model, no stimulation")
KEYS = ("band", "start", "dur", "amp", "sign", "lag", "kind")
for r in res[:3]:
    i = CONDS.index({k: r[k] for k in KEYS}) + 1
    axes[0].plot(t_r, sc[i], lw=1.5, ls="--",
                 label=f"a={r['sign']*r['amp']:+.2f} {r['start']:+.0f}.."
                       f"{r['start']+r['dur']:.0f}s")
    axes[0].axvspan(r["start"], r["start"] + r["dur"], color="#F0E442", alpha=0.12, lw=0)
axes[0].set_xlabel("recording time (s)")
axes[0].set_ylabel("transition score")
axes[0].legend(fontsize=6.5, frameon=False)
axes[0].set_title("Best suppressing conditions", loc="left", fontweight="bold")
axes[1].scatter([r["suppression_at_stim_end"] for r in res],
                [r["suppression_at_rollout_end"] for r in res],
                c=["#D55E00" if r["viable"] else "0.7" for r in res], s=18)
lim = max(0.05, max(abs(r["suppression_at_stim_end"]) for r in res) * 1.1)
axes[1].plot([-lim, lim], [-lim, lim], "k--", lw=0.7, label="no rebound")
axes[1].axhline(0, color="0.6", lw=0.6)
axes[1].axvline(0.3 * rise_d, color="#0072B2", ls=":", lw=1,
                label="30% of observed rise")
axes[1].set_xlabel("suppression at stimulation end")
axes[1].set_ylabel("suppression at rollout end")
axes[1].legend(fontsize=6.5, frameon=False)
axes[1].set_title("Does suppression outlast the stimulus?", loc="left",
                  fontweight="bold")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(RESULT_DIR / f"beta_delta_control.{ext}", dpi=250, bbox_inches="tight")
plt.close(fig)

print("\n" + "=" * 88)
ok = print_check(n_supp > 0, f"some condition suppresses the transition while on "
                             f"({n_supp}/{NCOND})")
print_check(n_term > 0, f"some condition remains suppressed after stimulation ends "
                        f"({n_term}/{NCOND})")

save_metrics(RESULT_DIR / "metrics.json", {
    "task": "abstract beta->delta control term on the autonomous cross-frequency ODE",
    "control_term": "dx_delta/dt += u(t) * a * H_k(x_beta(t-lag) - theta_beta)",
    "scope": "a is a DESIGN parameter, not fitted: the delayed beta->delta "
             "pathway lost 0.137 of held-out R2 when selection and reporting "
             "were separated. This simulates an abstract intervention on that "
             "pathway; it is not evidence the pathway exists.",
    "limitation": "the base field is monostable (1-2 attractors, all ictal, none "
                  "pre-ictal), so suppression under forcing cannot be "
                  "distinguished from termination. Both endpoints are reported.",
    "sweep": {"starts": STARTS, "durations": DURS, "amplitudes": AMPS,
              "signs": SIGNS, "lags": LAGS, "n_conditions": NCOND},
    "observed_rise": rise_d,
    "unstimulated_rise": float(base.max() - base[:int(1/DT)].mean()),
    "n_suppress_during": n_supp, "n_terminate_after": n_term,
    "results": res,
})
print(f"[DONE] {RESULT_DIR}")
