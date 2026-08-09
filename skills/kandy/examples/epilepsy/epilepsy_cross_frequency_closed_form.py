#!/usr/bin/env python3
"""Integrate the printed equations by hand -- no KANDy, no splines, no network.

Everything else in this directory integrates the fitted KAN.  This file takes
the five lines that ``epilepsy_cross_frequency_seizure_freedom.py`` PRINTS,
retypes them as plain numpy, and integrates those.  If the closed form is the
deliverable then it has to stand on its own, away from the object it was
extracted from, and that is the only thing tested here.

The equations (channel-mean band log-amplitude envelopes, standardized):

    dx_i/dt = (T_i - x_i) / tau_i
    T_i     = X_C_i + X_R_i * tanh( S_i * f_i(x) / X_R_i )

    f_delta = -0.6452*delta - 0.2496*gamma + 0.0638
    f_theta =  0.0940 - 0.6048*theta
    f_alpha =  0.3333
    f_beta  =  0.2306
    f_gamma =  6.1026 - 0.1136*(8.5784 - 5.9214*ReLU(delta - 0.2159))**2

Reconstruction re-injects the MEASURED PHASE, because these equations model an
envelope and contain no waveform.  That concession is worth almost all of the
apparent score, so three baselines run beside it and the only number reported
as the model's is the difference:

    flat      constant amplitude x the same measured phase
    shuffled  the model trajectory in scrambled temporal order
    reversed  the model trajectory run backwards

A shuffled trajectory carries no dynamics whatsoever.  If it scores near the
model, the reconstruction is reporting the phase it was handed.

Run:
    cd skills/kandy/examples && python epilepsy/epilepsy_cross_frequency_closed_form.py
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from _utils import ensure_outdir, print_check, save_metrics

RESULT_DIR = ensure_outdir(REPO_ROOT / "results" / "Epilepsy-Cross-Frequency")
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 50.0)}
NB = 5

# --- the fitted constants, copied from metrics.json -------------------------
TAU = np.array([12.6165, 15.8849, 20.0, 20.0, 20.0])
X_C = np.array([-0.1402, -0.2734, -0.5740, -0.1110, -0.0502])
X_R = np.array([4.5323, 4.2649, 3.8850, 3.4308, 3.5886])
S = np.array([13.6808, 13.1734, 10.8474, 9.7217, 9.6823])
TH = np.array([0.2159, -1.0028, -1.2172, -1.3362, -1.2859])
SMOOTH_TAU, DS_STATE, EDGE_S = 0.15, 10, 2.0


def f_of(x):
    """The five printed right-hand sides, retyped."""
    d, th, al, be, ga = x[0], x[1], x[2], x[3], x[4]
    return np.array([
        -0.6452 * d - 0.2496 * ga + 0.0638,
        0.0940 - 0.6048 * th,
        0.3333,
        0.2306,
        6.1026 - 0.1136 * (8.5784 - 5.9214 * max(d - TH[0], 0.0)) ** 2,
    ])


def rhs(x):
    T = X_C + X_R * np.tanh(S * f_of(x) / X_R)
    return (T - x) / TAU


# ---------------------------------------------------------------------------
# Rebuild the same state the equations were fitted to
# ---------------------------------------------------------------------------
payload = pickle.load((REPO_ROOT / "data" / "Dory_2021_04_12_006_preprocessed.pkl").open("rb"))
X = np.asarray(payload["X"], float)
FS = float(payload["fs"])
T_ALL = np.asarray(payload["t"], float)
LABELS = list(payload["labels"])
NCH = len(LABELS)


def causal_ewma(y, tau_s):
    al = 1.0 - np.exp(-1.0 / (tau_s * FS))
    out = np.empty_like(y)
    acc = y[0].copy()
    for i in range(len(y)):
        acc = acc + al * (y[i] - acc)
        out[i] = acc
    return out


AMP = np.zeros((len(X), NCH, NB))
PHASE = np.zeros((len(X), NCH, NB))
BSIG = np.zeros((len(X), NCH, NB))
for b, n in enumerate(BAND_NAMES):
    lo, hi = BANDS[n]
    bb, aa = butter(4, [lo / (0.5 * FS), hi / (0.5 * FS)], btype="band")
    filt = filtfilt(bb, aa, X, axis=0)
    z = hilbert(filt, axis=0)
    AMP[:, :, b], PHASE[:, :, b], BSIG[:, :, b] = np.abs(z), np.angle(z), filt
TRUE = BSIG.sum(2)

SM = np.stack([causal_ewma(np.log(AMP[:, :, b] + 1e-9), SMOOTH_TAU)
               for b in range(NB)], axis=2)
MU = SM.reshape(-1, NCH * NB).mean(0).reshape(NCH, NB)
SD = SM.reshape(-1, NCH * NB).std(0).reshape(NCH, NB) + 1e-12

mean_log = SM.mean(axis=1)                       # channel mean, per band
edge = int(round(EDGE_S * FS))
sl0 = slice(edge, len(X) - edge)
STATE_FULL = (mean_log - mean_log[sl0].mean(0)) / (mean_log[sl0].std(0) + 1e-12)
STATE = STATE_FULL[sl0][::DS_STATE]
T_STATE = T_ALL[sl0][::DS_STATE]
DT_S = float(T_STATE[1] - T_STATE[0])

# ---------------------------------------------------------------------------
# Integrate the closed form
# ---------------------------------------------------------------------------
START, SECONDS = -8.0, 25.0
i0 = int(np.argmin(np.abs(T_STATE - START)))
n_steps = int(round(SECONDS / DT_S))
x = STATE[i0].copy()
traj = np.empty((n_steps + 1, NB))
traj[0] = x
for i in range(n_steps):
    k1 = rhs(x)
    k2 = rhs(x + 0.5 * DT_S * k1)
    k3 = rhs(x + 0.5 * DT_S * k2)
    k4 = rhs(x + DT_S * k3)
    x = x + (DT_S / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    traj[i + 1] = x
t_r = T_STATE[i0:i0 + n_steps + 1]
nr = len(t_r)
traj = traj[:nr]
ref = STATE[i0:i0 + nr]

print("=" * 74)
print("INTEGRATING THE PRINTED EQUATIONS -- no KANDy, no splines")
print("=" * 74)
print(f"from t={t_r[0]:+.1f}s for {SECONDS:.0f}s at {1/DT_S:.0f} Hz\n")
print(f"{'band':>7} {'x(0)':>7} {'x(end) model':>13} {'x(end) data':>12} "
      f"{'fixed point':>12}")
fp = x.copy()
for _ in range(20000):                                   # settle to the attractor
    fp = fp + DT_S * rhs(fp)
for b, n in enumerate(BAND_NAMES):
    print(f"{n:>7} {traj[0, b]:7.2f} {traj[-1, b]:13.2f} {ref[-1, b]:12.2f} "
          f"{fp[b]:12.2f}")

PRE = (T_STATE >= -13) & (T_STATE <= -2)
POST = (T_STATE >= 18) & (T_STATE <= 28)
pre_c, post_c = STATE[PRE].mean(0), STATE[POST].mean(0)
axis = post_c - pre_c
an = float(np.linalg.norm(axis))
u = axis / an


def tscore(a):
    return (np.atleast_2d(a) - pre_c) @ u / an


s_data, s_model = tscore(ref).ravel(), tscore(traj).ravel()
base = s_data[:int(1 / DT_S)].mean()
print(f"\ntransition score: data rise {s_data.max()-base:+.2f}, "
      f"closed form {s_model.max()-s_model[:int(1/DT_S)].mean():+.2f}; "
      f"endpoint data {s_data[-1]:+.2f} vs model {s_model[-1]:+.2f}")

# ---------------------------------------------------------------------------
# Reconstruct, with the baselines that decide whether it means anything
# ---------------------------------------------------------------------------
m0 = int(np.argmin(np.abs(T_ALL - t_r[0])))
m1 = int(np.argmin(np.abs(T_ALL - t_r[-1])))
win = slice(m0, m1 + 1)
t_full = T_ALL[win]
rng = np.random.default_rng(0)


def up(a):
    return np.column_stack([np.interp(t_full, t_r, a[:, b]) for b in range(NB)])


def resynth(state_full):
    out = np.zeros((len(t_full), NCH))
    for c in range(NCH):
        for b in range(NB):
            out[:, c] += np.exp(MU[c, b] + SD[c, b] * state_full[:, b]) \
                * np.cos(PHASE[win, c, b])
    return out


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


variants = {
    "closed-form ODEs": up(traj),
    "FLAT amplitude": np.zeros((len(t_full), NB)),
    "SHUFFLED closed form": up(traj[rng.permutation(nr)]),
    "REVERSED closed form": up(traj[::-1]),
}
scores = {}
print(f"\n{'amplitude source (measured phase in all cases)':>44} | {'mean R2':>8}")
print("-" * 56)
for k, v in variants.items():
    rec = resynth(v)
    scores[k] = float(np.mean([r2(TRUE[win][:, c], rec[:, c]) for c in range(NCH)]))
    print(f"{k:>44} | {scores[k]:8.3f}")
gain = scores["closed-form ODEs"] - scores["FLAT amplitude"]
shuf = scores["closed-form ODEs"] - scores["SHUFFLED closed form"]
print(f"\nmodel - flat     = {gain:+.3f}   <- the only quantity that is the model's")
print(f"model - shuffled = {shuf:+.3f}   <- how much the TIME ORDER is worth")

per_ch = {}
rec_m = resynth(up(traj))
rec_f = resynth(np.zeros((len(t_full), NB)))
print(f"\n{'channel':>14} | {'model':>7} {'flat':>7} {'gain':>7}")
for c, lab in enumerate(LABELS):
    a, b_ = r2(TRUE[win][:, c], rec_m[:, c]), r2(TRUE[win][:, c], rec_f[:, c])
    per_ch[lab] = {"model": a, "flat": b_, "gain": a - b_}
    print(f"{lab:>14} | {a:7.3f} {b_:7.3f} {a - b_:+7.3f}")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
OI = {"verm": "#D55E00", "blue": "#0072B2", "purple": "#CC79A7", "yellow": "#F0E442"}
fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
axes[0].axvspan(1, 8, color=OI["yellow"], alpha=0.22, lw=0)
for b, n in enumerate(BAND_NAMES):
    axes[0].plot(t_r, ref[:, b], lw=0.7, alpha=0.5)
    axes[0].plot(t_r, traj[:, b], lw=1.8, ls="--", label=n)
axes[0].set_ylabel("band amplitude (sd)")
axes[0].legend(ncol=5, frameon=False, fontsize=7)
axes[0].set_title("Closed-form ODEs integrated by hand: thin = recording, "
                  "thick dashed = equations", loc="left", fontweight="bold")
axes[1].axvspan(1, 8, color=OI["yellow"], alpha=0.22, lw=0)
axes[1].plot(t_r, s_data, color="0.3", lw=1.4, label="recording")
axes[1].plot(t_r, s_model, color=OI["verm"], lw=2.2, label="closed form")
axes[1].axhline(0, color="0.7", lw=0.6)
axes[1].axhline(1, color="0.7", lw=0.6)
axes[1].set_ylabel("transition score")
axes[1].set_xlabel("recording time (s), onset at 0")
axes[1].legend(frameon=False, ncol=2)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(RESULT_DIR / f"closed_form_integration.{ext}", dpi=250,
                bbox_inches="tight")
plt.close(fig)

ok = print_check(np.isfinite(traj).all(), "closed form integrates without blowing up")
ok &= print_check(gain > 0, f"closed form beats a constant amplitude ({gain:+.3f})")
ok &= print_check(shuf > 0, f"time order of the closed form is worth something "
                            f"({shuf:+.3f})")

save_metrics(RESULT_DIR / "closed_form_metrics.json", {
    "task": "integrate the printed equations directly, with no KANDy object",
    "passed": bool(ok),
    "equations": {
        "delta": "-0.6452*delta - 0.2496*gamma + 0.0638",
        "theta": "0.0940 - 0.6048*theta",
        "alpha": "0.3333", "beta": "0.2306",
        "gamma": "6.1026 - 0.1136*(8.5784 - 5.9214*ReLU(delta-0.2159))**2",
        "wrapper": "dx/dt = (X_C + X_R*tanh(S*f/X_R) - x)/tau"},
    "constants": {"tau": TAU.tolist(), "X_C": X_C.tolist(), "X_R": X_R.tolist(),
                  "S": S.tolist(), "relu_threshold": TH.tolist()},
    "fixed_point": fp.tolist(),
    "endpoint": {"model": traj[-1].tolist(), "data": ref[-1].tolist()},
    "reconstruction_r2": scores,
    "model_minus_flat": gain, "model_minus_shuffled": shuf,
    "per_channel": per_ch,
    "note": "measured phase is injected; FLAT and SHUFFLED are the baselines "
            "that isolate what the equations contribute.",
})
print(f"[DONE] {RESULT_DIR}")
