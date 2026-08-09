#!/usr/bin/env python3
"""Driven (non-autonomous) reconstruction of all six recordings.

The autonomous envelope model predicts band amplitudes and contains no
waveform.  This script accepts an explicit time-dependent drive so that the
integrated model produces the six traces themselves.  With t in the right-hand
side this is an OBSERVER / data-assimilation model, not a predictive one: it
cannot be run past the end of the recording, and its reconstruction score is
partly the drive.  That is stated here rather than discovered later, and every
number below carries the baselines that separate the two contributions.

    ds_c/dt is not modelled directly.  Instead

        s_c(t) = sum_b A_c,b(x(t)) * cos( phi_b(t) + delta_c,b )

    x(t)      integrated from the autonomous envelope ODEs (no time argument)
    phi_b(t)  the DRIVE: one phase per band, injected
    delta_c,b a fitted CONSTANT offset per channel and band

What makes this worth running rather than circular
--------------------------------------------------
Handing over all 30 measured phase series (6 channels x 5 bands) reconstructs
six traces by construction and establishes nothing.  The question with content
is how FEW drive channels are enough:

    30 phases   per channel and band          -- the trivial upper bound
     5 phases   one per band, shared across all six channels, plus a constant
                offset per channel            -- only works if the six regions
                                                share a phase organisation
     0 phases   fully autonomous              -- measured already: spectrogram
                                                correlation 0.238

If 5 works nearly as well as 30, that is a real finding about spatial phase
coherence and a 6x compression of the drive.  If it does not, the six channels
are phase-independent and only the trivial version reconstructs them.

Baselines on every line, because the drive flatters everything:
    flat        constant amplitude x the same drive
    shuffled    the model's amplitude trajectory in scrambled time order

Run:
    cd skills/kandy/examples && python epilepsy/epilepsy_driven_reconstruction.py
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

RESULT_DIR = ensure_outdir(REPO_ROOT / "results" / "Epilepsy-Driven-Reconstruction")
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 50.0)}
NB = 5
SMOOTH_TAU, DS_STATE, EDGE_S = 0.15, 10, 2.0

# fitted constants of the autonomous envelope model (from metrics.json)
TAU = np.array([12.6165, 15.8849, 20.0, 20.0, 20.0])
X_C = np.array([-0.1402, -0.2734, -0.5740, -0.1110, -0.0502])
X_R = np.array([4.5323, 4.2649, 3.8850, 3.4308, 3.5886])
S_SC = np.array([13.6808, 13.1734, 10.8474, 9.7217, 9.6823])
TH0 = 0.2159


def f_of(x):
    d, th, al, be, ga = x
    return np.array([
        -0.6452 * d - 0.2496 * ga + 0.0638,
        0.0940 - 0.6048 * th,
        0.3333,
        0.2306,
        6.1026 - 0.1136 * (8.5784 - 5.9214 * max(d - TH0, 0.0)) ** 2,
    ])


def rhs(x):
    return (X_C + X_R * np.tanh(S_SC * f_of(x) / X_R) - x) / TAU


# ---------------------------------------------------------------------------
payload = pickle.load((REPO_ROOT / "data" / "Dory_2021_04_12_006_preprocessed.pkl").open("rb"))
X = np.asarray(payload["X"], float)
FS = float(payload["fs"])
T_ALL = np.asarray(payload["t"], float)
LABELS = list(payload["labels"])
NCH = len(LABELS)

AMP = np.zeros((len(X), NCH, NB))
PH = np.zeros((len(X), NCH, NB))
BS = np.zeros((len(X), NCH, NB))
for b, n in enumerate(BAND_NAMES):
    lo, hi = BANDS[n]
    bb, aa = butter(4, [lo / (0.5 * FS), hi / (0.5 * FS)], btype="band")
    filt = filtfilt(bb, aa, X, axis=0)
    z = hilbert(filt, axis=0)
    AMP[:, :, b], PH[:, :, b], BS[:, :, b] = np.abs(z), np.angle(z), filt
TRUE = BS.sum(2)


def causal_ewma(y, tau_s):
    al = 1.0 - np.exp(-1.0 / (tau_s * FS))
    out = np.empty_like(y)
    acc = y[0].copy()
    for i in range(len(y)):
        acc = acc + al * (y[i] - acc)
        out[i] = acc
    return out


SM = np.stack([causal_ewma(np.log(AMP[:, :, b] + 1e-9), SMOOTH_TAU)
               for b in range(NB)], axis=2)
MU = SM.reshape(-1, NCH * NB).mean(0).reshape(NCH, NB)
SD = SM.reshape(-1, NCH * NB).std(0).reshape(NCH, NB) + 1e-12
mean_log = SM.mean(axis=1)
edge = int(round(EDGE_S * FS))
sl0 = slice(edge, len(X) - edge)
STATE = ((mean_log - mean_log[sl0].mean(0)) / (mean_log[sl0].std(0) + 1e-12))[sl0][::DS_STATE]
T_STATE = T_ALL[sl0][::DS_STATE]
DT_S = float(T_STATE[1] - T_STATE[0])

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
traj = traj[:len(t_r)]

m0 = int(np.argmin(np.abs(T_ALL - t_r[0])))
m1 = int(np.argmin(np.abs(T_ALL - t_r[-1])))
win = slice(m0, m1 + 1)
t_full = T_ALL[win]
AMPW, PHW, TRUEW = AMP[win], PH[win], TRUE[win]
rng = np.random.default_rng(0)

A_MODEL = np.column_stack([np.interp(t_full, t_r, traj[:, b]) for b in range(NB)])
A_SHUF = np.column_stack([np.interp(t_full, t_r, traj[rng.permutation(len(traj)), b])
                          for b in range(NB)])
A_FLAT = np.zeros_like(A_MODEL)

# ---------------------------------------------------------------------------
# The two drives
# ---------------------------------------------------------------------------
# 30-phase drive: every channel keeps its own measured phase.
PHI_FULL = PHW

# 5-phase drive: one phase per band, the circular mean across channels, with a
# fitted constant offset per channel.  If the six regions share a phase
# organisation, this reconstructs all six from a sixth of the information.
PHI_SHARED = np.zeros((len(t_full), NB))
OFFSET = np.zeros((NCH, NB))
for b in range(NB):
    z = np.exp(1j * PHW[:, :, b])
    ref = np.angle(z.mean(axis=1))
    PHI_SHARED[:, b] = ref
    for c in range(NCH):
        OFFSET[c, b] = np.angle(np.mean(np.exp(1j * (PHW[:, c, b] - ref))))
plv = np.array([[float(np.abs(np.mean(np.exp(1j * (PHW[:, c, b] - PHI_SHARED[:, b])))))
                 for b in range(NB)] for c in range(NCH)])

print("=" * 78)
print("DRIVEN RECONSTRUCTION -- non-autonomous by choice; the drive is declared")
print("=" * 78)
print(f"window {t_full[0]:+.1f}..{t_full[-1]:+.1f}s, {len(t_full)} samples, "
      f"{NCH} channels\n")
print("phase locking of each channel to the shared per-band phase (PLV):")
print(f"{'channel':>14} | " + " ".join(f"{n[:3]:>5}" for n in BAND_NAMES))
for c, lab in enumerate(LABELS):
    print(f"{lab:>14} | " + " ".join(f"{plv[c, b]:5.2f}" for b in range(NB)))
print(f"{'mean':>14} | " + " ".join(f"{plv[:, b].mean():5.2f}" for b in range(NB)))
print("  PLV ~ 1 means the channel follows the shared phase up to a constant "
      "offset;\n  PLV ~ 0 means it does not and only its own phase will "
      "reconstruct it.\n")


def resynth(A_state, phi, per_channel_phase):
    out = np.zeros((len(t_full), NCH))
    for c in range(NCH):
        for b in range(NB):
            amp = np.exp(MU[c, b] + SD[c, b] * A_state[:, b])
            ph = phi[:, c, b] if per_channel_phase else phi[:, b] + OFFSET[c, b]
            out[:, c] += amp * np.cos(ph)
    return out


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


cases = {
    "30-phase drive + model amp": (A_MODEL, PHI_FULL, True, 30),
    "30-phase drive + FLAT amp": (A_FLAT, PHI_FULL, True, 30),
    "30-phase drive + SHUFFLED amp": (A_SHUF, PHI_FULL, True, 30),
    " 5-phase drive + model amp": (A_MODEL, PHI_SHARED, False, 5),
    " 5-phase drive + FLAT amp": (A_FLAT, PHI_SHARED, False, 5),
}
recs, scores = {}, {}
print(f"{'configuration':>32} {'drive dim':>10} | {'mean R2':>8}")
print("-" * 56)
for k, (A, phi, pc, dim) in cases.items():
    rec = resynth(A, phi, pc)
    recs[k] = rec
    scores[k] = float(np.mean([r2(TRUEW[:, c], rec[:, c]) for c in range(NCH)]))
    print(f"{k:>32} {dim:10d} | {scores[k]:8.3f}")

g30 = scores["30-phase drive + model amp"] - scores["30-phase drive + FLAT amp"]
g5 = scores[" 5-phase drive + model amp"] - scores[" 5-phase drive + FLAT amp"]
cost = scores["30-phase drive + model amp"] - scores[" 5-phase drive + model amp"]
print(f"\nmodel's own contribution, 30-phase drive : {g30:+.3f}")
print(f"model's own contribution,  5-phase drive : {g5:+.3f}")
print(f"cost of compressing the drive 30 -> 5    : {cost:+.3f}")

print(f"\n{'channel':>14} | {'30-phase':>9} {'5-phase':>9} {'flat(30)':>9}")
per_ch = {}
for c, lab in enumerate(LABELS):
    a = r2(TRUEW[:, c], recs["30-phase drive + model amp"][:, c])
    b_ = r2(TRUEW[:, c], recs[" 5-phase drive + model amp"][:, c])
    f_ = r2(TRUEW[:, c], recs["30-phase drive + FLAT amp"][:, c])
    per_ch[lab] = {"drive30": a, "drive5": b_, "flat30": f_}
    print(f"{lab:>14} | {a:9.3f} {b_:9.3f} {f_:9.3f}")

# ---------------------------------------------------------------------------
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False})
OI = {"verm": "#D55E00", "blue": "#0072B2", "yellow": "#F0E442"}
fig, axes = plt.subplots(NCH, 1, figsize=(11, 1.5 * NCH), sharex=True)
# Draw a window CENTRED on onset, not the first N samples: the record starts at
# -8 s, so "the first 8 seconds" ends exactly at t=0 and hides the entire event.
SHOW_FROM, SHOW_TO = -3.0, 10.0
show = slice(int(np.argmin(np.abs(t_full - SHOW_FROM))),
             int(np.argmin(np.abs(t_full - SHOW_TO))))
for c, lab in enumerate(LABELS):
    ax = axes[c]
    ax.axvspan(1, 8, color=OI["yellow"], alpha=0.2, lw=0)
    ax.plot(t_full[show], TRUEW[show, c], color="0.3", lw=0.5, label="recording")
    ax.plot(t_full[show], recs["30-phase drive + model amp"][show, c],
            color=OI["verm"], lw=0.5, alpha=0.85,
            label=f"30-phase drive (R2={per_ch[lab]['drive30']:.2f})")
    # The FLAT-amplitude trace belongs on these axes.  It is the same drive
    # with the model deleted -- a constant amplitude -- and it looks almost
    # identical.  That is the point: a waveform overlay cannot distinguish a
    # good amplitude model from no amplitude model at all, because the eye
    # reads peak ALIGNMENT and alignment comes entirely from the injected
    # phase.  Showing only the model trace here would be flattering it by
    # construction.
    ax.plot(t_full[show], recs["30-phase drive + FLAT amp"][show, c],
            color="#7B3294", lw=0.5, alpha=0.75,
            label=f"30-phase drive + NO model (R2={per_ch[lab]['flat30']:.2f})")
    ax.set_ylabel(lab, fontsize=7)
    ax.legend(ncol=3, frameon=False, fontsize=6, loc="upper right")
axes[-1].set_xlabel("recording time (s), onset at 0")
axes[0].set_title("Driven reconstruction: orange (with model) and purple "
                  "(model deleted, constant amplitude) are nearly identical -- "
                  "the injected phase, not the model, is doing the work",
                  loc="left", fontweight="bold", fontsize=8.5)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(RESULT_DIR / f"driven_reconstruction.{ext}", dpi=250,
                bbox_inches="tight")
plt.close(fig)

ok = print_check(scores["30-phase drive + model amp"] > 0.5,
                 "driven model reconstructs the six traces "
                 f"({scores['30-phase drive + model amp']:.3f})")
ok &= print_check(g30 > 0, f"model amplitude beats flat under the same drive "
                           f"({g30:+.3f})")
print_check(scores[" 5-phase drive + model amp"] > scores["30-phase drive + FLAT amp"],
            "a 5-dimensional drive beats a 30-dimensional drive with no model "
            f"({scores[' 5-phase drive + model amp']:.3f} vs "
            f"{scores['30-phase drive + FLAT amp']:.3f})")

save_metrics(RESULT_DIR / "metrics.json", {
    "task": "non-autonomous (observer) reconstruction of all six recordings",
    "passed": bool(ok),
    "note": "t enters through an injected phase drive, so this is data "
            "assimilation, not prediction: it cannot run past the recording. "
            "Reported R2 is mostly the drive; the model's contribution is the "
            "difference from the FLAT row under the same drive.",
    "reconstruction_r2": scores,
    "model_contribution": {"drive30": g30, "drive5": g5},
    "drive_compression_cost": cost,
    "plv_to_shared_phase": {lab: dict(zip(BAND_NAMES, plv[c].tolist()))
                            for c, lab in enumerate(LABELS)},
    "phase_offsets_rad": {lab: dict(zip(BAND_NAMES, OFFSET[c].tolist()))
                          for c, lab in enumerate(LABELS)},
    "per_channel": per_ch,
})
print(f"[DONE] {RESULT_DIR}")
