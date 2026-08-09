#!/usr/bin/env python3
"""Seizure-replay figure for the stimulated model: is it actually seizure-free?

``epilepsy_dbs_beta_gamma_actuator.py`` found that a closed-loop negative beta
drive holds the transition score below threshold, but its rollout is 25 s and
the post-pulse trajectory was STILL RISING when the window ended.  With a
monostable field and tau = 12-20 s that is exactly when you must not stop
looking: "did not re-cross the threshold in the 5 s we watched" is a different
claim from "seizure-free".

So this rolls out much longer -- past the end of the recording -- and reports
where the stimulated run ends up rather than where it happened to be when the
plot ran out of x-axis.  The recording is drawn only where it exists; beyond
that the comparison is model against model, and the panel says so.

Three trajectories, band by band:
    recording          the data
    no stimulation     the fitted closed-form ODE, which seizes
    stimulated         the winning condition from the sweep

Reported for each: peak transition score over the FULL horizon, whether it ever
crosses the seizure threshold, when it crosses if it does, and where it settles.

Run:
    cd skills/kandy/examples && python epilepsy/epilepsy_dbs_seizure_free_replay.py
"""

from __future__ import annotations

import json
import os
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

from _utils import ensure_outdir, print_check, save_metrics


def ef(n, d):
    return float(os.environ.get(n, d))


SOURCE = REPO_ROOT / "results" / "Epilepsy-Cross-Frequency"
DBS = REPO_ROOT / "results" / "Epilepsy-DBS-Beta-Gamma"
RESULT_DIR = Path(ensure_outdir(DBS))
arr = np.load(SOURCE / "cross_frequency_arrays.npz", allow_pickle=True)
met = json.loads((SOURCE / "metrics.json").read_text())

BAND = [str(v) for v in arr["bands"]]
NB = len(BAND)
BI = {n: i for i, n in enumerate(BAND)}
DELTA, BETA, GAMMA = BI["delta"], BI["beta"], BI["gamma"]
T = arr["t"]
STATE = arr["state"]
DT = float(T[1] - T[0])
TAU = np.asarray(met["model_definition"]["tau_s"], float)
X_C = np.asarray(met["model_definition"]["x_c"], float)
X_R = np.asarray(met["model_definition"]["x_r"], float)
T_SCALE = np.asarray(met["model_definition"]["t_scale"], float)
THRESH = np.asarray(met["model_definition"]["relu_thresholds"], float)
PRE = tuple(met["windows"]["pre_ictal"])
POST = tuple(met["windows"]["post_ictal"])

# the winning condition, overridable
AMP = ef("KANDY_SF_AMP", -1.0)
STIM_START = ef("KANDY_SF_STIM_START", 2.0)
STIM_DUR = ef("KANDY_SF_STIM_DUR", 10.0)
ROLL_START = ef("KANDY_SF_ROLL_START", -8.0)
HORIZON = ef("KANDY_SF_HORIZON", 90.0)        # far past the recording, on purpose
HK = 1.0
THRESH_FRAC = 0.5


def from_inner(v):
    return X_C + X_R * np.tanh(v / X_R)


def f_closed(x):
    x = np.atleast_2d(np.asarray(x, float))
    d, th, _al, _be, ga = [x[:, i] for i in range(NB)]
    gd = np.maximum(d - THRESH[DELTA], 0.0)
    return np.column_stack([
        -0.6452 * d - 0.2496 * ga + 0.0638,
        0.0940 - 0.6048 * th,
        np.full(len(x), 0.3333),
        np.full(len(x), 0.2306),
        6.1026 - 0.1136 * (8.5784 - 5.9214 * gd) ** 2,
    ])


def Hs(v, b):
    return 1.0 / (1.0 + np.exp(-np.clip(HK * (v - THRESH[b]), -60, 60)))


pre_c = STATE[(T >= PRE[0]) & (T <= PRE[1])].mean(0)
post_c = STATE[(T >= POST[0]) & (T <= POST[1])].mean(0)
axis = post_c - pre_c
an = float(np.linalg.norm(axis))
uax = axis / an


def tscore(x):
    return (np.atleast_2d(x) - pre_c) @ uax / an


i0 = int(np.argmin(np.abs(T - ROLL_START)))
n_rec = len(T) - i0 - 1                       # rows of recording after the start
nst = int(round(HORIZON / DT))
t_roll = T[i0] + np.arange(nst + 1) * DT
s_ref = tscore(STATE[i0:]).ravel()
obs_rise = float(s_ref.max() - s_ref[:max(1, int(1 / DT))].mean())
SEIZE_TH = THRESH_FRAC * obs_rise


def rollout(stim):
    x = STATE[i0].copy()
    tr = np.empty((nst + 1, NB))
    tr[0] = x

    def deriv(y, tt):
        yy = np.atleast_2d(y)
        out = (from_inner(T_SCALE * f_closed(yy)) - yy)[0] / TAU
        if stim and (STIM_START <= tt < STIM_START + STIM_DUR):
            out[BETA] += AMP * float(Hs(yy[0, BETA], BETA))
        return out

    for k in range(nst):
        t0 = float(t_roll[k])
        k1 = deriv(x, t0)
        k2 = deriv(x + 0.5 * DT * k1, t0 + 0.5 * DT)
        k3 = deriv(x + 0.5 * DT * k2, t0 + 0.5 * DT)
        k4 = deriv(x + DT * k3, t0 + DT)
        x = x + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(x).all():
            return tr[:k + 1]
        tr[k + 1] = x
    return tr


off = rollout(False)
on = rollout(True)
s_off, s_on = tscore(off).ravel(), tscore(on).ravel()
base_off = s_off[:max(1, int(1 / DT))].mean()
base_on = s_on[:max(1, int(1 / DT))].mean()

print("=" * 88)
print("IS THE STIMULATED MODEL SEIZURE-FREE?")
print("=" * 88)
print(f"closed-loop beta drive a={AMP:+.2f}, pulse {STIM_START:+.1f}.."
      f"{STIM_START+STIM_DUR:.1f}s")
print(f"horizon {HORIZON:.0f}s -- the recording only covers the first "
      f"{n_rec*DT:.0f}s of it")
print(f"seizure threshold {SEIZE_TH:+.3f} (50% of the observed rise "
      f"{obs_rise:+.3f})\n")


def report(name, s, base):
    peak = float(s.max() - base)
    cross = np.where(s - base >= SEIZE_TH)[0]
    t_cross = float(t_roll[cross[0]]) if len(cross) else None
    post = s[int((STIM_START + STIM_DUR - t_roll[0]) / DT):]
    return {"peak_rise": peak, "crosses": bool(len(cross)),
            "t_cross": t_cross, "end_score": float(s[-1]),
            "post_pulse_peak": float(post.max() - base) if len(post) else np.nan}


r_off, r_on = report("off", s_off, base_off), report("on", s_on, base_on)
print(f"{'':>18} {'peak rise':>10} {'crosses?':>9} {'when':>8} {'end score':>10}")
for nm, r in (("no stimulation", r_off), ("stimulated", r_on)):
    print(f"{nm:>18} {r['peak_rise']:+10.3f} {str(r['crosses']):>9} "
          + (f"{r['t_cross']:+7.1f}s" if r["t_cross"] is not None else f"{'--':>8}")
          + f" {r['end_score']:+10.3f}")
print(f"\npeak AFTER the pulse ends: no-stim {r_off['post_pulse_peak']:+.3f}, "
      f"stimulated {r_on['post_pulse_peak']:+.3f}")

SEIZURE_FREE = bool(not r_on["crosses"])
if SEIZURE_FREE:
    print(f"\n  The stimulated trajectory NEVER crosses the threshold across the "
          f"full {HORIZON:.0f}s,\n  including {HORIZON - STIM_START - STIM_DUR:.0f}s "
          "after the stimulus is switched off.")
else:
    print(f"\n  The stimulated trajectory DELAYS the crossing to "
          f"t={r_on['t_cross']:+.1f}s but does not\n  prevent it -- with a "
          "monostable field the system returns to its only attractor.\n  This is "
          "delay, not termination, and the 25 s window in the sweep was too\n"
          "  short to see it.")

# ---------------------------------------------------------------------------
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(NB + 1, 1, figsize=(11, 1.55 * (NB + 1)), sharex=True)
t_rec = T[i0:]
for b in range(NB):
    a = axes[b]
    a.axvspan(STIM_START, STIM_START + STIM_DUR, color="#56B4E9", alpha=0.18, lw=0)
    a.plot(t_rec, STATE[i0:, b], color="0.55", lw=0.7, label="recording")
    a.plot(t_roll, off[:, b], color="#D55E00", lw=1.5, label="no stimulation")
    a.plot(t_roll, on[:, b], color="#0072B2", lw=1.8, label="stimulated")
    a.axvline(0, color="0.4", ls=":", lw=0.8)
    a.set_ylabel(BAND[b], fontweight="bold")
axes[0].legend(ncol=3, frameon=False, fontsize=7)
a = axes[-1]
a.axvspan(STIM_START, STIM_START + STIM_DUR, color="#56B4E9", alpha=0.18, lw=0)
a.plot(t_rec, s_ref, color="0.35", lw=1.2, label="recording")
a.plot(t_roll, s_off, color="#D55E00", lw=2.0, label="no stimulation")
a.plot(t_roll, s_on, color="#0072B2", lw=2.2, label="stimulated")
a.axhline(SEIZE_TH + base_off, color="0.3", ls=":", lw=1.2,
          label=f"seizure threshold")
a.axvline(t_rec[-1], color="0.6", ls="-.", lw=1.0)
a.text(t_rec[-1], a.get_ylim()[0], "  recording ends", fontsize=6.5,
       color="0.45", va="bottom")
a.axvline(0, color="0.4", ls=":", lw=0.8)
a.set_ylabel("transition\nscore")
a.set_xlabel("recording time (s), seizure onset at 0")
a.legend(ncol=4, frameon=False, fontsize=7)
axes[0].set_title(
    f"Closed-loop beta drive a={AMP:+.1f}, pulse {STIM_START:+.0f}..."
    f"{STIM_START+STIM_DUR:.0f}s (shaded) -- "
    + ("SEIZURE-FREE over the full horizon"
       if SEIZURE_FREE else
       f"delays the crossing to t={r_on['t_cross']:+.1f}s, does not prevent it"),
    loc="left", fontweight="bold")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(RESULT_DIR / f"stimulated_seizure_replay.{ext}", dpi=200,
                bbox_inches="tight")
plt.close(fig)

print()
ok = print_check(r_off["crosses"], "the unstimulated model does seize, so there "
                                   "is something to prevent")
print_check(SEIZURE_FREE, f"the stimulated model never crosses threshold over "
                          f"{HORIZON:.0f}s")
save_metrics(RESULT_DIR / "seizure_free_replay.json", {
    "task": "long-horizon replay of the stimulated vs unstimulated model",
    "condition": {"band": "beta", "mode": "closed-loop", "amp": AMP,
                  "start": STIM_START, "duration": STIM_DUR},
    "horizon_s": HORIZON, "recording_covers_s": float(n_rec * DT),
    "seizure_threshold": SEIZE_TH, "observed_rise": obs_rise,
    "no_stimulation": r_off, "stimulated": r_on,
    "seizure_free": SEIZURE_FREE,
    "note": "the sweep's 25 s window ends while the post-pulse trajectory is "
            "still rising; this horizon is deliberately longer than the "
            "recording so the outcome is observed rather than cropped.",
})
print(f"[DONE] {RESULT_DIR}/stimulated_seizure_replay.png")
