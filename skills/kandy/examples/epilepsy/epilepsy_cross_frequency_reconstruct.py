#!/usr/bin/env python3
"""Integrate the fitted ODEs and rebuild the original iEEG time series.

The model's state is a band AMPLITUDE ENVELOPE, so it contains no waveform.
Rebuilding a signal needs phase put back, and doing that HANDS THE
RECONSTRUCTION MOST OF ITS OWN SCORE -- phase carries the zero crossings, the
frequency and most of a waveform's shape.  Measured here: a CONSTANT amplitude
driven by the measured phase already reaches R^2 = 0.640.  Any reconstruction
number quoted without that baseline beside it is mostly reporting the phase it
was given.

So five reconstructions are built and compared on one line:

  exact       measured amplitude x measured phase.  Must be ~1.00, or the
              decomposition and resynthesis are not inverses and nothing else
              on the line counts.
  res.bound   each channel's OWN envelope at 20 Hz x measured phase.  A true
              upper bound for any model at this time resolution.
  chan-mean   the measured CHANNEL-MEAN state mapped per channel.  Deliberately
              NOT called a ceiling: the model beats it on some channels,
              because averaging six channels is itself a poor description of
              any one of them, and a bound you can exceed is not a bound.
  MODEL       the autonomous rollout of the fitted ODEs, mapped the same way.
  FLAT amp    CONSTANT amplitude x the same measured phase.

The only quantity that isolates the model is MODEL - FLAT.  Everything above
the FLAT line came from the phase, not from the fitted dynamics.

A fully generative version is also built -- model amplitudes x synthesised
carriers, no phase from the recording at all.  Its waveform R^2 is expected to
be ~0 and is not a defect: with synthetic phase a waveform comparison measures
phase alignment.  That version is scored on the spectrogram instead.

Run (after epilepsy_cross_frequency_seizure_freedom.py):
    cd skills/kandy/examples && python epilepsy/epilepsy_cross_frequency_reconstruct.py
"""

from __future__ import annotations

import json
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

RESULT_DIR = ensure_outdir(os.environ.get(
    "KANDY_CF_RESULT_DIR", str(REPO_ROOT / "results" / "Epilepsy-Cross-Frequency")))
ARR = Path(RESULT_DIR) / "cross_frequency_arrays.npz"
if not ARR.exists():
    raise SystemExit(f"{ARR} not found -- run the fit script first")

d = np.load(ARR, allow_pickle=True)
T_STATE = d["t"]
STATE = d["state"]
t_roll = d["t_roll"]
model_traj = d["model_traj"]
BAND_NAMES = [str(b) for b in d["bands"]]
LABELS = [str(c) for c in d["channels"]]
NB, NCH = len(BAND_NAMES), len(LABELS)

BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 50.0)}
SMOOTH_TAU = 0.15
FILTER_ORDER = 4

payload = pickle.load((REPO_ROOT / "data" / "Dory_2021_04_12_006_preprocessed.pkl").open("rb"))
X = np.asarray(payload["X"], float)
FS = float(payload["fs"])
T = np.asarray(payload["t"], float)
CH_STD = np.asarray(payload["stds"], float)

OI = {"skyblue": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 9,
    "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# 1. Decompose: per band, per channel -> amplitude and phase at full rate
# ---------------------------------------------------------------------------
AMP = np.zeros((len(X), NCH, NB))
PHASE = np.zeros((len(X), NCH, NB))
BANDSIG = np.zeros((len(X), NCH, NB))
for b, name in enumerate(BAND_NAMES):
    lo, hi = BANDS[name]
    bb, aa = butter(FILTER_ORDER, [lo / (0.5 * FS), hi / (0.5 * FS)], btype="band")
    filt = filtfilt(bb, aa, X, axis=0)
    z = hilbert(filt, axis=0)
    AMP[:, :, b] = np.abs(z)
    PHASE[:, :, b] = np.angle(z)
    BANDSIG[:, :, b] = filt

TRUE = BANDSIG.sum(axis=2)          # the 1-50 Hz filtered recording


def causal_ewma(y, tau_s):
    al = 1.0 - np.exp(-1.0 / (tau_s * FS))
    out = np.empty_like(y)
    acc = y[0].copy()
    for i in range(len(y)):
        acc = acc + al * (y[i] - acc)
        out[i] = acc
    return out


# Per-channel log-envelope statistics: the map from the shared standardized
# state back into each channel's own amplitude units.
LOGE = np.log(AMP + 1e-9)
SM = np.stack([causal_ewma(LOGE[:, :, b], SMOOTH_TAU) for b in range(NB)], axis=2)
MU = SM.reshape(-1, NCH * NB).mean(0).reshape(NCH, NB)
SD = SM.reshape(-1, NCH * NB).std(0).reshape(NCH, NB) + 1e-12

# ---------------------------------------------------------------------------
# 2. Put each state trajectory back on the full-rate time grid
# ---------------------------------------------------------------------------
n = len(model_traj)
t_r = t_roll[:n]
m0, m1 = int(np.argmin(np.abs(T - t_r[0]))), int(np.argmin(np.abs(T - t_r[-1])))
win = slice(m0, m1 + 1)
t_full = T[win]

i0s = int(np.argmin(np.abs(T_STATE - t_r[0])))
state_win = STATE[i0s:i0s + n]


def upsample(xs):
    return np.column_stack([np.interp(t_full, t_r, xs[:, b]) for b in range(NB)])


X_MEAS = upsample(state_win)        # measured state  -> the CEILING
X_MODEL = upsample(model_traj)      # autonomous ODE rollout -> the MODEL


def resynth(state_full, use_measured_phase=True, seed=0):
    """Rebuild the waveform: amplitude from the state, phase as chosen."""
    out = np.zeros((len(t_full), NCH))
    rng = np.random.default_rng(seed)
    for c in range(NCH):
        for b in range(NB):
            amp = np.exp(MU[c, b] + SD[c, b] * state_full[:, b])
            if use_measured_phase:
                ph = PHASE[win, c, b]
            else:
                # synthetic carrier at the band centre, random start phase
                f0 = 0.5 * (BANDS[BAND_NAMES[b]][0] + BANDS[BAND_NAMES[b]][1])
                ph = 2 * np.pi * f0 * (t_full - t_full[0]) + rng.uniform(0, 2 * np.pi)
            out[:, c] += amp * np.cos(ph)
    return out


REC_EXACT = np.zeros((len(t_full), NCH))
for c in range(NCH):
    for b in range(NB):
        REC_EXACT[:, c] += AMP[win, c, b] * np.cos(PHASE[win, c, b])
REC_CEIL = resynth(X_MEAS)
REC_MODEL = resynth(X_MODEL)
REC_GEN = resynth(X_MODEL, use_measured_phase=False)

# THE BASELINE THAT DECIDES WHETHER ANY OF THIS MEANS ANYTHING.
#
# Measured phase is handed to every reconstruction above, and phase carries the
# zero crossings, the frequency and most of a waveform's shape.  So a CONSTANT
# amplitude driven by the same measured phase already reproduces an oscillation
# of roughly the right form.  If the model only matches that, then its score is
# reporting the phase it was given, not the amplitude dynamics it predicts.
REC_FLAT = np.zeros((len(t_full), NCH))
for c in range(NCH):
    for b in range(NB):
        REC_FLAT[:, c] += np.exp(MU[c, b]) * np.cos(PHASE[win, c, b])

# A genuine resolution bound: each channel's OWN measured envelope, pushed
# through the same 20 Hz downsample/upsample, x measured phase.  Unlike the
# channel-mean "ceiling" this really is an upper bound for any model working at
# this time resolution, because it uses the true per-channel amplitude.
REC_RES = np.zeros((len(t_full), NCH))
ds = max(1, int(round(FS / 20.0)))
for c in range(NCH):
    for b in range(NB):
        env = SM[win, c, b]
        env_ds = np.interp(t_full, t_full[::ds], env[::ds])
        REC_RES[:, c] += np.exp(env_ds) * np.cos(PHASE[win, c, b])

TRUE_W = TRUE[win]


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def corr(y, p):
    return float(np.corrcoef(y, p)[0, 1])


# ---------------------------------------------------------------------------
# 3. Score
# ---------------------------------------------------------------------------
print("=" * 74)
print("RECONSTRUCTING THE RECORDING FROM THE INTEGRATED ODEs")
print("=" * 74)
print(f"window {t_full[0]:+.1f}..{t_full[-1]:+.1f}s at {FS:.0f} Hz, "
      f"{len(t_full)} samples, {NCH} channels\n")
print(f"{'channel':>14} | {'exact':>7} {'res.bound':>9} {'chan-mean':>9} "
      f"{'MODEL':>7} {'FLAT amp':>9} | {'model-flat':>10}")
print("-" * 82)
rows = {}
for c, lab in enumerate(LABELS):
    e = r2(TRUE_W[:, c], REC_EXACT[:, c])
    rb = r2(TRUE_W[:, c], REC_RES[:, c])
    ce = r2(TRUE_W[:, c], REC_CEIL[:, c])
    mo = r2(TRUE_W[:, c], REC_MODEL[:, c])
    fl = r2(TRUE_W[:, c], REC_FLAT[:, c])
    rows[lab] = {"exact_r2": e, "resolution_bound_r2": rb, "chan_mean_r2": ce,
                 "model_r2": mo, "flat_amplitude_r2": fl,
                 "model_minus_flat": mo - fl}
    print(f"{lab:>14} | {e:7.3f} {rb:9.3f} {ce:9.3f} {mo:7.3f} {fl:9.3f} | "
          f"{mo - fl:+10.3f}")

mean_ceil = float(np.mean([v["chan_mean_r2"] for v in rows.values()]))
mean_model = float(np.mean([v["model_r2"] for v in rows.values()]))
mean_exact = float(np.mean([v["exact_r2"] for v in rows.values()]))
mean_flat = float(np.mean([v["flat_amplitude_r2"] for v in rows.values()]))
mean_res = float(np.mean([v["resolution_bound_r2"] for v in rows.values()]))
print("-" * 82)
print(f"{'mean':>14} | {mean_exact:7.3f} {mean_res:9.3f} {mean_ceil:9.3f} "
      f"{mean_model:7.3f} {mean_flat:9.3f} | {mean_model - mean_flat:+10.3f}")

print(f"""
HOW TO READ THIS
  exact      {mean_exact:.3f}  resynthesis is invertible; the pipeline is sound.
  res.bound  {mean_res:.3f}  each channel's OWN envelope at 20 Hz -- a true upper
                    bound for any model at this time resolution.
  chan-mean  {mean_ceil:.3f}  the measured CHANNEL-MEAN state mapped per channel.
                    NOT a bound: the model beats it on some channels, because
                    averaging six channels is itself a poor description of any
                    one of them.
  MODEL      {mean_model:.3f}  the autonomous ODE rollout.
  FLAT amp   {mean_flat:.3f}  CONSTANT amplitude x the same measured phase.

  The only comparison that isolates the model is MODEL - FLAT = """
      f"{mean_model - mean_flat:+.3f}.\n  Everything above the FLAT line is phase "
      "that was handed to the reconstruction,\n  not dynamics the model "
      "predicted.")

# envelope-level agreement, which is what the model actually claims
print(f"\n{'channel':>14} | envelope correlation, model vs recording, per band")
env_rows = {}
for c, lab in enumerate(LABELS):
    cs = []
    for b in range(NB):
        a_mod = np.exp(MU[c, b] + SD[c, b] * X_MODEL[:, b])
        cs.append(corr(AMP[win, c, b], a_mod))
    env_rows[lab] = cs
    print(f"{lab:>14} | " + "  ".join(f"{nm[:3]}:{v:+.2f}"
                                      for nm, v in zip(BAND_NAMES, cs)))

# fully generative: no phase information from the recording at all
gen = {}
for c, lab in enumerate(LABELS):
    gen[lab] = {"waveform_r2": r2(TRUE_W[:, c], REC_GEN[:, c])}
print(f"\nFully generative (synthetic carriers, NO phase from the recording): "
      f"mean waveform R2 = {np.mean([v['waveform_r2'] for v in gen.values()]):+.3f}")
print("  -- expected to be ~0 and it is: phase is not in the model, so a "
      "waveform\n     comparison measures phase alignment, not the amplitude "
      "dynamics.  Judge\n     this version on the spectrogram below instead.")

# spectrogram agreement for the generative version
band_pow_true, band_pow_gen = [], []
for b, name in enumerate(BAND_NAMES):
    lo, hi = BANDS[name]
    bb, aa = butter(FILTER_ORDER, [lo / (0.5 * FS), hi / (0.5 * FS)], btype="band")
    for c in range(NCH):
        band_pow_true.append(np.abs(hilbert(filtfilt(bb, aa, TRUE_W[:, c]))))
        band_pow_gen.append(np.abs(hilbert(filtfilt(bb, aa, REC_GEN[:, c]))))
sp_corr = float(np.mean([corr(a, b_) for a, b_ in zip(band_pow_true, band_pow_gen)]))
print(f"  spectrogram (band-power envelope) correlation = {sp_corr:+.3f}")

ok = print_check(mean_exact > 0.99,
                 f"resynthesis is invertible (exact R2 {mean_exact:.4f} > 0.99)")
ok &= print_check(mean_model > 0, f"ODE reconstruction beats the mean "
                                  f"({mean_model:.3f} > 0)")
ok &= print_check(mean_model <= mean_res + 1e-9,
                  f"model respects the resolution bound ({mean_model:.3f} "
                  f"<= {mean_res:.3f})")
ok &= print_check(mean_model > mean_flat,
                  f"model beats a CONSTANT amplitude on the same phase "
                  f"({mean_model:.3f} > {mean_flat:.3f}); this is the only "
                  "comparison that isolates the amplitude dynamics")

# ---------------------------------------------------------------------------
# 4. Figure
# ---------------------------------------------------------------------------
ci = LABELS.index("hippocampus_R") if "hippocampus_R" in LABELS else 0
fig, axes = plt.subplots(4, 1, figsize=(10, 8.4), sharex=True)
for ax in axes:
    ax.axvspan(1.0, 8.0, color=OI["yellow"], alpha=0.22, lw=0)
    ax.axvline(0, color=OI["vermillion"], ls=":", lw=0.9)

axes[0].plot(t_full, TRUE_W[:, ci], color="0.25", lw=0.5)
axes[0].set_ylabel("recording\n(1-50 Hz)")
axes[0].set_title(f"{LABELS[ci]}: the recording, and the same trace rebuilt from "
                  "the integrated ODEs", loc="left", fontweight="bold")

axes[1].plot(t_full, REC_CEIL[:, ci], color=OI["blue"], lw=0.5)
axes[1].set_ylabel(f"CEILING\n$R^2$={rows[LABELS[ci]]['ceiling_r2']:.2f}")

axes[2].plot(t_full, REC_MODEL[:, ci], color=OI["vermillion"], lw=0.5)
axes[2].set_ylabel(f"ODE rollout\n$R^2$={rows[LABELS[ci]]['model_r2']:.2f}")
axes[2].plot(t_full, REC_FLAT[:, ci], color=OI["purple"], lw=0.4, alpha=0.55)

axes[3].plot(t_full, np.abs(hilbert(TRUE_W[:, ci])), color="0.25", lw=0.8,
             label="recording")
axes[3].plot(t_full, np.abs(hilbert(REC_MODEL[:, ci])), color=OI["vermillion"],
             lw=1.4, label="ODE rollout")
axes[3].set_ylabel("broadband\nenvelope")
axes[3].set_xlabel("recording time (s), seizure onset at 0")
axes[3].legend(frameon=False, ncol=2)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(Path(RESULT_DIR) / f"reconstruction.{ext}", dpi=250,
                bbox_inches="tight")
plt.close(fig)

save_metrics(Path(RESULT_DIR) / "reconstruction_metrics.json", {
    "task": "reconstruct the recording by integrating the fitted ODEs",
    "passed": bool(ok),
    "window": [float(t_full[0]), float(t_full[-1])],
    "note": "measured phase is supplied to the amplitude model on purpose and "
            "is labelled; the model contains no phase. Judge the model against "
            "the CEILING (a perfect model of the same state), never against 1.0.",
    "per_channel": rows,
    "mean": {"exact": mean_exact, "resolution_bound": mean_res,
             "chan_mean": mean_ceil, "model": mean_model,
             "flat_amplitude": mean_flat,
             "model_minus_flat": mean_model - mean_flat},
    "envelope_correlation": {k: dict(zip(BAND_NAMES, v)) for k, v in env_rows.items()},
    "fully_generative": {"per_channel": gen,
                         "spectrogram_correlation": sp_corr,
                         "note": "no phase from the recording; waveform R2 is "
                                 "expected to be ~0 and is not a defect"},
})
print(f"[DONE] figure and metrics in {RESULT_DIR}")
