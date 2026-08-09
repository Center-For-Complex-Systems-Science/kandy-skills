#!/usr/bin/env python3
"""Stochastic replay: the fitted ODE plus fitted noise.

Every deterministic rollout in this directory is a smooth line, and the reason
is not the model form -- removing the relaxation term changed the residual
wiggle from 0.007 to 0.008, and fitting a fast derivative instead diverges in
under a second at any regularization.  The fast component of the band envelope
is simply not predictable from the state (k-NN R^2 ~ 0 against a Lorenz-96
control of 0.86), so the correct output of a deterministic field is the
conditional mean, which is smooth.

The honest way to get a trace that looks like the recording is therefore to say
which part is deterministic and which is not, and simulate both.  The noise is
FITTED, not chosen:

    residual = data - its own 5 s trend
    std        delta 0.774, theta 0.641, alpha 0.465, beta 0.301, gamma 0.300
    autocorr   0.20-0.25 s in every band (lag-1 correlation ~0.9)
    cross-band mean |correlation| 0.288, alpha-beta 0.40, beta-gamma 0.57

Two consequences fix the naive choice:

  * WHITE noise is wrong.  With a 0.25 s autocorrelation the residual is
    strongly coloured, and white noise pushed through a tau = 12-20 s system
    would be filtered to almost nothing.  An Ornstein-Uhlenbeck process at
    tau_n = 0.25 s is the right generator.
  * INDEPENDENT per-band noise is wrong.  The fast bands are correlated at
    0.40-0.57, so the driving noise needs the full 5x5 covariance via its
    Cholesky factor, or alpha/beta/gamma will wander apart in a way the
    recording never does.

Two injection points are compared, because they are different claims:

    process      dx = F(x) dt + eta      the noise is part of the dynamics and
                                         gets low-passed by tau
    observation  y  = x + eta            the field carries the slow state and
                                         the texture is unmodelled variability

VALIDATION IS ON ENSEMBLE STATISTICS, NOT APPEARANCE.  Noise makes any model
look convincing; a smooth wrong model plus tuned noise passes the eye test and
adds nothing.  So the ensemble is scored on variance, autocorrelation time,
spectral slope and the cross-band correlation matrix against the recording, and
a single realization is never offered as evidence.

Run:
    cd skills/kandy/examples && python epilepsy/epilepsy_stochastic_replay.py
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
from scipy.signal import savgol_filter, welch

from _utils import ensure_outdir, print_check, save_metrics


def ef(n, d):
    return float(os.environ.get(n, d))


def ei(n, d):
    return int(os.environ.get(n, d))


SOURCE = REPO_ROOT / "results" / "Epilepsy-Cross-Frequency"
RESULT_DIR = Path(ensure_outdir(REPO_ROOT / "results" / "Epilepsy-Stochastic-Replay"))
arr = np.load(SOURCE / "cross_frequency_arrays.npz", allow_pickle=True)
met = json.loads((SOURCE / "metrics.json").read_text())
BAND = [str(v) for v in arr["bands"]]
NB = len(BAND)
BI = {n: i for i, n in enumerate(BAND)}
DELTA, BETA = BI["delta"], BI["beta"]
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

N_REAL = ei("KANDY_ST_N", 200)
ROLL_START = ef("KANDY_ST_START", -8.0)
ROLL_SEC = ef("KANDY_ST_SEC", 25.0)
TREND_S = 5.0
SEED = 42
rng = np.random.default_rng(SEED)


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


def drift(x):
    return (from_inner(T_SCALE * f_closed(x)) - np.atleast_2d(x)) / TAU


# ---------------------------------------------------------------------------
# 1. Fit the noise from the residual
# ---------------------------------------------------------------------------
w = int(TREND_S / DT) | 1
RES = STATE - savgol_filter(STATE, w, 2, axis=0)
SIG = np.cov(RES.T)
CHOL = np.linalg.cholesky(SIG + 1e-9 * np.eye(NB))
tau_n = []
for b in range(NB):
    x = RES[:, b] - RES[:, b].mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac /= ac[0]
    tau_n.append(float(np.argmax(ac < 1 / np.e) * DT) if np.any(ac < 1 / np.e) else DT)
TAU_N = float(np.mean(tau_n))

print("=" * 84)
print("STOCHASTIC REPLAY -- fitted ODE + fitted noise")
print("=" * 84)
print(f"noise fitted from the residual about a {TREND_S:.0f}s trend")
print(f"  per-band std      " + " ".join(f"{n}={RES[:, b].std():.2f}"
                                         for b, n in enumerate(BAND)))
print(f"  OU timescale      {TAU_N:.2f}s (white noise would be wrong)")
print(f"  cross-band corr   mean |off-diag| "
      f"{np.mean(np.abs(np.corrcoef(RES.T)[~np.eye(NB, dtype=bool)])):.3f} "
      "(independent noise would be wrong)")

i0 = int(np.argmin(np.abs(T - ROLL_START)))
nst = min(int(round(ROLL_SEC / DT)), len(T) - i0 - 1)
t_r = T[i0:i0 + nst + 1]
ref = STATE[i0:i0 + nst + 1]

pre_c = STATE[(T >= PRE[0]) & (T <= PRE[1])].mean(0)
post_c = STATE[(T >= POST[0]) & (T <= POST[1])].mean(0)
axis = post_c - pre_c
an = float(np.linalg.norm(axis))
uax = axis / an


def tscore(x):
    return (np.atleast_2d(x) - pre_c) @ uax / an


def ou_noise(n_real, n_step):
    """Coloured, cross-correlated noise matching the fitted residual."""
    a = float(np.exp(-DT / TAU_N))
    s = np.sqrt(max(1.0 - a * a, 1e-9))
    e = np.zeros((n_step + 1, n_real, NB))
    e[0] = rng.standard_normal((n_real, NB)) @ CHOL.T
    for k in range(n_step):
        e[k + 1] = a * e[k] + s * (rng.standard_normal((n_real, NB)) @ CHOL.T)
    return e


def simulate(mode, n_real, stim=None):
    """mode: 'deterministic' | 'process' | 'observation'."""
    x = np.tile(STATE[i0], (n_real, 1))
    out = np.empty((nst + 1, n_real, NB))
    out[0] = x
    eta = ou_noise(n_real, nst) if mode != "deterministic" else None
    for k in range(nst):
        tn = float(t_r[k])

        def d_(y, tt):
            g = drift(y)
            if stim is not None and stim[0] <= tt < stim[0] + stim[1]:
                g[:, BETA] += stim[2] / (1.0 + np.exp(-(y[:, BETA] - THRESH[BETA])))
            return g

        k1 = d_(x, tn)
        k2 = d_(x + 0.5 * DT * k1, tn + 0.5 * DT)
        k3 = d_(x + 0.5 * DT * k2, tn + 0.5 * DT)
        k4 = d_(x + DT * k3, tn + DT)
        x = x + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if mode == "process":
            # OU increment enters the dynamics and is low-passed by tau
            x = x + (eta[k + 1] - eta[k]) * np.sqrt(2.0 * TAU_N / np.maximum(TAU, 1e-6))
        x = np.clip(x, -8, 8)
        out[k + 1] = x
    if mode == "observation":
        out = out + eta            # texture added at the output, state untouched
    return out


# ---------------------------------------------------------------------------
# 2. Ensemble statistics vs the recording -- the actual validation
# ---------------------------------------------------------------------------
def stats(a):
    """a: (n_step, NB) one realization."""
    tr = savgol_filter(a, w, 2, axis=0)
    r = a - tr
    ac_t = []
    for b in range(NB):
        x = r[:, b] - r[:, b].mean()
        c = np.correlate(x, x, "full")[len(x) - 1:]
        c = c / max(c[0], 1e-12)
        ac_t.append(np.argmax(c < 1 / np.e) * DT if np.any(c < 1 / np.e) else np.nan)
    cm = np.corrcoef(r.T)
    off = cm[~np.eye(NB, dtype=bool)]
    f, P = welch(r[:, BETA], 1.0 / DT, nperseg=min(128, len(r)))
    m = (f > 0.1) & (f < 5)
    slope = np.polyfit(np.log10(f[m] + 1e-9), np.log10(P[m] + 1e-18), 1)[0]
    return {"wiggle_std": float(r.std(axis=0).mean()),
            "autocorr_s": float(np.nanmean(ac_t)),
            "crossband": float(np.mean(np.abs(off))),
            "spec_slope": float(slope)}


ref_st = stats(ref)
print(f"\n[VALIDATION] ensemble statistics, not appearance ({N_REAL} realizations)")
print(f"{'':>16} {'wiggle std':>11} {'autocorr':>9} {'cross-band':>11} {'spec slope':>11}")
print(f"{'RECORDING':>16} {ref_st['wiggle_std']:11.3f} {ref_st['autocorr_s']:8.2f}s "
      f"{ref_st['crossband']:11.3f} {ref_st['spec_slope']:11.2f}")
ens = {}
for mode in ("deterministic", "process", "observation"):
    n = 1 if mode == "deterministic" else N_REAL
    sim = simulate(mode, n)
    st = [stats(sim[:, i, :]) for i in range(n)]
    ens[mode] = {k: float(np.mean([s[k] for s in st])) for k in ref_st}
    ens[mode]["sim"] = sim
    e = ens[mode]
    print(f"{mode:>16} {e['wiggle_std']:11.3f} {e['autocorr_s']:8.2f}s "
          f"{e['crossband']:11.3f} {e['spec_slope']:11.2f}")


def err(m):
    e = ens[m]
    return float(np.mean([abs(e[k] - ref_st[k]) / max(abs(ref_st[k]), 1e-9)
                          for k in ("wiggle_std", "autocorr_s", "crossband")]))


best_mode = min(("process", "observation"), key=err)
print(f"\n  relative error vs the recording: process {err('process'):.2f}, "
      f"observation {err('observation'):.2f}  ->  {best_mode} wins")

# ---------------------------------------------------------------------------
# 3. P(seizure) -- the deterministic single-trajectory claim, redone
# ---------------------------------------------------------------------------
s_ref = tscore(ref).ravel()
obs_rise = float(s_ref.max() - s_ref[:int(1 / DT)].mean())
TH_S = 0.5 * obs_rise
print(f"\n[P(SEIZURE)] threshold {TH_S:+.3f}; {N_REAL} realizations each")
p_res = {}
for label, stim in (("no stimulation", None),
                    ("closed-loop beta a=-1, +2..12s", (2.0, 10.0, -1.0))):
    sim = simulate(best_mode, N_REAL, stim=stim)
    sc = np.stack([tscore(sim[:, i, :]).ravel() for i in range(N_REAL)])
    base = sc[:, :int(1 / DT)].mean(axis=1)
    rise = sc.max(axis=1) - base
    p = float(np.mean(rise >= TH_S))
    p_res[label] = {"p_seizure": p, "mean_rise": float(rise.mean()),
                    "sd_rise": float(rise.std())}
    print(f"  {label:>32}: P(seizure) = {p:.3f}, rise {rise.mean():+.2f} "
          f"+/- {rise.std():.2f}")
d_p = p_res["no stimulation"]["p_seizure"] - \
    p_res["closed-loop beta a=-1, +2..12s"]["p_seizure"]
print(f"  {'reduction in P(seizure)':>32}: {d_p:+.3f}")

# ---------------------------------------------------------------------------
# 4. Figure
# ---------------------------------------------------------------------------
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
sim = ens[best_mode]["sim"]
det = ens["deterministic"]["sim"][:, 0, :]
fig, axes = plt.subplots(NB + 1, 1, figsize=(11, 1.55 * (NB + 1)), sharex=True)
for b in range(NB):
    a = axes[b]
    a.axvspan(1, 8, color="#F0E442", alpha=0.18, lw=0)
    for i in range(min(12, N_REAL)):
        a.plot(t_r, sim[:, i, b], color="#0072B2", lw=0.5, alpha=0.28)
    a.plot(t_r, ref[:, b], color="0.2", lw=1.0, label="recording")
    a.plot(t_r, det[:, b], color="#D55E00", lw=1.8, label="deterministic")
    a.plot(t_r, sim[:, :, b].mean(1), color="#0072B2", lw=1.6, ls="--",
           label=f"stochastic mean ({best_mode})")
    a.axvline(0, color="0.4", ls=":", lw=0.8)
    a.set_ylabel(BAND[b], fontweight="bold")
axes[0].legend(ncol=3, frameon=False, fontsize=7)
a = axes[-1]
a.axvspan(1, 8, color="#F0E442", alpha=0.18, lw=0)
scq = np.stack([tscore(sim[:, i, :]).ravel() for i in range(N_REAL)])
a.fill_between(t_r, np.percentile(scq, 5, axis=0), np.percentile(scq, 95, axis=0),
               color="#0072B2", alpha=0.20, lw=0, label="stochastic 5-95%")
a.plot(t_r, s_ref, color="0.2", lw=1.3, label="recording")
a.plot(t_r, tscore(det).ravel(), color="#D55E00", lw=1.8, label="deterministic")
a.axhline(TH_S + s_ref[:int(1 / DT)].mean(), color="0.4", ls=":", lw=1.1,
          label="seizure threshold")
a.set_ylabel("transition\nscore")
a.set_xlabel("recording time (s), onset at 0")
a.legend(ncol=4, frameon=False, fontsize=7)
axes[0].set_title(
    f"Fitted ODE + fitted {best_mode} noise (OU, tau={TAU_N:.2f}s, "
    f"cross-band covariance) -- {N_REAL} realizations",
    loc="left", fontweight="bold")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(RESULT_DIR / f"stochastic_replay.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)

print()
ok = print_check(abs(ens[best_mode]["wiggle_std"] - ref_st["wiggle_std"])
                 < 0.5 * ref_st["wiggle_std"],
                 f"ensemble reproduces the recording's fluctuation amplitude "
                 f"({ens[best_mode]['wiggle_std']:.3f} vs {ref_st['wiggle_std']:.3f})")
ok &= print_check(abs(ens[best_mode]["crossband"] - ref_st["crossband"]) < 0.15,
                  f"ensemble reproduces the cross-band correlation "
                  f"({ens[best_mode]['crossband']:.3f} vs {ref_st['crossband']:.3f})")
print_check(d_p > 0.05, f"stimulation reduces P(seizure) by more than 0.05 "
                        f"({d_p:+.3f})")

save_metrics(RESULT_DIR / "metrics.json", {
    "task": "stochastic replay: fitted closed-form ODE plus fitted OU noise",
    "noise": {"source": f"residual about a {TREND_S}s trend", "ou_tau_s": TAU_N,
              "per_band_std": RES.std(axis=0).tolist(),
              "covariance": SIG.tolist(),
              "white_noise_rejected_because": "residual autocorrelation is "
                                              f"{TAU_N:.2f}s, not zero",
              "independent_noise_rejected_because": "cross-band correlations "
                                                    "reach 0.57"},
    "recording_stats": ref_st, "ensemble_stats": {k: {kk: vv for kk, vv in v.items()
                                                      if kk != "sim"}
                                                  for k, v in ens.items()},
    "best_injection": best_mode,
    "p_seizure": p_res, "p_reduction": d_p, "n_realizations": N_REAL,
    "caveat": "noise makes any model look convincing; validation is on ensemble "
              "variance, autocorrelation and cross-band correlation, never on a "
              "single realization's appearance.",
})
print(f"[DONE] {RESULT_DIR}/stochastic_replay.png")
