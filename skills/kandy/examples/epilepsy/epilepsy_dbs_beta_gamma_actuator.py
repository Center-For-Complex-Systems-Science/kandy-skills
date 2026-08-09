#!/usr/bin/env python3
"""DBS-style intervention with the actuator on BETA and GAMMA, not delta.

Built on ``epilepsy_cross_frequency_dbs_intervention.py``, which is the right
experiment aimed at the wrong state.  Two things are kept from it because they
are the parts that make the result trustworthy:

  * ``rise`` is the MAXIMUM transition score over the whole rollout, not the
    value at pulse end.  With a finite pulse, a suppression that rebounds pushes
    the max back up and is correctly scored as still seizing.  That makes the
    metric structurally unable to mistake "held down while forced" for
    "terminated".
  * the baseline must SEIZE before any intervention is scored, otherwise every
    knob looks like a cure.

What changes is the actuator.  That script drives delta through a beta->delta
term, and delta is nearly invisible to the quantity being scored: across the
event delta moves 0.22 -> -0.05 while theta/alpha/beta/gamma each move ~1.5 sd.
Measured directly by sweeping the actuator band, best viable suppression was
delta +0.147, theta +0.427, beta +0.441, alpha +0.567, gamma +1.044 -- so the
lever was attached to the one band that cannot move the score.  Its own gate
table agrees: the fitted beta->delta coefficient improves delta's transition
R^2 by +0.001, and 0 of its conditions terminate even at 50x amplitude.

ACTUATORS HERE
--------------
    beta   additive only.  T_beta is a fitted CONSTANT (0.2306), so there is no
           fitted edge to modulate -- a gain form would be multiplying zero.
    gamma  additive AND gain.  T_gamma carries a fitted delta dependence,
           6.1026 - 0.1136*(8.5784 - 5.9214*ReLU(delta-th))^2, so the gain form
           scales a coefficient the data actually supports.

and two control laws for the additive case, which is the open- vs closed-loop
distinction that matters clinically:

    open    dx_b/dt += a * u(t)                    blind periodic stimulation
    closed  dx_b/dt += a * u(t) * H(x_b - theta_b) responsive: only while the
                                                   target band is above
                                                   threshold

VIABILITY IS REPORTED, NOT JUST CHECKED.  At large amplitude the gamma and
alpha drives leave the range the recording spans -- that is the integrator being
pushed off-manifold, not an intervention, and the count is printed so a
"successful" suppression that is off-manifold cannot be mistaken for a result.

Run:
    cd skills/kandy/examples && python epilepsy/epilepsy_dbs_beta_gamma_actuator.py
"""

from __future__ import annotations

import json
import os
import sys
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

from _utils import ensure_outdir, print_check, save_metrics


def ef(n, d):
    return float(os.environ.get(n, d))


def el(n, d):
    return [float(v) for v in os.environ.get(n, d).split(",") if v.strip()]


SOURCE = REPO_ROOT / "results" / "Epilepsy-Cross-Frequency"
RESULT_DIR = Path(ensure_outdir(REPO_ROOT / "results" / "Epilepsy-DBS-Beta-Gamma"))
if not (SOURCE / "cross_frequency_arrays.npz").exists():
    raise SystemExit("run epilepsy_cross_frequency_seizure_freedom.py first")

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
PRE_ICTAL = tuple(met["windows"]["pre_ictal"])
POST_ICTAL = tuple(met["windows"]["post_ictal"])
ROLL_START = ef("KANDY_BG_START", float(met["replay"]["start_time"]))
ROLL_SEC = ef("KANDY_BG_SEC", float(met["replay"]["seconds"]))
THRESH_FRAC = 0.5
HK = ef("KANDY_BG_HK", 1.0)

PULSE_DURS = el("KANDY_BG_DURS", "4,10,33")
STARTS = el("KANDY_BG_STARTS", "-4,-2,0,2,4")
AMPS = el("KANDY_BG_AMPS", "-3,-2,-1,-0.5,-0.25,0.25,0.5,1,2,3")
GAIN_AMPS = el("KANDY_BG_GAIN", "-2,-1,0,0.5,1,2,3,5")


def from_inner(v):
    return X_C + X_R * np.tanh(v / X_R)


def f_closed(x):
    """The printed closed-form equations, vectorized."""
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


def f_gamma_gain(x, g):
    """T_gamma with its FITTED delta-dependence scaled by g (g=1 is intact)."""
    x = np.atleast_2d(np.asarray(x, float))
    gd = np.maximum(x[:, DELTA] - THRESH[DELTA], 0.0)
    return 6.1026 - g * 0.1136 * (8.5784 - 5.9214 * gd) ** 2


def H(v, b):
    z = np.clip(HK * (np.asarray(v) - THRESH[b]), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def rhs_from_f(x, f):
    return (from_inner(T_SCALE * f) - x) / TAU


pre_c = STATE[(T >= PRE_ICTAL[0]) & (T <= PRE_ICTAL[1])].mean(0)
post_c = STATE[(T >= POST_ICTAL[0]) & (T <= POST_ICTAL[1])].mean(0)
axis = post_c - pre_c
an = float(np.linalg.norm(axis))
uax = axis / an


def tscore(x):
    return (np.atleast_2d(x) - pre_c) @ uax / an


i0 = int(np.argmin(np.abs(T - ROLL_START)))
nst = min(int(round(ROLL_SEC / DT)), len(T) - i0 - 1)
t_roll = T[i0:i0 + nst + 1]
ref = STATE[i0:i0 + nst + 1]
s_ref = tscore(ref).ravel()
obs_rise = float(s_ref.max() - s_ref[:max(1, int(1 / DT))].mean())
SEIZE_TH = THRESH_FRAC * obs_rise
LOK = STATE.min(0) - 0.1 * np.ptp(STATE, 0)
HOK = STATE.max(0) + 0.1 * np.ptp(STATE, 0)

print("=" * 90)
print("DBS-STYLE INTERVENTION -- actuator on BETA and GAMMA")
print("=" * 90)
print(f"observed rise {obs_rise:+.3f}; a rollout 'seizes' if its MAXIMUM score "
      f"rises >= {SEIZE_TH:+.3f}")
print("max-over-rollout is deliberate: a suppression that rebounds after the "
      "pulse\nends pushes the max back up and is scored as still seizing.\n")


def rollout(band=None, mode="none", amp=0.0, start=0.0, dur=0.0):
    x = STATE[i0].copy()
    tr = np.empty((nst + 1, NB))
    tr[0] = x

    def deriv(y, tt):
        yy = np.atleast_2d(y)
        f = f_closed(yy)
        p = float(start <= tt < start + dur)
        if mode == "gain":
            f[:, GAMMA] = f_gamma_gain(yy, 1.0 - amp * p)
        out = rhs_from_f(yy, f)[0]
        if mode == "open":
            out[band] += amp * p
        elif mode == "closed":
            out[band] += amp * p * float(H(yy[0, band], band))
        return out

    for k in range(nst):
        t0 = float(t_roll[k])
        k1 = deriv(x, t0)
        k2 = deriv(x + 0.5 * DT * k1, t0 + 0.5 * DT)
        k3 = deriv(x + 0.5 * DT * k2, t0 + 0.5 * DT)
        k4 = deriv(x + DT * k3, t0 + DT)
        x = x + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(x).all():
            tr[k + 1:] = np.nan
            return tr
        tr[k + 1] = x
    return tr


def summarize(tr, dur):
    ok = np.isfinite(tr).all(axis=1)
    n = int(ok.sum())
    t = tr[:n]
    s = tscore(t).ravel()
    base = float(s[:max(1, int(1 / DT))].mean())
    rise = float(s.max() - base)
    inside = np.all((t >= LOK) & (t <= HOK), axis=1)
    # which bands leave the observed range, so an off-manifold "success" is
    # visible rather than hidden inside a boolean
    off = [BAND[b] for b in range(NB)
           if not np.all((t[:, b] >= LOK[b]) & (t[:, b] <= HOK[b]))]
    return {"rise": rise, "complete": bool(n == len(tr)),
            "in_range": float(inside.mean()),
            "viable": bool(n == len(tr) and inside.mean() >= 0.9),
            "off_manifold_bands": off,
            "stops": bool(dur < ROLL_SEC - 2.0),
            "seizes": bool(rise >= SEIZE_TH)}


base_tr = rollout()
base_st = summarize(base_tr, 0.0)
print(f"[BASELINE] closed-form, no stimulation: rise {base_st['rise']:+.3f}, "
      f"seizes={base_st['seizes']}, viable={base_st['viable']}")
if not base_st["seizes"]:
    print("           the unstimulated model does not seize -- every "
          "intervention would\n           trivially 'succeed'. Sweep skipped.")

rows = []
if base_st["seizes"]:
    combos = []
    for b, nm in ((BETA, "beta"), (GAMMA, "gamma")):
        for mode in ("open", "closed"):
            for a, s_, d_ in product(AMPS, STARTS, PULSE_DURS):
                combos.append({"band": nm, "bidx": b, "mode": mode,
                               "amp": a, "start": s_, "dur": d_})
    for a, s_, d_ in product(GAIN_AMPS, STARTS, PULSE_DURS):
        combos.append({"band": "gamma", "bidx": GAMMA, "mode": "gain",
                       "amp": a, "start": s_, "dur": d_})
    print(f"\n[SWEEP] {len(combos)} conditions "
          f"(beta open/closed, gamma open/closed/gain)")
    for c in combos:
        tr = rollout(c["bidx"], c["mode"], c["amp"], c["start"], c["dur"])
        rows.append({**c, **summarize(tr, c["dur"])})

    print(f"\n{'band':>6} {'mode':>7} {'amp':>6} {'start':>6} {'dur':>5} | "
          f"{'rise':>7} {'viable':>7} {'stops':>6} {'off-manifold':>16}")
    for r in sorted(rows, key=lambda q: q["rise"])[:14]:
        print(f"{r['band']:>6} {r['mode']:>7} {r['amp']:+6.2f} {r['start']:+6.1f} "
              f"{r['dur']:5.0f} | {r['rise']:+7.3f} {str(r['viable']):>7} "
              f"{str(r['stops']):>6} {','.join(r['off_manifold_bands'])[:16]:>16}")

    def group(pred, label):
        sub = [r for r in rows if pred(r)]
        if not sub:
            return
        v = [r for r in sub if r["viable"]]
        bb = min(v, key=lambda q: q["rise"]) if v else None
        term = [r for r in v if not r["seizes"]]
        tstop = [r for r in term if r["stops"]]
        print(f"  {label:>28}: {len(v):3d}/{len(sub):3d} viable, best viable rise "
              + (f"{bb['rise']:+.3f}" if bb else "  n/a")
              + f", terminate {len(term):2d} (of which {len(tstop)} with a pulse "
                "that actually stops)")

    print("\n[BY ACTUATOR]  baseline rise "
          f"{base_st['rise']:+.3f}, seizure threshold {SEIZE_TH:+.3f}")
    for nm in ("beta", "gamma"):
        for md in ("open", "closed", "gain"):
            group(lambda r, n=nm, m=md: r["band"] == n and r["mode"] == m,
                  f"{nm} / {md}")

    viable_term = [r for r in rows if r["viable"] and not r["seizes"]]
    real_term = [r for r in viable_term if r["stops"]]
    n_off = sum(1 for r in rows if not r["viable"])
    print(f"\n  viable AND non-seizing: {len(viable_term)}/{len(rows)}")
    print(f"  of those, with a pulse that STOPS inside the rollout: {len(real_term)}")
    print(f"  conditions rejected as off-manifold: {n_off}/{len(rows)}")
    if real_term:
        b = min(real_term, key=lambda r: r["rise"])
        print(f"\n  TERMINATION FOUND: {b['band']} {b['mode']} amp {b['amp']:+.2f}, "
              f"pulse {b['start']:+.1f}..{b['start']+b['dur']:.1f}s, "
              f"rise {b['rise']:+.3f} vs baseline {base_st['rise']:+.3f}")
    elif viable_term:
        print("\n  Suppression only under stimulation that never switches off. "
              "With a\n  monostable field that is forcing, not termination.")

# ---------------------------------------------------------------------------
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
axes[0].plot(t_roll, s_ref, color="0.25", lw=1.3, label="recording")
axes[0].plot(t_roll, tscore(base_tr).ravel(), color="0.45", lw=2.0,
             label="no stimulation")
axes[0].axhline(SEIZE_TH, color="#D55E00", ls=":", lw=1.2,
                label=f"seizure threshold {SEIZE_TH:.2f}")
# Show the conditions that TERMINATE with a pulse that actually stops -- the
# result -- not merely the lowest-rise ones, which are all continuous drives.
_term = sorted([q for q in rows if q["viable"] and not q["seizes"] and q["stops"]],
               key=lambda q: q["rise"])
_cont = sorted([q for q in rows if q["viable"] and not q["seizes"] and not q["stops"]],
               key=lambda q: q["rise"])[:2]
for r, col, ls in list(zip(_term, ["#0072B2", "#009E73"], ["-", "-"])) \
        + list(zip(_cont, ["#CC79A7", "#E69F00"], ["--", "--"])):
    tr = rollout(r["bidx"], r["mode"], r["amp"], r["start"], r["dur"])
    tag = "pulse STOPS" if r["stops"] else "continuous"
    axes[0].plot(t_roll, tscore(tr).ravel(), lw=1.8, ls=ls, color=col,
                 label=f"{r['band']} {r['mode']} a={r['amp']:+.2f} ({tag})")
    if r["stops"]:
        # clip the shaded window to the rollout so it cannot imply the pulse
        # runs past the simulation
        axes[0].axvspan(max(r["start"], t_roll[0]),
                        min(r["start"] + r["dur"], t_roll[-1]),
                        color=col, alpha=0.10, lw=0)
        axes[0].axvline(r["start"] + r["dur"], color=col, ls=":", lw=1.0)
axes[0].set_xlim(t_roll[0], t_roll[-1])
axes[0].set_xlabel("recording time (s)")
axes[0].set_ylabel("transition score")
axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
axes[0].set_title("Interventions that terminate (solid) vs continuous drive (dashed)",
                  loc="left", fontweight="bold", fontsize=9)

# right: dose response, viable and off-manifold drawn as DISJOINT sets
for nm, col in (("beta", "#0072B2"), ("gamma", "#D55E00")):
    sub = [r for r in rows if r["band"] == nm and r["mode"] != "gain"]
    vi = [r for r in sub if r["viable"]]
    ba = [r for r in sub if not r["viable"]]
    axes[1].scatter([r["amp"] for r in vi], [r["rise"] for r in vi], s=20,
                    color=col, alpha=0.65, label=f"{nm}, viable")
    axes[1].scatter([r["amp"] for r in ba], [r["rise"] for r in ba], s=20,
                    facecolors="none", edgecolors=col, alpha=0.45, linewidths=0.7,
                    label=f"{nm}, off-manifold")
axes[1].axhline(SEIZE_TH, color="#D55E00", ls=":", lw=1.2)
axes[1].axhline(base_st["rise"], color="0.3", ls="--", lw=0.9, label="baseline")
axes[1].set_yscale("log")
axes[1].set_xlabel("stimulation amplitude  (negative = drive the band down)")
axes[1].set_ylabel("max transition score rise (log)")
axes[1].legend(fontsize=6, frameon=False, ncol=2)
axes[1].set_title("Dose response: only NEGATIVE drive suppresses",
                  loc="left", fontweight="bold", fontsize=9)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(RESULT_DIR / f"dbs_beta_gamma.{ext}", dpi=250, bbox_inches="tight")
plt.close(fig)

viable_term = [r for r in rows if r["viable"] and not r["seizes"]]
real_term = [r for r in viable_term if r["stops"]]
print("\n" + "=" * 90)
ok = print_check(base_st["seizes"], "the unstimulated model seizes, so the "
                                    "intervention question is not vacuous")
print_check(bool(viable_term), f"some viable condition keeps the rise below "
                               f"threshold ({len(viable_term)})")
print_check(bool(real_term), f"...with a pulse that actually switches off "
                             f"({len(real_term)})")

save_metrics(RESULT_DIR / "metrics.json", {
    "task": "DBS-style intervention with the actuator on beta and gamma",
    "why": "the delta actuator in epilepsy_cross_frequency_dbs_intervention.py "
           "drives a band that is nearly invisible to the transition score "
           "(delta moves 0.22->-0.05 across the event); measured best viable "
           "suppression by band: delta +0.147, theta +0.427, beta +0.441, "
           "alpha +0.567, gamma +1.044.",
    "scoring": "rise is the MAX over the whole rollout, so a rebound after the "
               "pulse is scored as still seizing.",
    "baseline": base_st, "observed_rise": obs_rise, "seizure_threshold": SEIZE_TH,
    "sweep": {"amps": AMPS, "gain_amps": GAIN_AMPS, "starts": STARTS,
              "durations": PULSE_DURS},
    "n_conditions": len(rows),
    "n_viable_nonseizing": len(viable_term),
    "n_terminating_with_finite_pulse": len(real_term),
    "results": rows,
})
print(f"[DONE] {RESULT_DIR}")
