#!/usr/bin/env python3
"""Publication figure for the cross-frequency seizure model.

Reads the arrays and metrics that
``epilepsy_cross_frequency_seizure_freedom.py`` writes, so the figure can be
re-drawn in seconds without repeating the ~35 minute fit.

The design problem this figure has to solve
-------------------------------------------
The autonomous model produces a SMOOTH curve while the recording wiggles, and
a naive overlay makes that look like a dead model.  It is not: the script's own
measurements show the wiggle is neither a rhythm (AR(1) surrogates matched to
each band's autocorrelation are just as spectrally peaky) nor predictable
(k-NN R^2 ~ 0 against a dimension-matched Lorenz-96 control of 0.86).  For a
stochastic fluctuation the correct output of a deterministic field is the
CONDITIONAL MEAN -- a smooth curve -- and a model that traced the wiggle would
be fitting noise.

So the figure plots the raw recording AND its local mean +/- 1 sd, and the
model is compared against the mean, with the band making explicit which part of
the signal is not predictable in principle.  The rolling statistics are a
DISPLAY device computed with a centred window; they are never used as state
variables (a centred filter leaks the future and makes a terrible state).

The figure does not hide the model's real defect.  Panel A marks the half-rise
of the recording and of the model, so the onset lag is visible rather than
argued about, and panel D shows the ictal envelope in recording units where the
model plainly undershoots.

Run:
    cd skills/kandy/examples && python epilepsy/epilepsy_cross_frequency_figure.py
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

RESULT_DIR = REPO_ROOT / "results" / "Epilepsy-Cross-Frequency"
ARR = RESULT_DIR / "cross_frequency_arrays.npz"
MET = RESULT_DIR / "metrics.json"
if not ARR.exists():
    raise SystemExit(f"{ARR} not found -- run epilepsy_cross_frequency_seizure_freedom.py first")

d = np.load(ARR, allow_pickle=True)
metrics = json.loads(MET.read_text()) if MET.exists() else {}

T = d["t"]
STATE = d["state"]
t_roll = d["t_roll"]
model_traj = d["model_traj"]
nocross_traj = d["nocross_traj"]
const_traj = d["const_traj"]
sym_traj = d["sym_traj"] if d["sym_traj"].size else None
TAU = d["tau"]
BAND_NAMES = [str(b) for b in d["bands"]]
LABELS = [str(c) for c in d["channels"]]
NB = len(BAND_NAMES)
DT = float(T[1] - T[0])

# Okabe-Ito: distinguishable under every common form of colour blindness, and
# separable in greyscale once paired with the line styles used below.
OI = {"orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
      "purple": "#CC79A7", "black": "#000000"}
C_DATA, C_MODEL, C_EQN = OI["black"], OI["vermillion"], OI["green"]
C_CUT, C_CONST = OI["blue"], OI["purple"]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------
PRE, ICT, POST = (-13.0, -2.0), (1.0, 8.0), (18.0, 28.0)


def wm(w):
    return (T >= w[0]) & (T <= w[1])


PRE_C, POST_C = STATE[wm(PRE)].mean(0), STATE[wm(POST)].mean(0)
AXIS = POST_C - PRE_C
AN = float(np.linalg.norm(AXIS))
U = AXIS / AN


def tscore(x):
    return (np.atleast_2d(x) - PRE_C) @ U / AN


S_DATA = tscore(STATE).ravel()
n = len(model_traj)
sl = slice(int(np.argmin(np.abs(T - t_roll[0]))), None)
s_ref = S_DATA[sl][:n]
s_model = tscore(model_traj).ravel()
s_cut = tscore(nocross_traj).ravel()[:n]
s_const = tscore(const_traj).ravel()[:n]
s_eqn = tscore(sym_traj).ravel()[:n] if sym_traj is not None else None

# Display-only local statistics.  Window = the model's fastest timescale, i.e.
# the finest structure the fitted field could possibly express.
WIN = max(3, int(round(float(TAU.min()) / DT)) | 1)


def roll(x, w, f):
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    return np.array([f(xp[i:i + w]) for i in range(len(x))])


s_mean = roll(s_ref, WIN, np.mean)
s_sd = roll(s_ref, WIN, np.std)


def half_time(s):
    base = s[:int(1.0 / DT)].mean()
    h = base + 0.5 * (s.max() - base)
    i = int(np.argmax(s >= h)) if np.any(s >= h) else len(s) - 1
    return t_roll[min(i, len(t_roll) - 1)]


t_half_d, t_half_m = half_time(s_ref), half_time(s_model)


def rmse(a):
    m = min(len(a), len(STATE[sl]))
    return float(np.sqrt(np.mean((a[:m] - STATE[sl][:m]) ** 2)))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(7.5, 8.6))
# Row 1 is a dedicated, invisible axis that holds panel a's legend, so a
# seven-entry legend never has to be squeezed on top of the data or allowed to
# collide with the titles of the panels below it.
gs = fig.add_gridspec(4, 2, height_ratios=[1.15, 0.22, 1.0, 1.0],
                      hspace=0.62, wspace=0.42)
axA = fig.add_subplot(gs[0, :])
axL = fig.add_subplot(gs[1, :])
axL.set_axis_off()
axB = fig.add_subplot(gs[2, 0])
axC = fig.add_subplot(gs[2, 1])
axD = fig.add_subplot(gs[3, :])

# ---- A: transition score, model against the conditional mean ---------------
axA.axvspan(*ICT, color=OI["yellow"], alpha=0.25, lw=0, zorder=0)
axA.fill_between(t_roll[:n], (s_mean - s_sd)[:n], (s_mean + s_sd)[:n],
                 color="0.80", alpha=0.55, lw=0, zorder=1,
                 label="recording, $\\pm$1 sd (unpredictable: k-NN $R^2\\approx0$)")
axA.plot(t_roll[:n], s_ref, color="0.55", lw=0.6, alpha=0.9, zorder=2,
         label="recording (raw)")
axA.plot(t_roll[:n], s_mean[:n], color=C_DATA, lw=1.8, zorder=4,
         label="recording, local mean")
axA.plot(t_roll[:n], s_model, color=C_MODEL, lw=2.2, zorder=5,
         label="KANDy, autonomous")
if s_eqn is not None:
    axA.plot(t_roll[:n], s_eqn, color=C_EQN, lw=1.4, ls=(0, (6, 2)), zorder=4,
             label="closed-form equation")
axA.plot(t_roll[:n], s_cut, color=C_CUT, lw=1.2, ls=(0, (2, 2)), zorder=3,
         label="all cross-band links cut")
axA.plot(t_roll[:n], s_const, color=C_CONST, lw=1.2, ls=(0, (1, 2)), zorder=3,
         label="constant-target baseline")

for tv, col, lab in ((t_half_d, C_DATA, "data"), (t_half_m, C_MODEL, "model")):
    axA.plot([tv], [0.5 * (s_ref.max() + s_ref[:int(1 / DT)].mean())], marker="v",
             color=col, ms=6, zorder=6, clip_on=False)
axA.annotate("", xy=(t_half_m, 1.52), xytext=(t_half_d, 1.52),
             arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.9))
axA.text(0.5 * (t_half_d + t_half_m), 1.58,
         f"onset lag {t_half_m - t_half_d:+.1f} s", ha="center", fontsize=7,
         color="0.25")
axA.axvline(0, color=OI["vermillion"], ls=":", lw=0.9, zorder=1)
axA.set_ylabel("Transition score\n(0 = pre-ictal, 1 = post-ictal)")
axA.set_xlabel("Recording time (s), seizure onset at 0")
axA.set_ylim(-0.55, 1.80)
_h, _l = axA.get_legend_handles_labels()
axL.legend(_h, _l, ncol=3, loc="center", frameon=False, handlelength=2.4,
           columnspacing=1.6, fontsize=7)
axA.set_title("Autonomous replay: no forcing, no nudging, no clipping",
              loc="left", fontweight="bold", pad=12)

# ---- B: the recovered equation as geometry ---------------------------------
bi, gi = BAND_NAMES.index("delta"), BAND_NAMES.index("gamma")
sc = axB.scatter(STATE[sl][:n, bi], STATE[sl][:n, gi], c=t_roll[:n],
                 cmap="cividis", s=5, alpha=0.55, lw=0, zorder=2)
axB.plot(model_traj[:, bi], model_traj[:, gi], color=C_MODEL, lw=2.2, zorder=4,
         label="model")
axB.plot(model_traj[0, bi], model_traj[0, gi], "o", color=C_MODEL, ms=6,
         mec="w", mew=0.8, zorder=5)
axB.plot(model_traj[-1, bi], model_traj[-1, gi], "s", color=C_MODEL, ms=6,
         mec="w", mew=0.8, zorder=5)

# delta-nullcline from the extracted equation, if it parsed
eq = (metrics.get("equations", {}) or {}).get("delta", {})
raw = eq.get("raw") if isinstance(eq, dict) else None
if raw:
    try:
        import sympy as sp

        dsym, gsym = sp.symbols("delta gamma")
        expr = sp.sympify(raw)
        free = {str(s) for s in expr.free_symbols}
        if free <= {"delta", "gamma"}:
            # T_scale/box squash are ~identity near the centre; the nullcline of
            # the fitted LINEAR target is where T_delta(delta, gamma) = delta.
            sol = sp.solve(sp.Eq(expr, dsym), dsym)
            if sol:
                gg = np.linspace(*axB.get_ylim(), 50)
                f = sp.lambdify(gsym, sol[0], "numpy")
                dd = np.asarray(f(gg), dtype=float) * np.ones_like(gg)
                axB.plot(dd, gg, color=OI["skyblue"], lw=1.6, ls="-.", zorder=3,
                         label=r"fitted $\delta$-nullcline")
    except Exception:                                            # noqa: BLE001
        pass
axB.set_xlabel("delta amplitude (sd)")
axB.set_ylabel("gamma amplitude (sd)")
axB.legend(frameon=False, loc="upper right")
axB.set_title(r"$T_\delta$: the one cross-band term", loc="left",
              fontweight="bold", pad=10)
cb = fig.colorbar(sc, ax=axB, pad=0.03, fraction=0.045)
cb.set_label("time (s)", fontsize=6.5, labelpad=1)
cb.ax.tick_params(labelsize=6)

# ---- C: is the model doing something?  numbers, not adjectives -------------
#
# NOT trajectory RMSE.  Over a 25 s autonomous rollout every trajectory that
# drifts from the pre-ictal to the post-ictal level scores about the same --
# measured here, KANDy 0.84 against a constant-target baseline 0.85, a 1% gap
# that would not support any claim.  The discriminating measurement is held-out
# prediction of the RATE on the transition block, by a model that never saw it.
#
tr = (metrics.get("heldout_r2", {}) or {}).get("transition", {})
if tr:
    arms = [("cross-band", "cross_band", C_MODEL),
            ("own-band only", "own_band_only", OI["skyblue"]),
            ("persistence", "zero_change", "0.65")]
    xs = np.arange(NB)
    w = 0.26
    for k, (lab, key, col) in enumerate(arms):
        vals = [tr.get(b, {}).get(key, np.nan) for b in BAND_NAMES]
        axC.bar(xs + (k - 1) * w, vals, width=w, color=col, label=lab,
                edgecolor="white", lw=0.5)
    axC.axhline(0, color="k", lw=0.7)
    axC.set_xticks(xs, BAND_NAMES, rotation=30, ha="right", fontsize=7)
    axC.set_ylabel("Held-out $R^2$ of $dx/dt$\non the transition")
    axC.legend(frameon=False, loc="upper right", fontsize=6.5)
    axC.set_title("Skill is real in delta and theta only", loc="left",
                  fontweight="bold", pad=10)
else:
    axC.set_axis_off()

# ---- D: recording units ----------------------------------------------------
ci = LABELS.index("hippocampus_R") if "hippocampus_R" in LABELS else 0
bshow = BAND_NAMES.index("beta")
chz = d[f"ch_{LABELS[ci]}"]
env_d = chz[sl][:n, bshow]
axD.axvspan(*ICT, color=OI["yellow"], alpha=0.25, lw=0, zorder=0)
axD.plot(t_roll[:n], env_d, color="0.55", lw=0.6, zorder=2,
         label=f"{LABELS[ci]} {BAND_NAMES[bshow]} (raw)")
axD.plot(t_roll[:n], roll(env_d, WIN, np.mean), color=C_DATA, lw=1.8, zorder=4,
         label="local mean")
axD.plot(t_roll[:n], model_traj[:, bshow], color=C_MODEL, lw=2.2, zorder=5,
         label="KANDy, autonomous")
axD.axvline(0, color=OI["vermillion"], ls=":", lw=0.9)
axD.set_xlabel("Recording time (s), seizure onset at 0")
axD.set_ylabel(f"{BAND_NAMES[bshow]} envelope (sd)")
axD.legend(ncol=3, frameon=False, loc="lower right", fontsize=6.5)
r_ch = float(np.corrcoef(env_d, model_traj[:, bshow])[0, 1])
axD.set_title(f"Seizure focus, {LABELS[ci]} (r = {r_ch:.2f}): the sustained "
              "shift is tracked, the ictal transient is not",
              loc="left", fontweight="bold", pad=10)

for ax, lab in ((axA, "a"), (axB, "b"), (axC, "c"), (axD, "d")):
    ax.text(-0.02, 1.16, lab, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="right")
    ax.yaxis.labelpad = 6

for ext in ("png", "pdf"):
    fig.savefig(RESULT_DIR / f"seizure_replay_figure.{ext}", dpi=300,
                bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# replay_vs_channels: the one autonomous trajectory against all six recordings
# ---------------------------------------------------------------------------
#
# The model's state is the CHANNEL MEAN, so it predicts one trajectory, not six.
# Laying it over each recording separately is therefore a test of how good that
# abstraction is region by region -- and it is not uniform.  Channels are
# ordered by agreement and coloured by it, so the split between the
# right-hemisphere limbic network and motor cortex is the first thing visible.
#
CH = {lab: d[f"ch_{lab}"] for lab in LABELS}
cmp_ = {}
for lab in LABELS:
    Sc = CH[lab][sl][:n]
    rs = [float(np.corrcoef(model_traj[:, b], Sc[:, b])[0, 1]) for b in range(NB)]
    cmp_[lab] = {"per_band": rs, "mean": float(np.mean(rs))}
order = sorted(LABELS, key=lambda k: -cmp_[k]["mean"])

fig2 = plt.figure(figsize=(7.5, 8.0))
# A spacer row keeps the summary bar clear of the band panels' tick labels.
gs2 = fig2.add_gridspec(NB + 2, 2, height_ratios=[1] * NB + [0.42, 1.15],
                        width_ratios=[3.1, 1.0], hspace=0.32, wspace=0.34)

# strong = tracks the model, weak = does not.  Two families, two hues.
strong = [OI["blue"], OI["skyblue"], OI["green"], "#4C9F70"]
weak = ["#B0B0B0", "#8A8A8A"]
ch_col, si, wi = {}, 0, 0
for lab in order:
    if cmp_[lab]["mean"] >= 0.25:
        ch_col[lab] = strong[min(si, len(strong) - 1)]
        si += 1
    else:
        ch_col[lab] = weak[min(wi, len(weak) - 1)]
        wi += 1

axes2 = []
for b, name in enumerate(BAND_NAMES):
    ax = fig2.add_subplot(gs2[b, 0])
    axes2.append(ax)
    ax.axvspan(*ICT, color=OI["yellow"], alpha=0.22, lw=0, zorder=0)
    for lab in order:
        ax.plot(t_roll[:n], CH[lab][sl][:n, b], color=ch_col[lab], lw=0.7,
                alpha=0.85 if cmp_[lab]["mean"] >= 0.25 else 0.55, zorder=2,
                label=lab if b == 0 else None)
    ax.plot(t_roll[:n], STATE[sl][:n, b], color="k", lw=1.4, zorder=4,
            label="channel mean (fit target)" if b == 0 else None)
    ax.plot(t_roll[:n], model_traj[:, b], color=C_MODEL, lw=2.4, zorder=5,
            label="KANDy, autonomous" if b == 0 else None)
    ax.axvline(0, color=OI["vermillion"], ls=":", lw=0.8, zorder=1)
    ax.set_ylabel(name, fontweight="bold")
    ax.set_xlim(t_roll[0], t_roll[n - 1])
    if b < NB - 1:
        ax.set_xticklabels([])
axes2[-1].set_xlabel("Recording time (s), seizure onset at 0", labelpad=2)
axes2[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.95),
                frameon=False, fontsize=6.5, columnspacing=1.2)

# right column: per-band correlation heat strip, channels ordered by agreement
axH = fig2.add_subplot(gs2[:NB, 1])
Mx = np.array([cmp_[lab]["per_band"] for lab in order])
im = axH.imshow(Mx, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
axH.set_xticks(range(NB), [b[:3] for b in BAND_NAMES], fontsize=6.5, rotation=0)
axH.set_yticks(range(len(order)), order, fontsize=6.5)
for i in range(len(order)):
    for j in range(NB):
        axH.text(j, i, f"{Mx[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                 color="white" if abs(Mx[i, j]) > 0.55 else "black")
axH.set_title("Per-band correlation\nwith the replay", fontsize=8, pad=8,
              fontweight="bold")
cb2 = fig2.colorbar(im, ax=axH, fraction=0.06, pad=0.06)
cb2.ax.tick_params(labelsize=6)

# bottom: mean agreement per channel
axS = fig2.add_subplot(gs2[NB + 1, :])
vals = [cmp_[lab]["mean"] for lab in order]
axS.bar(range(len(order)), vals, color=[ch_col[l] for l in order], width=0.65,
        edgecolor="white", lw=0.6)
for i, v in enumerate(vals):
    axS.text(i, v + 0.012 * np.sign(v or 1), f"{v:.2f}", ha="center",
             va="bottom" if v >= 0 else "top", fontsize=7)
axS.axhline(0, color="k", lw=0.7)
axS.set_xticks(range(len(order)), order, fontsize=7.5, rotation=12, ha="right")
axS.set_ylabel("Mean per-band\ncorrelation")
axS.set_ylim(min(0, min(vals)) - 0.06, max(vals) * 1.22)
axS.set_title("The model describes the right-hemisphere limbic network; "
              "motor cortex does not participate", loc="left", fontweight="bold",
              fontsize=9.5, pad=6)

fig2.suptitle("One autonomous trajectory against all six recordings",
              fontweight="bold", fontsize=12, y=0.995)
for ext in ("png", "pdf"):
    fig2.savefig(RESULT_DIR / f"replay_vs_channels.{ext}", dpi=300,
                 bbox_inches="tight")
plt.close(fig2)
print("[FIG] wrote " + str(RESULT_DIR) + "/replay_vs_channels.png/.pdf")
print("      channel agreement: "
      + ", ".join(f"{l}={cmp_[l]['mean']:.2f}" for l in order))
print(f"[FIG] wrote {RESULT_DIR}/seizure_replay_figure.png/.pdf")
print(f"      onset lag {t_half_m - t_half_d:+.1f}s; trajectory RMSE: "
      f"KANDy={rmse(model_traj):.3f}, constant-target={rmse(const_traj):.3f}, "
      f"cross-cut={rmse(nocross_traj):.3f} "
      "(RMSE barely separates KANDy from the constant baseline -- panel c is "
      "the discriminating measurement)")
