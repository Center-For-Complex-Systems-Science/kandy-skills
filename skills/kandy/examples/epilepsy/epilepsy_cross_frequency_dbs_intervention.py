#!/usr/bin/env python3
"""Abstract beta->delta intervention on the cross-frequency seizure model.

This is the control experiment that ``epilepsy_cross_frequency_seizure_freedom.py``
does not run.  The source model already has an autonomous five-band envelope
ODE and a significant directed beta->delta timing test, but its built-in
lesion sweep is static: a link is scaled for the whole rollout.  This script
adds a pulse in recording time and tests two intervention abstractions:

1. additive pulse

       dx_delta/dt = F_delta_augmented(x)
                    + a * u(t) * r_bd * H(beta(t-lag) - theta_b)

2. gain pulse

       f_delta(x, t) = f_delta_closed(x)
                       + (1 - a*u(t)) * c_bd * H(beta(t-lag) - theta_b)

where u(t) is a rectangular stimulation window, H is a smooth Heaviside
sigmoid, c_bd is fitted as a one-column target-map residual correction to the
printed closed-form delta equation, and r_bd is the same one-column correction
fit directly in rate space.  This is not clinical DBS: it is an abstract model
control on the beta->delta pathway in standardized band-amplitude coordinates.

Run:
    cd skills/kandy/examples && python epilepsy/epilepsy_cross_frequency_dbs_intervention.py
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


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_list(name: str, default: str) -> list[float]:
    return [float(v) for v in os.environ.get(name, default).split(",") if v.strip()]


SOURCE_DIR = REPO_ROOT / "results" / "Epilepsy-Cross-Frequency"
ARRAY_PATH = SOURCE_DIR / "cross_frequency_arrays.npz"
METRIC_PATH = SOURCE_DIR / "metrics.json"
RESULT_DIR = ensure_outdir(REPO_ROOT / "results" / "Epilepsy-Cross-Frequency-DBS")

if not ARRAY_PATH.exists() or not METRIC_PATH.exists():
    raise SystemExit(
        "Missing cross-frequency artifacts. Run "
        "`cd skills/kandy/examples && python epilepsy/epilepsy_cross_frequency_seizure_freedom.py` "
        "first."
    )

arrays = np.load(ARRAY_PATH, allow_pickle=True)
metrics = json.loads(METRIC_PATH.read_text())

BAND_NAMES = [str(v) for v in arrays["bands"]]
BI = {name: i for i, name in enumerate(BAND_NAMES)}
DELTA, BETA = BI["delta"], BI["beta"]
NB = len(BAND_NAMES)
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}

T = np.asarray(arrays["t"], float)
STATE = np.asarray(arrays["state"], float)
TAU = np.asarray(arrays["tau"], float)
X_C = np.asarray(arrays["x_c"], float)
X_R = np.asarray(arrays["x_r"], float)
THRESH = np.asarray(arrays["thresh"], float)
T_SCALE = np.asarray(arrays["t_scale"], float)
DT = float(T[1] - T[0])

PRE_ICTAL = (-13.0, -2.0)
ICTAL = (1.0, 8.0)
POST_ICTAL = (18.0, 28.0)
BASIN_TEST_WINDOWS = {"pre_ictal": (-13.0, -9.0), "post_ictal": (22.0, 28.0)}
TRANSITION_WINDOW = (-2.0, 9.0)
TRANSITION_THRESHOLD = 0.5

ROLL_START = env_float("KANDY_DBS_ROLL_START", -8.0)
ROLL_SECONDS = env_float("KANDY_DBS_ROLL_SECONDS", 25.0)
PULSE_DURATION = env_float("KANDY_DBS_DURATION", 4.0)
LINK_DELAY_S = env_float("KANDY_DBS_LINK_DELAY_S", 0.0)
HEAVISIDE_K = env_float("KANDY_DBS_HEAVISIDE_K", 64.0)
RIDGE = env_float("KANDY_DBS_RIDGE", 1.0e-3)
START_GRID = np.asarray(env_list("KANDY_DBS_START_GRID", "-1,0,1,2,3,4,5,6"), float)
ADD_AMP_GRID = np.asarray(
    env_list("KANDY_DBS_ADD_AMP_GRID", "-50,-30,-20,-10,-5,-3,-2,-1,0,1,2,3,5,10,20,30,50"),
    float,
)
GAIN_AMP_GRID = np.asarray(
    env_list("KANDY_DBS_GAIN_AMP_GRID", "0,0.5,1,2,3,5,10"),
    float,
)


def win_mask(window: tuple[float, float], on_rows: np.ndarray | None = None) -> np.ndarray:
    tt = T if on_rows is None else T[on_rows]
    return (tt >= window[0]) & (tt <= window[1])


def to_inner(target: np.ndarray) -> np.ndarray:
    return X_R * np.arctanh(np.clip((target - X_C) / X_R, -0.999, 0.999))


def from_inner(inner: np.ndarray) -> np.ndarray:
    return X_C + X_R * np.tanh(inner / X_R)


def f_closed(x: np.ndarray) -> np.ndarray:
    """The printed closed-form equations, vectorized."""
    x = np.atleast_2d(np.asarray(x, float))
    d, th, _al, _be, ga = [x[:, i] for i in range(NB)]
    gd = np.maximum(d - THRESH[DELTA], 0.0)
    return np.column_stack(
        [
            -0.6452 * d - 0.2496 * ga + 0.0638,
            0.0940 - 0.6048 * th,
            np.full(len(x), 0.3333),
            np.full(len(x), 0.2306),
            6.1026 - 0.1136 * (8.5784 - 5.9214 * gd) ** 2,
        ]
    )


def heaviside(v: np.ndarray | float) -> np.ndarray | float:
    z = np.clip(HEAVISIDE_K * (np.asarray(v) - THRESH[BETA]), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def target_from_f(f: np.ndarray) -> np.ndarray:
    return from_inner(T_SCALE * f)


def rhs_from_f(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    return (target_from_f(f) - x) / TAU


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    den = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum((y - pred) ** 2) / den) if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# Fit a one-column beta->delta residual term on the closed-form model.
# ---------------------------------------------------------------------------
lag_s = float(metrics.get("state", {}).get("increment_lag_s", 2.0))
lag_n = max(1, int(round(lag_s / DT)))
DX = (STATE[lag_n:] - STATE[:-lag_n]) / lag_s
rows = np.arange(len(DX))
t_rows = T[: len(DX)]

near_test = np.zeros(len(rows), dtype=bool)
test_rows_by_name = {}
guard = int(round(2.0 / DT))
for name, window in BASIN_TEST_WINDOWS.items():
    mask = (t_rows >= window[0]) & (t_rows <= window[1])
    r = rows[mask]
    test_rows_by_name[name] = r
    if len(r):
        near_test |= (rows >= r[0] - guard) & (rows <= r[-1] + guard)

trans_rows = rows[(t_rows >= TRANSITION_WINDOW[0]) & (t_rows <= TRANSITION_WINDOW[1])]
near_trans = np.zeros(len(rows), dtype=bool)
if len(trans_rows):
    near_trans |= (rows >= trans_rows[0] - guard) & (rows <= trans_rows[-1] + guard)
train_rows = rows[~near_test]
trainB_rows = rows[~(near_test | near_trans)]

T_TARGET = STATE[: len(DX)] + TAU * DX
F_TARGET = to_inner(T_TARGET) / T_SCALE
F_BASE = f_closed(STATE[: len(DX)])
resid_delta = F_TARGET[:, DELTA] - F_BASE[:, DELTA]
H_BETA = np.asarray(heaviside(STATE[: len(DX), BETA]), float)
RATE_BASE = rhs_from_f(STATE[: len(DX)], F_BASE)
rate_resid_delta = DX[:, DELTA] - RATE_BASE[:, DELTA]

den = float(H_BETA[trainB_rows] @ H_BETA[trainB_rows] + RIDGE)
coef_bd = float((H_BETA[trainB_rows] @ resid_delta[trainB_rows]) / den)
coef_bd_rate = float((H_BETA[trainB_rows] @ rate_resid_delta[trainB_rows]) / den)

def rate_delta_with_coef(idx: np.ndarray, coef: float) -> np.ndarray:
    f = f_closed(STATE[idx])
    f[:, DELTA] += coef * np.asarray(heaviside(STATE[idx, BETA]), float)
    return rhs_from_f(STATE[idx], f)[:, DELTA]


base_transition_r2 = r2(
    DX[trans_rows, DELTA], rhs_from_f(STATE[trans_rows], f_closed(STATE[trans_rows]))[:, DELTA]
)
aug_transition_r2 = r2(DX[trans_rows, DELTA], rate_delta_with_coef(trans_rows, coef_bd))
transition_gain = float(aug_transition_r2 - base_transition_r2)
aug_rate_transition_r2 = r2(
    DX[trans_rows, DELTA],
    RATE_BASE[trans_rows, DELTA] + coef_bd_rate * H_BETA[trans_rows],
)
transition_rate_gain = float(aug_rate_transition_r2 - base_transition_r2)


# ---------------------------------------------------------------------------
# Rollout helpers.
# ---------------------------------------------------------------------------
pre_c = STATE[win_mask(PRE_ICTAL)].mean(axis=0)
post_c = STATE[win_mask(POST_ICTAL)].mean(axis=0)
axis = post_c - pre_c
axis_norm = float(np.linalg.norm(axis))
axis_u = axis / axis_norm


def transition_score(x: np.ndarray) -> np.ndarray:
    return (np.atleast_2d(x) - pre_c) @ axis_u / axis_norm


i0 = int(np.argmin(np.abs(T - ROLL_START)))
n_steps = min(int(round(ROLL_SECONDS / DT)), len(T) - i0 - 1)
t_roll = T[i0 : i0 + n_steps + 1]
ref = STATE[i0 : i0 + n_steps + 1]
s_ref = transition_score(ref).ravel()
obs_base = float(s_ref[: max(1, int(1.0 / DT))].mean())
obs_rise = float(s_ref.max() - obs_base)
seizure_threshold = TRANSITION_THRESHOLD * obs_rise


def pulse(t: float, start: float, duration: float) -> float:
    return float(start <= t < start + duration)


def beta_lag_value(traj: np.ndarray, step: int, x_now: np.ndarray) -> float:
    lag_steps = int(round(LINK_DELAY_S / DT))
    if lag_steps <= 0:
        return float(x_now[BETA])
    idx = step - lag_steps
    if idx >= 0:
        return float(traj[idx, BETA])
    return float(STATE[i0 + idx, BETA])


def rhs_control(
    x: np.ndarray,
    beta_lag: float,
    t_now: float,
    mode: str,
    amp: float,
    start: float,
) -> np.ndarray:
    f = f_closed(x)
    h = float(heaviside(beta_lag))
    p = pulse(t_now, start, PULSE_DURATION)
    if mode == "additive":
        f[:, DELTA] += coef_bd * h
    elif mode == "gain":
        f[:, DELTA] += (1.0 - amp * p) * coef_bd * h
    elif mode == "baseline":
        f[:, DELTA] += coef_bd * h
    elif mode == "closed":
        pass
    else:
        raise ValueError(f"unknown mode {mode!r}")
    out = rhs_from_f(np.atleast_2d(x), f)[0]
    if mode == "additive":
        out[DELTA] += amp * p * coef_bd_rate * h
    return out


def rollout(mode: str, amp: float = 0.0, start: float = 0.0) -> np.ndarray:
    x = STATE[i0].copy()
    traj = np.empty((n_steps + 1, NB), dtype=float)
    traj[0] = x
    for step in range(n_steps):
        t0 = float(t_roll[step])

        def deriv(y: np.ndarray, tt: float) -> np.ndarray:
            return rhs_control(y, beta_lag_value(traj, step, y), tt, mode, amp, start)

        k1 = deriv(x, t0)
        k2 = deriv(x + 0.5 * DT * k1, t0 + 0.5 * DT)
        k3 = deriv(x + 0.5 * DT * k2, t0 + 0.5 * DT)
        k4 = deriv(x + DT * k3, t0 + DT)
        x = x + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(x).all():
            traj[step + 1 :] = np.nan
            return traj
        traj[step + 1] = x
    return traj


def summarize(traj: np.ndarray) -> dict:
    finite = np.isfinite(traj).all(axis=1)
    n = int(finite.sum())
    tr = traj[:n]
    s = transition_score(tr).ravel()
    base = float(s[: max(1, int(1.0 / DT))].mean())
    rise = float(s.max() - base)
    inside = np.all(
        (tr >= STATE.min(axis=0) - 0.1 * np.ptp(STATE, axis=0))
        & (tr <= STATE.max(axis=0) + 0.1 * np.ptp(STATE, axis=0)),
        axis=1,
    )
    ref_tr = STATE[i0 : i0 + len(tr)]
    rmse = float(np.sqrt(np.mean((tr[: len(ref_tr)] - ref_tr) ** 2)))
    half = base + 0.5 * rise
    idx_half = int(np.argmax(s >= half)) if np.any(s >= half) else len(s) - 1
    return {
        "rise": rise,
        "s_end": float(s[-1]),
        "t_half_s": float(t_roll[min(idx_half, len(t_roll) - 1)]),
        "t_peak_s": float(t_roll[int(np.argmax(s))]),
        "complete": bool(n == len(traj)),
        "in_observed_range_frac": float(inside.mean()),
        "traj_rmse_vs_data": rmse,
        "seizes": bool(rise >= seizure_threshold),
        "viable": bool(n == len(traj) and inside.mean() >= 0.9),
    }


def sweep(mode: str, amp_grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    rows_out: list[dict] = []
    rise = np.empty((len(START_GRID), len(amp_grid)), dtype=float)
    for si, start in enumerate(START_GRID):
        for ai, amp in enumerate(amp_grid):
            traj = rollout(mode, float(amp), float(start))
            st = summarize(traj)
            row = {"mode": mode, "start_s": float(start), "amp": float(amp), **st}
            rows_out.append(row)
            rise[si, ai] = st["rise"] if st["viable"] else np.nan
    return rows_out, rise


closed_traj = rollout("closed")
baseline_traj = rollout("baseline")
closed_st = summarize(closed_traj)
baseline_st = summarize(baseline_traj)
add_rows, add_rise = sweep("additive", ADD_AMP_GRID)
gain_rows, gain_rise = sweep("gain", GAIN_AMP_GRID)


def best_row(rows_in: list[dict]) -> dict:
    viable = [r for r in rows_in if r["viable"]]
    return min(viable, key=lambda r: r["rise"]) if viable else min(rows_in, key=lambda r: r["rise"])


def terminating(rows_in: list[dict]) -> list[dict]:
    out = [r for r in rows_in if r["viable"] and not r["seizes"]]
    return sorted(out, key=lambda r: (r["rise"], r["traj_rmse_vs_data"]))


best_add = best_row(add_rows)
best_gain = best_row(gain_rows)
best_add_traj = rollout("additive", best_add["amp"], best_add["start_s"])
best_gain_traj = rollout("gain", best_gain["amp"], best_gain["start_s"])
term_add = terminating(add_rows)
term_gain = terminating(gain_rows)

print("=" * 88)
print("CROSS-FREQUENCY BETA->DELTA ABSTRACT DBS INTERVENTION")
print("=" * 88)
print(f"source artifacts: {SOURCE_DIR}")
print(f"rollout from t={t_roll[0]:+.1f}s for {n_steps * DT:.1f}s at {1 / DT:.0f} Hz")
print(
    f"beta->delta residual coefficients c_bd={coef_bd:+.4f} target, "
    f"r_bd={coef_bd_rate:+.4f}/s rate, "
    f"H threshold={THRESH[BETA]:+.3f}, k={HEAVISIDE_K:g}, delay={LINK_DELAY_S:.2f}s"
)
print(
    f"delta transition R2: closed={base_transition_r2:+.3f}, "
    f"closed+target Heaviside={aug_transition_r2:+.3f}, "
    f"closed+rate Heaviside={aug_rate_transition_r2:+.3f}"
)
print(
    f"seizure threshold = {seizure_threshold:+.3f} "
    f"({TRANSITION_THRESHOLD:.0%} of observed rise {obs_rise:+.3f})"
)
print("\n[BASELINES]")
print(f"  data rise       {obs_rise:+.3f}")
print(f"  closed form     {closed_st['rise']:+.3f}  seizes={closed_st['seizes']}")
print(f"  + beta term     {baseline_st['rise']:+.3f}  seizes={baseline_st['seizes']}")
print("\n[INTERVENTIONS]")
print(
    f"  additive best: start={best_add['start_s']:+.1f}s, amp={best_add['amp']:+.2f}, "
    f"rise={best_add['rise']:+.3f}, seizes={best_add['seizes']}, viable={best_add['viable']}"
)
print(
    f"  gain best:     start={best_gain['start_s']:+.1f}s, amp={best_gain['amp']:+.2f}, "
    f"rise={best_gain['rise']:+.3f}, seizes={best_gain['seizes']}, viable={best_gain['viable']}"
)
print(f"  terminating additive settings: {len(term_add)}")
print(f"  terminating gain settings:     {len(term_gain)}")

# ---------------------------------------------------------------------------
# Figures and arrays.
# ---------------------------------------------------------------------------
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def score_line(traj: np.ndarray) -> np.ndarray:
    return transition_score(traj).ravel()


fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), constrained_layout=True)
ax = axes[0, 0]
ax.axvspan(*ICTAL, color="#f4a261", alpha=0.18, lw=0, label="ictal window")
ax.axvspan(best_add["start_s"], best_add["start_s"] + PULSE_DURATION, color="#0072B2", alpha=0.14, lw=0)
ax.plot(t_roll, s_ref, color="0.25", lw=1.6, label="recording")
ax.plot(t_roll, score_line(baseline_traj), color="#D55E00", lw=1.8, label="closed + beta term")
ax.plot(t_roll, score_line(best_add_traj), color="#0072B2", lw=1.8, label="best additive")
ax.plot(t_roll, score_line(best_gain_traj), color="#009E73", lw=1.8, label="best gain")
ax.axhline(obs_base + seizure_threshold, color="0.4", lw=0.8, ls=":")
ax.set_title("Transition score", loc="left", fontweight="bold")
ax.set_ylabel("score")
ax.legend(frameon=False, fontsize=7)

ax = axes[0, 1]
for b, name in enumerate(BAND_NAMES):
    alpha = 1.0 if name in ("delta", "beta") else 0.35
    lw = 1.8 if name in ("delta", "beta") else 0.8
    ax.plot(t_roll, best_add_traj[:, b], lw=lw, alpha=alpha, label=name)
ax.axvspan(best_add["start_s"], best_add["start_s"] + PULSE_DURATION, color="#0072B2", alpha=0.14, lw=0)
ax.set_title("Best additive pulse state", loc="left", fontweight="bold")
ax.set_ylabel("standardized envelope")
ax.legend(frameon=False, ncol=5, fontsize=7)

ax = axes[1, 0]
im = ax.imshow(
    add_rise,
    origin="lower",
    aspect="auto",
    extent=[ADD_AMP_GRID[0], ADD_AMP_GRID[-1], START_GRID[0], START_GRID[-1]],
    cmap="viridis",
)
ax.contour(
    ADD_AMP_GRID,
    START_GRID,
    add_rise,
    levels=[seizure_threshold],
    colors="white",
    linewidths=1.0,
)
ax.scatter([best_add["amp"]], [best_add["start_s"]], color="white", edgecolor="black", s=40)
ax.set_title("Additive pulse rise", loc="left", fontweight="bold")
ax.set_xlabel("amplitude, multiples of c_bd")
ax.set_ylabel("pulse start (s)")
fig.colorbar(im, ax=ax, label="transition rise")

ax = axes[1, 1]
im = ax.imshow(
    gain_rise,
    origin="lower",
    aspect="auto",
    extent=[GAIN_AMP_GRID[0], GAIN_AMP_GRID[-1], START_GRID[0], START_GRID[-1]],
    cmap="viridis",
)
ax.contour(
    GAIN_AMP_GRID,
    START_GRID,
    gain_rise,
    levels=[seizure_threshold],
    colors="white",
    linewidths=1.0,
)
ax.scatter([best_gain["amp"]], [best_gain["start_s"]], color="white", edgecolor="black", s=40)
ax.set_title("Gain pulse rise", loc="left", fontweight="bold")
ax.set_xlabel("a in G_delta,beta(t)=1-a*u(t)")
ax.set_ylabel("pulse start (s)")
fig.colorbar(im, ax=ax, label="transition rise")

for ext in ("png", "pdf"):
    fig.savefig(RESULT_DIR / f"beta_delta_dbs_intervention.{ext}", dpi=250, bbox_inches="tight")
plt.close(fig)

np.savez_compressed(
    RESULT_DIR / "dbs_intervention_arrays.npz",
    t_roll=t_roll,
    state_ref=ref,
    closed_traj=closed_traj,
    baseline_traj=baseline_traj,
    best_additive_traj=best_add_traj,
    best_gain_traj=best_gain_traj,
    start_grid=START_GRID,
    add_amp_grid=ADD_AMP_GRID,
    gain_amp_grid=GAIN_AMP_GRID,
    add_rise=add_rise,
    gain_rise=gain_rise,
    beta_delta_coef=np.array([coef_bd]),
    beta_delta_rate_coef=np.array([coef_bd_rate]),
    beta_threshold=np.array([THRESH[BETA]]),
)

# ---------------------------------------------------------------------------
# Rebuild full-rate time series from the integrated envelope trajectories.
# ---------------------------------------------------------------------------
# The envelope ODE contains no phase.  This reconstruction intentionally uses
# the measured analytic phase, so it is an observer/plausibility check rather
# than a fully generative waveform model.
payload = pickle.load((REPO_ROOT / "data" / "Dory_2021_04_12_006_preprocessed.pkl").open("rb"))
X_RAW = np.asarray(payload["X"], float)
T_RAW = np.asarray(payload["t"], float)
FS_RAW = float(payload["fs"])
LABELS = [str(v) for v in payload["labels"]]
NCH = len(LABELS)
SMOOTH_TAU = float(metrics.get("state", {}).get("smooth_tau_s", 0.15))

AMP = np.zeros((len(X_RAW), NCH, NB))
PHASE = np.zeros((len(X_RAW), NCH, NB))
BANDSIG = np.zeros((len(X_RAW), NCH, NB))
for b, name in enumerate(BAND_NAMES):
    lo, hi = BANDS[name]
    bb, aa = butter(4, [lo / (0.5 * FS_RAW), hi / (0.5 * FS_RAW)], btype="band")
    filt = filtfilt(bb, aa, X_RAW, axis=0)
    z = hilbert(filt, axis=0)
    AMP[:, :, b] = np.abs(z)
    PHASE[:, :, b] = np.angle(z)
    BANDSIG[:, :, b] = filt
TRUE_TS = BANDSIG.sum(axis=2)


def causal_ewma_full(y: np.ndarray, tau_s: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-1.0 / (tau_s * FS_RAW))
    out = np.empty_like(y)
    acc = y[0].copy()
    for i in range(len(y)):
        acc = acc + alpha * (y[i] - acc)
        out[i] = acc
    return out


SM = np.stack(
    [causal_ewma_full(np.log(AMP[:, :, b] + 1.0e-9), SMOOTH_TAU) for b in range(NB)],
    axis=2,
)
MU = SM.reshape(-1, NCH * NB).mean(axis=0).reshape(NCH, NB)
SD = SM.reshape(-1, NCH * NB).std(axis=0).reshape(NCH, NB) + 1.0e-12

m0 = int(np.argmin(np.abs(T_RAW - t_roll[0])))
m1 = int(np.argmin(np.abs(T_RAW - t_roll[-1])))
win_full = slice(m0, m1 + 1)
t_full = T_RAW[win_full]


def upsample_state(xs: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(t_full, t_roll, xs[:, b]) for b in range(NB)])


def resynth(state_full: np.ndarray) -> np.ndarray:
    out = np.zeros((len(t_full), NCH))
    for c in range(NCH):
        for b in range(NB):
            amp = np.exp(MU[c, b] + SD[c, b] * state_full[:, b])
            out[:, c] += amp * np.cos(PHASE[win_full, c, b])
    return out


def r2_channels(y: np.ndarray, pred: np.ndarray) -> dict:
    per = {}
    for c, lab in enumerate(LABELS):
        per[lab] = r2(y[:, c], pred[:, c])
    return {"mean": float(np.mean(list(per.values()))), "per_channel": per}


state_variants = {
    "measured_state_ceiling": upsample_state(ref),
    "closed_form": upsample_state(closed_traj),
    "closed_plus_beta_delta": upsample_state(baseline_traj),
    "best_additive": upsample_state(best_add_traj),
    "best_gain": upsample_state(best_gain_traj),
    "flat_amplitude": np.zeros((len(t_full), NB)),
}
recon = {name: resynth(state) for name, state in state_variants.items()}
truth_win = TRUE_TS[win_full]
recon_scores = {name: r2_channels(truth_win, pred) for name, pred in recon.items()}

print("\n[TIME SERIES] reconstructed six filtered traces using integrated envelopes + measured phase")
print(f"{'variant':>24} | {'mean R2':>8}")
print("-" * 37)
for name, score in recon_scores.items():
    print(f"{name:>24} | {score['mean']:8.3f}")
print("        note: measured phase is supplied; judge interventions by differences from closed_plus_beta_delta")

fig, axes = plt.subplots(NCH, 1, figsize=(11.5, 8.0), sharex=True, constrained_layout=True)
plot_series = {
    "recording": (truth_win, "0.25", 1.0, "-"),
    "closed+beta": (recon["closed_plus_beta_delta"], "#D55E00", 0.9, "-"),
    "best additive": (recon["best_additive"], "#0072B2", 0.9, "-"),
    "best gain": (recon["best_gain"], "#009E73", 0.9, "-"),
}
for c, lab in enumerate(LABELS):
    ax = axes[c]
    ax.axvspan(*ICTAL, color="#f4a261", alpha=0.12, lw=0)
    ax.axvspan(best_add["start_s"], best_add["start_s"] + PULSE_DURATION, color="#0072B2", alpha=0.08, lw=0)
    for name, (series, color, alpha, ls) in plot_series.items():
        ax.plot(t_full, series[:, c], color=color, alpha=alpha, lw=0.8, ls=ls, label=name)
    ax.set_ylabel(lab, rotation=0, ha="right", va="center")
    ax.set_yticks([])
axes[0].legend(frameon=False, ncol=4, fontsize=7, loc="upper right")
axes[0].set_title(
    "Phase-injected time-series reconstruction from integrated envelopes",
    loc="left",
    fontweight="bold",
)
axes[-1].set_xlabel("recording time (s), onset at 0")
for ext in ("png", "pdf"):
    fig.savefig(RESULT_DIR / f"time_series_reconstruction.{ext}", dpi=250, bbox_inches="tight")
plt.close(fig)

np.savez_compressed(
    RESULT_DIR / "time_series_reconstruction_arrays.npz",
    t_full=t_full,
    truth=truth_win,
    measured_state_reconstruction=recon["measured_state_ceiling"],
    closed_form_reconstruction=recon["closed_form"],
    baseline_reconstruction=recon["closed_plus_beta_delta"],
    best_additive_reconstruction=recon["best_additive"],
    best_gain_reconstruction=recon["best_gain"],
    flat_reconstruction=recon["flat_amplitude"],
    labels=np.asarray(LABELS),
)

gates = {
    "beta_delta_term_improves_delta_transition_r2": {
        "required": "> 0",
        "achieved": {"target_gain": transition_gain, "rate_gain": transition_rate_gain},
        "passed": bool(max(transition_gain, transition_rate_gain) > 0.0),
    },
    "augmented_baseline_still_replays_event": {
        "required": f"rise >= {seizure_threshold:+.3f}",
        "achieved": baseline_st["rise"],
        "passed": bool(baseline_st["seizes"] and baseline_st["viable"]),
    },
    "additive_pulse_finds_viable_termination": {
        "required": ">= 1 viable rollout below seizure threshold",
        "achieved": len(term_add),
        "passed": bool(len(term_add) >= 1),
    },
    "gain_pulse_finds_viable_termination": {
        "required": ">= 1 viable rollout below seizure threshold",
        "achieved": len(term_gain),
        "passed": bool(len(term_gain) >= 1),
    },
    "best_control_reduces_rise": {
        "required": "best rise < augmented baseline rise",
        "achieved": min(best_add["rise"], best_gain["rise"]) - baseline_st["rise"],
        "passed": bool(min(best_add["rise"], best_gain["rise"]) < baseline_st["rise"]),
    },
}

print("\n" + "=" * 88)
all_ok = True
for name, gate in gates.items():
    all_ok &= print_check(
        gate["passed"],
        f"{name}: achieved={gate['achieved']}, required={gate['required']}",
    )

save_metrics(
    RESULT_DIR / "metrics.json",
    {
        "task": "abstract beta->delta intervention on the cross-frequency envelope model",
        "passed": bool(all_ok),
        "source": {
            "arrays": str(ARRAY_PATH),
            "metrics": str(METRIC_PATH),
            "model": "printed closed-form cross-frequency equations plus one beta->delta Heaviside residual term",
        },
        "windows": {
            "roll_start_s": ROLL_START,
            "roll_seconds": n_steps * DT,
            "pulse_duration_s": PULSE_DURATION,
            "start_grid_s": START_GRID.tolist(),
            "ictal": list(ICTAL),
            "transition": list(TRANSITION_WINDOW),
        },
        "beta_delta_term": {
            "coef_f_space": coef_bd,
            "coef_rate_space": coef_bd_rate,
            "threshold": float(THRESH[BETA]),
            "heaviside_k": HEAVISIDE_K,
            "link_delay_s": LINK_DELAY_S,
            "fit_rows": int(len(trainB_rows)),
            "delta_transition_r2_closed": base_transition_r2,
            "delta_transition_r2_target_augmented": aug_transition_r2,
            "delta_transition_r2_rate_augmented": aug_rate_transition_r2,
            "transition_gain_target": transition_gain,
            "transition_gain_rate": transition_rate_gain,
            "note": "coefficient is fit on set-B rows, which exclude the transition window; "
                    "the target/tau scaling is inherited from the source model.",
        },
        "data": {"rise": obs_rise, "seizure_threshold": seizure_threshold},
        "baseline": {"closed_form": closed_st, "closed_plus_beta_delta": baseline_st},
        "time_series_reconstruction": {
            "scores": recon_scores,
            "figure": str(RESULT_DIR / "time_series_reconstruction.png"),
            "arrays": str(RESULT_DIR / "time_series_reconstruction_arrays.npz"),
            "note": "Measured phase is injected because the envelope ODE has no phase state; "
                    "this checks waveform plausibility, not fully generative prediction.",
        },
        "controls": {
            "additive": {
                "definition": "f_delta += c_bd*H_beta, then dx_delta/dt += amp*u(t)*r_bd*H_beta",
                "amp_grid": ADD_AMP_GRID.tolist(),
                "best": best_add,
                "n_terminating": len(term_add),
                "terminating": term_add[:20],
            },
            "gain": {
                "definition": "f_delta += (1 - amp*u(t))*c_bd*H_beta",
                "amp_grid": GAIN_AMP_GRID.tolist(),
                "best": best_gain,
                "n_terminating": len(term_gain),
                "terminating": term_gain[:20],
            },
        },
        "gates": gates,
        "note": "This is an abstract control on standardized band-amplitude dynamics, "
                "not a clinical DBS model. It has no electrode location, volume of "
                "tissue activation, stimulation frequency, or waveform-to-state map.",
    },
)

print(f"[DONE] {RESULT_DIR}")
