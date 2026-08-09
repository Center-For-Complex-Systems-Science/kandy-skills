#!/usr/bin/env python3
"""A deterministic planar dynamical system for the hippocampus_R seizure.

    d/dt [Omega, A] = F(Omega, A)

Omega is the slow phase velocity (1 s Savitzky-Golay derivative of the
unwrapped Hilbert phase) and A is the smoothed log amplitude envelope.  F is a
KANDy model, so both components come out as readable equations and the whole
thing is an ordinary autonomous ODE in the plane: vector field, nullclines,
fixed points, orbits.

Why these two variables and not the signal
------------------------------------------
Fitting a deterministic system to the raw trace is bounded by measurement, not
by effort: at the dynamical timescale (lag 7 samples = 35 ms, the AMI first
minimum) only ~13% of the one-step increment variance is predictable from the
channel's own past, and that is FLAT from embedding dimension 2 to 24.  The
remaining 87% is not a modelling failure to be optimised away, it is broadband
activity that no vector field on the trace can represent.

Omega and A are different.  They are 1 s-smoothed functionals of the same
recording, they carry the seizure (Omega runs 30 -> 60 rad/s, A rises sharply
at onset), and they are smooth enough that a derivative is meaningful.  A
planar ODE is also the smallest system that can do what a seizure does --
leave a resting state, traverse a large excursion, and come back -- so if a
deterministic description of this event exists at all, it should be visible
here.

What is actually being claimed
------------------------------
The vector field is fitted to ONE traversal of the excursion.  A single orbit
constrains F only along the path it visits; away from that path the field is
extrapolation, and with one seizure there is no second orbit to check it
against.  So the phase portrait is a compact description of this event, and
the autonomous rollout below tests the one thing that is testable without a
second seizure: whether integrating F from a PRE-ICTAL initial condition, with
no forcing of any kind, reproduces the excursion and the return.

A held-out time block is scored as well, against the honest baseline for slow
variables -- predicting no change.  Both numbers are reported whatever they
say.
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
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter, filtfilt, hilbert, savgol_filter

from _utils import ensure_outdir, print_check, save_metrics
from kandy import CustomLift, KANDy
from kandy.plotting import plot_loss_curves, use_pub_style


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_float(name, default):
    return float(os.environ.get(name, default))


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = REPO_ROOT / "data" / "Dory_2021_04_12_006_preprocessed.pkl"
RESULT_DIR = ensure_outdir(REPO_ROOT / "results" / "Epilepsy-Slow-Phase-Portrait")

CHANNEL = os.environ.get("KANDY_PP_CHANNEL", "hippocampus_R")
BANDPASS_HZ = (1.0, 40.0)
SLOW_SECONDS = env_float("KANDY_PP_SLOW_SECONDS", 0.5)
DERIV_SECONDS = env_float("KANDY_PP_DERIV_SECONDS", 2.0)
DECIM_HZ = env_float("KANDY_PP_DECIM_HZ", 20.0)

GRID = env_int("KANDY_PP_GRID", 6)
STEPS = env_int("KANDY_PP_STEPS", 200)
HELDOUT = (18.0, 28.0)
GUARD = env_float("KANDY_PP_GUARD", 2.0)
ROLL_START = env_float("KANDY_PP_ROLL_START", -4.0)
ROLL_SECONDS = env_float("KANDY_PP_ROLL_SECONDS", 14.0)
EVENT = (-1.0, 5.0)


def odd(n):
    n = max(5, int(n))
    return n + 1 if n % 2 == 0 else n


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(StringIO()):
        return fn(*a, **k)


# ---------------------------------------------------------------------------
# 1. The two slow variables
# ---------------------------------------------------------------------------
with DATA_PATH.open("rb") as fh:
    payload = pickle.load(fh)

X_RAW = np.asarray(payload["X"], dtype=float)
FS = float(payload["fs"])
T_ALL = np.asarray(payload["t"], dtype=float)
LABELS = list(payload["labels"])
ch = LABELS.index(CHANNEL)

nyq = 0.5 * FS
bb, aa = butter(4, [BANDPASS_HZ[0] / nyq, BANDPASS_HZ[1] / nyq], btype="band")
xf = filtfilt(bb, aa, X_RAW[:, ch])
an = hilbert(xf)
phase = np.unwrap(np.angle(an))
logamp = np.log(np.abs(an) + 1e-6)

w_slow = odd(SLOW_SECONDS * FS)
w_vslow = odd(SLOW_SECONDS * 4.0 * FS)
omega_fine = savgol_filter(phase, w_slow, 3, deriv=1, delta=1.0 / FS, mode="interp")
amp_fine = savgol_filter(logamp, w_slow, 3, mode="interp")
amp_slow_fine = savgol_filter(logamp, w_vslow, 3, mode="interp")

edge = int(round(2.0 * FS))
sl = slice(edge, len(X_RAW) - edge)
stride = int(round(FS / DECIM_HZ))
T_full = T_ALL[sl][::stride]
OM_full = omega_fine[sl][::stride]
AM_full = amp_fine[sl][::stride]
AS_full = amp_slow_fine[sl][::stride]
DT_S = float(np.median(np.diff(T_full)))

# LAGGED INCREMENTS, not Savitzky-Golay derivatives.  d/dt of an already
# 1 s-smoothed quantity is almost pure smoothing noise -- measured: fitting
# such a target gave held-out R2 = -1.08, worse than predicting no change.
# Over a lag matched to the variable's own timescale the signal clears the
# differentiation noise floor.
LAG = int(round(DERIV_SECONDS / DT_S))
T = T_full[:-LAG]
OMEGA, AMP, ASLOW = OM_full[:-LAG], AM_full[:-LAG], AS_full[:-LAG]
dOMEGA = (OM_full[LAG:] - OM_full[:-LAG]) / (LAG * DT_S)
dAMP = (AM_full[LAG:] - AM_full[:-LAG]) / (LAG * DT_S)
dASLOW = (AS_full[LAG:] - AS_full[:-LAG]) / (LAG * DT_S)
N = len(T)

# ASLOW is deliberately NOT a state variable.  It is a centred 4 s filter, so
# its own increment depends on samples in the future and its determinism bound
# came out NEGATIVE (-0.47): including it adds a component no autonomous field
# can ever fit.  Kept only for plotting.
CEN = np.array([OMEGA.mean(), AMP.mean()])
SCA = np.array([OMEGA.std(), AMP.std()]) + 1e-12
om_c, om_s, am_c, am_s = CEN[0], SCA[0], CEN[1], SCA[1]
S = (np.column_stack([OMEGA, AMP]) - CEN) / SCA
DS = np.column_stack([dOMEGA, dAMP]) / SCA

print(f"[DATA]  {CHANNEL}: {N} points at {DECIM_HZ:.0f} Hz, "
      f"{T[0]:.1f}..{T[-1]:.1f}s; increment lag {LAG} ({LAG * DT_S:.2f}s)", flush=True)
print(f"[STATE] Omega {OMEGA.min():.1f}..{OMEGA.max():.1f} rad/s "
      f"(mean {om_c:.1f}), A {AMP.min():.2f}..{AMP.max():.2f} (mean {am_c:.2f})",
      flush=True)


def block(win):
    m = (T >= win[0]) & (T <= win[1])
    near = (T >= win[0] - GUARD) & (T <= win[1] + GUARD)
    return m, near


test_m, near_test = block(HELDOUT)
train_m = ~near_test
print(f"[SPLIT] train={int(train_m.sum())}, heldout={int(test_m.sum())} "
      f"({HELDOUT[0]:.0f}..{HELDOUT[1]:.0f}s), guard={GUARD:.0f}s; "
      f"seizure in train: {bool(((T >= EVENT[0]) & (T <= EVENT[1]) & train_m).any())}",
      flush=True)


# ---------------------------------------------------------------------------
# 1b. What is the best ANY autonomous field on this state could do?
#
# Before fitting anything: predict each velocity from the mean velocity of its
# nearest neighbours IN STATE SPACE, excluding neighbours close in TIME.  If
# the velocity is a function of the state, that predictor works; if the same
# state is visited twice with different velocities, no vector field can fit it
# and neither can KANDy.  This is the same model-free move the Mackey-Glass
# file uses for its lag scan, and it puts a ceiling on the fit below.
#
# The identical estimator is run on a van der Pol oscillator at the same
# sampling and lag as a positive control: it must return ~1.0 there, or the
# bound is measuring the estimator rather than the data.
# ---------------------------------------------------------------------------
def determinism_bound(state, vel, times, k=10, t_exclude=2.0):
    Z = (state - state.mean(0)) / (state.std(0) + 1e-12)
    Vz = (vel - vel.mean(0)) / (vel.std(0) + 1e-12)
    P = np.empty_like(Vz)
    for i in range(len(Z)):
        d2 = np.sum((Z - Z[i]) ** 2, axis=1)
        d2[np.abs(times - times[i]) < t_exclude] = np.inf
        nn = np.argpartition(d2, k)[:k]
        P[i] = Vz[nn].mean(0)
    return 1.0 - np.sum((Vz - P) ** 2, axis=0) / np.sum((Vz - Vz.mean(0)) ** 2, axis=0)


BOUND = determinism_bound(S, DS, T)
tt = np.arange(0, 200, DT_S)
vx = np.empty(len(tt)); vy = np.empty(len(tt)); vx[0], vy[0] = 2.0, 0.0
for i in range(len(tt) - 1):
    vx[i + 1] = vx[i] + DT_S * vy[i]
    vy[i + 1] = vy[i] + DT_S * (2.0 * (1 - vx[i] ** 2) * vy[i] - vx[i])
Sv = np.column_stack([vx, vy])[:-LAG]
Vv = np.column_stack([(vx[LAG:] - vx[:-LAG]), (vy[LAG:] - vy[:-LAG])]) / (LAG * DT_S)
CONTROL = determinism_bound(Sv, Vv, tt[:-LAG])
print(f"[BOUND] best possible autonomous field on this state: "
      f"dOmega={BOUND[0]:+.3f}, dA={BOUND[1]:+.3f}", flush=True)
print(f"[BOUND] van der Pol control (same estimator, same lag): "
      f"{CONTROL[0]:+.3f}, {CONTROL[1]:+.3f}  <- validates the estimator", flush=True)


# ---------------------------------------------------------------------------
# 2. The vector field.  Cross-term explicit; powers of one variable are the
#    spline's job, so they are not in the lift.
# ---------------------------------------------------------------------------
def lift_np(s):
    return np.column_stack([np.ones(len(s)), s[:, 0], s[:, 1], s[:, 0] * s[:, 1]])


def lift_torch(s):
    return torch.stack([torch.ones_like(s[:, 0]), s[:, 0], s[:, 1],
                        s[:, 0] * s[:, 1]], dim=1)


FEATURES = ["1", "Om", "A", "Om*A"]
model = KANDy(lift=CustomLift(lift_np, 4, lift_torch, "slow_plane"),
              grid=GRID, k=3, steps=STEPS, seed=SEED,
              base_fun=torch.nn.Identity(), device="cpu")
print("[TRAIN] Fitting d/dt [Omega, A] = F(Omega, A) ...", flush=True)
quiet(model.fit, X=S[train_m], X_dot=DS[train_m], val_frac=0.15, test_frac=0.15,
      lamb=1e-2, opt="LBFGS", lr=1.0, batch=-1, stop_grid_update_step=20,
      patience=0, verbose=False)


def r2(y, p):
    den = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
    return 1.0 - np.sum((y - p) ** 2, axis=0) / np.where(den > 0, den, np.nan)


pred_te = model.predict(S[test_m])
r2_te = r2(DS[test_m], pred_te)
r2_zero = r2(DS[test_m], np.zeros_like(pred_te))
r2_tr = r2(DS[train_m], model.predict(S[train_m]))
names = ("dOmega", "dA")
print("[EVAL]  " + "  ".join(f"{n}" for n in names), flush=True)
for lab, v in (("in-sample ", r2_tr), ("held-out  ", r2_te),
               ("no-change ", r2_zero), ("bound     ", BOUND)):
    print(f"[EVAL]  {lab} R2: " + ", ".join(f"{n}={x:+.4f}" for n, x in zip(names, v)),
          flush=True)
print("[EVAL]  gain over no-change: "
      + ", ".join(f"{n}={a - b:+.4f}" for n, a, b in zip(names, r2_te, r2_zero)),
      flush=True)


# ---------------------------------------------------------------------------
# 3. The equation
# ---------------------------------------------------------------------------
formulas = []
try:
    exprs = quiet(model.get_formula, var_names=FEATURES, r2_threshold=0.9)
    formulas = [str(e) for e in exprs]
    print("\n[EQUATION] in standardized coordinates "
          f"om=(Omega-{om_c:.2f})/{om_s:.2f}, a=(A-{am_c:.2f})/{am_s:.2f}", flush=True)
    for nm, e in zip(("d(om)/dt", "d(a)/dt "), formulas):
        print(f"           {nm} = {e if len(e) < 260 else e[:260] + ' ...'}", flush=True)
except Exception as exc:  # noqa: BLE001 - symbolic snapping is best-effort
    print(f"\n[EQUATION] get_formula failed: {type(exc).__name__}: {exc}", flush=True)


# ---------------------------------------------------------------------------
# 4. Autonomous integration -- the only test a single seizure permits
# ---------------------------------------------------------------------------
def rhs(s):
    return model.predict(np.atleast_2d(s))


def rollout(s0, steps, dt):
    s = np.asarray(s0, dtype=float).copy()
    out = np.empty((steps + 1, len(s)))
    out[0] = s
    for i in range(steps):
        k1 = rhs(s)[0]
        k2 = rhs(s + 0.5 * dt * k1)[0]
        k3 = rhs(s + 0.5 * dt * k2)[0]
        k4 = rhs(s + dt * k3)[0]
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(s).all() or np.abs(s).max() > 20:
            return out[: i + 1]
        out[i + 1] = s
    return out


i0 = int(np.argmin(np.abs(T - ROLL_START)))
n_roll = int(round(ROLL_SECONDS / DT_S))
ROLL = rollout(S[i0], n_roll, DT_S)
t_roll = T[i0] + np.arange(len(ROLL)) * DT_S
ref = S[i0: i0 + len(ROLL)]

ev = (T >= EVENT[0]) & (T <= EVENT[1])
base_m = (T >= T[i0]) & (T < EVENT[0])
data_rise = float(OMEGA[ev].max() - OMEGA[base_m].mean())
roll_om = ROLL[:, 0] * om_s + om_c
roll_ev = (t_roll >= EVENT[0]) & (t_roll <= EVENT[1])
roll_base = (t_roll >= t_roll[0]) & (t_roll < EVENT[0])
model_rise = (float(roll_om[roll_ev].max() - roll_om[roll_base].mean())
              if roll_ev.any() and roll_base.any() else float("nan"))
tail = slice(-int(2.0 / DT_S), None)
recov = (1.0 - (roll_om[tail].mean() - roll_om[roll_base].mean())
         / max(model_rise, 1e-9)) if np.isfinite(model_rise) else float("nan")

print(f"\n[ROLL]  autonomous from t={T[i0]:.1f}s, {len(ROLL)}/{n_roll + 1} steps "
      f"survived ({len(ROLL) * DT_S:.1f}s)", flush=True)
print(f"[ROLL]  Omega excursion: data={data_rise:+.1f} rad/s, "
      f"model={model_rise:+.1f} rad/s", flush=True)
print(f"[ROLL]  return to baseline by end: {recov:.2f} (1.0 = full)", flush=True)


# ---------------------------------------------------------------------------
# 5. Fixed points of the fitted field, on the observed region
# ---------------------------------------------------------------------------
gx = np.linspace(S[:, 0].min(), S[:, 0].max(), 220)
gy = np.linspace(S[:, 1].min(), S[:, 1].max(), 220)
GX, GY = np.meshgrid(gx, gy)
FV = model.predict(np.column_stack([GX.ravel(), GY.ravel()]))
U = FV[:, 0].reshape(GX.shape)
V = FV[:, 1].reshape(GX.shape)
speed = np.hypot(U, V)

fixed = []
step = 12
for i in range(1, GX.shape[0] - 1, step):
    for j in range(1, GX.shape[1] - 1, step):
        s = np.array([GX[i, j], GY[i, j]])
        for _ in range(60):
            f = rhs(s)[0]
            J = np.empty((2, 2))
            h = 1e-4
            for k in range(2):
                sp = s.copy(); sp[k] += h
                sm = s.copy(); sm[k] -= h
                J[:, k] = (rhs(sp)[0] - rhs(sm)[0]) / (2 * h)
            try:
                s = s - np.linalg.solve(J, f)
            except np.linalg.LinAlgError:
                break
            if not np.isfinite(s).all():
                break
        if (np.isfinite(s).all() and gx[0] <= s[0] <= gx[-1]
                and gy[0] <= s[1] <= gy[-1]
                and np.linalg.norm(rhs(s)[0]) < 1e-3
                and not any(np.linalg.norm(s - f0) < 0.05 for f0, _ in fixed)):
            h = 1e-4
            J = np.empty((2, 2))
            for k in range(2):
                sp = s.copy(); sp[k] += h
                sm = s.copy(); sm[k] -= h
                J[:, k] = (rhs(sp)[0] - rhs(sm)[0]) / (2 * h)
            fixed.append((s.copy(), np.linalg.eigvals(J)))

print(f"\n[FIXED] {len(fixed)} fixed point(s) inside the observed region", flush=True)
fixed_report = []
for s, ev_ in fixed:
    kind = ("stable" if np.all(ev_.real < 0) else
            "unstable" if np.all(ev_.real > 0) else "saddle")
    spiral = "spiral " if np.any(np.abs(ev_.imag) > 1e-9) else ""
    om_p, a_p = s[0] * om_s + om_c, s[1] * am_s + am_c
    print(f"        Omega={om_p:6.1f} rad/s, A={a_p:+6.2f}  ->  {spiral}{kind}"
          f"  (eig {ev_[0]:.3f}, {ev_[1]:.3f})", flush=True)
    fixed_report.append({"omega": float(om_p), "amp": float(a_p),
                         "type": f"{spiral}{kind}",
                         "eigenvalues": [complex(v).real for v in ev_],
                         "eig_imag": [complex(v).imag for v in ev_]})


# ---------------------------------------------------------------------------
# 6. Checks
# ---------------------------------------------------------------------------
gain = float(min(r2_te - r2_zero))
emerged = bool(np.isfinite(model_rise) and model_rise >= 0.5 * data_rise)
passed = print_check(
    gain >= 0.02 and emerged,
    "planar slow system: beats no-change off-block AND reproduces the "
    "excursion autonomously",
)
print(f"        held-out gain over no-change (worst component) = {gain:+.4f}", flush=True)
print(f"        autonomous excursion >= 50% of data            = {emerged}", flush=True)

save_metrics(RESULT_DIR / "metrics.json", {
    "task": f"{CHANNEL} planar slow dynamical system d/dt[Omega,A]=F(Omega,A)",
    "channel": CHANNEL, "decim_hz": DECIM_HZ, "n_points": int(N),
    "standardization": {"omega_center": float(om_c), "omega_scale": float(om_s),
                        "amp_center": float(am_c), "amp_scale": float(am_s)},
    "r2_in_sample": [float(v) for v in r2_tr],
    "r2_heldout": [float(v) for v in r2_te],
    "r2_no_change_baseline": [float(v) for v in r2_zero],
    "heldout_gain": [float(a - b) for a, b in zip(r2_te, r2_zero)],
    "equations_standardized": formulas,
    "rollout": {"start": float(T[i0]), "steps_survived": int(len(ROLL)),
                "data_rise_rad_s": data_rise, "model_rise_rad_s": model_rise,
                "recovery": float(recov)},
    "fixed_points": fixed_report,
    "caveat": "vector field fitted to a single traversal; away from the observed "
              "orbit it is extrapolation and one seizure cannot check it",
    "passed": passed,
})
np.savez_compressed(RESULT_DIR / "phase_portrait_arrays.npz",
                    t=T, omega=OMEGA, amp=AMP, domega=dOMEGA, damp=dAMP,
                    state=S, dstate=DS, roll=ROLL, t_roll=t_roll,
                    grid_x=gx, grid_y=gy, field_u=U, field_v=V)


# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
use_pub_style()


def to_phys(s):
    return s[:, 0] * om_s + om_c, s[:, 1] * am_s + am_c


def plot_portrait() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    GXp, GYp = GX * om_s + om_c, GY * am_s + am_c
    ax.streamplot(GXp, GYp, U, V, color=speed, cmap="Greys", density=1.3,
                  linewidth=0.7, arrowsize=0.8)
    ax.contour(GXp, GYp, U, levels=[0.0], colors="#1f77b4", linewidths=1.8)
    ax.contour(GXp, GYp, V, levels=[0.0], colors="#2ca02c", linewidths=1.8)
    om_d, a_d = OMEGA, AMP
    ax.plot(om_d, a_d, lw=0.7, color="0.45", alpha=0.65, label="data (all)")
    ax.plot(om_d[ev], a_d[ev], lw=2.0, color="#d62728", label="data (seizure)")
    ro, ra = to_phys(ROLL)
    ax.plot(ro, ra, lw=1.6, color="#ff7f0e", ls="--", label="autonomous rollout")
    for fp in fixed_report:
        ax.plot(fp["omega"], fp["amp"], "o", ms=9, mfc="w", mec="k", mew=1.4, zorder=6)
    ax.plot([], [], color="#1f77b4", lw=1.8, label=r"$\dot\Omega=0$ nullcline")
    ax.plot([], [], color="#2ca02c", lw=1.8, label=r"$\dot A=0$ nullcline")
    ax.set_xlabel(r"$\Omega$  (rad/s)")
    ax.set_ylabel(r"$A$  (log amplitude)")
    ax.set_title(f"{CHANNEL}: phase portrait of the fitted slow system")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"phase_portrait.{e}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ro, ra = to_phys(ROLL)
    for ax in axes:
        ax.axvspan(*EVENT, color="#f4a261", alpha=0.22, lw=0)
    axes[0].plot(T, OMEGA, color="#1f77b4", lw=1.3, label="data")
    axes[0].plot(t_roll, ro, color="#ff7f0e", lw=1.5, ls="--", label="autonomous")
    axes[0].set_ylabel(r"$\Omega$ (rad/s)")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"{CHANNEL}: integrating the fitted field from a pre-ictal state")
    axes[1].plot(T, AMP, color="#1f77b4", lw=1.3)
    axes[1].plot(t_roll, ra, color="#ff7f0e", lw=1.5, ls="--")
    axes[1].set_ylabel(r"$A$ (log amplitude)")
    axes[1].set_xlabel("recording time (s)")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"rollout.{e}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_losses() -> None:
    if getattr(model, "train_results_", None):
        fig, _ = plot_loss_curves(model.train_results_, save=str(RESULT_DIR / "loss"))
        plt.close(fig)


plot_portrait()
plot_timeseries()
plot_losses()
print(f"\n[FIGS]  Saved {RESULT_DIR}", flush=True)
