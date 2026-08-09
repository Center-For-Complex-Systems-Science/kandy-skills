#!/usr/bin/env python3
"""Kuramoto-Sakaguchi fit to the epilepsy recording, with an identifiable readout.

Why this model and not a delay embedding
----------------------------------------
A model-free lag scan on this recording (the same binned-mean scan
``odes/mackey_glass_delay_example.py`` uses to recover tau) finds NO informative
delay, in either the band-passed signal or the log-amplitude envelope, at any
lag from 5 ms to 5 s.  The only cross-channel structure sits at 10-20 ms, which
at the measured intrinsic frequencies (omega ~ 43-55 rad/s) is a phase offset of
0.44-0.86 rad -- not a delay term.  A fixed phase offset in a coupled-oscillator
model is exactly the Sakaguchi frustration parameter, so that is what is fitted
here:

    dtheta_i/dt = omega_i + sum_{j != i} kappa_ij * sin(theta_j - theta_i + alpha)

The four channels are the ones whose cross-channel lag-scan gain rose above the
noise floor.

Why the coefficients are identifiable
-------------------------------------
Expanding the Sakaguchi term,

    kappa_ij * sin(d + alpha) = (kappa_ij cos alpha) sin(d) + (kappa_ij sin alpha) cos(d)

so the lift needs exactly two columns per UNORDERED pair (a, b), a < b:

    sin(d_ab), cos(d_ab)      with d_ab = theta_b - theta_a

Row a of the readout picks up (+kappa cos alpha, +kappa sin alpha); row b picks
up (-kappa cos alpha, +kappa sin alpha), because sin is odd and cos is even.
Both kappa and alpha therefore come back in closed form, per directed edge:

    kappa_{a<-b} = hypot(A_a, B_a),   alpha_{a<-b} = atan2(B_a,  A_a)
    kappa_{b<-a} = hypot(A_b, B_b),   alpha_{b<-a} = atan2(B_b, -A_b)

This is the property the adaptive-kappa variants lack: there, kappa is a pairwise
log-amplitude average, which is a rank-N linear map of N channel envelopes, so
the kappa columns are rank-deficient and their coefficients are not identifiable.
Constant kappa is the price of getting a transportable equation.

Structural check that costs nothing
-----------------------------------
Under the Kuramoto-Sakaguchi hypothesis the columns for pair (a, b) may load
only on rows a and b.  Mass on any other row is structure violation, and it is
reported as ``off_structure_fraction``.

Validation protocol
-------------------
Every number is quoted against a baseline computed on the identical held-out
rows, and every gate is defined RELATIVE to a baseline rather than as an
absolute constant:

    omega-only        dtheta_i/dt = omega_i          (no coupling at all)
    linear K-S        closed-form fit in the lift above
    KANDy             spline edges on the same lift
    persistence       dtheta_i/dt(t) = dtheta_i/dt(t-1)   [reference only --
                      it uses the past derivative, which the models do not see]

Splits are chronological with a guard band wide enough to cover the smoothing
windows, so no held-out row shares filter support with a training row.  Rollouts
are autonomous: the model gets an initial phase vector and nothing else -- no
q(t), no replay forcing, no trajectory nudging.  The decisive test is whether
the coupled model beats the UNCOUPLED omega-only rollout; if it does not, the
coupling terms are decoration.

Split modes (``KANDY_KS_SPLIT_MODE``)
    forward        train on the first 70%, test on the last 20% (default)
    event_holdout  train on everything outside the seizure window, test ON it
                   -- the digital-twin question: does an off-event fit predict
                   the event?

Known limitation: one recording, one seizure.  ``event_holdout`` tests
generalization to the event but cannot test generalization across events.
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

from _utils import ensure_outdir, print_check, save_metrics, wrap_pi
from kandy import CustomLift, KANDy
from kandy.plotting import plot_loss_curves, use_pub_style


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# ---------------------------------------------------------------------------
# 0. Parameters
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = REPO_ROOT / "data" / "Dory_2021_04_12_006_preprocessed.pkl"
RESULT_DIR = ensure_outdir(REPO_ROOT / "results" / "Epilepsy-Kuramoto-Sakaguchi")

CHANNELS = os.environ.get(
    "KANDY_KS_CHANNELS", "motor_R,posterior_R,striatum_R,hippocampus_R"
).split(",")
SPLIT_MODE = os.environ.get("KANDY_KS_SPLIT_MODE", "forward").strip().lower()

BANDPASS_HZ = (1.0, 40.0)
FILTER_ORDER = 4
EDGE_TRIM_SECONDS = 2.0
PHASE_SAVGOL_SECONDS = 0.155

TRAIN_FRAC = env_float("KANDY_KS_TRAIN_FRAC", 0.70)
VAL_FRAC = env_float("KANDY_KS_VAL_FRAC", 0.10)
GUARD_SECONDS = env_float("KANDY_KS_GUARD_SECONDS", 1.0)
EVENT_WINDOW = (-1.0, 4.0)

GRID = env_int("KANDY_KS_GRID", 4)
FIT_STEPS = env_int("KANDY_KS_STEPS", 60)
N_TRAIN_MAX = env_int("KANDY_KS_N_TRAIN_MAX", 4000)
FORMULA_R2_THRESHOLD = env_float("KANDY_KS_FORMULA_R2", 0.8)

ROLLOUT_HORIZONS = [0.10, 0.25, 0.50, 1.00]
N_ROLLOUT_STARTS = env_int("KANDY_KS_N_ROLLOUT_STARTS", 40)

# Gates are margins over a baseline, never absolute constants.
GATE_COUPLING_R2_GAIN = env_float("KANDY_KS_GATE_R2_GAIN", 0.02)
GATE_ROLLOUT_IMPROVEMENT = env_float("KANDY_KS_GATE_ROLLOUT", 0.05)


# ---------------------------------------------------------------------------
# 1. Data, phases, phase velocities
# ---------------------------------------------------------------------------
def odd_window(seconds: float, fs: float, minimum: int = 5) -> int:
    win = max(minimum, int(round(seconds * fs)))
    return win + 1 if win % 2 == 0 else win


def bandpass(X: np.ndarray, fs: float) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(
        FILTER_ORDER, [BANDPASS_HZ[0] / nyq, BANDPASS_HZ[1] / nyq], btype="band"
    )
    return filtfilt(b, a, X, axis=0)


def quiet_fit(model: KANDy, **kwargs) -> None:
    with contextlib.redirect_stdout(StringIO()):
        model.fit(**kwargs)


def r2_per_output(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    denom = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    denom = np.where(denom > 0, denom, np.nan)
    return 1.0 - np.sum((y_true - y_pred) ** 2, axis=0) / denom


with DATA_PATH.open("rb") as fh:
    payload = pickle.load(fh)

X_ALL = np.asarray(payload["X"], dtype=float)
FS = float(payload["fs"])
DT = float(payload["dt"])
T_ALL = np.asarray(payload["t"], dtype=float)
ALL_LABELS = list(payload["labels"])

missing = [name for name in CHANNELS if name not in ALL_LABELS]
if missing:
    raise ValueError(f"Unknown channels {missing}; recording has {ALL_LABELS}")
CH_IDX = [ALL_LABELS.index(name) for name in CHANNELS]
LABELS = list(CHANNELS)
M = len(LABELS)

PAIRS = [(a, b) for a in range(M) for b in range(a + 1, M)]
N_PAIR = len(PAIRS)
FEATURE_NAMES = [
    name for a, b in PAIRS for name in (f"sin(d{a}{b})", f"cos(d{a}{b})")
]

print("[DATA]  Loaded epilepsy recording", flush=True)
print(f"        {len(X_ALL)} samples, fs={FS:.1f} Hz, dt={DT:.4f}s", flush=True)
print(f"[SEL]   {M} channels: {LABELS}", flush=True)
print(f"        selected by model-free cross-channel lag-scan gain", flush=True)

X_FILT = bandpass(X_ALL[:, CH_IDX], FS)
analytic = hilbert(X_FILT, axis=0)
theta_unwrapped = np.unwrap(np.angle(analytic), axis=0)

phase_window = odd_window(PHASE_SAVGOL_SECONDS, FS)
theta_s = savgol_filter(theta_unwrapped, phase_window, 3, axis=0, mode="interp")
theta_dot_s = savgol_filter(
    theta_unwrapped, phase_window, 3, deriv=1, delta=DT, axis=0, mode="interp"
)

edge = int(round(EDGE_TRIM_SECONDS * FS))
valid = slice(edge, len(X_ALL) - edge)
T = T_ALL[valid]
THETA = theta_s[valid]
THETA_DOT = theta_dot_s[valid]
SIGNAL = X_FILT[valid]
N_ROW = len(T)

print(
    f"[PRE]   band-pass {BANDPASS_HZ[0]:.0f}-{BANDPASS_HZ[1]:.0f} Hz, "
    f"phase window {phase_window} samples, {N_ROW} rows "
    f"({T[0]:.1f}..{T[-1]:.1f}s)",
    flush=True,
)


# ---------------------------------------------------------------------------
# 2. Chronological split with a guard band
# ---------------------------------------------------------------------------
# The guard must exceed the widest smoothing support, or held-out rows share
# filter taps with training rows and the "held-out" R2 is inflated.
guard = max(int(round(GUARD_SECONDS * FS)), phase_window)
rows = np.arange(N_ROW)

if SPLIT_MODE == "forward":
    n_tr = int(N_ROW * TRAIN_FRAC)
    n_va = int(N_ROW * VAL_FRAC)
    train_rows = rows[: n_tr - guard]
    val_rows = rows[n_tr : n_tr + n_va - guard]
    test_rows = rows[n_tr + n_va + guard :]
elif SPLIT_MODE == "event_holdout":
    in_event = (T >= EVENT_WINDOW[0]) & (T <= EVENT_WINDOW[1])
    near_event = (T >= EVENT_WINDOW[0] - GUARD_SECONDS) & (
        T <= EVENT_WINDOW[1] + GUARD_SECONDS
    )
    test_rows = rows[in_event]
    outside = rows[~near_event]
    n_va = int(len(outside) * VAL_FRAC)
    train_rows = outside[:-n_va] if n_va else outside
    val_rows = outside[-n_va:] if n_va else outside[:0]
else:
    raise ValueError("KANDY_KS_SPLIT_MODE must be 'forward' or 'event_holdout'")

if len(train_rows) > N_TRAIN_MAX:
    rng = np.random.default_rng(SEED)
    fit_rows = np.sort(rng.choice(train_rows, N_TRAIN_MAX, replace=False))
else:
    fit_rows = train_rows

print(
    f"[SPLIT] mode={SPLIT_MODE}, guard={guard} samples ({guard * DT:.2f}s)",
    flush=True,
)
print(
    f"        train={len(train_rows)} ({T[train_rows[0]]:.1f}..{T[train_rows[-1]]:.1f}s), "
    f"val={len(val_rows)}, test={len(test_rows)} "
    f"({T[test_rows[0]]:.1f}..{T[test_rows[-1]]:.1f}s)",
    flush=True,
)
event_in_train = bool(
    np.any((T[train_rows] >= EVENT_WINDOW[0]) & (T[train_rows] <= EVENT_WINDOW[1]))
)
print(f"        seizure window in TRAIN: {event_in_train}", flush=True)


# ---------------------------------------------------------------------------
# 3. The Kuramoto-Sakaguchi lift
# ---------------------------------------------------------------------------
def ks_lift_np(theta: np.ndarray) -> np.ndarray:
    """[sin(d_ab), cos(d_ab)] for each unordered pair, d_ab = theta_b - theta_a."""
    feats = []
    for a, b in PAIRS:
        d = theta[:, b] - theta[:, a]
        feats.extend([np.sin(d), np.cos(d)])
    return np.column_stack(feats)


def ks_lift_torch(theta: torch.Tensor) -> torch.Tensor:
    feats = []
    for a, b in PAIRS:
        d = theta[:, b] - theta[:, a]
        feats.extend([torch.sin(d), torch.cos(d)])
    return torch.stack(feats, dim=1)


PHI = ks_lift_np(THETA)
design = np.column_stack([np.ones(N_ROW), PHI])
print(
    f"[LIFT]  {len(FEATURE_NAMES)} features "
    f"({N_PAIR} pairs x 2), design cond={np.linalg.cond(design[train_rows]):.1f}",
    flush=True,
)


# ---------------------------------------------------------------------------
# 4. Closed-form Kuramoto-Sakaguchi fit -> omega, kappa, alpha
# ---------------------------------------------------------------------------
coef, *_ = np.linalg.lstsq(design[train_rows], THETA_DOT[train_rows], rcond=None)
C = coef.T                                    # (M, 1 + 2*N_PAIR)
OMEGA_HAT = C[:, 0].copy()

kappa_dir = np.zeros((M, M))                  # kappa[i, j]: j drives i
alpha_dir = np.full((M, M), np.nan)
off_mass = 0.0
tot_mass = 0.0
for p, (a, b) in enumerate(PAIRS):
    cs, cc = 1 + 2 * p, 2 + 2 * p
    Aa, Ba = C[a, cs], C[a, cc]
    Ab, Bb = C[b, cs], C[b, cc]
    kappa_dir[a, b] = float(np.hypot(Aa, Ba))
    alpha_dir[a, b] = float(np.arctan2(Ba, Aa))
    kappa_dir[b, a] = float(np.hypot(Ab, Bb))
    alpha_dir[b, a] = float(np.arctan2(Bb, -Ab))
    col_mass = np.abs(C[:, [cs, cc]]).sum()
    tot_mass += col_mass
    off_mass += col_mass - np.abs(C[[a, b]][:, [cs, cc]]).sum()

off_structure_fraction = float(off_mass / max(tot_mass, 1e-12))

# Weighted circular mean of alpha across directed edges.
w = np.array([kappa_dir[i, j] for i in range(M) for j in range(M) if i != j])
al = np.array([alpha_dir[i, j] for i in range(M) for j in range(M) if i != j])
ALPHA = float(np.arctan2(np.sum(w * np.sin(al)), np.sum(w * np.cos(al))))
alpha_spread = float(
    1.0 - np.abs(np.sum(w * np.exp(1j * al))) / max(np.sum(w), 1e-12)
)

print(
    f"[READOUT] alpha per edge -> shared alpha = {ALPHA:+.3f} rad "
    f"({np.degrees(ALPHA):+.0f} deg), circular spread={alpha_spread:.3f}",
    flush=True,
)
print(
    f"[READOUT] off-structure coefficient mass = {off_structure_fraction:.3f} "
    "(0 = perfect Kuramoto-Sakaguchi structure)",
    flush=True,
)


def shared_alpha_refit(alpha: float, fit_idx: np.ndarray):
    """Refit omega_i and kappa_ij with alpha held fixed, one row at a time."""
    omega = np.zeros(M)
    kappa = np.zeros((M, M))
    for i in range(M):
        partners = [j for j in range(M) if j != i]
        cols = [np.ones(len(fit_idx))]
        for j in partners:
            d = THETA[fit_idx, j] - THETA[fit_idx, i]
            cols.append(np.sin(d + alpha))
        A = np.column_stack(cols)
        sol, *_ = np.linalg.lstsq(A, THETA_DOT[fit_idx, i], rcond=None)
        omega[i] = sol[0]
        for k, j in enumerate(partners):
            kappa[i, j] = sol[1 + k]
    return omega, kappa


OMEGA_KS, KAPPA_KS = shared_alpha_refit(ALPHA, train_rows)


def ks_rhs(theta: np.ndarray, omega: np.ndarray, kappa: np.ndarray, alpha: float):
    """dtheta/dt for the shared-alpha Kuramoto-Sakaguchi model."""
    th = np.atleast_2d(theta)
    out = np.tile(omega, (len(th), 1))
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            out[:, i] += kappa[i, j] * np.sin(th[:, j] - th[:, i] + alpha)
    return out


print("\n[EQUATION] shared-alpha Kuramoto-Sakaguchi fit (train rows only)")
print(f"           dtheta_i/dt = omega_i + sum_j kappa_ij*sin(theta_j-theta_i+alpha)")
print(f"           alpha = {ALPHA:+.4f} rad")
for i in range(M):
    terms = "  ".join(
        f"{KAPPA_KS[i, j]:+.3f}*sin(th_{j}-th_{i}+a)" for j in range(M) if j != i
    )
    print(
        f"           dth_{i}/dt [{LABELS[i]:>14}] = {OMEGA_KS[i]:+7.3f} "
        f"({OMEGA_KS[i] / (2 * np.pi):5.2f} Hz)  {terms}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# 5. KANDy on the same lift
# ---------------------------------------------------------------------------
phase_center = THETA_DOT[train_rows].mean(axis=0)
phase_scale = THETA_DOT[train_rows].std(axis=0) + 1e-12

model = KANDy(
    lift=CustomLift(
        fn=ks_lift_np,
        output_dim=len(FEATURE_NAMES),
        torch_fn=ks_lift_torch,
        name="kuramoto_sakaguchi_pairs",
    ),
    grid=GRID,
    k=3,
    steps=FIT_STEPS,
    seed=SEED,
    base_fun=torch.nn.Identity(),
    device="cpu",
)

print("\n[TRAIN] Fitting KANDy on the Kuramoto-Sakaguchi lift ...", flush=True)
quiet_fit(
    model,
    X=THETA[fit_rows],
    X_dot=(THETA_DOT[fit_rows] - phase_center) / phase_scale,
    val_frac=0.15,
    test_frac=0.15,
    lamb=0.0,
    opt="LBFGS",
    lr=1.0,
    batch=-1,
    stop_grid_update_step=10,
    patience=0,
    verbose=False,
)


def kandy_rhs(theta: np.ndarray) -> np.ndarray:
    return model.predict(np.atleast_2d(theta)) * phase_scale + phase_center


# ---------------------------------------------------------------------------
# 6. Held-out derivative skill vs. baselines
# ---------------------------------------------------------------------------
y_test = THETA_DOT[test_rows]
pred_omega_only = np.tile(OMEGA_KS, (len(test_rows), 1))
pred_linear_ks = ks_rhs(THETA[test_rows], OMEGA_KS, KAPPA_KS, ALPHA)
pred_linear_raw = design[test_rows] @ C.T
pred_kandy = kandy_rhs(THETA[test_rows])
prev_rows = np.maximum(test_rows - 1, 0)
pred_persistence = THETA_DOT[prev_rows]

scores = {
    "omega_only": r2_per_output(y_test, pred_omega_only),
    "linear_ks_shared_alpha": r2_per_output(y_test, pred_linear_ks),
    "linear_free_sin_cos": r2_per_output(y_test, pred_linear_raw),
    "kandy": r2_per_output(y_test, pred_kandy),
    "persistence_reference": r2_per_output(y_test, pred_persistence),
}

print("\n[EVAL]  held-out phase-velocity R2 (test rows only)")
for name, val in scores.items():
    tag = "  <- reference, uses past derivative" if "persistence" in name else ""
    print(
        f"        {name:>24}: mean={np.nanmean(val):+.4f}  "
        f"per-channel={np.round(val, 3)}{tag}",
        flush=True,
    )

coupling_gain = float(
    np.nanmean(scores["linear_ks_shared_alpha"]) - np.nanmean(scores["omega_only"])
)
kandy_gain = float(np.nanmean(scores["kandy"]) - np.nanmean(scores["linear_ks_shared_alpha"]))
print(
    f"        coupling gain over omega-only = {coupling_gain:+.4f}; "
    f"KANDy gain over linear K-S = {kandy_gain:+.4f}",
    flush=True,
)


# ---------------------------------------------------------------------------
# 7. Autonomous held-out rollouts -- coupled vs. uncoupled
# ---------------------------------------------------------------------------
def rollout(rhs, theta0: np.ndarray, steps: int, dt: float) -> np.ndarray:
    out = np.empty((steps + 1, M))
    x = theta0.astype(float).copy()
    out[0] = x
    for s in range(steps):
        k1 = rhs(x)[0]
        k2 = rhs(x + 0.5 * dt * k1)[0]
        k3 = rhs(x + 0.5 * dt * k2)[0]
        k4 = rhs(x + dt * k3)[0]
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(x).all():
            return out[: s + 1]
        out[s + 1] = x
    return out


def order_parameter(theta: np.ndarray) -> np.ndarray:
    return np.abs(np.mean(np.exp(1j * theta), axis=1))


max_steps = int(round(max(ROLLOUT_HORIZONS) / DT))
contiguous = test_rows[test_rows + max_steps < N_ROW]
starts = (
    contiguous[
        np.linspace(0, len(contiguous) - 1, N_ROLLOUT_STARTS).astype(int)
    ]
    if len(contiguous) >= N_ROLLOUT_STARTS
    else contiguous
)

rhs_map = {
    "omega_only": lambda th: ks_rhs(th, OMEGA_KS, np.zeros((M, M)), ALPHA),
    "linear_ks": lambda th: ks_rhs(th, OMEGA_KS, KAPPA_KS, ALPHA),
    "kandy": kandy_rhs,
}
rollout_stats = {name: {h: [] for h in ROLLOUT_HORIZONS} for name in rhs_map}
order_stats = {name: {h: [] for h in ROLLOUT_HORIZONS} for name in rhs_map}
example_traces = {}

print(
    f"\n[ROLLOUT] autonomous, no forcing/nudging, {len(starts)} held-out starts",
    flush=True,
)
for name, rhs in rhs_map.items():
    for si, start in enumerate(starts):
        traj = rollout(rhs, THETA[start], max_steps, DT)
        ref = THETA[start : start + len(traj)]
        r_pred, r_ref = order_parameter(traj), order_parameter(ref)
        for h in ROLLOUT_HORIZONS:
            n = min(int(round(h / DT)), len(traj) - 1)
            rollout_stats[name][h].append(
                float(np.sqrt(np.mean(wrap_pi(traj[: n + 1] - ref[: n + 1]) ** 2)))
            )
            order_stats[name][h].append(
                float(np.sqrt(np.mean((r_pred[: n + 1] - r_ref[: n + 1]) ** 2)))
            )
        if si == len(starts) // 2:
            example_traces[name] = {
                "t": T[start : start + len(traj)] - T[start],
                "pred": traj,
                "ref": ref,
                "r_pred": r_pred,
                "r_ref": r_ref,
                "start_time": float(T[start]),
            }

rollout_mean = {
    name: {h: float(np.mean(v)) for h, v in per_h.items()}
    for name, per_h in rollout_stats.items()
}
order_mean = {
    name: {h: float(np.mean(v)) for h, v in per_h.items()}
    for name, per_h in order_stats.items()
}

print(f"        {'horizon':>8} " + " ".join(f"{n:>12}" for n in rhs_map))
for h in ROLLOUT_HORIZONS:
    print(
        f"        {h:>7.2f}s " + " ".join(f"{rollout_mean[n][h]:>12.4f}" for n in rhs_map),
        flush=True,
    )
print("        (wrapped phase RMSE in rad; lower is better)", flush=True)

H_GATE = 0.25
rollout_improvement = float(
    1.0 - rollout_mean["linear_ks"][H_GATE] / max(rollout_mean["omega_only"][H_GATE], 1e-12)
)
print(
    f"        coupled vs uncoupled at {H_GATE:.2f}s: "
    f"{rollout_improvement:+.1%} phase-RMSE reduction",
    flush=True,
)


# ---------------------------------------------------------------------------
# 8. Symbolic formulas from the KAN
# ---------------------------------------------------------------------------
formulas = []
try:
    with contextlib.redirect_stdout(StringIO()):
        exprs = model.get_formula(
            var_names=FEATURE_NAMES, r2_threshold=FORMULA_R2_THRESHOLD
        )
    formulas = [str(e) for e in exprs]
    print(
        f"\n[SYMBOLIC] get_formula(r2_threshold={FORMULA_R2_THRESHOLD}) "
        f"-> {len(formulas)} expressions",
        flush=True,
    )
    for i, expr in enumerate(formulas):
        shown = expr if len(expr) < 240 else expr[:240] + " ..."
        print(f"           (dth_{i}/dt - c_{i})/s_{i} = {shown}", flush=True)
    print(
        "           c = " + np.array2string(phase_center, precision=3)
        + ", s = " + np.array2string(phase_scale, precision=3),
        flush=True,
    )
except Exception as exc:  # noqa: BLE001 - symbolic snapping is best-effort
    print(f"\n[SYMBOLIC] get_formula failed: {type(exc).__name__}: {exc}", flush=True)


# ---------------------------------------------------------------------------
# 9. Gates -- all relative to a baseline
# ---------------------------------------------------------------------------
gates = {
    "coupling_beats_omega_only_by": {
        "required": GATE_COUPLING_R2_GAIN,
        "achieved": coupling_gain,
        "passed": coupling_gain >= GATE_COUPLING_R2_GAIN,
    },
    "kandy_at_least_matches_linear_ks": {
        "required": -0.01,
        "achieved": kandy_gain,
        "passed": kandy_gain >= -0.01,
    },
    "coupled_rollout_beats_uncoupled_by": {
        "required": GATE_ROLLOUT_IMPROVEMENT,
        "achieved": rollout_improvement,
        "passed": rollout_improvement >= GATE_ROLLOUT_IMPROVEMENT,
    },
    "kuramoto_sakaguchi_structure_holds": {
        "required": "off_structure_fraction <= 0.25",
        "achieved": off_structure_fraction,
        "passed": off_structure_fraction <= 0.25,
    },
}
passed = print_check(
    all(g["passed"] for g in gates.values()),
    "coupling adds held-out skill over an uncoupled oscillator model",
)
for name, g in gates.items():
    print(f"        {'PASS' if g['passed'] else 'FAIL'}  {name}: {g['achieved']}", flush=True)


metrics = {
    "task": "epilepsy Kuramoto-Sakaguchi fit with identifiable readout",
    "data_path": str(DATA_PATH),
    "channels": LABELS,
    "channel_selection": "cross-channel lag-scan gain above noise floor",
    "split": {
        "mode": SPLIT_MODE,
        "guard_samples": int(guard),
        "guard_seconds": float(guard * DT),
        "n_train": int(len(train_rows)),
        "n_val": int(len(val_rows)),
        "n_test": int(len(test_rows)),
        "train_time_range": [float(T[train_rows[0]]), float(T[train_rows[-1]])],
        "test_time_range": [float(T[test_rows[0]]), float(T[test_rows[-1]])],
        "seizure_window_in_train": event_in_train,
    },
    "equation": {
        "form": "dtheta_i/dt = omega_i + sum_j kappa_ij*sin(theta_j-theta_i+alpha)",
        "alpha_rad": ALPHA,
        "alpha_deg": float(np.degrees(ALPHA)),
        "alpha_circular_spread": alpha_spread,
        "omega_rad_s": [float(v) for v in OMEGA_KS],
        "omega_hz": [float(v / (2 * np.pi)) for v in OMEGA_KS],
        "kappa": KAPPA_KS,
        "kappa_row_is_target": True,
        "off_structure_fraction": off_structure_fraction,
        "alpha_per_directed_edge": alpha_dir,
        "kappa_per_directed_edge_free_alpha": kappa_dir,
    },
    "heldout_r2": {k: [float(x) for x in v] for k, v in scores.items()},
    "heldout_r2_mean": {k: float(np.nanmean(v)) for k, v in scores.items()},
    "coupling_gain_over_omega_only": coupling_gain,
    "kandy_gain_over_linear_ks": kandy_gain,
    "rollout_phase_rmse_mean": rollout_mean,
    "rollout_order_parameter_rmse_mean": order_mean,
    "rollout_starts": int(len(starts)),
    "rollout_improvement_at_0.25s": rollout_improvement,
    "symbolic_formulas": formulas,
    "symbolic_output_center": [float(v) for v in phase_center],
    "symbolic_output_scale": [float(v) for v in phase_scale],
    "gates": gates,
    "passed": passed,
}
save_metrics(RESULT_DIR / "metrics.json", metrics)

np.savez_compressed(
    RESULT_DIR / "ks_arrays.npz",
    t=T,
    labels=np.asarray(LABELS),
    signal=SIGNAL,
    theta=THETA,
    theta_dot=THETA_DOT,
    train_rows=train_rows,
    test_rows=test_rows,
    omega=OMEGA_KS,
    kappa=KAPPA_KS,
    alpha=np.array([ALPHA]),
    kappa_free_alpha=kappa_dir,
    alpha_free=alpha_dir,
    linear_coefficients=C,
    pred_test_linear_ks=pred_linear_ks,
    pred_test_kandy=pred_kandy,
    pred_test_omega_only=pred_omega_only,
)


# ---------------------------------------------------------------------------
# 10. Figures
# ---------------------------------------------------------------------------
use_pub_style()


def plot_coupling() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    lim = np.abs(KAPPA_KS).max()
    im = axes[0].imshow(KAPPA_KS, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[0].set_xticks(range(M), LABELS, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticks(range(M), LABELS, fontsize=8)
    axes[0].set_xlabel("source j")
    axes[0].set_ylabel("target i")
    axes[0].set_title(rf"$\kappa_{{ij}}$ at shared $\alpha$={ALPHA:+.2f} rad")
    fig.colorbar(im, ax=axes[0])

    ax = axes[1]
    edges = [(i, j) for i in range(M) for j in range(M) if i != j]
    ax.scatter(
        [alpha_dir[i, j] for i, j in edges],
        [kappa_dir[i, j] for i, j in edges],
        s=34,
        c="#1f77b4",
    )
    ax.axvline(ALPHA, color="#d62728", ls="--", lw=1.2, label=rf"shared $\alpha$")
    ax.set_xlabel(r"per-edge $\alpha$ (rad)")
    ax.set_ylabel(r"per-edge $\kappa$")
    ax.set_title("free-alpha readout per directed edge")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "coupling.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "coupling.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"omega_only": "0.45", "linear_ks": "#1f77b4", "kandy": "#d62728"}
    for name in rhs_map:
        axes[0].plot(
            ROLLOUT_HORIZONS,
            [rollout_mean[name][h] for h in ROLLOUT_HORIZONS],
            marker="o",
            color=colors[name],
            label=name,
        )
        axes[1].plot(
            ROLLOUT_HORIZONS,
            [order_mean[name][h] for h in ROLLOUT_HORIZONS],
            marker="o",
            color=colors[name],
            label=name,
        )
    axes[0].set_ylabel("wrapped phase RMSE (rad)")
    axes[1].set_ylabel("order-parameter RMSE")
    for ax in axes:
        ax.set_xlabel("rollout horizon (s)")
        ax.legend(fontsize=8)
    fig.suptitle(
        f"Autonomous held-out rollouts, mean over {len(starts)} starts "
        f"({SPLIT_MODE} split)"
    )
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "rollout_vs_horizon.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "rollout_vs_horizon.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_example_rollout() -> None:
    if "linear_ks" not in example_traces:
        return
    fig, axes = plt.subplots(M + 1, 1, figsize=(10, 1.5 * (M + 1)), sharex=True)
    ref = example_traces["linear_ks"]["ref"]
    t_local = example_traces["linear_ks"]["t"]
    for ch, ax in enumerate(axes[:M]):
        ax.plot(t_local, np.cos(ref[:, ch]), color="#1f77b4", lw=1.0, label="data")
        for name, style in (("omega_only", ":"), ("linear_ks", "--"), ("kandy", "-.")):
            tr = example_traces[name]
            ax.plot(
                tr["t"],
                np.cos(tr["pred"][:, ch]),
                ls=style,
                lw=1.0,
                label=name if ch == 0 else None,
                color={"omega_only": "0.45", "linear_ks": "#d62728", "kandy": "#2ca02c"}[name],
            )
        ax.set_ylabel(LABELS[ch], rotation=0, ha="right", va="center", fontsize=8)
        ax.set_ylim(-1.15, 1.15)
    axes[0].legend(ncol=4, fontsize=7, loc="upper right")
    axes[-1].plot(t_local, example_traces["linear_ks"]["r_ref"], color="#1f77b4", lw=1.1)
    for name, style in (("omega_only", ":"), ("linear_ks", "--"), ("kandy", "-.")):
        tr = example_traces[name]
        axes[-1].plot(
            tr["t"],
            tr["r_pred"],
            ls=style,
            lw=1.1,
            color={"omega_only": "0.45", "linear_ks": "#d62728", "kandy": "#2ca02c"}[name],
        )
    axes[-1].set_ylabel("order", rotation=0, ha="right", va="center", fontsize=8)
    axes[-1].set_ylim(0, 1.05)
    axes[-1].set_xlabel("time since rollout start (s)")
    fig.suptitle(
        f"Held-out autonomous rollout from t={example_traces['linear_ks']['start_time']:.1f}s"
    )
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "example_rollout.png", dpi=250, bbox_inches="tight")
    fig.savefig(RESULT_DIR / "example_rollout.pdf", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_fit() -> None:
    fig, axes = plt.subplots(1, M, figsize=(3.2 * M, 3.3))
    axes = np.atleast_1d(axes)
    for ch, ax in enumerate(axes):
        ax.scatter(y_test[:, ch], pred_kandy[:, ch], s=4, alpha=0.25, label="KANDy")
        ax.scatter(y_test[:, ch], pred_linear_ks[:, ch], s=4, alpha=0.25, label="linear K-S")
        lo = float(np.min(y_test[:, ch]))
        hi = float(np.max(y_test[:, ch]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_title(
            f"{LABELS[ch]}\nK-S R2={scores['linear_ks_shared_alpha'][ch]:.3f}  "
            f"KAN R2={scores['kandy'][ch]:.3f}",
            fontsize=8,
        )
    axes[0].legend(fontsize=7)
    fig.supxlabel("measured phase velocity (rad/s)")
    fig.supylabel("predicted")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "phase_velocity_fit.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_losses() -> None:
    if getattr(model, "train_results_", None):
        fig, _ax = plot_loss_curves(model.train_results_, save=str(RESULT_DIR / "loss"))
        plt.close(fig)


plot_coupling()
plot_rollout_curves()
plot_example_rollout()
plot_velocity_fit()
plot_losses()
print(f"\n[FIGS]  Saved {RESULT_DIR}", flush=True)
