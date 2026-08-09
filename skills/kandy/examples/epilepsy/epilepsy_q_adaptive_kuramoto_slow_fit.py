#!/usr/bin/env python3
"""Latent adaptive-Kuramoto generator for the epilepsy residual q(t).

This script starts from the residual drive extracted by
``epilepsy_latent_drive_map_fit.py``:

    q(t) = dX_data/dt - F_baseline(X_data).

It does not replay that measured q(t).  Instead, it turns the leading residual
SVD coordinates into a small latent oscillator population, adds a slow envelope
coordinate, and learns a discrete-time KANDy map

    eta_{n+1} = eta_n + G_KANDy(eta_n),
    q_hat_n = D(eta_n),

where eta = [latent phases, adaptive coupling proxies, slow drift].

The slow coordinate is data-derived from a low-pass residual-energy envelope.
That makes this a candidate hidden-state model, not a causal claim and not yet
an exact digital twin.  The decisive check is whether the learned eta map can
autonomously roll q_hat(t) through the seizure window from only its initial
latent state.
"""

from __future__ import annotations

import contextlib
import os
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
from scipy.signal import hilbert, savgol_filter

from _utils import ensure_outdir, print_check, save_metrics, wrap_pi
from kandy import CustomLift, KANDy
from kandy.plotting import use_pub_style


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

SOURCE_DIR = REPO_ROOT / "results" / "Epilepsy-Latent-Delay-Map"
SOURCE_ARRAYS = SOURCE_DIR / "latent_map_arrays.npz"
RESULT_DIR = ensure_outdir(REPO_ROOT / "results" / "Epilepsy-Q-Adaptive-Kuramoto-Slow")

PHASE_MODE = os.environ.get("KANDY_Q_PHASE_MODE", "hilbert").strip().lower()
N_OSC = env_int("KANDY_Q_N_OSC", 6)
MAP_LAG_SAMPLES = env_int("KANDY_Q_MAP_LAG_SAMPLES", 5)
GRID = env_int("KANDY_Q_GRID", 4)
PHASE_STEPS = env_int("KANDY_Q_PHASE_STEPS", 48)
USE_PHASE_RESIDUAL = bool(env_int("KANDY_Q_USE_PHASE_RESIDUAL", 0))
OMEGA_STEPS = env_int("KANDY_Q_OMEGA_STEPS", 48)
KAPPA_STEPS = env_int("KANDY_Q_KAPPA_STEPS", 48)
SLOW_STEPS = env_int("KANDY_Q_SLOW_STEPS", 48)
N_TRAIN_MAX = env_int("KANDY_Q_N_TRAIN_MAX", 3000)
N_TEST_MAX = env_int("KANDY_Q_N_TEST_MAX", 1500)
TEST_STRIDE = env_int("KANDY_Q_TEST_STRIDE", 5)
SLOW_WINDOW_SECONDS = env_float("KANDY_Q_SLOW_WINDOW_SECONDS", 2.0)
EVENT_WINDOW = (-1.0, 4.0)
PRE_WINDOW = (-10.0, -2.0)
POST_WINDOW = (8.0, 20.0)
CLIP_MARGIN = env_float("KANDY_Q_CLIP_MARGIN", 0.15)

if not SOURCE_ARRAYS.exists():
    raise FileNotFoundError(
        "Residual arrays are missing. Run "
        "skills/kandy/examples/oscillators/epilepsy_latent_drive_map_fit.py first."
    )


def odd_window(seconds: float, dt: float, minimum: int = 5) -> int:
    win = max(minimum, int(round(seconds / dt)))
    if win % 2 == 0:
        win += 1
    return win


def zscore_columns(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-12
    return (X - mean) / std, mean, std


def quiet_fit(model: KANDy, **kwargs) -> None:
    with contextlib.redirect_stdout(StringIO()):
        model.fit(**kwargs)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    denom = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    denom = np.where(denom > 0, denom, np.nan)
    return 1.0 - np.sum((y_true - y_pred) ** 2, axis=0) / denom


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


arrays = np.load(SOURCE_ARRAYS, allow_pickle=True)
T = np.asarray(arrays["t"], dtype=float)
DT = float(np.median(np.diff(T)))
Q_NORM = np.asarray(arrays["q_normalized"], dtype=float)
Q_DOT = np.asarray(arrays["q_dot"], dtype=float)
Z = np.asarray(arrays["z"], dtype=float)
SVD_COMPONENTS = np.asarray(arrays["q_svd_components"], dtype=float)
SVD_VAR = np.asarray(arrays["q_svd_variance_ratio"], dtype=float)
SVD_CUM = np.asarray(arrays["q_svd_cumulative_variance"], dtype=float)

if PHASE_MODE not in {"hilbert", "paired"}:
    raise ValueError("KANDY_Q_PHASE_MODE must be 'hilbert' or 'paired'")

latent_rank = N_OSC if PHASE_MODE == "hilbert" else 2 * N_OSC
if Z.shape[1] < latent_rank:
    raise ValueError(
        f"N_OSC={N_OSC}, phase_mode={PHASE_MODE!r} needs "
        f"{latent_rank} residual SVD coordinates"
    )

PAIR_FEATURES = [(i, j) for i in range(N_OSC) for j in range(i + 1, N_OSC)]
PAIR_INDEX = {pair: idx for idx, pair in enumerate(PAIR_FEATURES)}
N_PAIR = len(PAIR_FEATURES)
OMEGA_START = N_OSC
KAPPA_START = 2 * N_OSC
N_ETA = 2 * N_OSC + N_PAIR + 1

print("[DATA]  Loaded residual q(t) arrays", flush=True)
print(f"        rows={len(T)}, dt={DT:.4f}s, q_dim={Q_NORM.shape[1]}", flush=True)
print(
    f"[SVD]   phase_mode={PHASE_MODE}, rank {latent_rank} "
    f"cumulative residual variance={SVD_CUM[latent_rank - 1]:.3f}",
    flush=True,
)

if PHASE_MODE == "paired":
    # Pair residual SVD modes into 2-D oscillator planes.  This is a strict
    # oscillator hypothesis: every pair is a latent rotating residual mode.
    Z_PAIR = Z[:, : 2 * N_OSC].reshape(len(Z), N_OSC, 2)
    radius = np.sqrt(np.sum(Z_PAIR**2, axis=2) + 1e-12)
    PHI = np.unwrap(np.arctan2(Z_PAIR[:, :, 1], Z_PAIR[:, :, 0]), axis=0)
else:
    # Hilbert phases are less arbitrary for residual SVD scores that are not
    # already quadrature-paired.  The coordinate still remains data-derived.
    analytic_z = hilbert(Z[:, :N_OSC], axis=0)
    radius = np.abs(analytic_z) + 1e-12
    PHI = np.unwrap(np.angle(analytic_z), axis=0)
LOG_AMP, log_amp_mean, log_amp_std = zscore_columns(np.log(radius + 1e-6))
omega_window = odd_window(0.505, DT)
omega_raw = np.gradient(PHI, DT, axis=0)
OMEGA, omega_mean, omega_std = zscore_columns(
    savgol_filter(omega_raw, omega_window, 3, axis=0, mode="interp")
)
KAPPA = np.column_stack(
    [0.5 * (LOG_AMP[:, i] + LOG_AMP[:, j]) for i, j in PAIR_FEATURES]
)

residual_energy = np.sqrt(np.mean(Q_NORM**2, axis=1))
slow_window = odd_window(SLOW_WINDOW_SECONDS, DT)
slow_raw = savgol_filter(residual_energy, slow_window, 3, axis=0, mode="interp")
SLOW, slow_mean, slow_std = zscore_columns(slow_raw[:, None])
SLOW = SLOW[:, 0]
ETA = np.column_stack([PHI, OMEGA, KAPPA, SLOW])

# A decoder converts the latent oscillator state back to q.  It is linear in
# oscillator coordinates because the residual SVD front end is linear.
PAIR_AVG = np.zeros((N_PAIR, N_OSC))
for row_idx, (i, j) in enumerate(PAIR_FEATURES):
    PAIR_AVG[row_idx, i] = 0.5
    PAIR_AVG[row_idx, j] = 0.5
PAIR_TO_OSC = np.linalg.pinv(PAIR_AVG)


def eta_to_features(eta: np.ndarray) -> np.ndarray:
    eta2 = np.atleast_2d(eta)
    phi = eta2[:, :N_OSC]
    omega = eta2[:, OMEGA_START:KAPPA_START]
    kappa = eta2[:, KAPPA_START : KAPPA_START + N_PAIR]
    slow = eta2[:, -1:]
    log_amp_hat = kappa @ PAIR_TO_OSC.T
    amp_hat = np.exp(log_amp_hat * log_amp_std + log_amp_mean)
    if PHASE_MODE == "paired":
        z_hat = np.empty((len(eta2), 2 * N_OSC), dtype=float)
        z_hat[:, 0::2] = amp_hat * np.cos(phi)
        z_hat[:, 1::2] = amp_hat * np.sin(phi)
    else:
        z_hat = amp_hat * np.cos(phi)
    order = np.abs(np.mean(np.exp(1j * phi), axis=1, keepdims=True))
    return np.column_stack(
        [
            z_hat,
            slow,
            order,
            kappa.mean(axis=1, keepdims=True),
            omega.mean(axis=1, keepdims=True),
            omega.std(axis=1, keepdims=True),
        ]
    )


decoder_features = eta_to_features(ETA)
decoder_design = np.column_stack([np.ones(len(decoder_features)), decoder_features])
ridge = 1e-6
decoder_q_norm = np.linalg.solve(
    decoder_design.T @ decoder_design + ridge * np.eye(decoder_design.shape[1]),
    decoder_design.T @ Q_NORM,
)
decoder_q_dot = np.linalg.solve(
    decoder_design.T @ decoder_design + ridge * np.eye(decoder_design.shape[1]),
    decoder_design.T @ Q_DOT,
)


def decode_q(eta: np.ndarray, dot: bool = False) -> np.ndarray:
    feats = eta_to_features(eta)
    design = np.column_stack([np.ones(len(feats)), feats])
    return design @ (decoder_q_dot if dot else decoder_q_norm)


q_recon = decode_q(ETA)
q_recon_r2 = r2(Q_NORM, q_recon)
energy_recon = np.sqrt(np.mean(q_recon**2, axis=1))

start_idx = np.arange(0, len(ETA) - MAP_LAG_SAMPLES)
future_idx = start_idx + MAP_LAG_SAMPLES
ETA_X = ETA[start_idx]
ETA_F = ETA[future_idx]
Y_PHASE = wrap_pi(ETA_F[:, :N_OSC] - ETA_X[:, :N_OSC])
OMEGA_PHYS = OMEGA * omega_std + omega_mean
KIN_PHASE = OMEGA_PHYS[start_idx] * (MAP_LAG_SAMPLES * DT)
Y_PHASE_RESID = wrap_pi(Y_PHASE - KIN_PHASE)
Y_OMEGA = ETA_F[:, OMEGA_START:KAPPA_START]
Y_KAPPA = ETA_F[:, KAPPA_START : KAPPA_START + N_PAIR]
Y_SLOW = ETA_F[:, -1:]

phase_delta_scale = Y_PHASE_RESID.std(axis=0) + 1e-12
Y_PHASE_S = Y_PHASE_RESID / phase_delta_scale

#
# Chronological split with a guard band, replacing a stride-5 interleave.
# Holding out every 5th row while the upstream q(t) was smoothed over 0.255 s
# and z over 0.505 s means held-out rows share filter support with training
# rows: the "held-out" R2 is inflated and cannot see the temporal
# extrapolation failure that shows up as a 13x rollout overshoot.
#
all_rows = np.arange(len(ETA_X))
GUARD = int(round(1.0 / DT))
n_tr = int(len(all_rows) * 0.60)
n_va = int(len(all_rows) * 0.20)
train_rows = all_rows[: n_tr - GUARD]
val_rows = all_rows[n_tr: n_tr + n_va - GUARD]
test_rows = all_rows[n_tr + n_va + GUARD:]
rng = np.random.default_rng(SEED)
if len(train_rows) > N_TRAIN_MAX:
    train_rows = np.sort(rng.choice(train_rows, N_TRAIN_MAX, replace=False))
if len(test_rows) > N_TEST_MAX:
    test_rows = np.sort(rng.choice(test_rows, N_TEST_MAX, replace=False))
print(f"[SPLIT] chronological: train={len(train_rows)} "
      f"({T[0]:.1f}..{T[train_rows[-1]]:.1f}s), val={len(val_rows)}, "
      f"test={len(test_rows)}, guard={GUARD * DT:.2f}s", flush=True)

print(
    "[ETA]   "
    f"osc={N_OSC}, pairs={N_PAIR}, eta_dim={N_ETA}, "
    f"map_lag={MAP_LAG_SAMPLES} samples ({MAP_LAG_SAMPLES * DT:.3f}s)",
    flush=True,
)
print(
    f"[DEC]   q decoder mean R2={np.nanmean(q_recon_r2):.3f}, "
    f"energy corr={safe_corr(residual_energy, energy_recon):.3f}",
    flush=True,
)


def split_eta_np(eta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        eta[:, :N_OSC],
        eta[:, OMEGA_START:KAPPA_START],
        eta[:, KAPPA_START : KAPPA_START + N_PAIR],
        eta[:, -1],
    )


def split_eta_torch(
    eta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        eta[:, :N_OSC],
        eta[:, OMEGA_START:KAPPA_START],
        eta[:, KAPPA_START : KAPPA_START + N_PAIR],
        eta[:, -1],
    )


def latent_order_np(phi: np.ndarray) -> np.ndarray:
    return np.abs(np.mean(np.exp(1j * phi), axis=1))


def latent_order_torch(phi: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(
        torch.mean(torch.sin(phi), dim=1) ** 2
        + torch.mean(torch.cos(phi), dim=1) ** 2
        + 1e-12
    )


def phase_lift_np(eta: np.ndarray) -> np.ndarray:
    phi, omega, kappa, slow = split_eta_np(eta)
    features = [np.ones(len(eta)), slow]
    for i in range(N_OSC):
        features.extend([omega[:, i], slow * omega[:, i]])
    for idx, (i, j) in enumerate(PAIR_FEATURES):
        delta = phi[:, j] - phi[:, i]
        s = np.sin(delta)
        c = np.cos(delta)
        k = kappa[:, idx]
        features.extend([k * s, k * c, s, c, k, slow * s, slow * c])
    features.append(latent_order_np(phi))
    return np.column_stack(features)


def phase_lift_torch(eta: torch.Tensor) -> torch.Tensor:
    phi, omega, kappa, slow = split_eta_torch(eta)
    features = [torch.ones_like(slow), slow]
    for i in range(N_OSC):
        features.extend([omega[:, i], slow * omega[:, i]])
    for idx, (i, j) in enumerate(PAIR_FEATURES):
        delta = phi[:, j] - phi[:, i]
        s = torch.sin(delta)
        c = torch.cos(delta)
        k = kappa[:, idx]
        features.extend([k * s, k * c, s, c, k, slow * s, slow * c])
    features.append(latent_order_torch(phi))
    return torch.stack(features, dim=1)


def omega_lift_np(eta: np.ndarray) -> np.ndarray:
    phi, omega, kappa, slow = split_eta_np(eta)
    features = [np.ones(len(eta)), slow, latent_order_np(phi)]
    for i in range(N_OSC):
        features.extend([omega[:, i], slow * omega[:, i], omega[:, i] ** 2])
    for idx, (i, j) in enumerate(PAIR_FEATURES):
        delta = phi[:, j] - phi[:, i]
        features.extend([np.sin(delta), np.cos(delta), kappa[:, idx], slow * kappa[:, idx]])
    return np.column_stack(features)


def omega_lift_torch(eta: torch.Tensor) -> torch.Tensor:
    phi, omega, kappa, slow = split_eta_torch(eta)
    features = [torch.ones_like(slow), slow, latent_order_torch(phi)]
    for i in range(N_OSC):
        features.extend([omega[:, i], slow * omega[:, i], omega[:, i] ** 2])
    for idx, (i, j) in enumerate(PAIR_FEATURES):
        delta = phi[:, j] - phi[:, i]
        features.extend([torch.sin(delta), torch.cos(delta), kappa[:, idx], slow * kappa[:, idx]])
    return torch.stack(features, dim=1)


def kappa_lift_np(eta: np.ndarray) -> np.ndarray:
    phi, omega, kappa, slow = split_eta_np(eta)
    features = [np.ones(len(eta)), slow, latent_order_np(phi)]
    for idx, (i, j) in enumerate(PAIR_FEATURES):
        delta = phi[:, j] - phi[:, i]
        features.extend(
            [
                np.sin(delta),
                np.cos(delta),
                kappa[:, idx],
                slow * kappa[:, idx],
                omega[:, j] - omega[:, i],
            ]
        )
    return np.column_stack(features)


def kappa_lift_torch(eta: torch.Tensor) -> torch.Tensor:
    phi, omega, kappa, slow = split_eta_torch(eta)
    features = [torch.ones_like(slow), slow, latent_order_torch(phi)]
    for idx, (i, j) in enumerate(PAIR_FEATURES):
        delta = phi[:, j] - phi[:, i]
        features.extend(
            [
                torch.sin(delta),
                torch.cos(delta),
                kappa[:, idx],
                slow * kappa[:, idx],
                omega[:, j] - omega[:, i],
            ]
        )
    return torch.stack(features, dim=1)


def slow_lift_np(eta: np.ndarray) -> np.ndarray:
    phi, omega, kappa, slow = split_eta_np(eta)
    order = latent_order_np(phi)
    mean_k = kappa.mean(axis=1)
    std_k = kappa.std(axis=1)
    mean_omega = omega.mean(axis=1)
    std_omega = omega.std(axis=1)
    return np.column_stack(
        [
            np.ones(len(eta)),
            slow,
            slow**2,
            slow**3,
            order,
            mean_k,
            std_k,
            mean_omega,
            std_omega,
            slow * order,
            slow * mean_k,
            slow * mean_omega,
        ]
    )


def slow_lift_torch(eta: torch.Tensor) -> torch.Tensor:
    phi, omega, kappa, slow = split_eta_torch(eta)
    order = latent_order_torch(phi)
    mean_k = torch.mean(kappa, dim=1)
    std_k = torch.std(kappa, dim=1, unbiased=False)
    mean_omega = torch.mean(omega, dim=1)
    std_omega = torch.std(omega, dim=1, unbiased=False)
    return torch.stack(
        [
            torch.ones_like(slow),
            slow,
            slow**2,
            slow**3,
            order,
            mean_k,
            std_k,
            mean_omega,
            std_omega,
            slow * order,
            slow * mean_k,
            slow * mean_omega,
        ],
        dim=1,
    )


PHASE_LIFT_DIM = 2 + 2 * N_OSC + 7 * N_PAIR + 1
OMEGA_LIFT_DIM = 3 + 3 * N_OSC + 4 * N_PAIR
KAPPA_LIFT_DIM = 3 + 5 * N_PAIR
SLOW_LIFT_DIM = 12

model_phase = KANDy(
    lift=CustomLift(
        fn=phase_lift_np,
        output_dim=PHASE_LIFT_DIM,
        torch_fn=phase_lift_torch,
        name="q_latent_phase",
    ),
    grid=GRID,
    k=3,
    steps=PHASE_STEPS,
    seed=SEED,
    base_fun=torch.nn.Identity(),
    device="cpu",
)
model_omega = KANDy(
    lift=CustomLift(
        fn=omega_lift_np,
        output_dim=OMEGA_LIFT_DIM,
        torch_fn=omega_lift_torch,
        name="q_latent_omega",
    ),
    grid=GRID,
    k=3,
    steps=OMEGA_STEPS,
    seed=SEED,
    base_fun=torch.nn.Identity(),
    device="cpu",
)
model_kappa = KANDy(
    lift=CustomLift(
        fn=kappa_lift_np,
        output_dim=KAPPA_LIFT_DIM,
        torch_fn=kappa_lift_torch,
        name="q_latent_kappa",
    ),
    grid=GRID,
    k=3,
    steps=KAPPA_STEPS,
    seed=SEED,
    base_fun=torch.nn.Identity(),
    device="cpu",
)
model_slow = KANDy(
    lift=CustomLift(
        fn=slow_lift_np,
        output_dim=SLOW_LIFT_DIM,
        torch_fn=slow_lift_torch,
        name="q_latent_slow",
    ),
    grid=GRID,
    k=3,
    steps=SLOW_STEPS,
    seed=SEED,
    base_fun=torch.nn.Identity(),
    device="cpu",
)

if USE_PHASE_RESIDUAL:
    print("[TRAIN] Fitting latent phase-residual map ...", flush=True)
    quiet_fit(
        model_phase,
        X=ETA_X[train_rows],
        X_dot=Y_PHASE_S[train_rows],
        val_frac=0.1,
        test_frac=0.1,
        lamb=0.0,
        opt="LBFGS",
        lr=1.0,
        batch=-1,
        stop_grid_update_step=8,
        patience=0,
        verbose=False,
    )
else:
    print("[TRAIN] Skipping phase-residual map; using known phi += dt*omega row", flush=True)
#
# lamb is selected by MULTI-STEP rollout error on the validation block, not by
# one-step R2.  The failure being fixed is compounding: the omega and kappa maps
# score 0.92/0.88 one-step and still overshoot the measured drive energy by 13x
# after 240 steps.  A one-step criterion is blind to that by construction, so it
# would keep choosing lamb=0.  The criterion below rolls the coupled system the
# way it is actually deployed -- slow coordinate driven from the recording --
# and scores decoded drive energy, which is the quantity the sweep reads.
#
LAMB_GRID = [float(v) for v in os.environ.get(
    "KANDY_Q_LAMB_GRID", "0.0,0.001,0.01,0.1").split(",")]
VAL_ROLL_STEPS = env_int("KANDY_Q_VAL_ROLL_STEPS", 40)
VAL_ROLL_STARTS = env_int("KANDY_Q_VAL_ROLL_STARTS", 16)


def _fit_all(lamb):
    for mdl, tgt, stp in ((model_omega, Y_OMEGA, OMEGA_STEPS),
                          (model_kappa, Y_KAPPA, KAPPA_STEPS),
                          (model_slow, Y_SLOW, SLOW_STEPS)):
        quiet_fit(mdl, X=ETA_X[train_rows], X_dot=tgt[train_rows],
                  val_frac=0.1, test_frac=0.1, lamb=lamb, opt="LBFGS", lr=1.0,
                  batch=-1, stop_grid_update_step=8, patience=0, verbose=False)


def _val_rollout_error(steps, n_starts):
    """Coupled multi-step rollout on the validation block, slow driven from
    data, scored on decoded drive energy against the recording."""
    cand = val_rows[val_rows + steps * MAP_LAG_SAMPLES < len(ETA) - 1]
    if len(cand) == 0:
        return float("inf")
    starts = cand[np.linspace(0, len(cand) - 1, min(n_starts, len(cand))).astype(int)]
    eta = ETA[starts].copy()
    err, cnt = 0.0, 0
    for i in range(steps):
        om_next = model_omega.predict(eta)
        k_next = model_kappa.predict(eta)
        j = np.minimum(starts + (i + 1) * MAP_LAG_SAMPLES, len(SLOW) - 1)
        om_phys = om_next * omega_std + omega_mean
        eta = eta.copy()
        eta[:, :N_OSC] = eta[:, :N_OSC] + om_phys * (MAP_LAG_SAMPLES * DT)
        eta[:, OMEGA_START:KAPPA_START] = om_next
        eta[:, KAPPA_START : KAPPA_START + N_PAIR] = k_next
        eta[:, -1] = SLOW[j]
        eta = np.minimum(np.maximum(eta, eta_lo), eta_hi)
        if not np.isfinite(eta).all():
            return float("inf")
        e_pred = np.sqrt(np.mean(decode_q(eta) ** 2, axis=1))
        err += float(np.sum((e_pred - residual_energy[j]) ** 2))
        cnt += len(j)
    return err / max(cnt, 1)


# rollout clip box, needed by the lamb-selection rollout below
eta_lo = np.concatenate(
    [
        np.full(N_OSC, -np.inf),
        np.min(ETA[:, OMEGA_START:KAPPA_START], axis=0),
        np.min(ETA[:, KAPPA_START : KAPPA_START + N_PAIR], axis=0),
        [np.min(ETA[:, -1])],
    ]
)
eta_hi = np.concatenate(
    [
        np.full(N_OSC, np.inf),
        np.max(ETA[:, OMEGA_START:KAPPA_START], axis=0),
        np.max(ETA[:, KAPPA_START : KAPPA_START + N_PAIR], axis=0),
        [np.max(ETA[:, -1])],
    ]
)
finite_span = eta_hi - eta_lo
finite_span[~np.isfinite(finite_span)] = 0.0
eta_lo = eta_lo - CLIP_MARGIN * finite_span
eta_hi = eta_hi + CLIP_MARGIN * finite_span

print("[TRAIN] Selecting lamb by multi-step rollout error on the validation "
      f"block ({VAL_ROLL_STEPS} steps x {VAL_ROLL_STARTS} starts) ...", flush=True)
_best = None
_table = []
for _lamb in LAMB_GRID:
    _fit_all(_lamb)
    _e = _val_rollout_error(VAL_ROLL_STEPS, VAL_ROLL_STARTS)
    _one = float(np.nanmean(r2(Y_OMEGA[test_rows], model_omega.predict(ETA_X[test_rows]))))
    _table.append({"lamb": _lamb, "val_rollout_mse": _e, "omega_one_step_r2": _one})
    print(f"        lamb={_lamb:<6g} val rollout MSE={_e:12.4f}   "
          f"(omega one-step R2={_one:+.3f})", flush=True)
    if _best is None or _e < _best[0]:
        _best = (_e, _lamb)
LAMB_SEL = _best[1]
print(f"[TRAIN] selected lamb={LAMB_SEL:g} (one-step R2 alone would have "
      f"chosen {max(_table, key=lambda r: r['omega_one_step_r2'])['lamb']:g})",
      flush=True)
_fit_all(LAMB_SEL)


def predict_deltas(eta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eta2 = np.atleast_2d(eta)
    omega_scaled = eta2[:, OMEGA_START:KAPPA_START]
    omega_phys = omega_scaled * omega_std + omega_mean
    if USE_PHASE_RESIDUAL:
        phase_residual = model_phase.predict(eta2) * phase_delta_scale
    else:
        phase_residual = np.zeros((len(eta2), N_OSC), dtype=float)
    dphi = omega_phys * (MAP_LAG_SAMPLES * DT) + phase_residual
    omega_next = model_omega.predict(eta2)
    k_next = model_kappa.predict(eta2)
    s_next = model_slow.predict(eta2)
    return dphi, omega_next, k_next, s_next


if USE_PHASE_RESIDUAL:
    pred_phase_s = model_phase.predict(ETA_X[test_rows])
else:
    pred_phase_s = np.zeros((len(test_rows), N_OSC), dtype=float)
pred_omega = model_omega.predict(ETA_X[test_rows])
pred_kappa = model_kappa.predict(ETA_X[test_rows])
pred_slow = model_slow.predict(ETA_X[test_rows])
pred_phase_full = (
    KIN_PHASE[test_rows]
    + pred_phase_s * phase_delta_scale
)
pred_phase_full = wrap_pi(pred_phase_full)
phase_inc_r2 = r2(Y_PHASE[test_rows], pred_phase_full)
omega_state_r2 = r2(Y_OMEGA[test_rows], pred_omega)
kappa_state_r2 = r2(Y_KAPPA[test_rows], pred_kappa)
slow_state_r2 = r2(Y_SLOW[test_rows], pred_slow)

eta_lo = np.concatenate(
    [
        np.full(N_OSC, -np.inf),
        np.min(ETA[:, OMEGA_START:KAPPA_START], axis=0),
        np.min(ETA[:, KAPPA_START : KAPPA_START + N_PAIR], axis=0),
        [np.min(ETA[:, -1])],
    ]
)
eta_hi = np.concatenate(
    [
        np.full(N_OSC, np.inf),
        np.max(ETA[:, OMEGA_START:KAPPA_START], axis=0),
        np.max(ETA[:, KAPPA_START : KAPPA_START + N_PAIR], axis=0),
        [np.max(ETA[:, -1])],
    ]
)
finite_span = eta_hi - eta_lo
finite_span[~np.isfinite(finite_span)] = 0.0
eta_lo = eta_lo - CLIP_MARGIN * finite_span
eta_hi = eta_hi + CLIP_MARGIN * finite_span


def rollout_eta(start: int, steps: int) -> np.ndarray:
    eta = ETA[start].copy()
    out = np.empty((steps + 1, N_ETA), dtype=float)
    out[0] = eta
    for step in range(steps):
        dphi, omega_next, k_next, s_next = predict_deltas(eta)
        eta[:N_OSC] = eta[:N_OSC] + dphi[0]
        eta[OMEGA_START:KAPPA_START] = omega_next[0]
        eta[KAPPA_START : KAPPA_START + N_PAIR] = k_next[0]
        eta[-1] = s_next[0, 0]
        eta = np.minimum(np.maximum(eta, eta_lo), eta_hi)
        if not np.isfinite(eta).all():
            return out[: step + 1]
        out[step + 1] = eta
    return out


def summarize_window(name: str, window: tuple[float, float]) -> dict:
    mask = (T >= window[0]) & (T <= window[1])
    if not np.any(mask):
        return {"name": name, "samples": 0}
    return {
        "name": name,
        "samples": int(mask.sum()),
        "q_energy_mean": float(residual_energy[mask].mean()),
        "slow_mean": float(SLOW[mask].mean()),
        "latent_order_mean": float(latent_order_np(PHI[mask]).mean()),
        "mean_kappa": float(KAPPA[mask].mean()),
        "mean_abs_omega": float(np.mean(np.abs(OMEGA[mask]))),
    }


event_start = max(EVENT_WINDOW[0], float(T[0]))
event_end = min(EVENT_WINDOW[1], float(T[-1]))
event_idx = int(np.argmin(np.abs(T - event_start)))
event_steps = int(round((event_end - event_start) / (MAP_LAG_SAMPLES * DT)))
event_eta_pred = rollout_eta(event_idx, event_steps)
event_trace_idx = event_idx + np.arange(len(event_eta_pred)) * MAP_LAG_SAMPLES
event_trace_idx = event_trace_idx[event_trace_idx < len(ETA)]
event_eta_pred = event_eta_pred[: len(event_trace_idx)]
event_eta_ref = ETA[event_trace_idx]
event_time = T[event_trace_idx]
event_q_pred = decode_q(event_eta_pred)
event_q_ref = Q_NORM[event_trace_idx]
event_q_dot_pred = decode_q(event_eta_pred, dot=True)
event_q_dot_ref = Q_DOT[event_trace_idx]
event_energy_pred = np.sqrt(np.mean(event_q_pred**2, axis=1))
event_energy_ref = np.sqrt(np.mean(event_q_ref**2, axis=1))
event_order_pred = latent_order_np(event_eta_pred[:, :N_OSC])
event_order_ref = latent_order_np(event_eta_ref[:, :N_OSC])

q_corr = [safe_corr(event_q_pred[:, i], event_q_ref[:, i]) for i in range(Q_NORM.shape[1])]
q_dot_corr = [safe_corr(event_q_dot_pred[:, i], event_q_dot_ref[:, i]) for i in range(Q_DOT.shape[1])]
event_metrics = {
    "start_time": float(event_time[0]),
    "end_time": float(event_time[-1]),
    "samples": int(len(event_time)),
    "q_rmse": float(np.sqrt(np.mean((event_q_pred - event_q_ref) ** 2))),
    "q_corr_mean": float(np.nanmean(q_corr)),
    "q_corr_per_dim": q_corr,
    "q_dot_rmse": float(np.sqrt(np.mean((event_q_dot_pred - event_q_dot_ref) ** 2))),
    "q_dot_corr_mean": float(np.nanmean(q_dot_corr)),
    "q_energy_corr": safe_corr(event_energy_pred, event_energy_ref),
    "q_energy_rmse": float(np.sqrt(np.mean((event_energy_pred - event_energy_ref) ** 2))),
    "slow_corr": safe_corr(event_eta_pred[:, -1], event_eta_ref[:, -1]),
    "slow_rmse": float(np.sqrt(np.mean((event_eta_pred[:, -1] - event_eta_ref[:, -1]) ** 2))),
    "omega_rmse": float(
        np.sqrt(
            np.mean(
                (
                    event_eta_pred[:, OMEGA_START:KAPPA_START]
                    - event_eta_ref[:, OMEGA_START:KAPPA_START]
                )
                ** 2
            )
        )
    ),
    "latent_order_corr": safe_corr(event_order_pred, event_order_ref),
    "latent_order_rmse": float(np.sqrt(np.mean((event_order_pred - event_order_ref) ** 2))),
}

print(
    "[EVAL]  one-step R2 means "
    f"phase={np.nanmean(phase_inc_r2):.3f}, "
    f"omega_next={np.nanmean(omega_state_r2):.3f}, "
    f"kappa_next={np.nanmean(kappa_state_r2):.3f}, "
    f"slow_next={np.nanmean(slow_state_r2):.3f}",
    flush=True,
)
print(
    "[EVENT] q-generator rollout "
    f"{event_metrics['start_time']:.2f}s..{event_metrics['end_time']:.2f}s: "
    f"q_corr={event_metrics['q_corr_mean']:.3f}, "
    f"energy_corr={event_metrics['q_energy_corr']:.3f}, "
    f"slow_corr={event_metrics['slow_corr']:.3f}",
    flush=True,
)

# ---------------------------------------------------------------------------
# Seizure-disruption sweep on the latent q generator
#
# Three knobs, each of which reduces to the fitted model at its nominal value:
#
#   lam_slow  how strongly the slow drift s drives the FAST blocks.  Because a
#             KANDy output is a sum of univariate functions of lift columns,
#             the slow variable's contribution can be isolated exactly: evaluate
#             each fast model at eta and again at eta with s pinned to its
#             pre-ictal value, then blend.  lam=1 is the fit, lam=0 removes the
#             slow drive entirely.  This is the knob that asks whether the slow
#             coordinate is what produces the event.
#   d_slow    additive offset on the predicted next s -- excitability.
#   g_kappa   gain on the adaptive coupling's deviation from its pre-ictal mean.
#
# The seizure is scored as the rise in decoded residual-drive energy over the
# event window relative to the pre-ictal part of the SAME rollout, so the
# baseline cell measures whether this model produces an event at all.  If it
# does not, the sweep says so rather than reporting a large disruptor count.
# ---------------------------------------------------------------------------
if bool(env_int("KANDY_Q_SWEEP", 1)):
    # Keep the pre-window SHORT.  A long one looks like a better baseline but
    # gives the rollout time to drift off the data before the event window is
    # reached -- at -10 s the fitted rollout was already spiking at 2-3x the
    # data by -9 s, so both "seizure" and "termination" were measured on a
    # trajectory that had left the recording long before t=0.
    SW_PRE = (env_float("KANDY_Q_PRE0", -2.0), env_float("KANDY_Q_PRE1", -1.0))
    SW_EVENT = (-1.0, 4.0)
    N_LAM = env_int("KANDY_Q_N_LAM", 9)
    N_DS = env_int("KANDY_Q_N_DS", 9)
    N_GK = env_int("KANDY_Q_N_GK", 9)
    LAM_GRID = np.linspace(0.0, 2.0, N_LAM)
    DS_GRID = np.linspace(-2.0, 2.0, N_DS) * float(np.std(SLOW))
    GK_GRID = np.linspace(0.0, 2.0, N_GK)

    pre_mask_d = (T >= SW_PRE[0]) & (T <= SW_PRE[1])
    S_REF = float(np.mean(SLOW[pre_mask_d]))
    K_REF = KAPPA[pre_mask_d].mean(axis=0)

    DRIVE_SLOW = bool(env_int("KANDY_Q_DRIVE_SLOW", 1))

    def step_eta(eta, lam, ds, gk, s_meas=None):
        """One map step with the three knobs applied.  lam=1, ds=0, gk=1 is the
        fitted model exactly.

        With s_meas supplied the slow coordinate is EXOGENOUS -- held at its
        measured value rather than generated.  That is the only configuration in
        which this model actually carries the event: the slow map has one-step
        R2 0.82 but iterated a few hundred steps it runs to its clip floor and
        stays, so a fully autonomous rollout has no seizure in it to disrupt.
        Driving s from the recording puts the real driver back in and makes the
        question well posed -- at the cost that the claim becomes semi-
        autonomous: can the FAST dynamics be detuned enough to suppress the
        drive excursion while the measured slow driver is still pushing?"""
        eta_ref = eta.copy()
        eta_ref[:, -1] = S_REF
        om_full = model_omega.predict(eta)
        om_ref = model_omega.predict(eta_ref)
        k_full = model_kappa.predict(eta)
        k_ref = model_kappa.predict(eta_ref)
        om_next = om_ref + lam[:, None] * (om_full - om_ref)
        k_next = k_ref + lam[:, None] * (k_full - k_ref)
        k_next = K_REF + gk[:, None] * (k_next - K_REF)
        s_next = ((s_meas if s_meas is not None
                   else model_slow.predict(eta)[:, 0]) + ds)
        om_phys = om_next * omega_std + omega_mean
        out = eta.copy()
        out[:, :N_OSC] = eta[:, :N_OSC] + om_phys * (MAP_LAG_SAMPLES * DT)
        out[:, OMEGA_START:KAPPA_START] = om_next
        out[:, KAPPA_START : KAPPA_START + N_PAIR] = k_next
        out[:, -1] = s_next
        return np.minimum(np.maximum(out, eta_lo), eta_hi)

    def sweep_rollout(start_idx, steps, lam, ds, gk):
        n = len(lam)
        eta = np.tile(ETA[start_idx], (n, 1))
        energy = np.empty((steps + 1, n))
        slow_tr = np.empty((steps + 1, n))
        order_tr = np.empty((steps + 1, n))

        def record(i, e):
            q = decode_q(e)
            energy[i] = np.sqrt(np.mean(q ** 2, axis=1))
            slow_tr[i] = e[:, -1]
            order_tr[i] = latent_order_np(e[:, :N_OSC])

        record(0, eta)
        for i in range(steps):
            j = min(start_idx + (i + 1) * MAP_LAG_SAMPLES, len(SLOW) - 1)
            sm = np.full(n, SLOW[j]) if DRIVE_SLOW else None
            eta = step_eta(eta, lam, ds, gk, sm)
            if not np.isfinite(eta).all():
                eta = np.nan_to_num(eta, nan=0.0, posinf=0.0, neginf=0.0)
            record(i + 1, eta)
        return energy, slow_tr, order_tr

    sw_start = int(np.argmin(np.abs(T - SW_PRE[0])))
    sw_steps = int(round((SW_EVENT[1] - SW_PRE[0]) / (MAP_LAG_SAMPLES * DT)))
    sw_time = T[sw_start] + np.arange(sw_steps + 1) * (MAP_LAG_SAMPLES * DT)
    pre_m = sw_time <= SW_PRE[1]
    ev_m = (sw_time >= SW_EVENT[0]) & (sw_time <= SW_EVENT[1])

    LL, DD = np.meshgrid(LAM_GRID, DS_GRID, indexing="ij")
    n_plane = LL.size
    lam_all = np.concatenate([LL.ravel(), np.ones(N_GK)])
    ds_all = np.concatenate([DD.ravel(), np.zeros(N_GK)])
    gk_all = np.concatenate([np.ones(n_plane), GK_GRID])

    print(f"\n[SWEEP] {N_LAM} slow-drive gains x {N_DS} excitability offsets "
          f"(+ {N_GK} coupling gains) = {len(lam_all)} rollouts of "
          f"{sw_steps} steps from t={T[sw_start]:.1f}s", flush=True)
    en, sl_tr, or_tr = sweep_rollout(sw_start, sw_steps, lam_all, ds_all, gk_all)
    rise = en[ev_m].max(axis=0) - en[pre_m].mean(axis=0)
    finite = np.isfinite(en).all(axis=0)

    data_energy = residual_energy[
        (T >= SW_PRE[0]) & (T <= SW_EVENT[1])]
    d_pre = residual_energy[(T >= SW_PRE[0]) & (T <= SW_PRE[1])].mean()
    d_ev = residual_energy[(T >= SW_EVENT[0]) & (T <= SW_EVENT[1])].max()
    DATA_RISE = float(d_ev - d_pre)

    d_energy_ev = np.interp(sw_time, T, residual_energy)
    def energy_corr(col):
        v = en[:, col]
        if not np.isfinite(v).all() or v.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(v[ev_m], d_energy_ev[ev_m])[0, 1])

    base_i = int(np.argmin((lam_all - 1.0) ** 2 + (ds_all / max(np.std(SLOW), 1e-9)) ** 2
                           + (gk_all - 1.0) ** 2))
    BASE_RISE = float(rise[base_i])
    print(f"[SWEEP] data q-energy rise over the event = {DATA_RISE:+.4f}", flush=True)
    print(f"[SWEEP] slow coordinate is {'DRIVEN from the recording' if DRIVE_SLOW else 'generated'}",
          flush=True)
    print(f"[SWEEP] baseline cell (lam=1, ds=0, gk=1) rise = {BASE_RISE:+.4f} "
          f"({BASE_RISE / max(DATA_RISE, 1e-12):.0%} of data), "
          f"energy corr with data over the event = {energy_corr(base_i):+.3f}",
          flush=True)
    well_posed = bool(BASE_RISE >= 0.5 * DATA_RISE)
    print(f"[SWEEP] the disruption question is "
          f"{'WELL POSED' if well_posed else 'VACUOUS -- this model produces no event to disrupt'}",
          flush=True)

    # A cell only counts as a termination if its trajectory stays physically
    # sane.  Without this, argmin(rise) selects the DECODER BLOW-UP: eta is
    # clipped to the observed box but amp = exp(log_amp*std + mean) is
    # exponential, so a state pinned at the clip boundary decodes to an enormous
    # drive.  Such a cell blows up in the pre-window and then collapses, giving
    # a hugely negative "rise" (measured: -53.1 at lam=1.50, d_s=+0.50) that
    # looks like perfect suppression and is nothing of the kind.
    d_max = float(residual_energy[(T >= SW_PRE[0]) & (T <= SW_EVENT[1])].max())
    d_mean = float(residual_energy[(T >= SW_PRE[0]) & (T <= SW_EVENT[1])].mean())
    sane = (finite & (en.max(axis=0) <= 3.0 * d_max)
            & (en.mean(axis=0) <= 3.0 * d_mean) & (en.mean(axis=0) >= 0.2 * d_mean))
    print(f"[SWEEP] {int(sane.sum())}/{len(sane)} cells stay in a plausible energy "
          f"band (max <= 3x data max, mean within 0.2-3x data mean)", flush=True)

    SUPPRESS = env_float("KANDY_Q_SUPPRESS_FRAC", 0.5)
    disrupt = sane & (rise < SUPPRESS * BASE_RISE)
    rise_map = rise[:n_plane].reshape(LL.shape)
    dis_map = disrupt[:n_plane].reshape(LL.shape)
    print(f"[SWEEP] plane: {int(dis_map.sum())}/{dis_map.size} cells suppress the "
          f"rollout excursion below {SUPPRESS:.0%} of baseline", flush=True)
    print(f"        coupling-gain scan: {int(disrupt[n_plane:].sum())}/{N_GK}", flush=True)
    print("        rise vs lam_slow (ds=0): "
          + " ".join(f"{v:.3f}" for v in rise_map[:, N_DS // 2]), flush=True)
    print("        rise vs d_slow (lam=1):  "
          + " ".join(f"{v:.3f}" for v in rise_map[N_LAM // 2, :]), flush=True)

    # ---- termination replay: baseline vs the most-suppressed cell -----------
    # Re-run just those two, keeping the full latent trajectory so the decoded
    # drive can be plotted, not only its energy.
    def sweep_rollout_full(start_idx, steps, lam, ds, gk):
        n = len(lam)
        eta = np.tile(ETA[start_idx], (n, 1))
        traj = np.empty((steps + 1, n, N_ETA))
        traj[0] = eta
        for i in range(steps):
            j = min(start_idx + (i + 1) * MAP_LAG_SAMPLES, len(SLOW) - 1)
            sm = np.full(n, SLOW[j]) if DRIVE_SLOW else None
            eta = step_eta(eta, lam, ds, gk, sm)
            if not np.isfinite(eta).all():
                eta = np.nan_to_num(eta, nan=0.0, posinf=0.0, neginf=0.0)
            traj[i + 1] = eta
        return traj

    k_best = int(np.argmin(np.where(sane, rise, np.inf)))
    LAM_B, DS_B, GK_B = (float(lam_all[k_best]), float(ds_all[k_best]),
                         float(gk_all[k_best]))
    det = sweep_rollout_full(sw_start, sw_steps,
                             np.array([1.0, LAM_B]), np.array([0.0, DS_B]),
                             np.array([1.0, GK_B]))
    q_base = decode_q(det[:, 0, :])
    q_term = decode_q(det[:, 1, :])
    e_base = np.sqrt(np.mean(q_base ** 2, axis=1))
    e_term = np.sqrt(np.mean(q_term ** 2, axis=1))
    r_base = latent_order_np(det[:, 0, :N_OSC])
    r_term = latent_order_np(det[:, 1, :N_OSC])
    print(f"[TERM]  baseline rise={rise[base_i]:+.3f} (corr {energy_corr(base_i):+.3f}) "
          f"-> terminated rise={rise[k_best]:+.3f} (corr {energy_corr(k_best):+.3f}) "
          f"at lam={LAM_B:.2f}, d_s={DS_B:+.3f}, g_kappa={GK_B:.2f}", flush=True)

    use_pub_style()
    d_m = (T >= SW_PRE[0]) & (T <= SW_EVENT[1])
    figt, axt = plt.subplots(5, 1, figsize=(11, 11), sharex=True)
    for ax in axt:
        ax.axvspan(*SW_EVENT, color="#f4a261", alpha=0.22, lw=0)
    axt[0].plot(T[d_m], residual_energy[d_m], color="#1f77b4", lw=1.2, label="data")
    axt[0].plot(sw_time, e_base, color="#d62728", lw=1.5,
                label=r"model, fitted ($\lambda$=1)")
    axt[0].plot(sw_time, e_term, color="#2ca02c", lw=1.7,
                label=rf"terminated ($\lambda$={LAM_B:.2f})")
    axt[0].set_ylabel("drive energy")
    axt[0].legend(fontsize=8, ncol=3)
    axt[0].set_title("Seizure replay in the latent q generator, and its termination"
                     + ("  (slow coordinate driven from the recording)"
                        if DRIVE_SLOW else ""))
    axt[1].plot(T[d_m], SLOW[d_m], color="#1f77b4", lw=1.2)
    axt[1].plot(sw_time, det[:, 0, -1], color="#d62728", lw=1.5)
    axt[1].plot(sw_time, det[:, 1, -1], color="#2ca02c", lw=1.7)
    axt[1].set_ylabel("slow $s$")
    axt[2].plot(T[d_m], latent_order_np(PHI[d_m]), color="#1f77b4", lw=1.2)
    axt[2].plot(sw_time, r_base, color="#d62728", lw=1.5)
    axt[2].plot(sw_time, r_term, color="#2ca02c", lw=1.7)
    axt[2].set_ylabel("latent order $R$")
    for k, ax in enumerate(axt[3:]):
        ax.plot(T[d_m], Q_NORM[d_m, k], color="#1f77b4", lw=0.8, alpha=0.7)
        ax.plot(sw_time, q_base[:, k], color="#d62728", lw=1.1)
        ax.plot(sw_time, q_term[:, k], color="#2ca02c", lw=1.3)
        ax.set_ylabel(f"$q_{k}$")
    axt[-1].set_xlabel("recording time (s)")
    figt.tight_layout()
    for _e in ("png", "pdf"):
        figt.savefig(RESULT_DIR / f"q_seizure_termination.{_e}", dpi=250,
                     bbox_inches="tight")
    plt.close(figt)

    sweep_metrics = {
        "windows": {"pre": list(SW_PRE), "event": list(SW_EVENT)},
        "slow_driven_from_data": DRIVE_SLOW,
        "baseline_energy_corr": energy_corr(base_i),
        "terminated_energy_corr": energy_corr(k_best),
        "termination": {"lam_slow": LAM_B, "d_slow": DS_B, "g_kappa": GK_B,
                        "baseline_rise": float(rise[base_i]),
                        "terminated_rise": float(rise[k_best]),
                        "suppression": 1.0 - float(rise[k_best]) / max(float(rise[base_i]), 1e-12)},
        "lam_slow_grid": [float(v) for v in LAM_GRID],
        "d_slow_grid": [float(v) for v in DS_GRID],
        "g_kappa_grid": [float(v) for v in GK_GRID],
        "data_rise": DATA_RISE, "baseline_rise": BASE_RISE,
        "baseline_fraction_of_data": BASE_RISE / max(DATA_RISE, 1e-12),
        "question_well_posed": well_posed,
        "rise_map": rise_map, "disruptor_map": dis_map.astype(int),
        "n_disruptors_plane": int(dis_map.sum()),
        "n_sane_cells": int(sane.sum()),
        "sanity_band": {"max_le": 3.0 * d_max, "mean_in": [0.2 * d_mean, 3.0 * d_mean]},
        "coupling_scan_rise": [float(v) for v in rise[n_plane:]],
        "suppress_fraction": SUPPRESS,
    }

    use_pub_style()
    figs, axs = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ext = [DS_GRID[0], DS_GRID[-1], LAM_GRID[0], LAM_GRID[-1]]
    im = axs[0].imshow(rise_map, origin="lower", aspect="auto", extent=ext, cmap="magma")
    axs[0].contour(DS_GRID, LAM_GRID, rise_map, levels=[SUPPRESS * BASE_RISE],
                   colors="w", linewidths=1.6)
    axs[0].scatter([0.0], [1.0], marker="*", s=180, c="w", edgecolors="k", zorder=5)
    axs[0].set_xlabel(r"excitability offset $\delta_s$")
    axs[0].set_ylabel(r"slow-drive gain $\lambda$")
    axs[0].set_title("event q-energy rise")
    figs.colorbar(im, ax=axs[0], fraction=0.046)
    axs[1].axhline(BASE_RISE, color="0.5", ls=":", lw=1.2, label="baseline")
    axs[1].axhline(DATA_RISE, color="#1f77b4", ls="--", lw=1.2, label="data")
    axs[1].plot(LAM_GRID, rise_map[:, N_DS // 2], "o-", color="#d62728",
                label=r"vs $\lambda$ ($\delta_s$=0)")
    axs[1].plot(GK_GRID, rise[n_plane:], "s-", color="#2ca02c",
                label=r"vs $g_\kappa$")
    axs[1].set_xlabel("gain")
    axs[1].set_ylabel("event q-energy rise")
    axs[1].set_title("1-D scans")
    axs[1].legend(fontsize=7)
    axs[2].plot(sw_time, en[:, base_i], color="#d62728", lw=1.4, label="baseline rollout")
    axs[2].plot(sw_time, en[:, k_best], color="#2ca02c", lw=1.4,
                label=f"most suppressed ($\lambda$={lam_all[k_best]:.2f}, "
                      f"$\delta_s$={ds_all[k_best]:+.2f}, $g$={gk_all[k_best]:.2f})")
    mask_d = (T >= SW_PRE[0]) & (T <= SW_EVENT[1])
    axs[2].plot(T[mask_d], residual_energy[mask_d], color="#1f77b4", lw=1.2, label="data")
    axs[2].axvspan(*SW_EVENT, color="#f4a261", alpha=0.2, lw=0)
    axs[2].set_xlabel("recording time (s)")
    axs[2].set_ylabel("residual drive energy")
    axs[2].set_title("the event, and the most suppressed cell")
    axs[2].legend(fontsize=7)
    figs.suptitle("Disrupting the seizure in the latent q generator")
    figs.tight_layout()
    for _e in ("png", "pdf"):
        figs.savefig(RESULT_DIR / f"q_disruption_sweep.{_e}", dpi=250, bbox_inches="tight")
    plt.close(figs)
else:
    sweep_metrics = None


thresholds = {
    "q_decoder_r2_min": 0.55,
    "phase_increment_r2_min": 0.25,
    "omega_state_r2_min": 0.75,
    "slow_state_r2_min": 0.80,
    "event_q_corr_min": 0.25,
    "event_energy_corr_min": 0.35,
}
passed = print_check(
    float(np.nanmean(q_recon_r2)) >= thresholds["q_decoder_r2_min"]
    and float(np.nanmean(phase_inc_r2)) >= thresholds["phase_increment_r2_min"]
    and float(np.nanmean(omega_state_r2)) >= thresholds["omega_state_r2_min"]
    and float(np.nanmean(slow_state_r2)) >= thresholds["slow_state_r2_min"]
    and event_metrics["q_corr_mean"] >= thresholds["event_q_corr_min"]
    and event_metrics["q_energy_corr"] >= thresholds["event_energy_corr_min"],
    "latent adaptive-Kuramoto plus slow drift autonomously regenerates q(t)",
)

metrics = {
    "task": "latent adaptive-Kuramoto residual q generator with slow drift",
    "source_arrays": str(SOURCE_ARRAYS),
    "n_oscillators": N_OSC,
    "n_pairs": N_PAIR,
    "eta_dim": N_ETA,
    "q_dim": int(Q_NORM.shape[1]),
    "dt": DT,
    "fit": {
        "phase_mode": PHASE_MODE,
        "phase_update": (
            "known phi_next = phi + dt*omega row"
            + (" plus KANDy residual" if USE_PHASE_RESIDUAL else "")
        ),
        "use_phase_residual": USE_PHASE_RESIDUAL,
        "grid": GRID,
        "phase_steps": PHASE_STEPS,
        "omega_steps": OMEGA_STEPS,
        "kappa_steps": KAPPA_STEPS,
        "slow_steps": SLOW_STEPS,
        "map_lag_samples": MAP_LAG_SAMPLES,
        "map_lag_seconds": MAP_LAG_SAMPLES * DT,
        "n_train": int(len(train_rows)),
        "n_heldout": int(len(test_rows)),
        "slow_window_seconds": SLOW_WINDOW_SECONDS,
        "clip_margin": CLIP_MARGIN,
    },
    "regularization": {"grid": LAMB_GRID, "selected": LAMB_SEL,
                       "criterion": "multi-step rollout MSE on a held-out time "
                                    "block, slow driven from data",
                       "table": _table},
    "svd_variance_ratio": [float(v) for v in SVD_VAR],
    "svd_cumulative_variance": [float(v) for v in SVD_CUM],
    "q_decoder_r2": [float(v) for v in q_recon_r2],
    "q_decoder_r2_mean": float(np.nanmean(q_recon_r2)),
    "phase_increment_r2": [float(v) for v in phase_inc_r2],
    "phase_increment_r2_mean": float(np.nanmean(phase_inc_r2)),
    "omega_state_r2": [float(v) for v in omega_state_r2],
    "omega_state_r2_mean": float(np.nanmean(omega_state_r2)),
    "kappa_state_r2": [float(v) for v in kappa_state_r2],
    "kappa_state_r2_mean": float(np.nanmean(kappa_state_r2)),
    "slow_state_r2": [float(v) for v in slow_state_r2],
    "slow_state_r2_mean": float(np.nanmean(slow_state_r2)),
    "window_summaries": [
        summarize_window("pre", PRE_WINDOW),
        summarize_window("event", EVENT_WINDOW),
        summarize_window("post", POST_WINDOW),
    ],
    "event_rollout": event_metrics,
    "thresholds": thresholds,
    "disruption_sweep": sweep_metrics,
    "passed": passed,
}
save_metrics(RESULT_DIR / "metrics.json", metrics)

torch.save(
    {
        "phase_state_dict": model_phase.model_.state_dict() if USE_PHASE_RESIDUAL else None,
        "omega_state_dict": model_omega.model_.state_dict(),
        "kappa_state_dict": model_kappa.model_.state_dict(),
        "slow_state_dict": model_slow.model_.state_dict(),
        "decoder_q_norm": torch.tensor(decoder_q_norm),
        "decoder_q_dot": torch.tensor(decoder_q_dot),
        "pair_features": PAIR_FEATURES,
        "pair_to_osc": torch.tensor(PAIR_TO_OSC),
        "log_amp_mean": torch.tensor(log_amp_mean),
        "log_amp_std": torch.tensor(log_amp_std),
        "omega_mean": torch.tensor(omega_mean),
        "omega_std": torch.tensor(omega_std),
        "slow_mean": torch.tensor(slow_mean),
        "slow_std": torch.tensor(slow_std),
        "metadata": metrics["fit"],
    },
    RESULT_DIR / "fitted_q_adaptive_kuramoto_slow_state.pt",
)
np.savez_compressed(
    RESULT_DIR / "q_adaptive_kuramoto_slow_arrays.npz",
    t=T,
    eta=ETA,
    phi=PHI,
    omega=OMEGA,
    kappa=KAPPA,
    slow=SLOW,
    residual_energy=residual_energy,
    q_normalized=Q_NORM,
    q_dot=Q_DOT,
    q_recon=q_recon,
    q_recon_energy=energy_recon,
    event_time=event_time,
    event_eta_ref=event_eta_ref,
    event_eta_pred=event_eta_pred,
    event_q_ref=event_q_ref,
    event_q_pred=event_q_pred,
    event_q_dot_ref=event_q_dot_ref,
    event_q_dot_pred=event_q_dot_pred,
    event_energy_ref=event_energy_ref,
    event_energy_pred=event_energy_pred,
)


use_pub_style()
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
for ax in axes:
    ax.axvspan(EVENT_WINDOW[0], EVENT_WINDOW[1], color="#f4a261", alpha=0.18, lw=0)

axes[0].plot(T, residual_energy, color="#555555", lw=0.8, alpha=0.65, label="q energy")
axes[0].plot(T, slow_raw, color="#1f77b4", lw=1.3, label="slow envelope")
axes[0].set_ylabel("residual")
axes[0].legend(loc="upper right", frameon=True)

axes[1].plot(T, SLOW, color="#1f77b4", lw=0.9, label="slow true")
axes[1].plot(event_time, event_eta_pred[:, -1], color="#d62728", lw=1.1, label="slow rollout")
axes[1].set_ylabel("s")
axes[1].legend(loc="upper right", frameon=True)

axes[2].plot(T, latent_order_np(PHI), color="#2ca02c", lw=0.8, label="latent order true")
axes[2].plot(event_time, event_order_pred, color="#d62728", lw=1.0, label="order rollout")
axes[2].set_ylabel("R")
axes[2].legend(loc="upper right", frameon=True)

axes[3].plot(T, Q_NORM[:, 0], color="#1f77b4", lw=0.6, alpha=0.65, label="q dim 0 true")
axes[3].plot(T, q_recon[:, 0], color="#444444", lw=0.8, alpha=0.8, label="q dim 0 decoded")
axes[3].plot(event_time, event_q_pred[:, 0], color="#d62728", lw=1.0, label="q dim 0 rollout")
axes[3].set_ylabel("q0")
axes[3].set_xlabel("time (s)")
axes[3].legend(loc="upper right", frameon=True)
fig.suptitle("Residual q(t) as latent adaptive Kuramoto plus slow drift", y=0.995)
fig.tight_layout()
fig.savefig(RESULT_DIR / "q_latent_slow_timeseries.png", dpi=250, bbox_inches="tight")
fig.savefig(RESULT_DIR / "q_latent_slow_timeseries.pdf", dpi=250, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
for mode, ax in enumerate(axes):
    ax.plot(event_time, event_q_ref[:, mode], color="#1f77b4", lw=0.9, label="true")
    ax.plot(event_time, event_q_pred[:, mode], color="#d62728", lw=0.9, label="rollout")
    ax.set_ylabel(f"q{mode}")
axes[-1].set_xlabel("time (s)")
axes[0].legend(loc="upper right", frameon=True)
fig.suptitle("Autonomous residual-drive rollout in the seizure window", y=0.995)
fig.tight_layout()
fig.savefig(RESULT_DIR / "q_event_rollout.png", dpi=250, bbox_inches="tight")
fig.savefig(RESULT_DIR / "q_event_rollout.pdf", dpi=250, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.6, 5.2))
sc = ax.scatter(
    event_eta_pred[:, -1],
    event_order_pred,
    c=event_time,
    s=18,
    cmap="viridis",
    edgecolors="none",
)
ax.plot(event_eta_ref[:, -1], event_order_ref, color="#1f77b4", lw=1.0, alpha=0.7, label="true path")
ax.set_xlabel("slow coordinate s")
ax.set_ylabel("latent order R")
ax.legend(loc="best", frameon=True)
fig.colorbar(sc, ax=ax, label="time (s)")
fig.tight_layout()
fig.savefig(RESULT_DIR / "q_slow_order_portrait.png", dpi=250, bbox_inches="tight")
fig.savefig(RESULT_DIR / "q_slow_order_portrait.pdf", dpi=250, bbox_inches="tight")
plt.close(fig)

print(f"[FIGS]  Saved figures to {RESULT_DIR}", flush=True)
