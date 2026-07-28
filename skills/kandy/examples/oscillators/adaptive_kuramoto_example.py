#!/usr/bin/env python3
"""KANDy example: Adaptive Kuramoto-Sakaguchi oscillators.

N coupled phase oscillators with adaptive directed coupling weights:
    dtheta_i/dt  = omega_i + sum_j kappa_ij * sin(theta_j - theta_i + alpha)
    dkappa_ij/dt = -eps * [kappa_ij + sin(theta_j - theta_i + beta)]

The phase equation is bilinear in kappa_ij and the phase-difference
trigonometric functions. KANDy is separable, so that product is a structural
feature, not something the spline edges can invent from separate kappa and
sin/cos inputs.

Lifts used here:
    phase model:
        kappa_ij * sin(theta_j - theta_i)
        kappa_ij * cos(theta_j - theta_i)

    kappa model:
        one independent sin/cos feature per unordered oscillator pair
        directed kappa_ij features

The kappa lift intentionally does not include both cos(theta_j - theta_i) and
cos(theta_i - theta_j), since those are identical columns and make symbolic
coefficients non-identifiable.
"""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-kandy")

import numpy as np
import torch
import sympy as sp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from kandy import KANDy, CustomLift
from kandy.plotting import plot_loss_curves, use_pub_style

# ---------------------------------------------------------------------------
# 0. Reproducibility / parameters
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

N = 5
ALPHA = -np.pi / 4
BETA = -np.pi / 2
EPS = 0.1

N_TRAIN = 5000
FIT_STEPS = 35
T_EVAL = 500
DT_SIM = 0.05
KAPPA_SAMPLE_RANGE = 0.7

rng = np.random.default_rng(SEED)
OMEGA = rng.uniform(-0.5, 0.5, size=N)

# Directed off-diagonal pairs for kappa_ij.
OD_PAIRS = [(i, j) for i in range(N) for j in range(N) if i != j]
N_OD = len(OD_PAIRS)
N_STATE = N + N_OD

# Independent phase pairs for sin/cos(theta_b - theta_a), a < b.
PAIR_FEATURES = [(i, j) for i in range(N) for j in range(i + 1, N)]
PAIR_INDEX = {pair: idx for idx, pair in enumerate(PAIR_FEATURES)}
N_PAIR = len(PAIR_FEATURES)

PHASE_FEATURE_NAMES = (
    [f"ks{i}{j}" for (i, j) in OD_PAIRS]
    + [f"kc{i}{j}" for (i, j) in OD_PAIRS]
)
KAPPA_FEATURE_NAMES = (
    [f"s{i}{j}" for (i, j) in PAIR_FEATURES]
    + [f"c{i}{j}" for (i, j) in PAIR_FEATURES]
    + [f"k{i}{j}" for (i, j) in OD_PAIRS]
)

print(f"[PARAMS] N={N}, N_OD={N_OD}, N_STATE={N_STATE}")
print(f"         alpha={ALPHA:.4f}, beta={BETA:.4f}, eps={EPS}")
print(f"         KAN_phase: [{len(PHASE_FEATURE_NAMES)}, {N}]")
print(f"         KAN_kappa: [{len(KAPPA_FEATURE_NAMES)}, {N_OD}]")
print(f"         omega = {OMEGA.round(4)}")

# ---------------------------------------------------------------------------
# 1. State helpers and exact dynamics
# ---------------------------------------------------------------------------

def unpack_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split state rows into theta and off-diagonal kappa coordinates."""
    return state[:, :N], state[:, N:]


def adaptive_rhs_batch(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized adaptive Kuramoto-Sakaguchi RHS.

    state: (T, N + N_OD), ordered as theta_0..theta_N, then kappa_ij for
    OD_PAIRS. Returns (dtheta, dkappa_od).
    """
    theta, kappa_od = unpack_state(state)
    dtheta = np.tile(OMEGA, (state.shape[0], 1))
    dkappa = np.zeros((state.shape[0], N_OD))

    for idx, (i, j) in enumerate(OD_PAIRS):
        delta = theta[:, j] - theta[:, i]
        dtheta[:, i] += kappa_od[:, idx] * np.sin(delta + ALPHA)
        dkappa[:, idx] = -EPS * (kappa_od[:, idx] + np.sin(delta + BETA))

    return dtheta, dkappa


def adaptive_rhs_ivp(_t: float, state: np.ndarray) -> np.ndarray:
    """Single-state RHS for solve_ivp."""
    dtheta, dkappa = adaptive_rhs_batch(state[None, :])
    return np.concatenate([dtheta.ravel(), dkappa.ravel()])


def simulate(T_end: float, dt: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one validation trajectory; return (t, state_traj)."""
    rng0 = np.random.default_rng(seed)
    theta0 = rng0.uniform(0, 2 * np.pi, size=N)
    kappa0 = rng0.uniform(-0.5, 0.5, size=N_OD)
    y0 = np.concatenate([theta0, kappa0])

    t_eval = np.arange(0, T_end, dt)
    sol = solve_ivp(
        adaptive_rhs_ivp,
        [0, T_end],
        y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    return sol.t, sol.y.T


# ---------------------------------------------------------------------------
# 2. Lifts
# ---------------------------------------------------------------------------

def phase_lift_np(state: np.ndarray) -> np.ndarray:
    """Lift for dtheta: [kappa_ij*sin(delta_ij), kappa_ij*cos(delta_ij)]."""
    theta, kappa_od = unpack_state(state)
    sin_products = []
    cos_products = []
    for idx, (i, j) in enumerate(OD_PAIRS):
        delta = theta[:, j] - theta[:, i]
        sin_products.append(kappa_od[:, idx] * np.sin(delta))
        cos_products.append(kappa_od[:, idx] * np.cos(delta))
    return np.column_stack(sin_products + cos_products)


def phase_lift_torch(state: torch.Tensor) -> torch.Tensor:
    """Torch version of phase_lift_np for rollout-loss-capable fits."""
    theta = state[:, :N]
    kappa_od = state[:, N:]
    sin_products = []
    cos_products = []
    for idx, (i, j) in enumerate(OD_PAIRS):
        delta = theta[:, j] - theta[:, i]
        sin_products.append(kappa_od[:, idx] * torch.sin(delta))
        cos_products.append(kappa_od[:, idx] * torch.cos(delta))
    return torch.stack(sin_products + cos_products, dim=1)


def kappa_lift_np(state: np.ndarray) -> np.ndarray:
    """Lift for dkappa: independent pair sin/cos features plus directed kappa."""
    theta, kappa_od = unpack_state(state)
    sin_feats = []
    cos_feats = []
    for i, j in PAIR_FEATURES:
        delta = theta[:, j] - theta[:, i]
        sin_feats.append(np.sin(delta))
        cos_feats.append(np.cos(delta))
    kappa_feats = [kappa_od[:, idx] for idx in range(N_OD)]
    return np.column_stack(sin_feats + cos_feats + kappa_feats)


def kappa_lift_torch(state: torch.Tensor) -> torch.Tensor:
    """Torch version of kappa_lift_np for rollout-loss-capable fits."""
    theta = state[:, :N]
    kappa_od = state[:, N:]
    sin_feats = []
    cos_feats = []
    for i, j in PAIR_FEATURES:
        delta = theta[:, j] - theta[:, i]
        sin_feats.append(torch.sin(delta))
        cos_feats.append(torch.cos(delta))
    kappa_feats = [kappa_od[:, idx] for idx in range(N_OD)]
    return torch.stack(sin_feats + cos_feats + kappa_feats, dim=1)


# ---------------------------------------------------------------------------
# 3. Independent training data
# ---------------------------------------------------------------------------
print("[DATA]  Sampling independent states on the theta/kappa domain ...")
train_rng = np.random.default_rng(SEED + 200)
theta_train = train_rng.uniform(0, 2 * np.pi, size=(N_TRAIN, N))
kappa_train = train_rng.uniform(
    -KAPPA_SAMPLE_RANGE,
    KAPPA_SAMPLE_RANGE,
    size=(N_TRAIN, N_OD),
)
STATE_TRAIN = np.hstack([theta_train, kappa_train])
THETA_DOT, KAPPA_DOT = adaptive_rhs_batch(STATE_TRAIN)

phase_design = np.column_stack([np.ones(N_TRAIN), phase_lift_np(STATE_TRAIN)])
kappa_design = np.column_stack([np.ones(N_TRAIN), kappa_lift_np(STATE_TRAIN)])
print(f"[DATA]  Training samples: {N_TRAIN}")
print(f"[DATA]  Phase lift condition number: {np.linalg.cond(phase_design):.2e}")
print(f"[DATA]  Kappa lift condition number: {np.linalg.cond(kappa_design):.2e}")

print("[DATA]  Simulating validation trajectory ...")
t_true, STATE_TRUE = simulate(
    T_end=(T_EVAL + 1) * DT_SIM,
    dt=DT_SIM,
    seed=SEED + 1,
)
T_EVAL = min(T_EVAL, len(STATE_TRUE))
STATE_TRUE = STATE_TRUE[:T_EVAL]
print(f"[DATA]  Validation snapshots: {T_EVAL}, dt={DT_SIM:.3f}")

# ---------------------------------------------------------------------------
# 4. Fit KANDy models
# ---------------------------------------------------------------------------
phase_lift = CustomLift(
    fn=phase_lift_np,
    output_dim=len(PHASE_FEATURE_NAMES),
    torch_fn=phase_lift_torch,
    name="adaptive_phase_products",
)
kappa_lift = CustomLift(
    fn=kappa_lift_np,
    output_dim=len(KAPPA_FEATURE_NAMES),
    torch_fn=kappa_lift_torch,
    name="adaptive_kappa_linear",
)

model_phase = KANDy(
    lift=phase_lift,
    grid=5,
    k=3,
    steps=FIT_STEPS,
    seed=SEED,
    base_fun=torch.nn.Identity(),
    device="cpu",
)
model_kappa = KANDy(
    lift=kappa_lift,
    grid=5,
    k=3,
    steps=FIT_STEPS,
    seed=SEED,
    base_fun=torch.nn.Identity(),
    device="cpu",
)

print("\n[TRAIN] Fitting phase model ...")
model_phase.fit(
    X=STATE_TRAIN,
    X_dot=THETA_DOT,
    val_frac=0.15,
    test_frac=0.15,
    lamb=0.0,
    patience=0,
    verbose=True,
)

print("\n[TRAIN] Fitting kappa model ...")
model_kappa.fit(
    X=STATE_TRAIN,
    X_dot=KAPPA_DOT,
    val_frac=0.15,
    test_frac=0.15,
    lamb=0.0,
    patience=0,
    verbose=True,
)

# ---------------------------------------------------------------------------
# 5. Coupled rollout validation
# ---------------------------------------------------------------------------

def wrap_pi(x: np.ndarray) -> np.ndarray:
    """Wrap phase differences to (-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def learned_rhs(state: np.ndarray) -> np.ndarray:
    """Combined learned RHS for the raw state vector."""
    dtheta = model_phase.predict(state).ravel()
    dkappa = model_kappa.predict(state).ravel()
    return np.concatenate([dtheta, dkappa])


def rollout_learned(state0: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
    """RK4 rollout using the two learned KANDy models."""
    state = state0.copy().astype(np.float64)
    traj = [state.copy()]

    for _ in range(n_steps - 1):
        k1 = learned_rhs(state)
        k2 = learned_rhs(state + 0.5 * dt * k1)
        k3 = learned_rhs(state + 0.5 * dt * k2)
        k4 = learned_rhs(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(state.copy())

    return np.array(traj)


print(f"\n[ROLLOUT] Rolling out {T_EVAL} steps ...", flush=True)
STATE_PRED = rollout_learned(STATE_TRUE[0], T_EVAL, DT_SIM)

true_theta = STATE_TRUE[:, :N]
pred_theta = STATE_PRED[:, :N]
true_kappa = STATE_TRUE[:, N:]
pred_kappa = STATE_PRED[:, N:]

rmse_theta = np.sqrt(np.mean(wrap_pi(pred_theta - true_theta) ** 2))
rmse_kappa = np.sqrt(np.mean((pred_kappa - true_kappa) ** 2))
print(f"[EVAL]  Wrapped phase RMSE theta: {rmse_theta:.6f}", flush=True)
print(f"[EVAL]  Kappa RMSE: {rmse_kappa:.6f}", flush=True)

r_true = np.abs(np.mean(np.exp(1j * true_theta), axis=1))
r_pred = np.abs(np.mean(np.exp(1j * pred_theta), axis=1))
rmse_r = np.sqrt(np.mean((r_pred - r_true) ** 2))
print(f"[EVAL]  Order parameter RMSE r: {rmse_r:.6f}", flush=True)

signal_true = np.cos(true_theta)
signal_pred = np.cos(pred_theta)
rmse_signal = np.sqrt(np.mean((signal_pred - signal_true) ** 2))
print(f"[EVAL]  Oscillator signal RMSE cos(theta): {rmse_signal:.6f}", flush=True)

# ---------------------------------------------------------------------------
# 6. Effective symbolic readout
# ---------------------------------------------------------------------------

def effective_linear_readout(
    model: KANDy,
    lift_fn,
    state: np.ndarray,
) -> np.ndarray:
    """Least-squares linear surrogate of model.predict(state) on [1, lift]."""
    phi = lift_fn(state)
    design = np.column_stack([np.ones(len(phi)), phi])
    coef, *_ = np.linalg.lstsq(design, model.predict(state), rcond=None)
    return coef.T


def true_phase_readout() -> np.ndarray:
    """Exact phase coefficients in [1, k*sin(delta), k*cos(delta)]."""
    coef = np.zeros((N, 1 + len(PHASE_FEATURE_NAMES)))
    coef[:, 0] = OMEGA
    for idx, (i, _j) in enumerate(OD_PAIRS):
        coef[i, 1 + idx] = np.cos(ALPHA)
        coef[i, 1 + N_OD + idx] = np.sin(ALPHA)
    return coef


def true_kappa_readout() -> np.ndarray:
    """Exact kappa coefficients in [1, pair sin, pair cos, directed kappa]."""
    coef = np.zeros((N_OD, 1 + len(KAPPA_FEATURE_NAMES)))
    for out_idx, (i, j) in enumerate(OD_PAIRS):
        pair = (i, j) if i < j else (j, i)
        pair_idx = PAIR_INDEX[pair]
        sin_sign = 1.0 if i < j else -1.0

        coef[out_idx, 1 + pair_idx] = -EPS * sin_sign * np.cos(BETA)
        coef[out_idx, 1 + N_PAIR + pair_idx] = -EPS * np.sin(BETA)
        coef[out_idx, 1 + 2 * N_PAIR + out_idx] = -EPS
    return coef


def formulas_from_readout(
    coef: np.ndarray,
    feature_names: list[str],
    threshold: float = 5e-4,
) -> list[sp.Expr]:
    """Convert linear coefficients to compact SymPy formulas."""
    symbols = [sp.Symbol(name) for name in feature_names]
    formulas = []
    for row in coef:
        expr = sp.Float(round(float(row[0]), 4))
        for value, symbol in zip(row[1:], symbols):
            if abs(value) >= threshold:
                expr += sp.Float(round(float(value), 4)) * symbol
        formulas.append(sp.expand(expr))
    return formulas


A_phase = effective_linear_readout(model_phase, phase_lift_np, STATE_TRAIN)
A_kappa = effective_linear_readout(model_kappa, kappa_lift_np, STATE_TRAIN)
A_phase_true = true_phase_readout()
A_kappa_true = true_kappa_readout()

phase_coef_err = np.max(np.abs(A_phase - A_phase_true))
kappa_coef_err = np.max(np.abs(A_kappa - A_kappa_true))

phase_formulas = formulas_from_readout(A_phase, PHASE_FEATURE_NAMES)
kappa_formulas = formulas_from_readout(A_kappa, KAPPA_FEATURE_NAMES)

print("\n[SYMBOLIC] Effective linear formulas")
print("           ksij = kappa_ij*sin(theta_j-theta_i)")
print("           kcij = kappa_ij*cos(theta_j-theta_i)")
print(f"           phase max coefficient error = {phase_coef_err:.2e}")
print(f"           kappa max coefficient error = {kappa_coef_err:.2e}\n")

checks_ok = (
    rmse_theta < 5e-2
    and rmse_kappa < 1e-2
    and rmse_signal < 5e-2
    and rmse_r < 5e-3
    and phase_coef_err < 1e-3
    and kappa_coef_err < 1e-3
)
print(
    f"[CHECK] {'PASS' if checks_ok else 'FAIL'}: rollout and coefficients within thresholds\n",
    flush=True,
)

print("--- Phase dynamics dtheta/dt ---")
for i, formula in enumerate(phase_formulas):
    print(f"  dtheta{i}/dt = {formula}")

print("\n--- Coupling dynamics dkappa/dt ---")
for idx, ((i, j), formula) in enumerate(zip(OD_PAIRS, kappa_formulas)):
    print(f"  dk{i}{j}/dt = {formula}")

# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/Adaptive-Kuramoto", exist_ok=True)

t_roll = np.arange(T_EVAL) * DT_SIM
colors = plt.cm.tab10.colors

# 7a. Phase trajectories
fig, axes = plt.subplots(N, 1, figsize=(8, 2 * N), sharex=True)
for i, ax in enumerate(axes):
    ax.plot(t_roll, true_theta[:, i], color=colors[i], lw=1.2, label="True")
    ax.plot(t_roll, pred_theta[:, i], color=colors[i], lw=1.0, ls="--", label="KANDy")
    ax.set_ylabel(f"$\\theta_{i}$")
    if i == 0:
        ax.legend(loc="upper right", fontsize=7)
axes[-1].set_xlabel("time")
fig.suptitle("Adaptive Kuramoto: phase trajectories", fontsize=11)
fig.tight_layout()
fig.savefig("results/Adaptive-Kuramoto/phase_trajectories.png", dpi=300, bbox_inches="tight")
fig.savefig("results/Adaptive-Kuramoto/phase_trajectories.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7b. Oscillator signals x_i(t) = cos(theta_i)
fig, axes = plt.subplots(N, 1, figsize=(8, 2 * N), sharex=True)
for i, ax in enumerate(axes):
    ax.plot(t_roll, signal_true[:, i], color=colors[i], lw=1.2, label="True")
    ax.plot(t_roll, signal_pred[:, i], color=colors[i], lw=1.0, ls="--", label="KANDy")
    ax.set_ylabel(f"$\\cos(\\theta_{i})$")
    ax.set_ylim(-1.1, 1.1)
    if i == 0:
        ax.legend(loc="upper right", fontsize=7)
axes[-1].set_xlabel("time")
fig.suptitle("Adaptive Kuramoto: oscillator signals", fontsize=11)
fig.tight_layout()
fig.savefig("results/Adaptive-Kuramoto/oscillator_signals.png", dpi=300, bbox_inches="tight")
fig.savefig("results/Adaptive-Kuramoto/oscillator_signals.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7c. Coupling weight trajectories for the first four directed couplings.
n_show = min(4, N_OD)
fig, axes = plt.subplots(n_show, 1, figsize=(8, 2 * n_show), sharex=True)
if n_show == 1:
    axes = [axes]
for k_idx, ax in enumerate(axes):
    i, j = OD_PAIRS[k_idx]
    ax.plot(t_roll, true_kappa[:, k_idx], color=colors[k_idx % 10], lw=1.2)
    ax.plot(t_roll, pred_kappa[:, k_idx], color=colors[k_idx % 10], lw=1.0, ls="--")
    ax.set_ylabel(f"$\\kappa_{{{i}{j}}}$")
axes[-1].set_xlabel("time")
fig.suptitle("Adaptive Kuramoto: coupling weights", fontsize=11)
fig.tight_layout()
fig.savefig("results/Adaptive-Kuramoto/kappa_trajectories.png", dpi=300, bbox_inches="tight")
fig.savefig("results/Adaptive-Kuramoto/kappa_trajectories.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7d. Order parameter
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(t_roll, r_true, color="steelblue", lw=1.4, label="True $r(t)$")
ax.plot(t_roll, r_pred, color="tomato", lw=1.0, ls="--", label="KANDy $r(t)$")
ax.set_xlabel("time")
ax.set_ylabel("$r(t)$")
ax.set_title(f"Kuramoto order parameter  (RMSE={rmse_r:.4f})")
ax.legend(fontsize=8)
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig("results/Adaptive-Kuramoto/order_parameter.png", dpi=300, bbox_inches="tight")
fig.savefig("results/Adaptive-Kuramoto/order_parameter.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7e. Coefficient heatmaps
def coefficient_heatmap(true_coef, learned_coef, feature_names, out_names, save):
    fig, axes = plt.subplots(1, 2, figsize=(12, max(3.5, 0.22 * len(out_names))), sharey=True)
    true_terms = true_coef[:, 1:]
    learned_terms = learned_coef[:, 1:]
    vmax = max(np.max(np.abs(true_terms)), np.max(np.abs(learned_terms)))
    for ax, mat, title in [
        (axes[0], true_terms, "True"),
        (axes[1], learned_terms, "KANDy linear readout"),
    ]:
        im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=60, ha="right", fontsize=6)
        ax.set_xlabel("lift feature")
    axes[0].set_yticks(range(len(out_names)))
    axes[0].set_yticklabels(out_names, fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.85, label="coefficient")
    fig.savefig(f"{save}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{save}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


coefficient_heatmap(
    A_phase_true,
    A_phase,
    PHASE_FEATURE_NAMES,
    [f"dtheta{i}/dt" for i in range(N)],
    "results/Adaptive-Kuramoto/phase_linear_coefficients",
)
coefficient_heatmap(
    A_kappa_true,
    A_kappa,
    KAPPA_FEATURE_NAMES,
    [f"dk{i}{j}/dt" for (i, j) in OD_PAIRS],
    "results/Adaptive-Kuramoto/kappa_linear_coefficients",
)

# 7f. Loss curves
if hasattr(model_phase, "train_results_") and model_phase.train_results_:
    fig, ax = plot_loss_curves(
        model_phase.train_results_,
        save="results/Adaptive-Kuramoto/loss_phase",
    )
    plt.close(fig)

if hasattr(model_kappa, "train_results_") and model_kappa.train_results_:
    fig, ax = plot_loss_curves(
        model_kappa.train_results_,
        save="results/Adaptive-Kuramoto/loss_kappa",
    )
    plt.close(fig)

print("[FIGS]  Saved results/Adaptive-Kuramoto/")
