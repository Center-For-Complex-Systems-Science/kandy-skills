#!/usr/bin/env python3
"""KANDy example: Standard Kuramoto oscillators.

N coupled phase oscillators with fixed global coupling:
    dtheta_i/dt = omega_i + (K/N) sum_{j != i} sin(theta_j - theta_i)

This is the simplest Kuramoto model: no adaptive coupling, no phase lag.
It is a useful diagnostic before the adaptive variant because the true RHS is
linear in pairwise sine phase differences.

Parameters:
    N   = 5 oscillators
    K   = 1.0
    omega_i ~ Uniform(-1, 1)

Koopman lift:
    s_ij = sin(theta_j - theta_i), one feature for each unordered pair i < j.

The one-feature-per-pair convention matters. The older directed lift included
both sin(theta_i - theta_j) and sin(theta_j - theta_i), which are exact
negatives. That made the coefficient matrix non-identifiable and allowed the
symbolic extraction step to return arbitrary mixtures of equivalent terms.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-kandy")

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

import numpy as np
import torch
import sympy as sp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from kandy import KANDy, CustomLift
from kandy.plotting import plot_loss_curves, use_pub_style
from _utils import ensure_outdir, print_check, save_metrics, wrap_pi

# ---------------------------------------------------------------------------
# 0. Reproducibility / parameters
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

N = 5
K = 1.0
K_OVER_N = K / N

N_TRAIN = 6000
T_EVAL = 500
DT_SIM = 0.05

rng = np.random.default_rng(SEED)
OMEGA = rng.uniform(-1.0, 1.0, size=N)
RESULT_DIR = ensure_outdir("results/Kuramoto")

# Unordered phase pairs. s_ij = sin(theta_j - theta_i), i < j.
PAIR_FEATURES = [(i, j) for i in range(N) for j in range(i + 1, N)]
N_FEAT = len(PAIR_FEATURES)
FEATURE_NAMES = [f"s{i}{j}" for (i, j) in PAIR_FEATURES]
FEATURE_DEFS = [
    f"{name}=sin(theta{j}-theta{i})"
    for name, (i, j) in zip(FEATURE_NAMES, PAIR_FEATURES)
]

print(f"[PARAMS] N={N}, K={K}, N_FEAT={N_FEAT}")
print(f"         KAN: [{N_FEAT}, {N}]")
print(f"         omega = {OMEGA.round(4)}")
print(f"         K/N = {K_OVER_N:.4f}")

# ---------------------------------------------------------------------------
# 1. Physics and training data
# ---------------------------------------------------------------------------

def kuramoto_rhs_batch(theta: np.ndarray) -> np.ndarray:
    """Vectorized RHS of the standard Kuramoto model.

    theta: (T, N) phase angles
    returns: (T, N) phase velocities
    """
    dtheta = np.tile(OMEGA, (theta.shape[0], 1))
    for i in range(N):
        for j in range(N):
            if i != j:
                dtheta[:, i] += K_OVER_N * np.sin(theta[:, j] - theta[:, i])
    return dtheta


def kuramoto_rhs(_t: float, theta: np.ndarray) -> np.ndarray:
    """Single-state RHS for solve_ivp."""
    return kuramoto_rhs_batch(theta[None, :]).ravel()


def simulate(T_end: float, dt: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one validation trajectory; return (t, theta_traj)."""
    rng0 = np.random.default_rng(seed)
    theta0 = rng0.uniform(0, 2 * np.pi, size=N)
    t_eval = np.arange(0, T_end, dt)
    sol = solve_ivp(
        kuramoto_rhs,
        [0, T_end],
        theta0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    return sol.t, sol.y.T


print("[DATA]  Sampling independent phase states on the N-torus ...")
train_rng = np.random.default_rng(SEED + 100)
THETA_TRAIN = train_rng.uniform(0, 2 * np.pi, size=(N_TRAIN, N))
THETA_DOT = kuramoto_rhs_batch(THETA_TRAIN)
print(f"[DATA]  Training samples: {len(THETA_TRAIN)}")

print("[DATA]  Simulating validation trajectory ...")
t_true, THETA_TRUE = simulate(T_end=(T_EVAL + 1) * DT_SIM, dt=DT_SIM, seed=SEED + 1)
T_EVAL = min(T_EVAL, len(THETA_TRUE))
THETA_TRUE = THETA_TRUE[:T_EVAL]
print(f"[DATA]  Validation snapshots: {T_EVAL}, dt={DT_SIM:.3f}")

# ---------------------------------------------------------------------------
# 2. Koopman lift: one antisymmetric sine feature per unordered pair
# ---------------------------------------------------------------------------

def build_lift(theta: np.ndarray) -> np.ndarray:
    """Lift (T, N) -> (T, N*(N-1)/2)."""
    return np.column_stack(
        [np.sin(theta[:, j] - theta[:, i]) for (i, j) in PAIR_FEATURES]
    )


def build_lift_torch(theta: torch.Tensor) -> torch.Tensor:
    """Torch lift for rollout-loss-capable fits."""
    return torch.stack(
        [torch.sin(theta[:, j] - theta[:, i]) for (i, j) in PAIR_FEATURES],
        dim=1,
    )


Phi_train = build_lift(THETA_TRAIN)
design_cond = np.linalg.cond(np.column_stack([np.ones(len(Phi_train)), Phi_train]))
print(f"[DATA]  Lift design condition number: {design_cond:.2e}")
print("[DATA]  Feature definitions:")
for definition in FEATURE_DEFS:
    print(f"         {definition}")

# ---------------------------------------------------------------------------
# 3. KANDy model
# ---------------------------------------------------------------------------
lift = CustomLift(
    fn=build_lift,
    output_dim=N_FEAT,
    torch_fn=build_lift_torch,
    name="kuramoto_pair_sines",
)

model = KANDy(
    lift=lift,
    grid=5,
    k=3,
    steps=60,
    seed=SEED,
    base_fun=torch.nn.Identity(),
    device="cpu",
)

print("\n[TRAIN] Fitting KAN ...")
model.fit(
    X=THETA_TRAIN,
    X_dot=THETA_DOT,
    val_frac=0.15,
    test_frac=0.15,
    lamb=0.0,
    patience=0,
    verbose=True,
)

# ---------------------------------------------------------------------------
# 4. Rollout validation -- before any symbolic/linear reporting
# ---------------------------------------------------------------------------

print(f"\n[ROLLOUT] Rolling out {T_EVAL} steps ...", flush=True)
pred = model.rollout(THETA_TRUE[0], T=T_EVAL, dt=DT_SIM, integrator="rk4")

rmse_theta = np.sqrt(np.mean(wrap_pi(pred - THETA_TRUE) ** 2))
print(f"[EVAL]  Wrapped phase RMSE theta: {rmse_theta:.6f}", flush=True)

r_true = np.abs(np.mean(np.exp(1j * THETA_TRUE), axis=1))
r_pred = np.abs(np.mean(np.exp(1j * pred), axis=1))
rmse_r = np.sqrt(np.mean((r_pred - r_true) ** 2))
print(f"[EVAL]  Order parameter RMSE r: {rmse_r:.6f}", flush=True)

signal_true = np.cos(THETA_TRUE)
signal_pred = np.cos(pred)
rmse_signal = np.sqrt(np.mean((signal_pred - signal_true) ** 2))
print(f"[EVAL]  Oscillator signal RMSE cos(theta): {rmse_signal:.6f}", flush=True)

# ---------------------------------------------------------------------------
# 5. Symbolic/effective linear readout
# ---------------------------------------------------------------------------
# The true Kuramoto RHS is linear in the lifted pair-sine features. For this
# case the most stable symbolic report is the effective linear map from the
# fitted KANDy model, as recommended in references/api.md for coefficient
# recovery. PyKAN auto_symbolic is edge-local and can move constants between
# affine edge wrappers even when the predictive model is numerically correct.

def effective_linear_readout(model: KANDy, theta: np.ndarray) -> np.ndarray:
    """Least-squares linear surrogate of model.predict(theta) on [1, phi(theta)]."""
    phi = build_lift(theta)
    design = np.column_stack([np.ones(len(phi)), phi])
    coef, *_ = np.linalg.lstsq(design, model.predict(theta), rcond=None)
    return coef.T  # (N, 1 + N_FEAT)


def true_linear_readout() -> np.ndarray:
    """Exact coefficient matrix in the [1, s_ij] basis."""
    coef = np.zeros((N, 1 + N_FEAT))
    coef[:, 0] = OMEGA
    for k_idx, (i, j) in enumerate(PAIR_FEATURES, start=1):
        coef[i, k_idx] = K_OVER_N
        coef[j, k_idx] = -K_OVER_N
    return coef


A_eff = effective_linear_readout(model, THETA_TRAIN)
A_true = true_linear_readout()
coef_err = np.max(np.abs(A_eff - A_true))

symbols = [sp.Symbol(name) for name in FEATURE_NAMES]
formulas = []
for row in A_eff:
    expr = sp.Float(round(float(row[0]), 4))
    for coeff, symbol in zip(row[1:], symbols):
        if abs(coeff) >= 5e-4:
            expr += sp.Float(round(float(coeff), 4)) * symbol
    formulas.append(sp.expand(expr))

print("\n[SYMBOLIC] Effective linear formulas in the independent pair basis:")
print(f"           max |learned - true| coefficient error = {coef_err:.2e}\n")
for i, formula in enumerate(formulas):
    print(f"  dtheta{i}/dt = {formula}", flush=True)

    true_terms = [f"{OMEGA[i]:.4f}"]
    for (a, b), name in zip(PAIR_FEATURES, FEATURE_NAMES):
        if i == a:
            true_terms.append(f"+ {K_OVER_N:.2f}*{name}")
        elif i == b:
            true_terms.append(f"- {K_OVER_N:.2f}*{name}")
    print(f"    [TRUE] = {' '.join(true_terms)}", flush=True)
print(flush=True)

thresholds = {
    "rmse_theta": 1e-2,
    "rmse_signal": 5e-3,
    "rmse_order_parameter": 5e-3,
    "max_coefficient_error": 1e-3,
}
checks_ok = (
    rmse_theta < thresholds["rmse_theta"]
    and rmse_signal < thresholds["rmse_signal"]
    and rmse_r < thresholds["rmse_order_parameter"]
    and coef_err < thresholds["max_coefficient_error"]
)
metrics = {
    "seed": SEED,
    "n_oscillators": N,
    "n_train": N_TRAIN,
    "n_eval": T_EVAL,
    "dt": DT_SIM,
    "rmse_theta": rmse_theta,
    "rmse_signal": rmse_signal,
    "rmse_order_parameter": rmse_r,
    "max_coefficient_error": coef_err,
    "thresholds": thresholds,
    "passed": print_check(
        checks_ok,
        "rollout and coefficients within thresholds",
    ),
}
save_metrics(RESULT_DIR / "metrics.json", metrics)

# ---------------------------------------------------------------------------
# 6. Figure helpers
# ---------------------------------------------------------------------------

def plot_phase_trajectories(
    outdir: Path,
    t_roll: np.ndarray,
    theta_true: np.ndarray,
    theta_pred: np.ndarray,
) -> None:
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(N, 1, figsize=(8, 2 * N), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t_roll, theta_true[:, i], color=colors[i], lw=1.2, label="True")
        ax.plot(t_roll, theta_pred[:, i], color=colors[i], lw=1.0, ls="--", label="KANDy")
        ax.set_ylabel(f"$\\theta_{i}$")
        if i == 0:
            ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("time")
    fig.suptitle("Kuramoto: phase trajectories", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "phase_trajectories.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "phase_trajectories.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_oscillator_signals(
    outdir: Path,
    t_roll: np.ndarray,
    signal_true: np.ndarray,
    signal_pred: np.ndarray,
) -> None:
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(N, 1, figsize=(8, 2 * N), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t_roll, signal_true[:, i], color=colors[i], lw=1.2, label="True")
        ax.plot(t_roll, signal_pred[:, i], color=colors[i], lw=1.0, ls="--", label="KANDy")
        ax.set_ylabel(f"$\\cos(\\theta_{i})$")
        ax.set_ylim(-1.1, 1.1)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("time")
    fig.suptitle("Kuramoto: oscillator signals", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "oscillator_signals.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "oscillator_signals.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_order_parameter(
    outdir: Path,
    t_roll: np.ndarray,
    r_true: np.ndarray,
    r_pred: np.ndarray,
    rmse_r: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t_roll, r_true, color="steelblue", lw=1.4, label="True $r(t)$")
    ax.plot(t_roll, r_pred, color="tomato", lw=1.0, ls="--", label="KANDy $r(t)$")
    ax.set_xlabel("time")
    ax.set_ylabel("$r(t)$")
    ax.set_title(f"Kuramoto order parameter  (RMSE={rmse_r:.4f})")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(outdir / "order_parameter.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "order_parameter.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_linear_coefficients(
    outdir: Path,
    true_coef: np.ndarray,
    learned_coef: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    vmax = max(np.max(np.abs(true_coef[:, 1:])), np.max(np.abs(learned_coef[:, 1:])))
    for ax, mat, title in [
        (axes[0], true_coef[:, 1:], "True"),
        (axes[1], learned_coef[:, 1:], "KANDy linear readout"),
    ]:
        im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(N_FEAT))
        ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("lift feature")
    axes[0].set_yticks(range(N))
    axes[0].set_yticklabels([f"$d\\theta_{i}/dt$" for i in range(N)])
    fig.colorbar(im, ax=axes, shrink=0.85, label="coefficient")
    fig.suptitle("Kuramoto coefficient recovery", fontsize=11)
    fig.savefig(outdir / "linear_coefficients.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "linear_coefficients.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_loss(outdir: Path, model: KANDy) -> None:
    if hasattr(model, "train_results_") and model.train_results_ is not None:
        fig, _ax = plot_loss_curves(
            model.train_results_,
            save=str(outdir / "loss_curves"),
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
use_pub_style()
t_roll = np.arange(T_EVAL) * DT_SIM
plot_phase_trajectories(RESULT_DIR, t_roll, THETA_TRUE, pred)
plot_oscillator_signals(RESULT_DIR, t_roll, signal_true, signal_pred)
plot_order_parameter(RESULT_DIR, t_roll, r_true, r_pred, rmse_r)
plot_linear_coefficients(RESULT_DIR, A_true, A_eff)
plot_training_loss(RESULT_DIR, model)

print(f"[FIGS]  Saved {RESULT_DIR}/")
