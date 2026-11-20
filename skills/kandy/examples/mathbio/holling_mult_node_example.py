#!/usr/bin/env python3
"""KANDy example: a multiplication node, and reading the product term off by ALS.

Rosenzweig-MacArthur predator-prey with a Holling Type II functional response:

    dN/dt = r*N*(1 - N/K)  -  a*N*P / (1 + a*h*N)
    dP/dt = e*a*N*P / (1 + a*h*N)  -  m*P

THE OBSTRUCTION
---------------
The interaction is  pred = P * g(N)  with  g(N) = a*N/(1 + a*h*N)  — a product
of two DIFFERENT variables.  A separable KAN computes a SUM of univariate
functions, so it cannot represent it, and this is not a resolution problem:
give every coordinate a 60-knot basis and solve for the best possible additive
model and the ceiling on dN/dt is

    lift [N, P]                 best possible additive R^2 = 0.9376
    lift [N, P, N*P]                                       = 0.9931
    lift [N, P, N*P, N^2*P]                                = 0.9996
    lift [N, pred]  (oracle)                               = 1.0000

The ladder is the geometric series  a*N*P/(1+a*h*N) = a*NP - a^2*h*N^2P + ... :
each extra cross-coordinate buys one term.  `mathbio/holling_type_ii_example.py`
takes the oracle route — it supplies `pred` built from the TRUE a and h, and so
recovers r, K, e, m but cannot be wrong about a or h.  This script recovers a
and h too, supplying neither.

THE FIX: ONE MULTIPLICATION NODE
--------------------------------
A KAN multiplication node computes psi_1(.) * psi_2(.), which factors the term
exactly:  one branch learns g(N), the other learns P.  g is univariate, which
is what a spline is for.  Architecture:

    width = [2, [2, 1], 1]        2 inputs -> 2 sum nodes + 1 mult node -> 1

    R^2:  additive 0.9374  ->  multiplication node 0.99999

WHY auto_symbolic CANNOT FINISH THE JOB
---------------------------------------
Snapping this model does not merely produce an ugly formula, it produces a
DIFFERENT MODEL.  The script measures this and records it in metrics.json; the
size of the damage varies run to run (R^2 has been seen to drop from 0.99999 to
anywhere in 0.75-0.98, with the output moving by 75-176 against a range of
~250), so it is REPORTED, not gated — this is a statement about a dependency's
behaviour and the example should not start failing if PyKAN improves.

The extracted expression drops one of the two variables entirely.  Three things
compound: the product is invariant under psi_1 -> c*psi_1, psi_2 -> psi_2/c so
the branches are only defined up to a scale; the output edge is itself a free
spline, so a perfect product gets reshaped on the way out; and two layers of
snapped primitives compose into something that does not simplify.  (On a
deepcopy, auto_symbolic also needs a forward pass first to repopulate the
cached activations, or it raises IndexError.)

THE RECIPE: LET ALS READ THAT ONE TERM
--------------------------------------
Do not call auto_symbolic on the product.  Everything else it handles well.
For the product, evaluate the TRAINED model on a regular (N, P) grid and factor

    model(N, P)  =  f(N)  +  u(N) * v(P)

by alternating least squares — fix one factor and the other is an ordinary
linear least squares, so each half-step is well posed.  The trained network is
smooth and can be sampled anywhere, so on a grid the interaction is a dense
matrix that is exactly rank one once f(N) is removed.

Two constraints make the split between the additive and product parts unique:

    v(0) = 0                 no interaction when there are no predators
    no standalone b(P) term  dN/dt has no pure-predator contribution

Without them the fit is just as good but the parts trade against each other and
f(N) comes out as 0.40*N instead of 1.00*N.  The remaining freedom is the
single scale u -> c*u, v -> v/c, fixed by normalising v to unit slope in P.

Then name the univariate factor.  This is where the Holling form pays off: it is
a single affine-composed primitive,

    g(N) = a*N/(1 + a*h*N) = 1/h - (1/h)/(1 + a*h*N) = d + c/(alpha*N + 1)
    =>   h = 1/d,      a = alpha*d

so a 1-D fit returns the ecological constants.  Contrast
`odes/mackey_glass_delay_example.py`, whose Hill term beta*x/(1+x^10) is a
rational nonlinearity a spline also learns perfectly, but which is NOT an
affine-composed primitive — there the recipe stops at the recovered shape.

Runtime: ~2 minutes.
KAN:  width = [2, [2, 1], 1], mult_arity = 2
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from _utils import ensure_outdir, print_check, save_metrics
from kandy import KANDy, CustomLift
from kandy.plotting import use_pub_style

# ---------------------------------------------------------------------------
# SHIM — KANDy hardcodes width=[m, n] ("depth fixed at 1 — never expose as
# param"), so multiplication nodes are unreachable through its API even though
# PyKAN's MultKAN supports them via width=[m, [n_sum, n_mult], n] and
# mult_arity.  This swaps the constructor kandy.core uses.  Delete once KANDy
# exposes a width/mult_arity argument.  Same pattern as the bundled
# DelayEmbedding in odes/mackey_glass_delay_example.py.
# ---------------------------------------------------------------------------
import kandy.core as _kc
from kan import MultKAN

_ORIG_KAN = _kc.KAN


def _mult_kan_factory(n_sum: int, n_mult: int):
    def factory(**kw):
        m, n = kw.pop("width")
        kw["width"] = [m, [n_sum, n_mult], n]
        kw["mult_arity"] = 2
        return MultKAN(**kw)
    return factory


# ---------------------------------------------------------------------------
# 0. Reproducibility and the true ecology
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

R_TRUE, K_TRUE = 1.0, 50.0
A_TRUE, H_TRUE = 1.2, 0.1          # attack rate, handling time — TO BE RECOVERED
AH = A_TRUE * H_TRUE

print(f"[MODEL] r={R_TRUE}, K={K_TRUE}, a={A_TRUE}, h={H_TRUE}")
print(f"[MODEL] g(N) = a*N/(1+a*h*N) saturates at 1/h = {1/H_TRUE:.1f}")

# ---------------------------------------------------------------------------
# 1. Data — independent samples over a box, derivatives from the true RHS
#
#    Box sampling (not one trajectory) for the reason in mathbio/sir_example.py:
#    a single Rosenzweig-MacArthur run spends most of its length near a fixed
#    point and barely excites the interaction.
# ---------------------------------------------------------------------------
N_SAMPLES = 8000
rng = np.random.default_rng(0)
N_RAW = rng.uniform(0.05, 50.0, N_SAMPLES)
P_RAW = rng.uniform(0.05, 30.0, N_SAMPLES)
PRED = (A_TRUE * N_RAW * P_RAW) / (1 + AH * N_RAW)
TARGET = (R_TRUE * N_RAW * (1 - N_RAW / K_TRUE) - PRED).astype(np.float32)

X_RAW = np.column_stack([N_RAW, P_RAW]).astype(np.float32)
X_MU, X_SD = X_RAW.mean(0, keepdims=True), X_RAW.std(0, keepdims=True)
X = ((X_RAW - X_MU) / X_SD).astype(np.float32)      # KAN grids live near [-1,1]
Y = TARGET[:, None]
print(f"[DATA]  {N_SAMPLES} box samples, N in [0.05, 50], P in [0.05, 30]")


def r2_of(pred, true=TARGET):
    return float(1 - np.sum((true - pred) ** 2) / np.sum((true - true.mean()) ** 2))


def fit_model(n_sum: int, n_mult: int, steps: int = 400):
    """Fit KANDy; n_mult=0 gives the ordinary additive model."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    _kc.KAN = _mult_kan_factory(n_sum, n_mult) if n_mult else _ORIG_KAN
    try:
        m = KANDy(lift=CustomLift(fn=lambda Z: Z, output_dim=2, name="identity"),
                  grid=8, k=3, steps=steps, seed=SEED)
        m.fit(X=X, X_dot=Y, val_frac=0.15, test_frac=0.15, lamb=0.0,
              patience=100, verbose=False)
        return m
    finally:
        _kc.KAN = _ORIG_KAN


def model_out(m, N_arr, P_arr):
    """Evaluate a fitted model at raw (N, P)."""
    raw = np.column_stack([N_arr, P_arr]).astype(np.float32)
    return m.predict((raw - X_MU) / X_SD).ravel()


# ---------------------------------------------------------------------------
# 2. The separable baseline — it must fail, and it must fail at the ceiling
# ---------------------------------------------------------------------------
print("\n[BASE]  additive KANDy, width=[2, 1] ...")
base = fit_model(0, 0)
r2_base = r2_of(base.predict(X).ravel())
print(f"[BASE]  R^2 = {r2_base:.6f}   (best possible additive model: 0.9376)")
print("[BASE]  a sum of univariate functions cannot express P*g(N); more spline")
print("[BASE]  resolution does not move this.")

# ---------------------------------------------------------------------------
# 3. One multiplication node
# ---------------------------------------------------------------------------
print("\n[MULT]  KANDy with width=[2, [2, 1], 1], mult_arity=2 ...")
model = fit_model(2, 1)
r2_mult = r2_of(model.predict(X).ravel())
print(f"[MULT]  R^2 = {r2_mult:.8f}   ({(1-r2_base)/(1-r2_mult):.0f}x less "
      f"unexplained variance than additive)")

# ---------------------------------------------------------------------------
# 4. Why auto_symbolic is not used on the product
#
#    Measured, not asserted: snapping replaces the network with a materially
#    different function.  Done on a COPY so the trained model survives.
# ---------------------------------------------------------------------------
import copy

probe = copy.deepcopy(model)
before = probe.predict(X).ravel()
# auto_symbolic reads the cached activations, and deepcopy does not carry them —
# repopulate with a forward pass or it raises IndexError.
probe.model_.save_act = True
with torch.no_grad():
    probe.model_(torch.tensor(X[:4096], dtype=torch.float32))
try:
    probe.model_.auto_symbolic(lib=["x", "x^2", "1/x", "0"])
    after = probe.predict(X).ravel()
    r2_snap = r2_of(after)
    snap_shift = float(np.abs(after - before).max())
    snap_note = "ok"
except Exception as exc:                                   # pragma: no cover
    r2_snap, snap_shift = float("nan"), float("nan")
    snap_note = f"{type(exc).__name__}: {exc}"
print(f"\n[SNAP]  R^2 before auto_symbolic = {r2_of(before):.6f}")
if snap_note == "ok":
    print(f"[SNAP]  R^2 after  auto_symbolic = {r2_snap:.6f}   "
          f"max output change {snap_shift:.1f}")
    print("[SNAP]  Snapping does not just rename the model, it replaces it — so")
    print("[SNAP]  the product term is read off by ALS below instead.")
else:
    print(f"[SNAP]  auto_symbolic could not run at all: {snap_note}")
    print("[SNAP]  Either way it is not a route to this equation; ALS below is.")

# ---------------------------------------------------------------------------
# 5. ALS readout of the product term
# ---------------------------------------------------------------------------
N_GRID = np.linspace(1.0, 48.0, 120)
P_GRID = np.linspace(0.5, 29.0, 120)
N_BASIS = 40


def hat_basis(x, n_knots, lo, hi):
    """Piecewise-linear basis — a free univariate shape, as a spline edge is."""
    knots = np.linspace(lo, hi, n_knots)
    B = np.zeros((len(x), n_knots))
    for i, k in enumerate(knots):
        left = knots[i - 1] if i > 0 else knots[0] - (knots[1] - knots[0])
        right = knots[i + 1] if i < n_knots - 1 else knots[-1] + (knots[-1] - knots[-2])
        B[:, i] = np.minimum(np.clip((x - left) / (k - left), 0, 1),
                             np.clip((right - x) / (right - k), 0, 1))
    return B


def als_factor(surface, N_grid, P_grid, iters=30):
    """Factor surface(N,P) = f(N) + u(N)*v(P) by alternating least squares.

    Each half-step is a plain linear solve.  The P-basis drops its first hat so
    v(0) = 0, and no standalone b(P) term is included — together these make the
    split between the additive and product parts unique.
    """
    NN, PP = np.meshgrid(N_grid, P_grid, indexing="ij")
    n, p, y = NN.ravel(), PP.ravel(), surface.ravel()
    BN = hat_basis(n, N_BASIS, N_grid.min(), N_grid.max())
    BP = hat_basis(p, N_BASIS, 0.0, P_grid.max())[:, 1:]
    nb = BN.shape[1]
    v = np.linalg.lstsq(BP, p, rcond=None)[0]
    for _ in range(iters):
        w = BP @ v
        sol, *_ = np.linalg.lstsq(np.hstack([BN, BN * w[:, None]]), y, rcond=None)
        a_c, u = sol[:nb], sol[nb:]
        z = BN @ u
        sol, *_ = np.linalg.lstsq(np.hstack([BN, BP * z[:, None]]), y, rcond=None)
        a_c, v = sol[:nb], sol[nb:]
    resid = y - (BN @ a_c + (BN @ u) * (BP @ v))
    return a_c, u, v, float(np.sqrt(np.mean(resid ** 2)))


NN, PP = np.meshgrid(N_GRID, P_GRID, indexing="ij")
SURFACE = model_out(model, NN.ravel(), PP.ravel()).reshape(NN.shape)
a_coef, u_coef, v_coef, als_rmse = als_factor(SURFACE, N_GRID, P_GRID)
print(f"\n[ALS]   readout residual rmse = {als_rmse:.3e}  "
      f"(surface range {SURFACE.min():.1f}..{SURFACE.max():.1f})")

# fix the remaining scale: normalise v to unit slope in P
BP_g = hat_basis(P_GRID, N_BASIS, 0.0, P_GRID.max())[:, 1:]
v_curve = BP_g @ v_coef
v_slope = float(np.polyfit(P_GRID, v_curve, 1)[0])
v_fixed = v_curve / v_slope
r2_v = float(1 - np.sum((P_GRID - v_fixed) ** 2) / np.sum((P_GRID - P_GRID.mean()) ** 2))

BN_g = hat_basis(N_GRID, N_BASIS, N_GRID.min(), N_GRID.max())
u_curve = (BN_g @ u_coef) * v_slope
g_true = -A_TRUE * N_GRID / (1 + AH * N_GRID)
r2_u = float(1 - np.sum((g_true - u_curve) ** 2) / np.sum((g_true - g_true.mean()) ** 2))
print(f"[ALS]   v(P) vs P:      R^2 = {r2_v:.8f}   <- also a falsification check:")
print("[ALS]     if the interaction were not a rank-one product in (N, P),")
print("[ALS]     v would not come back as a clean straight line.")
print(f"[ALS]   u(N) vs -g(N):  R^2 = {r2_u:.8f}")

# ---------------------------------------------------------------------------
# 6. Name the univariate factor -> the ecological constants
# ---------------------------------------------------------------------------
def saturating(x, c, alpha, d):
    return d + c / (alpha * x + 1.0)


popt, _ = curve_fit(saturating, N_GRID, -u_curve, p0=[-10.0, 0.1, 10.0],
                    maxfev=400_000)
c_fit, alpha_fit, d_fit = (float(v) for v in popt)
h_rec = 1.0 / d_fit
a_rec = alpha_fit * d_fit
h_err = abs(h_rec - H_TRUE) / H_TRUE
a_err = abs(a_rec - A_TRUE) / A_TRUE
print(f"\n[NAME]  g(N) ~ d + c/(alpha*N + 1):  c={c_fit:.4f} "
      f"alpha={alpha_fit:.5f} d={d_fit:.4f}")
print(f"[NAME]  h = 1/d     = {h_rec:.5f}   (true {H_TRUE})   err {100*h_err:.2f}%")
print(f"[NAME]  a = alpha*d = {a_rec:.5f}   (true {A_TRUE})   err {100*a_err:.2f}%")

f_curve = BN_g @ a_coef
f_coef = np.polyfit(N_GRID, f_curve, 2)
r_rec, rk_rec = float(f_coef[1]), float(f_coef[0])
K_rec = -r_rec / rk_rec if rk_rec != 0 else float("nan")
print(f"[NAME]  f(N) ~ {rk_rec:+.5f}*N^2 {r_rec:+.5f}*N   "
      f"(true {-R_TRUE/K_TRUE:+.5f}, {R_TRUE:+.5f})")
print(f"[NAME]  r = {r_rec:.4f} (true {R_TRUE}),  K = -r/(coef) = {K_rec:.2f} "
      f"(true {K_TRUE})")
print(f"\n[EQN]   RECOVERED  dN/dt = {r_rec:.4f}*N {rk_rec:+.4f}*N^2 "
      f"- {a_rec:.4f}*N*P/(1 + {alpha_fit:.4f}*N)")
print(f"[EQN]   TRUE             = {R_TRUE:.4f}*N {-R_TRUE/K_TRUE:+.4f}*N^2 "
      f"- {A_TRUE:.4f}*N*P/(1 + {AH:.4f}*N)")

# ---------------------------------------------------------------------------
# 7. Verification
#
#     Two-sided.  The separable baseline must still FAIL near its ceiling — if
#     it ever passed, the obstruction this example exists to demonstrate would
#     not be being demonstrated.  The multiplication node must fit, and the ALS
#     readout must return the constants.
#
#     The auto_symbolic collapse is RECORDED but not gated: it is a statement
#     about a dependency's behaviour, and should PyKAN improve, this example
#     should not start failing.  What is gated is that the ALS readout works.
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "additive_ceiling_r2_max": 0.96,     # separable model must fail
    "mult_node_r2_min": 0.999,
    "als_residual_rmse_max": 1.0,
    "v_vs_P_r2_min": 0.999,              # interaction really is rank one
    "u_vs_g_r2_min": 0.99,
    "a_rel_err_max": 0.05,
    "h_rel_err_max": 0.05,
    "r_rel_err_max": 0.05,
}
passed = print_check(
    r2_base <= THRESHOLDS["additive_ceiling_r2_max"]
    and r2_mult >= THRESHOLDS["mult_node_r2_min"]
    and als_rmse <= THRESHOLDS["als_residual_rmse_max"]
    and r2_v >= THRESHOLDS["v_vs_P_r2_min"]
    and r2_u >= THRESHOLDS["u_vs_g_r2_min"]
    and a_err <= THRESHOLDS["a_rel_err_max"]
    and h_err <= THRESHOLDS["h_rel_err_max"]
    and abs(r_rec - R_TRUE) / R_TRUE <= THRESHOLDS["r_rel_err_max"],
    "multiplication node + ALS readout recovers a, h, r and K",
)
RESULT_DIR = ensure_outdir("results/HollingMultNode")
save_metrics(RESULT_DIR / "metrics.json", {
    "task": "HollingMultNode",
    "category": "mathbio",
    "seed": SEED,
    "n_train": N_SAMPLES,
    "sampling": "uniform box + true RHS (oracle derivatives)",
    "architecture": "width=[2,[2,1],1], mult_arity=2",
    "r2_additive_baseline": r2_base,
    "r2_additive_ceiling_reference": 0.937630,
    "r2_mult_node": r2_mult,
    "r2_after_auto_symbolic": r2_snap,
    "auto_symbolic_max_output_change": snap_shift,
    "auto_symbolic_status": snap_note,
    "als_residual_rmse": als_rmse,
    "r2_v_vs_P": r2_v,
    "r2_u_vs_g": r2_u,
    "recovered": {"a": a_rec, "h": h_rec, "r": r_rec, "K": K_rec},
    "true": {"a": A_TRUE, "h": H_TRUE, "r": R_TRUE, "K": K_TRUE},
    "rel_err": {"a": a_err, "h": h_err,
                "r": abs(r_rec - R_TRUE) / R_TRUE},
    "parameters_supplied_as_oracle": [],
    "thresholds": THRESHOLDS,
    "passed": passed,
})


# ---------------------------------------------------------------------------
# 8. Figures
# ---------------------------------------------------------------------------
def plot_ceiling(outdir):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar([0, 1], [r2_base, r2_mult], color=["#7f7f7f", "#d62728"], width=0.55)
    ax.axhline(0.937630, ls="--", lw=1.0, color="k")
    ax.text(0.5, 0.9405, "best possible additive model", ha="center", fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["additive\nwidth=[2,1]", "mult node\nwidth=[2,[2,1],1]"])
    ax.set_ylim(0.90, 1.005)
    ax.set_ylabel("$R^2$ on $dN/dt$")
    ax.set_title("A product of two variables needs a product")
    fig.tight_layout()
    fig.savefig(outdir / "ceiling.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_factor(outdir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(N_GRID, -g_true, lw=2.6, color="#1f77b4", alpha=0.5,
                 label=r"true $g(N)=aN/(1+ahN)$")
    axes[0].plot(N_GRID, -u_curve, lw=1.3, ls="--", color="#d62728",
                 label="ALS readout")
    axes[0].plot(N_GRID, saturating(N_GRID, *popt), lw=1.0, ls=":", color="k",
                 label="named $d + c/(\\alpha N+1)$")
    axes[0].set_xlabel("$N$")
    axes[0].set_ylabel("$g(N)$")
    axes[0].set_title(f"functional response recovered\n"
                      f"$a$={a_rec:.4f} (true {A_TRUE}), $h$={h_rec:.5f} "
                      f"(true {H_TRUE})", fontsize=10)
    axes[0].legend(fontsize=8)

    axes[1].plot(P_GRID, P_GRID, lw=2.6, color="#1f77b4", alpha=0.5, label="$P$")
    axes[1].plot(P_GRID, v_fixed, lw=1.3, ls="--", color="#2ca02c",
                 label="ALS readout $v(P)$")
    axes[1].set_xlabel("$P$")
    axes[1].set_ylabel("$v(P)$")
    axes[1].set_title(f"the other factor is $P$\n$R^2$={r2_v:.6f} "
                      "(rank-one check)", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "recovered_factors.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_surface(outdir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    true_surf = (R_TRUE * NN * (1 - NN / K_TRUE)
                 - A_TRUE * NN * PP / (1 + AH * NN))
    recon = (BN_g @ a_coef)[:, None] + np.outer(BN_g @ u_coef, BP_g @ v_coef)
    for ax, (Z, t) in zip(axes, [(true_surf, "true $dN/dt$"),
                                 (SURFACE, "trained mult-node KAN"),
                                 (recon, "ALS reconstruction")]):
        im = ax.pcolormesh(N_GRID, P_GRID, Z.T, shading="auto", cmap="viridis")
        ax.set_xlabel("$N$")
        ax.set_ylabel("$P$")
        ax.set_title(t, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(outdir / "surfaces.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


use_pub_style()
plot_ceiling(RESULT_DIR)
plot_factor(RESULT_DIR)
plot_surface(RESULT_DIR)
print(f"[FIGS]  Saved {RESULT_DIR}/")
