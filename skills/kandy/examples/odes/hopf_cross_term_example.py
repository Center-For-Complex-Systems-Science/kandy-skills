#!/usr/bin/env python3
"""KANDy example: the cross-term obstruction in 2D — a plain KAN vs. a lift.

Hopf (Stuart-Landau) normal form in Cartesian coordinates:

    dx/dt = mu*x - om*y - x*(x^2 + y^2)  =  mu*x - om*y - x^3 - x*y^2
    dy/dt = om*x + mu*y - y*(x^2 + y^2)  =  om*x + mu*y - x^2*y - y^3

THE POINT: two inputs, two outputs, two DIFFERENT mixed monomials
------------------------------------------------------------------
The KAN paper demonstrates that a network can represent a product by plotting
f(x, y) = x*y, which a depth-2 KAN builds from the quadratic identity
x*y = ((x+y)^2 - (x-y)^2)/4.  That demo has one output, so a single hidden
representation can be dedicated to the one product that output needs.

A dynamical system is not like that.  Lorenz needs x*y routed into dz/dt and
x*z routed into dy/dt *simultaneously*, from one shared hidden layer.  This
script is the smallest system with that same structure: dx/dt needs x*y^2,
dy/dt needs x^2*y — different mixed monomials, one per output, from only two
inputs.  It is therefore the 2D analogue of the Lorenz obstruction and can be
plotted the way the KAN paper plots x*y.

THE DELIVERABLE IS AN EQUATION
------------------------------
Everything below is scored on whether a closed-form RHS can be READ OFF the
fit, with correct coefficients.  Prediction accuracy is reported only where it
shows a model is ALSO wrong, never as the comparison itself.

What the script measures (numbers below are from an actual run)
--------------------------------------------------------------
1. WITHOUT THE LIFT THERE IS NO EQUATION TO READ, AND THE FIT IS WRONG TOO.
   A separable model computes a SUM of univariate functions.  We solve
   directly for the best possible additive model - a 60-knot basis per
   coordinate, least squares, no network involved - and it is capped at
   R^2 = 0.962.  A single-layer KAN on raw (x, y) lands on that ceiling
   (0.961), as it must; widening it, refining the grid or training longer
   cannot move this.  The two edges it does have are then bent away from the
   true terms to absorb what they cannot represent (first figure, top left),
   so even the coefficients it CAN express come out wrong.

   Pointwise R^2 badly understates the damage: at R^2 = 0.96 the rolled-out
   limit cycle collapses from radius 1.000 to 0.551, a 45% error.

2. WITH THE LIFT THE EQUATION COMES BACK EXACTLY.
   phi = [x, y, x*y^2, x^2*y] makes every term linear in a lifted coordinate,
   so each edge is a single named quantity.  Fitting the edges recovers

       dx/dt = +1.0000*x -1.0000*x^3 -2.0000*y -1.0000*(x*y^2)
       dy/dt = +2.0000*x +1.0000*y -1.0000*y^3 -1.0000*(x^2*y)

   against a truth of mu = 1, om = 2 and unit cubic and cross terms - max
   deviation 2e-5 over all sixteen coefficients, with the two unused edges
   identically zero.  R^2 = 1.000000 and the rollout radius is 1.000001.

   Note which terms are in the lift: the mixed monomials ONLY.  x^3 and y^3
   are powers of a SINGLE variable, so the spline edge on x learns
   mu*x - x^3 by itself.  Putting x^3 in the lift alongside x would make the
   decomposition non-unique (cf. mathbio/lotka_volterra_competition_example.py).

3. WHY DEPTH IS BESIDE THE POINT.
   The obvious objection is "just add a layer".  A depth-2 KAN [2,5,2] on raw
   (x, y) does fit well - R^2 = 0.99999, rollout radius 1.001 - because it can
   build products by composition, as the KAN paper's x*y demo does.  It is not
   a competing result: it returns no equation.  Each product is smeared across
   the hidden layer, so the layer-0 edges (third figure) are ten smooth curves
   with no term to name and no coefficient to extract, and composition means
   there is no decomposition to snap symbolically in the first place.

   Depth buys you a better black box.  This script is about the other thing.

THE TRAP: DO NOT TRAIN ON THE LIMIT CYCLE
-----------------------------------------
On the limit cycle x^2 + y^2 = mu, so the cross term collapses:

    x*y^2 = x*(mu - x^2) = mu*x - x^3

which is a function of x ALONE.  Train on converged trajectories and the cross
term is exactly representable additively - the separable model scores ~1.0 and
the obstruction is invisible.  It reappears the moment the model is asked
about a state off the cycle, which is exactly what a rollout does.  This
script therefore samples the vector field over a DISK.  Same lesson as the
conserved-manifold degeneracy in mathbio/sir_example.py: a model fitted on an
invariant set only learns the dynamics restricted to that set.

Lift  phi: R^2 -> R^4   theta = [x, y, x*y^2, x^2*y]
KAN:  width = [4, 2],  base_fun='zero'
"""

import contextlib
import io
import os
import shutil
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The KAN fit is not bit-reproducible under multi-threaded BLAS even at fixed
# seed. This script compares four fits against each other, so an unpinned
# thread schedule would show up as a difference between MODELS. Pin it.
torch.set_num_threads(1)

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from _utils import ensure_outdir, print_check, save_metrics

from kan import KAN
from kandy import KANDy, CustomLift, rk4_integrate_numpy
from kandy.plotting import (
    fit_polynomial,
    get_edge_activation,
    use_pub_style,
)

# ---------------------------------------------------------------------------
# 0. Reproducibility and parameters
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

MU = 1.0            # supercritical: stable limit cycle at r = sqrt(MU) = 1
OM = 2.0            # rotation rate

R_TRAIN = 1.6       # training disk radius (limit cycle sits at r = 1)
R_EXTRAP = 2.2      # extrapolation annulus: R_TRAIN < r < R_EXTRAP
N_TRAIN = 4000
N_EXTRAP = 1500

STEPS = 200         # LBFGS steps for every model (identical budget)
GRID = 5

OUTDIR = ensure_outdir(Path(__file__).resolve().parents[1] / "results" / "HopfCrossTerm")


def rhs(state):
    """True vector field. Accepts (2,) or (N, 2)."""
    s = np.atleast_2d(np.asarray(state, dtype=float))
    x, y = s[:, 0], s[:, 1]
    r2 = x * x + y * y
    out = np.column_stack([MU * x - OM * y - x * r2,
                           OM * x + MU * y - y * r2])
    return out[0] if np.ndim(state) == 1 else out


def sample_annulus(n, r_lo, r_hi, rng):
    """Uniform samples in an annulus (r_lo = 0 gives a disk)."""
    u = rng.uniform(0.0, 1.0, n)
    r = np.sqrt(r_lo ** 2 + u * (r_hi ** 2 - r_lo ** 2))
    th = rng.uniform(0.0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(th), r * np.sin(th)])


# ---------------------------------------------------------------------------
# 1. Data — sample the vector field OFF the attractor (see docstring)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)

X_disk = sample_annulus(N_TRAIN, 0.0, R_TRAIN, rng)
Xdot_disk = rhs(X_disk)

X_out = sample_annulus(N_EXTRAP, R_TRAIN, R_EXTRAP, rng)
Xdot_out = rhs(X_out)

# The same count of samples, but confined to the limit cycle, to quantify the
# trap: there the cross term is NOT distinguishable from a function of x alone.
th = rng.uniform(0.0, 2 * np.pi, N_TRAIN)
X_cycle = np.sqrt(MU) * np.column_stack([np.cos(th), np.sin(th)])
Xdot_cycle = rhs(X_cycle)

print(f"[DATA]  {N_TRAIN} states in disk r < {R_TRAIN}, "
      f"{N_EXTRAP} in annulus {R_TRAIN} < r < {R_EXTRAP}")


# ---------------------------------------------------------------------------
# 2. The additive ceiling — the bound every separable model obeys
# ---------------------------------------------------------------------------
def best_additive_r2(features, target, n_knots=60):
    """Best possible R^2 for sum_i f_i(feature_i), by least squares.

    Each feature gets its own dense hat-function basis, so f_i is effectively
    an arbitrary univariate function. This is the ceiling for ANY separable
    model -- a single-layer KAN, a GAM, SINDy on a dictionary with no cross
    terms -- and it does not move if you add knots, widen the network, or
    train longer. Returns one R^2 per target column.
    """
    blocks = []
    for j in range(features.shape[1]):
        v = features[:, j]
        knots = np.linspace(v.min(), v.max(), n_knots)
        h = knots[1] - knots[0]
        blocks.append(np.clip(1.0 - np.abs(v[:, None] - knots[None, :]) / h, 0.0, None))
    design = np.column_stack(blocks + [np.ones(len(features))])

    r2 = []
    for k in range(target.shape[1]):
        coef, *_ = np.linalg.lstsq(design, target[:, k], rcond=None)
        resid = target[:, k] - design @ coef
        ss_tot = np.sum((target[:, k] - target[:, k].mean()) ** 2)
        r2.append(float(1.0 - np.sum(resid ** 2) / ss_tot))
    return r2


ceiling_raw = best_additive_r2(X_disk, Xdot_disk)
ceiling_cycle = best_additive_r2(X_cycle, Xdot_cycle)

print("\n[CEILING] Best possible additive model, 60 knots per coordinate:")
print(f"  raw (x, y) over the disk    : R^2 = [{ceiling_raw[0]:.4f}, {ceiling_raw[1]:.4f}]"
      "   <- structural gap")
print(f"  raw (x, y) on the LIMIT CYCLE: R^2 = [{ceiling_cycle[0]:.4f}, {ceiling_cycle[1]:.4f}]"
      "   <- the trap: gap vanishes")


# ---------------------------------------------------------------------------
# 3. Lifts
# ---------------------------------------------------------------------------
def phi_np(X):
    X = np.atleast_2d(X)
    x, y = X[:, 0], X[:, 1]
    return np.column_stack([x, y, x * y ** 2, x ** 2 * y])


def phi_torch(X):
    x, y = X[:, 0], X[:, 1]
    return torch.stack([x, y, x * y ** 2, x ** 2 * y], dim=1)


def identity_np(X):
    return np.atleast_2d(X)


def identity_torch(X):
    return X


lift_full = CustomLift(fn=phi_np, output_dim=4, torch_fn=phi_torch, name="hopf_cross")
lift_none = CustomLift(fn=identity_np, output_dim=2, torch_fn=identity_torch, name="identity")

LIFT_NAMES = ["x", "y", "x*y^2", "x^2*y"]


# ---------------------------------------------------------------------------
# 4. Fit: (a) single-layer KAN on raw (x, y), (b) same with the lift
# ---------------------------------------------------------------------------
def r2_per_output(y_true, y_pred):
    out = []
    for k in range(y_true.shape[1]):
        ss_res = np.sum((y_true[:, k] - y_pred[:, k]) ** 2)
        ss_tot = np.sum((y_true[:, k] - y_true[:, k].mean()) ** 2)
        out.append(float(1.0 - ss_res / ss_tot))
    return out


# KANDy logs a line per LBFGS step to stdout regardless of verbose=, and
# PyKAN draws a tqdm bar on stderr. Between them they bury the comparison, so
# silence both streams around the fits; the printed metrics are the output.
@contextlib.contextmanager
def quiet():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


def fit_kandy(lift, label):
    model = KANDy(lift=lift, grid=GRID, k=3, steps=STEPS, seed=SEED,
                  base_fun="zero")
    with quiet():
        model.fit(X=X_disk, X_dot=Xdot_disk, val_frac=0.15, test_frac=0.15,
                  lamb=0.0, patience=50, verbose=False)
    r2_in = r2_per_output(Xdot_disk, model.predict(X_disk))
    r2_ex = r2_per_output(Xdot_out, model.predict(X_out))
    print(f"[FIT]   {label:<28} in-domain R^2 = [{r2_in[0]:.6f}, {r2_in[1]:.6f}]"
          f"   extrapolation R^2 = [{r2_ex[0]:.4f}, {r2_ex[1]:.4f}]")
    return model, r2_in, r2_ex


print("\n[FIT]   single-layer KAN (KANDy), identical budget for both lifts:")
model_raw, r2_raw, r2_raw_out = fit_kandy(lift_none, "(a) no lift, raw (x,y)")
model_lift, r2_lift, r2_lift_out = fit_kandy(lift_full, "(b) phi=[x,y,xy^2,x^2y]")
print(f"        (a) sits on the additive ceiling "
      f"[{ceiling_raw[0]:.6f}, {ceiling_raw[1]:.6f}] — as it must")


# ---------------------------------------------------------------------------
# 5. Fit: deeper PyKAN on raw (x, y) — the generous baseline
#
# Depth is what lets a KAN build a product at all, so a fair comparison has to
# give the un-lifted network real depth and width. These use the PyKAN default
# base_fun='silu' (not KANDy's 'zero'), which is strictly more expressive.
# ---------------------------------------------------------------------------
def fit_deep_kan(width, label):
    torch.manual_seed(SEED)
    dataset = {
        "train_input": torch.tensor(X_disk, dtype=torch.float32),
        "train_label": torch.tensor(Xdot_disk, dtype=torch.float32),
        "test_input": torch.tensor(X_out, dtype=torch.float32),
        "test_label": torch.tensor(Xdot_out, dtype=torch.float32),
    }
    with quiet():
        net = KAN(width=width, grid=GRID, k=3, seed=SEED, device="cpu",
                  auto_save=False)
        net.fit(dataset, opt="LBFGS", steps=STEPS, lamb=0.0)
    with torch.no_grad():
        pred_in = net(dataset["train_input"]).numpy()
        pred_out = net(dataset["test_input"]).numpy()
    r2_in = r2_per_output(Xdot_disk, pred_in)
    r2_ex = r2_per_output(Xdot_out, pred_out)
    print(f"[DEEP]  {label:<12} in-domain R^2 = [{r2_in[0]:.4f}, {r2_in[1]:.4f}]"
          f"   extrapolation R^2 = [{r2_ex[0]:.4f}, {r2_ex[1]:.4f}]")
    return net, r2_in, r2_ex


print("\n[FIT]   (c) deeper PyKAN on raw (x, y), base_fun='silu'")
deep2, r2_deep2, r2_deep2_out = fit_deep_kan([2, 5, 2], "[2,5,2]")
deep3, r2_deep3, r2_deep3_out = fit_deep_kan([2, 5, 5, 2], "[2,5,5,2]")


# ---------------------------------------------------------------------------
# 6. Read the lifted model's edges as polynomials
#
# A single-layer KAN output is exactly sum_i edge_ij(theta_i), so each edge can
# be fitted independently. get_formula() is the wrong tool here: PyKAN snaps
# one primitive under an affine composition c*f(a*x+b)+d, which cannot express
# mu*x - x^3 (same limitation as v - v^3/3 in fitzhugh_nagumo_example.py).
# Fitting each edge as a polynomial and summing recovers the RHS exactly.
# ---------------------------------------------------------------------------
theta_t = torch.tensor(phi_np(X_disk), dtype=torch.float32,
                       device=model_lift.device)
model_lift.model_.save_act = True
model_lift.model_(theta_t)

# True (linear, cubic) coefficient of each edge, u = the lifted coordinate.
TRUE_EDGE = {
    ("dx/dt", "x"): (MU, -1.0), ("dx/dt", "y"): (-OM, 0.0),
    ("dx/dt", "x*y^2"): (-1.0, 0.0), ("dx/dt", "x^2*y"): (0.0, 0.0),
    ("dy/dt", "x"): (OM, 0.0), ("dy/dt", "y"): (MU, -1.0),
    ("dy/dt", "x*y^2"): (0.0, 0.0), ("dy/dt", "x^2*y"): (-1.0, 0.0),
}

print("\n[EDGES] Lifted model, per-edge cubic fits   (learned | true):")
edge_fits = {}
for j, out_name in enumerate(["dx/dt", "dy/dt"]):
    for i, in_name in enumerate(LIFT_NAMES):
        xe, ye = get_edge_activation(model_lift.model_, 0, i, j)
        fit = fit_polynomial(xe, ye, degree=3)
        cubic, _, linear, _ = fit["params"]   # np.polyfit order, highest first
        t_lin, t_cub = TRUE_EDGE[(out_name, in_name)]
        # An edge that is identically zero carries no signal, so its R^2 is
        # a fit to numerical noise and means nothing. Report the amplitude.
        amp = float(np.ptp(ye))
        dead = amp < 1e-3
        r2_str = "  (dead edge, R^2 n/a)" if dead else f"  R^2={fit['r2']:.5f}"
        print(f"  {out_name:<6} <- {in_name:<7}: "
              f"{linear:+.4f}*u {cubic:+.4f}*u^3   |   "
              f"{t_lin:+.4f}*u {t_cub:+.4f}*u^3{r2_str}")
        edge_fits[(out_name, in_name)] = {
            "linear": float(linear), "cubic": float(cubic),
            "true_linear": t_lin, "true_cubic": t_cub,
            "amplitude": amp, "dead": dead,
            "r2": None if dead else float(fit["r2"]),
        }

# The two identifying numbers: the cross-term edges must be LINEAR with slope
# -1, and the self-edges must carry the cubic.
cross_dx = edge_fits[("dx/dt", "x*y^2")]["linear"]
cross_dy = edge_fits[("dy/dt", "x^2*y")]["linear"]
cubic_dx = edge_fits[("dx/dt", "x")]["cubic"]
cubic_dy = edge_fits[("dy/dt", "y")]["cubic"]

edge_max_err = max(max(abs(v["linear"] - v["true_linear"]),
                       abs(v["cubic"] - v["true_cubic"]))
                   for v in edge_fits.values())


def assemble_rhs(coef_key, tol=1e-3):
    """Sum the edges into a closed-form RHS.

    A single-layer KAN output is exactly sum_i edge_ij(theta_i), so the
    equation is just the concatenation of the per-edge fits -- no symbolic
    snapper needed, and no composition to unwind. This is the deliverable.
    """
    lines = []
    for out_name in ["dx/dt", "dy/dt"]:
        terms = []
        for in_name in LIFT_NAMES:
            v = edge_fits[(out_name, in_name)]
            u = in_name if in_name in ("x", "y") else f"({in_name})"
            for coef, power in ((v[coef_key + "linear"], u),
                                (v[coef_key + "cubic"], f"{u}^3")):
                if abs(coef) > tol:
                    terms.append(f"{coef:+.4f}*{power}")
        lines.append(f"  {out_name} = " + " ".join(terms))
    return "\n".join(lines)


print("\n[EQUATION] Recovered by summing the edges:")
print(assemble_rhs(""))
print("[EQUATION] Truth:")
print(assemble_rhs("true_"))
print(f"[EQUATION] max coefficient deviation: {edge_max_err:.2e}")


# ---------------------------------------------------------------------------
# 7. Rollout — where the additive model actually breaks
# ---------------------------------------------------------------------------
Y0 = np.array([0.25, 0.0])          # start well inside the cycle
T_END, DT_ROLL = 12.0, 0.005

ts, traj_true = rk4_integrate_numpy(rhs, Y0, (0.0, T_END), DT_ROLL)


def rollout(model):
    _, tr = rk4_integrate_numpy(lambda s: model.predict(s), Y0,
                                (0.0, T_END), DT_ROLL)
    return tr


def deep_rollout(net):
    def f(s):
        with torch.no_grad():
            return net(torch.tensor(s[None, :], dtype=torch.float32)).numpy()[0]
    _, tr = rk4_integrate_numpy(f, Y0, (0.0, T_END), DT_ROLL)
    return tr


traj_raw = rollout(model_raw)
traj_lift = rollout(model_lift)
traj_deep2 = deep_rollout(deep2)


def final_radius(tr):
    """Mean orbit radius over the last two time units (the saturated cycle)."""
    tail = tr[int(0.83 * len(tr)):]
    tail = tail[np.all(np.isfinite(tail), axis=1)]
    if len(tail) < 10:
        return float("nan")
    return float(np.mean(np.hypot(tail[:, 0], tail[:, 1])))


r_true = final_radius(traj_true)
r_raw = final_radius(traj_raw)
r_lift = final_radius(traj_lift)
r_deep2 = final_radius(traj_deep2)

print(f"\n[ROLLOUT] saturated radius (truth = sqrt(mu) = {np.sqrt(MU):.3f}):")
print(f"  truth            {r_true:.4f}")
print(f"  raw single-layer {r_raw:.4f}   ({100*abs(r_raw-r_true)/r_true:.1f}% error)")
print(f"  deep [2,5,2]     {r_deep2:.4f}   ({100*abs(r_deep2-r_true)/r_true:.1f}% error)")
print(f"  lifted           {r_lift:.4f}   ({100*abs(r_lift-r_true)/r_true:.1f}% error)")


# ---------------------------------------------------------------------------
# 8. Checks
# ---------------------------------------------------------------------------
TH = {
    "additive_ceiling_max": 0.99,      # a real structural gap must exist
    "cycle_ceiling_min": 0.999,        # ...and must vanish on the limit cycle
    "raw_at_ceiling_tol": 0.01,        # the un-lifted KAN must SIT on it
    "lift_r2_min": 0.9999,
    # Extrapolation is deliberately loose. A KAN edge is a B-spline, and no
    # spline extrapolates past its grid -- outside the training disk EVERY
    # model here degrades (lifted 0.911, deep [2,5,2] 0.825, deep [2,5,5,2]
    # 0.764 on the worse output). That is a property of the spline basis, not
    # of the lift, so this threshold is a regression guard, not evidence.
    "lift_extrap_r2_min": 0.85,
    "cross_slope_tol": 0.05,           # cross-term edges: slope -1
    "cubic_tol": 0.05,                 # self edges: cubic coefficient -1
    "lift_radius_rel_err_max": 0.02,
    "raw_radius_rel_err_min": 0.20,    # the additive model must visibly break
}

checks = [
    print_check(max(ceiling_raw) < TH["additive_ceiling_max"],
                f"additive ceiling off-attractor {max(ceiling_raw):.4f} "
                f"< {TH['additive_ceiling_max']} (obstruction is real)"),
    print_check(min(ceiling_cycle) > TH["cycle_ceiling_min"],
                f"additive ceiling ON the limit cycle {min(ceiling_cycle):.4f} "
                f"> {TH['cycle_ceiling_min']} (trap is real)"),
    print_check(min(r2_lift) > TH["lift_r2_min"],
                f"lifted in-domain R^2 {min(r2_lift):.6f} > {TH['lift_r2_min']}"),
    print_check(max(abs(a - b) for a, b in zip(r2_raw, ceiling_raw))
                < TH["raw_at_ceiling_tol"],
                f"un-lifted single-layer KAN sits on the additive ceiling "
                f"({min(r2_raw):.4f} vs {min(ceiling_raw):.4f})"),
    print_check(min(r2_lift_out) > TH["lift_extrap_r2_min"],
                f"lifted extrapolation R^2 {min(r2_lift_out):.4f} "
                f"> {TH['lift_extrap_r2_min']} (spline guard, see TH comment)"),
    print_check(abs(r_raw - r_true) / r_true > TH["raw_radius_rel_err_min"],
                f"un-lifted rollout radius error "
                f"{100*abs(r_raw-r_true)/r_true:.1f}% > "
                f"{100*TH['raw_radius_rel_err_min']:.0f}% "
                f"(R^2={min(r2_raw):.3f} hides a broken limit cycle)"),
    print_check(abs(cross_dx + 1.0) < TH["cross_slope_tol"]
                and abs(cross_dy + 1.0) < TH["cross_slope_tol"],
                f"cross-term edge slopes ({cross_dx:.3f}, {cross_dy:.3f}) "
                f"within {TH['cross_slope_tol']} of -1"),
    print_check(abs(cubic_dx + 1.0) < TH["cubic_tol"]
                and abs(cubic_dy + 1.0) < TH["cubic_tol"],
                f"self-edge cubic coefficients ({cubic_dx:.3f}, {cubic_dy:.3f}) "
                f"within {TH['cubic_tol']} of -1"),
    print_check(edge_max_err < TH["cubic_tol"],
                f"ALL 16 lifted edge coefficients within {TH['cubic_tol']} of "
                f"truth (max deviation {edge_max_err:.4f})"),
    print_check(abs(r_lift - r_true) / r_true < TH["lift_radius_rel_err_max"],
                f"lifted rollout radius error "
                f"{100*abs(r_lift-r_true)/r_true:.2f}% < "
                f"{100*TH['lift_radius_rel_err_max']:.0f}%"),
]
passed = all(checks)


# ---------------------------------------------------------------------------
# 9. Figures
# ---------------------------------------------------------------------------
def refresh_acts(model, X):
    """Repopulate model.acts from a full-batch forward pass.

    get_edge_activation reads whatever the LAST forward pass left behind, and
    the rollout above called predict() one state at a time -- so without this
    the edge plots would be drawn from a single sample.
    """
    model.model_.save_act = True
    model.model_(torch.tensor(model.lift(X), dtype=torch.float32,
                              device=model.device))


def plot_edge_comparison(path):
    """The headline figure: the SAME output, dx/dt, without the lift and with.

    Laid out so the missing edges are visible as missing. Without the lift
    dx/dt has exactly two edges to work with; with it, four -- and the two
    extra ones are precisely the mixed monomials.
    """
    refresh_acts(model_raw, X_disk)
    refresh_acts(model_lift, X_disk)

    fig, axes = plt.subplots(2, 4, figsize=(15.0, 6.8))
    analytic = {
        "x": lambda u: MU * u - u ** 3, "y": lambda u: -OM * u,
        "x*y^2": lambda u: -u, "x^2*y": lambda u: 0.0 * u,
    }

    # Row 0 -- raw (x, y). Two edges exist; the other two panels are the point.
    for i, name in enumerate(["x", "y"]):
        ax = axes[0, i]
        xe, ye = get_edge_activation(model_raw.model_, 0, i, 0)
        ax.plot(xe, ye - np.mean(ye), lw=2.2, color="#c44e52", label="KAN edge")
        ref = analytic[name](xe)
        ax.plot(xe, ref - np.mean(ref), "--", lw=1.4, color="k",
                label="analytic term")
        ax.set_title(f"dx/dt  $\\leftarrow$  {name}", fontsize=10)
        ax.set_xlabel(name)
        ax.axhline(0, color="0.7", lw=0.6)
    axes[0, 0].legend(fontsize=8, frameon=False)
    axes[0, 0].set_ylabel("no lift\nedge output")

    for c, missing in enumerate(["x\\cdot y^2", "x^2\\cdot y"], start=2):
        ax = axes[0, c]
        ax.text(0.5, 0.55, "no such edge", ha="center", va="center",
                fontsize=13, color="#c44e52", transform=ax.transAxes)
        ax.text(0.5, 0.36, f"${missing}$ is not an input,\nand a separable KAN\n"
                           "cannot manufacture it",
                ha="center", va="center", fontsize=9, color="0.35",
                transform=ax.transAxes)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linestyle(":")
            s.set_color("#c44e52")

    # Row 1 -- lifted: every edge matches its analytic term.
    amps = []
    for i, name in enumerate(LIFT_NAMES):
        ax = axes[1, i]
        xe, ye = get_edge_activation(model_lift.model_, 0, i, 0)
        yc = ye - np.mean(ye)
        amps.append(np.ptp(yc))
        ax.plot(xe, yc, lw=2.2, color="#4c72b0", label="KAN edge")
        ref = analytic[name](xe)
        ax.plot(xe, ref - np.mean(ref), "--", lw=1.4, color="k",
                label="analytic term")
        fitted = edge_fits[("dx/dt", name)]
        tag = ("dead edge" if fitted["dead"] else
               f"{fitted['linear']:+.3f}u {fitted['cubic']:+.3f}u$^3$")
        ax.set_title(f"dx/dt  $\\leftarrow$  {name}\n{tag}", fontsize=10)
        ax.set_xlabel(name)
        ax.axhline(0, color="0.7", lw=0.6)
    # Shared y-limits so the zero edge reads as zero, not as magnified noise.
    span = 0.6 * max(amps)
    for ax in axes[1]:
        ax.set_ylim(-span, span)
    axes[1, 0].set_ylabel("with lift\nedge output")
    axes[1, 0].legend(fontsize=8, frameon=False)

    fig.suptitle(
        "Hopf normal form: two inputs, two outputs, two different mixed "
        "monomials ($xy^2$ into $\\dot{x}$, $x^2y$ into $\\dot{y}$)\n"
        "top — no lift: two edges, bent away from the true terms to absorb "
        f"what they cannot represent. No equation to read (R$^2$ capped at "
        f"{max(ceiling_raw):.3f}, limit cycle collapses to r={r_raw:.3f})\n"
        "bottom — same network on $\\phi=[x,y,xy^2,x^2y]$: every edge lands on "
        f"its analytic term, coefficients exact to {edge_max_err:.0e}",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_deep_edges(path):
    """The depth-2 KAN's first layer: expressive, but nothing is nameable."""
    deep2.save_act = True
    deep2(torch.tensor(X_disk, dtype=torch.float32))
    n_hidden = 5
    fig, axes = plt.subplots(2, n_hidden, figsize=(3.0 * n_hidden, 5.0))
    for i, name in enumerate(["x", "y"]):
        for j in range(n_hidden):
            ax = axes[i, j]
            xe, ye = get_edge_activation(deep2, 0, i, j)
            ax.plot(xe, ye, lw=1.8, color="#55a868")
            ax.set_title(f"$h_{j}$ $\\leftarrow$ {name}", fontsize=9)
            ax.axhline(0, color="0.7", lw=0.6)
            ax.set_xlabel(name)
    fig.suptitle(
        "\"Just add a layer\": depth-2 KAN [2,5,2] on raw (x, y), layer-0 edges. "
        f"It fits well (R$^2$={min(r2_deep2):.5f}) — and returns NO EQUATION.\n"
        "Each product is built by composition and smeared across the hidden "
        "layer: ten smooth curves, not one of which names a term of the RHS "
        "or carries a coefficient you could read off.",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_fields(path):
    """Where the additive model's error lives, in the plane."""
    g = np.linspace(-R_TRAIN, R_TRAIN, 220)
    GX, GY = np.meshgrid(g, g)
    mask = GX ** 2 + GY ** 2 <= R_TRAIN ** 2
    grid = np.column_stack([GX.ravel(), GY.ravel()])

    true_f = rhs(grid)[:, 1].reshape(GX.shape)
    preds = {
        "no lift (single layer)": model_raw.predict(grid)[:, 1].reshape(GX.shape),
        "deep KAN [2,5,2]": deep2(torch.tensor(grid, dtype=torch.float32)
                                  ).detach().numpy()[:, 1].reshape(GX.shape),
        "lift $[x,y,xy^2,x^2y]$": model_lift.predict(grid)[:, 1].reshape(GX.shape),
    }

    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.4))
    vmax = np.nanmax(np.abs(np.where(mask, true_f, np.nan)))
    axes[0, 0].pcolormesh(GX, GY, np.where(mask, true_f, np.nan),
                          cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    axes[0, 0].set_title("truth:  dy/dt")
    axes[1, 0].axis("off")

    emax = 0.0
    errs = {}
    for name, p in preds.items():
        errs[name] = np.where(mask, p - true_f, np.nan)
        emax = max(emax, np.nanmax(np.abs(errs[name])))

    for c, (name, p) in enumerate(preds.items(), start=1):
        axes[0, c].pcolormesh(GX, GY, np.where(mask, p, np.nan), cmap="RdBu_r",
                              vmin=-vmax, vmax=vmax, shading="auto")
        axes[0, c].set_title(name, fontsize=10)
        im = axes[1, c].pcolormesh(GX, GY, errs[name], cmap="RdBu_r",
                                   vmin=-emax, vmax=emax, shading="auto")
        rmse = float(np.sqrt(np.nanmean(errs[name] ** 2)))
        axes[1, c].set_title(f"error (RMSE {rmse:.3f})", fontsize=10)
        fig.colorbar(im, ax=axes[1, c], fraction=0.046)

    for ax in axes.ravel():
        if ax.has_data():
            ax.set_aspect("equal")
            circ = plt.Circle((0, 0), np.sqrt(MU), fill=False, color="k",
                              lw=1.0, ls=":")
            ax.add_patch(circ)

    fig.suptitle(
        "dy/dt over the training disk. The un-lifted error has the quadrant "
        "structure of the missing $x^2y$ term;\nthe dotted circle is the limit "
        "cycle, where that error nearly vanishes -- which is why "
        "on-attractor training hides it.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_rollout(path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.0))
    series = [
        (traj_true, "truth", "k", 2.0),
        (traj_raw, f"no lift (r={r_raw:.3f})", "#c44e52", 1.4),
        (traj_deep2, f"deep [2,5,2] (r={r_deep2:.3f})", "#55a868", 1.4),
        (traj_lift, f"lifted (r={r_lift:.3f})", "#4c72b0", 1.4),
    ]
    for tr, label, color, lw in series:
        good = np.all(np.isfinite(tr), axis=1)
        ax1.plot(tr[good, 0], tr[good, 1], color=color, lw=lw, label=label,
                 alpha=0.9)
        ax2.plot(ts[good], np.hypot(tr[good, 0], tr[good, 1]), color=color,
                 lw=lw, label=label)
    ax1.set_xlabel("x"), ax1.set_ylabel("y"), ax1.set_aspect("equal")
    ax1.set_title("phase plane"), ax1.legend(fontsize=8, frameon=False)
    ax2.set_xlabel("t"), ax2.set_ylabel("radius")
    ax2.axhline(np.sqrt(MU), color="0.6", ls=":", lw=1.0)
    ax2.set_title("approach to the limit cycle")
    fig.suptitle("Rollout from (0.25, 0) -- a trajectory that crosses the "
                 "region the additive model cannot represent", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_kan_diagram(model, in_vars, path, title):
    """PyKAN's own model.plot() diagram, straight from the library.

    Edge opacity is tanh(beta * |activation|), so an edge that carries nothing
    fades out. That is what makes this the visual confirmation: with the lift,
    x*y^2 feeds dx/dt and NOTHING else, x^2*y feeds dy/dt and nothing else --
    the routing of the two cross terms is visible as two live edges and two
    faded ones. Without the lift there are only two inputs, so there is no
    edge that could carry a cross term at all.

    Called directly rather than via kandy.plotting.plot_kan_architecture
    because that wrapper does not expose `folder`, and model.plot() writes a
    PNG per spline as a side effect.
    """
    net = model.model_
    net.save_act = True
    with torch.no_grad():
        net(torch.tensor(model.lift(X_disk), dtype=torch.float32,
                         device=model.device))
    plt.figure()
    net.plot(folder=str(OUTDIR / "_splines"), beta=6.0, in_vars=in_vars,
             out_vars=["$\\dot{x}$", "$\\dot{y}$"], title=title)
    fig = plt.gcf()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_diagram_comparison(path, left, right):
    """The two model.plot() diagrams side by side.

    Note what the left panel does NOT show: it looks like a perfectly healthy
    fitted network. The obstruction is not a broken-looking diagram, it is the
    absence of any node that could carry a cross term -- which is exactly why
    it goes unnoticed. The confirmation is the comparison, not either panel.
    """
    n_live = sum(1 for v in edge_fits.values() if not v["dead"])
    n_total = len(edge_fits)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.0))
    for ax, img, cap in (
        (axes[0], left,
         "no lift — inputs are $x$ and $y$, all 4 edges live.\n"
         "The diagram looks healthy and gives no warning; yet R$^2$ is capped "
         f"at {max(ceiling_raw):.3f}\nand the limit cycle collapses to "
         f"r={r_raw:.3f}. There is simply no node that could carry $xy^2$."),
        (axes[1], right,
         "with lift — $xy^2$ and $x^2y$ enter as their own input nodes.\n"
         f"Only {n_live} of {n_total} possible edges survive: each cross term "
         "feeds exactly ONE output,\nwhich is the true structure of the "
         "equations. The routing is the answer."),
    ):
        ax.imshow(plt.imread(img))
        ax.axis("off")
        ax.set_title(cap, fontsize=9.5)
    fig.suptitle("PyKAN model.plot(): the cross-term obstruction, "
                 "as the network sees it", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=160)
    plt.close(fig)


use_pub_style()
diag_raw = plot_kan_diagram(
    model_raw, ["$x$", "$y$"],
    OUTDIR / "hopf_cross_term_kanplot_nolift.png", "no lift")
diag_lift = plot_kan_diagram(
    model_lift, ["$x$", "$y$", "$xy^2$", "$x^2y$"],
    OUTDIR / "hopf_cross_term_kanplot_lift.png", "with lift")
plot_diagram_comparison(OUTDIR / "hopf_cross_term_kanplot.png",
                        diag_raw, diag_lift)

plot_edge_comparison(OUTDIR / "hopf_cross_term_edges.png")
plot_deep_edges(OUTDIR / "hopf_cross_term_deep_edges.png")
plot_fields(OUTDIR / "hopf_cross_term_fields.png")
plot_rollout(OUTDIR / "hopf_cross_term_rollout.png")

# model.plot() writes one PNG per spline as a side effect; the composed
# figures above are what matter, so don't leave the scratch behind.
shutil.rmtree(OUTDIR / "_splines", ignore_errors=True)
print(f"\n[PLOTS] Saved {len(list(OUTDIR.glob('*.png')))} figures to {OUTDIR}")

save_metrics(OUTDIR / "metrics.json", {
    "task": "HopfCrossTerm",
    "category": "odes",
    "seed": SEED,
    "mu": MU, "omega": OM,
    "n_train": N_TRAIN, "r_train": R_TRAIN, "r_extrap": R_EXTRAP,
    "additive_ceiling_disk": ceiling_raw,
    "additive_ceiling_limit_cycle": ceiling_cycle,
    "r2_raw_singlelayer": r2_raw,
    "r2_raw_singlelayer_extrap": r2_raw_out,
    "r2_deep_2_5_2": r2_deep2,
    "r2_deep_2_5_2_extrap": r2_deep2_out,
    "r2_deep_2_5_5_2": r2_deep3,
    "r2_deep_2_5_5_2_extrap": r2_deep3_out,
    "r2_lifted": r2_lift,
    "r2_lifted_extrap": r2_lift_out,
    "edge_fits": {f"{k[0]}<-{k[1]}": v for k, v in edge_fits.items()},
    "equation_recovered": assemble_rhs(""),
    "equation_true": assemble_rhs("true_"),
    "edge_max_coefficient_error": edge_max_err,
    "radius_true": r_true,
    "radius_raw": r_raw,
    "radius_deep_2_5_2": r_deep2,
    "radius_lifted": r_lift,
    "thresholds": TH,
    "passed": passed,
})
