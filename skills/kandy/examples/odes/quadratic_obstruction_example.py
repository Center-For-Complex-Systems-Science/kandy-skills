#!/usr/bin/env python3
"""KANDy example: the quadratic obstruction — why hidden width 2 is not enough.

The simplest nonlinear holomorphic vector field, z_dot = z^2 with z = x + iy:

    dx/dt = x^2 - y^2
    dy/dt = 2*x*y

THE POINT: one product is cheap, two INDEPENDENT quadratics are not
--------------------------------------------------------------------
The KAN paper shows a [2,2,1] network representing f(x,y) = x*y, built from
the identity x*y = ((x+y)^2 - (x-y)^2)/4.  Both hidden units are consumed to
manufacture that single product.  The natural next question for dynamics is
what happens when the output is 2-dimensional, because then ONE shared hidden
layer has to serve both equations at once.

The answer depends entirely on what the second output needs, and the obvious
test cases give the wrong impression:

    target            best [2,2,2] R^2      why
    ----------------------------------------------------------------------
    (x*y, x)              1.000000          FREE. With h1 = x+y, h2 = x-y
                                            you already have x = (h1+h2)/2.
                                            A LINEAR second output costs
                                            nothing -- do not use this as a
                                            demonstration, it succeeds.
    (x*y, x^2)            0.999835          APPROXIMABLE. Exact representation
                                            needs one direction to degenerate
                                            while its coefficient blows up, so
                                            the span is only reached in a
                                            limit (border rank). Numerically
                                            there is no clean failure.
    (x*y, x^2 - y^2)      ~0.87-0.91        OBSTRUCTED, and this is the one.
                                            Width 3 jumps to 0.99999.

WHY: A SPAN OF SQUARES ALWAYS CONTAINS A SQUARE
-----------------------------------------------
With linear inner activations and quadratic outer ones -- the regime the KAN
paper's x*y construction lives in -- a [2,w,2] network computes

    h_j = a_j . v          (v = (x,y),  j = 1..w)
    z_k = sum_j c_kj h_j^2 + (linear)

so the quadratic parts of BOTH outputs lie in one shared span

    S = span{ (a_1.v)^2, ..., (a_w.v)^2 }

Every spanning element is a perfect square, i.e. a rank-one quadratic form.
Now write the required space for z_dot = z^2:

    V = span{ x^2 - y^2 ,  2*x*y }

and take a general element a*(x^2 - y^2) + b*(2*x*y).  As a symmetric matrix
that is [[a, b], [b, -a]], with

    det = -(a^2 + b^2)   <   0   for every (a, b) != (0, 0)

A perfect square has det = 0.  So V, which is exactly the space of HARMONIC
(trace-free) quadratic forms, contains no perfect square at all -- while any
S built from w = 2 squares is spanned by two of them.  V cannot fit inside S,
and unlike the x^2 case it is not even a limit of such spans: scanning every
pair of directions leaves at least 16.7% of V unrepresented.  Hence w >= 3,
and w = 3 suffices: a_1 = x, a_2 = y, a_3 = x+y spans all of {x^2, x*y, y^2}.

This is the 2-input shadow of the Lorenz situation.  There V = span{x*y, x*z},
whose general element x*(b*y + c*z) is again never a perfect square, so one
shared hidden layer must be strictly wider than the number of products -- and
the width needed grows with the number of DISTINCT products the equations
demand, not with the number of state variables.

TWO TIERS OF EVIDENCE -- DO NOT CONFLATE THEM
---------------------------------------------
The proof above holds only in the QUADRATIC CATEGORY (quadratic outer
activations).  A real KAN edge is a spline, which is strictly more expressive,
so the proved bound does NOT predict what a trained network scores.  Measured
here, the two tiers are far apart at width 2:

    proved bound, quadratic activations   R^2 <= 0.4862   (deterministic)
    trained PyKAN with splines            R^2  ~ 0.87-0.91 across seeds/runs
    trained PyKAN, width 3                R^2  ~ 0.99999

Splines nearly double the bound and STILL fail.  That is the useful reading:
the obstruction is proved in the quadratic category and survives empirically
when the activations are relaxed to arbitrary splines -- two independent
results pointing the same way, not one predicting the other.

Quote the spline numbers as a RANGE.  The bound is deterministic, but the
trained values are not reproducible at fixed seed even with threads pinned
(best-seed width 2 measured 0.912 / 0.907 / 0.902 on three runs of this
script).  The gap to width 3 is ~0.1 and the check threshold is 0.97, so the
conclusion is far outside that noise -- but do not cite a single decimal
figure as if it were derived.

WHAT THIS SCRIPT DOES ABOUT IT
------------------------------
1. Computes the obstruction with no network involved: scan all width-2
   direction pairs and report the best achievable R^2 in the quadratic
   category.  This is a bound, not a training outcome.
2. Trains real PyKAN networks at widths 2..5, several seeds each, and reports
   the measured curve ALONGSIDE that bound -- see the gap above.
3. Runs the (x*y, x) control at width 2, which SUCCEEDS -- proving width 2 is
   not merely underpowered.  It is this particular pair of quadratics that it
   cannot serve.
4. Fits KANDy with phi = [x^2 - y^2, x*y] and recovers the equations exactly
   at DEPTH ONE.  The lift does not make the network deeper; it removes the
   need for depth, because the products are supplied instead of manufactured.

Lift  phi: R^2 -> R^2   theta = [x^2 - y^2, x*y]
KAN:  width = [2, 2],  base_fun='zero'
"""

import contextlib
import io
import shutil
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

torch.set_num_threads(1)

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from _utils import ensure_outdir, print_check, save_metrics

from kan import KAN
from kandy import KANDy, CustomLift
from kandy.plotting import fit_polynomial, get_edge_activation, use_pub_style

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

N_TRAIN, N_TEST = 3000, 1200
BOX = 1.0                      # sample (x, y) in [-BOX, BOX]^2
WIDTHS = [2, 3, 4, 5]
N_SEEDS = 3
STEPS = 250
GRID = 10
GRID_RANGE = [-2.2, 2.2]       # must cover hidden pre-activations, not just x

OUTDIR = ensure_outdir(EXAMPLE_ROOT / "results" / "QuadraticObstruction")


@contextlib.contextmanager
def quiet():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


# ---------------------------------------------------------------------------
# 1. Data — the vector field of z_dot = z^2
# ---------------------------------------------------------------------------
def rhs(state):
    s = np.atleast_2d(np.asarray(state, dtype=float))
    x, y = s[:, 0], s[:, 1]
    out = np.column_stack([x ** 2 - y ** 2, 2 * x * y])
    return out[0] if np.ndim(state) == 1 else out


def control_rhs(state):
    """The (x*y, x) control target -- representable at width 2."""
    s = np.atleast_2d(np.asarray(state, dtype=float))
    x, y = s[:, 0], s[:, 1]
    return np.column_stack([x * y, x])


rng = np.random.default_rng(SEED)
X = rng.uniform(-BOX, BOX, (N_TRAIN, 2))
X_te = rng.uniform(-BOX, BOX, (N_TEST, 2))
Y, Y_te = rhs(X), rhs(X_te)

print(f"[DATA]  {N_TRAIN} samples in [-{BOX}, {BOX}]^2, target dz/dt = z^2")


def r2_per_output(y_true, y_pred):
    return [float(1 - np.sum((y_true[:, k] - y_pred[:, k]) ** 2) /
                  np.sum((y_true[:, k] - y_true[:, k].mean()) ** 2))
            for k in range(y_true.shape[1])]


# ---------------------------------------------------------------------------
# 2. The obstruction, with no network involved
#
# Best possible [2,w,2] model in the quadratic category: pick w directions,
# then solve exactly for the output coefficients. Only the directions are
# non-linear parameters, so a scan over them gives a genuine bound.
# ---------------------------------------------------------------------------
def best_quadratic_model_r2(width, target_fn, n_starts=40):
    Yt = target_fn(X_te)

    def design(thetas, pts):
        cols = [(np.cos(t) * pts[:, 0] + np.sin(t) * pts[:, 1]) ** 2
                for t in thetas]
        cols += [pts[:, 0], pts[:, 1], np.ones(len(pts))]   # free linear part
        return np.column_stack(cols)

    def sse(thetas):
        A, B = design(thetas, X), design(thetas, X_te)
        tot = 0.0
        for k in range(2):
            coef, *_ = np.linalg.lstsq(A, target_fn(X)[:, k], rcond=None)
            tot += np.sum((Yt[:, k] - B @ coef) ** 2)
        return tot

    starts = np.random.default_rng(SEED).uniform(0, np.pi, (n_starts, width))
    best = min((minimize(sse, s, method="Nelder-Mead",
                         options={"xatol": 1e-10, "fatol": 1e-12,
                                  "maxiter": 8000}) for s in starts),
               key=lambda r: r.fun)
    A, B = design(best.x, X), design(best.x, X_te)
    pred = np.column_stack([B @ np.linalg.lstsq(A, target_fn(X)[:, k],
                                                rcond=None)[0]
                            for k in range(2)])
    return min(r2_per_output(Yt, pred)), best.x


print("\n[BOUND] Best achievable R^2 with quadratic activations (no training):")
bound = {}
for w in WIDTHS:
    bound[w], _ = best_quadratic_model_r2(w, rhs)
    print(f"  width {w}: R^2 <= {bound[w]:.6f}"
          + ("   <- obstructed" if bound[w] < 0.99 else ""))
bound_ctrl, _ = best_quadratic_model_r2(2, control_rhs)
print(f"  width 2 on the (x*y, x) control: R^2 <= {bound_ctrl:.6f}"
      "   <- no obstruction")


# ---------------------------------------------------------------------------
# 3. Train real PyKAN networks — does the measured curve match the bound?
# ---------------------------------------------------------------------------
def train_kan(width, target_fn, seed):
    Yt, Yv = target_fn(X), target_fn(X_te)
    ds = {"train_input": torch.tensor(X, dtype=torch.float32),
          "train_label": torch.tensor(Yt, dtype=torch.float32),
          "test_input": torch.tensor(X_te, dtype=torch.float32),
          "test_label": torch.tensor(Yv, dtype=torch.float32)}
    with quiet():
        net = KAN(width=[2, width, 2], grid=GRID, k=3, seed=seed, device="cpu",
                  auto_save=False, grid_range=GRID_RANGE)
        # update_grid=True crashes PyKAN's curve2coef when a hidden unit
        # collapses (it swallows the lstsq error and returns an unbound
        # variable) -- which is exactly what the obstructed width-2 fit does.
        net.fit(ds, opt="LBFGS", steps=STEPS, lamb=0.0, update_grid=False)
        with torch.no_grad():
            pred = net(ds["test_input"]).numpy()
    return min(r2_per_output(Yv, pred)), net


print("\n[TRAIN] PyKAN [2, w, 2] on dz/dt = z^2:")
measured, nets = {}, {}
for w in WIDTHS:
    scores = []
    for s in range(N_SEEDS):
        sc, net = train_kan(w, rhs, s)
        scores.append(sc)
        if s == 0:
            nets[w] = net
    measured[w] = scores
    print(f"  width {w}: R^2 = {max(scores):.6f} best, "
          f"{min(scores):.6f} worst   (bound {bound[w]:.6f})")

ctrl_scores = [train_kan(2, control_rhs, s)[0] for s in range(N_SEEDS)]
print(f"  width 2 on the (x*y, x) control: R^2 = {max(ctrl_scores):.6f} best "
      f"-> width 2 is NOT simply underpowered")


# ---------------------------------------------------------------------------
# 4. KANDy: supply the products instead of manufacturing them
# ---------------------------------------------------------------------------
def phi_np(Xa):
    Xa = np.atleast_2d(Xa)
    x, y = Xa[:, 0], Xa[:, 1]
    return np.column_stack([x ** 2 - y ** 2, x * y])


def phi_torch(Xa):
    x, y = Xa[:, 0], Xa[:, 1]
    return torch.stack([x ** 2 - y ** 2, x * y], dim=1)


lift = CustomLift(fn=phi_np, output_dim=2, torch_fn=phi_torch, name="harmonic")
LIFT_NAMES = ["x^2-y^2", "x*y"]

model = KANDy(lift=lift, grid=5, k=3, steps=200, seed=SEED, base_fun="zero")
with quiet():
    model.fit(X=X, X_dot=Y, val_frac=0.15, test_frac=0.15, lamb=0.0,
              patience=50, verbose=False)
r2_kandy = r2_per_output(Y_te, model.predict(X_te))
print(f"\n[KANDY] depth-1 with phi = [x^2-y^2, x*y]: R^2 = "
      f"[{r2_kandy[0]:.6f}, {r2_kandy[1]:.6f}]")

model.model_.save_act = True
model.model_(torch.tensor(phi_np(X), dtype=torch.float32, device=model.device))

TRUE_EDGE = {("dx/dt", "x^2-y^2"): 1.0, ("dx/dt", "x*y"): 0.0,
             ("dy/dt", "x^2-y^2"): 0.0, ("dy/dt", "x*y"): 2.0}
edge_fits, terms = {}, {"dx/dt": [], "dy/dt": []}
for j, out_name in enumerate(["dx/dt", "dy/dt"]):
    for i, in_name in enumerate(LIFT_NAMES):
        xe, ye = get_edge_activation(model.model_, 0, i, j)
        slope = float(fit_polynomial(xe, ye, degree=1)["params"][0])
        edge_fits[f"{out_name}<-{in_name}"] = {
            "slope": slope, "true_slope": TRUE_EDGE[(out_name, in_name)]}
        if abs(slope) > 1e-3:
            terms[out_name].append(f"{slope:+.4f}*({in_name})")
edge_err = max(abs(v["slope"] - v["true_slope"]) for v in edge_fits.values())

print("[EQUATION] Recovered at depth 1:")
for k, v in terms.items():
    print(f"  {k} = " + " ".join(v))
print("[EQUATION] Truth:\n  dx/dt = +1.0000*(x^2-y^2)\n  dy/dt = +2.0000*(x*y)")
print(f"[EQUATION] max coefficient deviation: {edge_err:.2e}")


# ---------------------------------------------------------------------------
# 5. Checks
# ---------------------------------------------------------------------------
TH = {"bound_w2_max": 0.97, "bound_w3_min": 0.999, "trained_w2_max": 0.97,
      "trained_w3_min": 0.999, "control_min": 0.999, "kandy_min": 0.9999,
      "edge_tol": 0.01}

checks = [
    print_check(bound[2] < TH["bound_w2_max"],
                f"width-2 bound {bound[2]:.4f} < {TH['bound_w2_max']} "
                "(obstruction provable without training)"),
    print_check(bound[3] > TH["bound_w3_min"],
                f"width-3 bound {bound[3]:.6f} > {TH['bound_w3_min']} "
                "(three squares suffice)"),
    print_check(max(measured[2]) < TH["trained_w2_max"],
                f"trained width-2 R^2 {max(measured[2]):.4f} < "
                f"{TH['trained_w2_max']} over {N_SEEDS} seeds"),
    print_check(max(measured[3]) > TH["trained_w3_min"],
                f"trained width-3 R^2 {max(measured[3]):.6f} > "
                f"{TH['trained_w3_min']} (the jump is real)"),
    print_check(max(ctrl_scores) > TH["control_min"],
                f"(x*y, x) control at width 2 R^2 {max(ctrl_scores):.6f} > "
                f"{TH['control_min']} (width 2 is not simply underpowered)"),
    print_check(min(r2_kandy) > TH["kandy_min"],
                f"KANDy depth-1 with lift R^2 {min(r2_kandy):.6f} > "
                f"{TH['kandy_min']}"),
    print_check(edge_err < TH["edge_tol"],
                f"all 4 lifted edge slopes within {TH['edge_tol']} of truth "
                f"(max deviation {edge_err:.2e})"),
]
passed = all(checks)


# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
def plot_width_scan(path):
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(WIDTHS, [bound[w] for w in WIDTHS], "k--o", lw=1.6, ms=6,
            label="algebraic bound (quadratic activations, no training)")
    lo = [min(measured[w]) for w in WIDTHS]
    hi = [max(measured[w]) for w in WIDTHS]
    ax.fill_between(WIDTHS, lo, hi, color="#4c72b0", alpha=0.25,
                    label=f"trained PyKAN, {N_SEEDS}-seed spread")
    ax.plot(WIDTHS, hi, "-o", color="#4c72b0", lw=2.0, ms=6,
            label="trained PyKAN, best seed")
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.annotate(f"trained splines cap at {max(measured[2]):.3f}\n"
                f"({N_SEEDS} seeds: {min(measured[2]):.3f}-"
                f"{max(measured[2]):.3f})",
                xy=(2, max(measured[2])), xytext=(2.18, 0.86),
                fontsize=9, color="#4c72b0",
                arrowprops=dict(arrowstyle="->", color="#4c72b0", lw=1.2))
    ax.annotate(f"proved bound for quadratic\nactivations: {bound[2]:.3f}",
                xy=(2, bound[2]), xytext=(2.18, 0.56), fontsize=9, color="k",
                arrowprops=dict(arrowstyle="->", color="k", lw=1.2))
    ax.annotate("width 3: both reach 1.0 —\n$x^2$, $y^2$, $(x{+}y)^2$ span\n"
                "every quadratic form",
                xy=(3, max(measured[3])), xytext=(3.15, 0.80), fontsize=9,
                color="#55a868",
                arrowprops=dict(arrowstyle="->", color="#55a868", lw=1.2))
    ax.set_xticks(WIDTHS)
    ax.set_xlabel("hidden width $w$  in  KAN [2, $w$, 2]")
    ax.set_ylabel("$R^2$ (worse of the two outputs)")
    ax.set_ylim(0.40, 1.04)
    ax.set_title("The quadratic obstruction for $\\dot z = z^2$\n"
                 "proved for quadratic activations (black); measured for real "
                 "splines (blue).\nSplines beat the bound and still fail at "
                 "width 2 — both jump at width 3.", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def kan_diagram(net, path, title, in_vars):
    net.save_act = True
    with torch.no_grad():
        net(torch.tensor(X, dtype=torch.float32))
    plt.figure()
    net.plot(folder=str(OUTDIR / "_splines"), beta=6.0, in_vars=in_vars,
             out_vars=["$\\dot{x}$", "$\\dot{y}$"], title=title)
    fig = plt.gcf()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_diagrams(path, imgs):
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 6.4))
    caps = [
        f"[2,2,2] — OBSTRUCTED, R$^2$={max(measured[2]):.3f}\n"
        "topology looks healthy; the tell is the strained,\n"
        "non-quadratic edge shapes it bends into and fails anyway",
        f"[2,3,2] — works, R$^2$={max(measured[3]):.5f}\n"
        "three squares span every quadratic form,\n"
        "so both outputs can be served at once",
        "KANDy depth-1 on $\\phi=[x^2{-}y^2,\\,xy]$, R$^2$="
        f"{min(r2_kandy):.5f}\nno hidden layer at all: the products are\n"
        "supplied, so each edge is one named term",
    ]
    for ax, img, cap in zip(axes, imgs, caps):
        ax.imshow(plt.imread(img))
        ax.axis("off")
        ax.set_title(cap, fontsize=9.5)
    fig.suptitle("$\\dot z = z^2$: width 2 cannot serve two harmonic "
                 "quadratics — width 3 can, and a lift makes depth unnecessary",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_error_fields(path):
    g = np.linspace(-BOX, BOX, 200)
    GX, GY = np.meshgrid(g, g)
    grid = np.column_stack([GX.ravel(), GY.ravel()])
    truth = rhs(grid)[:, 0].reshape(GX.shape)

    with torch.no_grad():
        p2 = nets[2](torch.tensor(grid, dtype=torch.float32)
                     ).numpy()[:, 0].reshape(GX.shape)
        p3 = nets[3](torch.tensor(grid, dtype=torch.float32)
                     ).numpy()[:, 0].reshape(GX.shape)
    pk = model.predict(grid)[:, 0].reshape(GX.shape)

    fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.6))
    vmax = np.abs(truth).max()
    axes[0, 0].pcolormesh(GX, GY, truth, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          shading="auto")
    axes[0, 0].set_title("truth:  $\\dot x = x^2 - y^2$")
    axes[1, 0].axis("off")

    panels = [("[2,2,2]", p2), ("[2,3,2]", p3), ("KANDy + lift", pk)]
    emax = max(np.abs(p - truth).max() for _, p in panels)
    for c, (name, p) in enumerate(panels, start=1):
        axes[0, c].pcolormesh(GX, GY, p, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                              shading="auto")
        axes[0, c].set_title(name, fontsize=10)
        im = axes[1, c].pcolormesh(GX, GY, p - truth, cmap="RdBu_r",
                                   vmin=-emax, vmax=emax, shading="auto")
        axes[1, c].set_title(f"error (RMSE {np.sqrt(np.mean((p-truth)**2)):.4f})",
                             fontsize=10)
        fig.colorbar(im, ax=axes[1, c], fraction=0.046)
    for ax in axes.ravel():
        if ax.has_data():
            ax.set_aspect("equal")
    fig.suptitle("The width-2 failure is structural, not noise: the clean "
                 "saddle of $x^2-y^2$ is broken into vertical bands,\n"
                 "and the residual is smooth and organised rather than "
                 "scattered", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=160)
    plt.close(fig)


use_pub_style()
plot_width_scan(OUTDIR / "quadratic_obstruction_width_scan.png")
imgs = [
    kan_diagram(nets[2], OUTDIR / "_kanplot_w2.png", "[2,2,2]", ["$x$", "$y$"]),
    kan_diagram(nets[3], OUTDIR / "_kanplot_w3.png", "[2,3,2]", ["$x$", "$y$"]),
    kan_diagram(model.model_, OUTDIR / "_kanplot_lift.png", "KANDy + lift",
                ["$x^2{-}y^2$", "$xy$"]),
]
plot_diagrams(OUTDIR / "quadratic_obstruction_kanplot.png", imgs)
plot_error_fields(OUTDIR / "quadratic_obstruction_fields.png")
shutil.rmtree(OUTDIR / "_splines", ignore_errors=True)
for p in imgs:
    Path(p).unlink(missing_ok=True)
print(f"\n[PLOTS] Saved {len(list(OUTDIR.glob('*.png')))} figures to {OUTDIR}")

save_metrics(OUTDIR / "metrics.json", {
    "task": "QuadraticObstruction", "category": "odes", "seed": SEED,
    "n_train": N_TRAIN, "widths": WIDTHS, "n_seeds": N_SEEDS,
    "algebraic_bound": {str(w): bound[w] for w in WIDTHS},
    "algebraic_bound_control_w2": bound_ctrl,
    "trained_r2": {str(w): measured[w] for w in WIDTHS},
    "trained_control_w2": ctrl_scores,
    "r2_kandy_depth1": r2_kandy,
    "edge_fits": edge_fits,
    "edge_max_error": edge_err,
    "equation_recovered": {k: " ".join(v) for k, v in terms.items()},
    "thresholds": TH, "passed": passed,
})
