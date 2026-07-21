#!/usr/bin/env python3
"""KANDy example: DMDLift — letting the data choose the lift.

Every other example here hands KANDy a lift built from domain knowledge: you
know the cross-terms (`S*I`, `m^3h*V`, `u*w_x`) and you supply them.  This one
does the opposite.  `DMDLift` runs Extended DMD on the trajectory, extracts the
leading Koopman eigenfunctions, and uses THOSE as the lift — no physics input
beyond a generic polynomial dictionary.

The system (Brunton, Brunton, Proctor & Kutz) is the standard benchmark for
this, because the right answer is known in closed form:

    dx1/dt = mu * x1
    dx2/dt = lambda * (x2 - x1^2)                    mu = -0.05, lambda = -1

It has an EXACT finite Koopman invariant subspace spanned by the three
observables [x1, x2, x1^2], in which the nonlinear system is exactly linear:

    d/dt [x1, x2, x1^2] = [[mu, 0,  0     ],   [x1  ]
                           [0,  lam, -lam ],   [x2  ]
                           [0,  0,   2*mu ]] @ [x1^2]

so the Koopman eigenvalues are exactly {mu, 2*mu, lambda} and the
eigenfunctions are {x1, x1^2, x2 - c*x1^2} with c = lambda/(lambda - 2*mu).

That gives something rare in this repo: a validation target that is not an R^2.
We can check the recovered spectrum against numbers known analytically.

Why this matters for a separable KAN
------------------------------------
The core constraint the skill exists to teach is that a KAN is separable, so
cross-terms must be supplied.  Koopman coordinates are the one principled way
out: in eigenfunction coordinates the dynamics DIAGONALISE, each coordinate
evolving as d(phi_i)/dt = mu_i * phi_i with no coupling at all.  A separable
model is then exactly the right hypothesis class.

The consequence is that dx/dt is exactly linear in these coordinates — this
script verifies it with least squares, which hits R^2 = 1.000000 on both
equations.  So the discovery here is the COORDINATES, not the fit: the same
"a high R^2 can be cheap" lesson as `odes/mackey_glass_delay_example.py`,
arrived at from the opposite direction.  The price is interpretability: the
formula is clean, but written in phi_k, not in anything a biologist or engineer
would recognise.

Three traps, all measured below
-------------------------------
1. **`n_modes` must cover the dictionary, and the default ranking fights you.**
   `sort_by='magnitude'` keeps the most PERSISTENT modes.  Here the physically
   essential eigenvalue lambda = -1 is the FASTEST decaying one, so with
   `n_modes=3` it is ranked below a spurious mode and dropped entirely.
   Persistence is not importance.

2. **Never concatenate trajectories.**  `DMDLift.fit` forms consecutive pairs
   `(X[:-1], X[1:])` internally, so gluing several initial conditions together
   injects one fake transition per seam.  This script measures the damage: one
   trajectory recovers the spectrum exactly, twelve concatenated ones smear
   lambda = -1 into roughly -0.8.  There is no multi-trajectory API, so this
   fails silently.

3. **Fit the LIFT on the trajectory, but the KAN on a REGION.**  `DMDLift.fit`
   needs consecutive pairs, so it must see an ordered trajectory.  The
   regression does not: along a single decaying trajectory the phi coordinates
   are strongly collinear (cond ~ 4.5e2), and although least squares still
   recovers the exact answer, the KAN's edge-by-edge symbolic snap picks a bad
   decomposition — measured formula R^2 ~ 0.4 on dx2/dt.  Evaluating the fitted
   lift on box-sampled states instead drops the conditioning to ~5e1 and lifts
   formula R^2 to ~0.999.

   The wrinkle is that `KANDy.fit` calls `lift.fit(X)` on whatever X you hand
   it, so you cannot simply pass box samples to a `KANDy(lift=DMDLift(...))` —
   it would refit the DMD on data that is not a trajectory.  Decouple: fit the
   DMDLift yourself, evaluate it, and give KANDy the resulting features behind
   an identity `CustomLift`.  That is what section 4 does.

Where DMDLift stops working
---------------------------
A finite Koopman invariant subspace is a special property, not a given.  The
script ends by running the same procedure on the Van der Pol oscillator, whose
limit cycle admits no finite polynomial invariant subspace: the recovered
"eigenvalues" are then effective modes, not exact ones, and there is nothing to
check them against.  DMDLift degrades gracefully into a useful modal basis —
but the exactness above is a property of the system, not of the method.

Runtime: well under a minute.
KAN:  width = [5, 2],  base_fun='zero'
"""

import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from kandy import KANDy, CustomLift, DMDLift, PolynomialLift
from kandy.plotting import plot_all_edges, plot_loss_curves, use_pub_style

# ---------------------------------------------------------------------------
# 0. Reproducibility and system parameters
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

MU, LAMBDA = -0.05, -1.0
C_MANIFOLD = LAMBDA / (LAMBDA - 2 * MU)        # slow-manifold coefficient
DT = 0.05
T_END = 40.0
TRUE_EIGS = np.array([MU, 2 * MU, LAMBDA])


def exact_trajectory(x0, t_end=T_END, dt=DT):
    """Closed-form solution — no integrator error to muddy the spectrum."""
    t = np.arange(0, t_end, dt)
    x1 = x0[0] * np.exp(MU * t)
    x2 = (x0[1] - C_MANIFOLD * x0[0] ** 2) * np.exp(LAMBDA * t) \
        + C_MANIFOLD * x1 ** 2
    return t, np.stack([x1, x2], axis=1)


def exact_rhs(X):
    x1, x2 = X.T
    return np.stack([MU * x1, LAMBDA * (x2 - x1 ** 2)], axis=1)


t_grid, traj = exact_trajectory([1.2, -0.7])
X_dot = exact_rhs(traj)
print(f"[SYS]  mu={MU}, lambda={LAMBDA}, c=lambda/(lambda-2mu)={C_MANIFOLD:.4f}")
print(f"[SYS]  exact Koopman eigenvalues: {np.round(TRUE_EIGS, 4)}")
print(f"[DATA] one trajectory, {len(traj)} samples at dt={DT}")


def continuous_eigs(lift):
    """Flow-map eigenvalues -> generator eigenvalues:  mu = log(lambda)/dt.

    NOTE: `_evals` is private — the library exposes no public accessor for the
    recovered spectrum yet.
    """
    return np.log(lift._evals.astype(complex)) / DT


def match_error(est, true=TRUE_EIGS):
    """Largest distance from each TRUE eigenvalue to its nearest estimate."""
    return max(float(np.min(np.abs(est - t))) for t in true)


# ---------------------------------------------------------------------------
# 1. Trap 1 — n_modes too small drops the fastest (and essential) mode
# ---------------------------------------------------------------------------
print("\n[TRAP-1] sort_by='magnitude' keeps the most PERSISTENT modes, but")
print("[TRAP-1] lambda=-1 is the fastest-decaying eigenvalue here:")
for n_modes in (3, 5):
    lift = DMDLift(n_modes=n_modes, dictionary=PolynomialLift(degree=2))
    lift.fit(traj)
    est = continuous_eigs(lift)
    got_lambda = np.min(np.abs(est - LAMBDA)) < 1e-6
    print(f"[TRAP-1]   n_modes={n_modes}: {np.round(np.sort(est.real), 4)}   "
          f"lambda=-1 {'RECOVERED' if got_lambda else 'MISSING'}")
print("[TRAP-1] The dictionary has 5 observables, so ask for all 5 — the mode")
print("[TRAP-1] that defines the slow manifold is the least persistent one.")

# ---------------------------------------------------------------------------
# 2. Trap 2 — concatenating trajectories injects fake transitions
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
segments = [exact_trajectory(rng.uniform(-1.5, 1.5, 2), t_end=20.0)[1]
            for _ in range(12)]
glued = np.concatenate(segments)

lift_single = DMDLift(n_modes=5, dictionary=PolynomialLift(degree=2)).fit(traj)
lift_glued = DMDLift(n_modes=5, dictionary=PolynomialLift(degree=2)).fit(glued)
err_single = match_error(continuous_eigs(lift_single))
err_glued = match_error(continuous_eigs(lift_glued))
print(f"\n[TRAP-2] one trajectory ({len(traj)} pts)      max eigenvalue error "
      f"= {err_single:.2e}")
print(f"[TRAP-2] 12 glued trajectories ({len(glued)} pts) max eigenvalue error "
      f"= {err_glued:.2e}")
print(f"[TRAP-2] {len(segments) - 1} seams = {len(segments) - 1} fake transitions. "
      "More data, worse answer.")
print(f"[TRAP-2] glued spectrum: "
      f"{np.round(np.sort(continuous_eigs(lift_glued).real), 4)}")

# ---------------------------------------------------------------------------
# 3. The recovered spectrum, against the analytic answer
# ---------------------------------------------------------------------------
lift = lift_single
est = continuous_eigs(lift)
print(f"\n[SPEC] recovered (log(lambda)/dt): "
      f"{np.round(np.sort(est.real), 5)}")
print(f"[SPEC] exact Koopman eigenvalues  : {np.round(TRUE_EIGS, 5)}")
for tv in TRUE_EIGS:
    k = int(np.argmin(np.abs(est - tv)))
    print(f"[SPEC]   {tv:+.4f}  ->  {est[k].real:+.8f}   "
          f"(error {abs(est[k] - tv):.2e})")
print("[SPEC] The two extra modes are spurious — artefacts of a 5-dimensional")
print("[SPEC] dictionary describing a 3-dimensional invariant subspace.")
print(f"[SPEC] lift feature names: {lift.feature_names}")

# ---------------------------------------------------------------------------
# 4. Trap 3 — the lift needs a trajectory, the regression needs a region
# ---------------------------------------------------------------------------
feature_names = lift.feature_names
Theta_traj = lift(traj)

# Ground truth check: dx/dt really is linear in these coordinates.
design = np.column_stack([Theta_traj, np.ones(len(Theta_traj))])
lsq_r2 = []
for i in range(2):
    c, *_ = np.linalg.lstsq(design, X_dot[:, i], rcond=None)
    lsq_r2.append(1.0 - np.sum((X_dot[:, i] - design @ c) ** 2)
                  / np.sum((X_dot[:, i] - X_dot[:, i].mean()) ** 2))
print(f"\n[LINEAR] least squares on the phi coordinates: R^2 = "
      f"{np.round(lsq_r2, 6)}")
print("[LINEAR] So the coordinates are right — anything below this is the fit.")

# Box-sampled states over the region the trajectory explores.  The lift is
# ALREADY fitted; we only evaluate it here, so no trajectory ordering is needed.
span = np.ptp(traj, axis=0)
lo, hi = traj.min(axis=0) - 0.3 * span, traj.max(axis=0) + 0.3 * span
box = np.random.default_rng(SEED).uniform(lo, hi, size=(4000, 2))
Theta_box, X_dot_box = lift(box), exact_rhs(box)

print(f"[TRAP-3] cond(phi) along the trajectory = "
      f"{np.linalg.cond(design):.3g}")
print(f"[TRAP-3] cond(phi) over the box         = "
      f"{np.linalg.cond(np.column_stack([Theta_box, np.ones(len(Theta_box))])):.3g}")


def fit_kan(theta, targets):
    """KANDy on PRE-COMPUTED Koopman features (identity lift).

    Passing DMDLift directly would make KANDy.fit refit the DMD on whatever
    X it receives — wrong for box samples, which are not a trajectory.
    """
    m = KANDy(lift=CustomLift(fn=lambda Z: Z, output_dim=theta.shape[1],
                              name="koopman_coords"),
              grid=3, k=3, steps=200, seed=SEED, base_fun="zero")
    m.fit(X=theta, X_dot=targets, val_frac=0.15, test_frac=0.15, lamb=0.0,
          patience=60, verbose=False)
    return m


results = {}
for tag, th, yd in [("trajectory", Theta_traj, X_dot),
                    ("box", Theta_box, X_dot_box)]:
    m = fit_kan(th, yd)
    p = m.predict(th)
    net = [1.0 - np.sum((yd[:, i] - p[:, i]) ** 2)
           / np.sum((yd[:, i] - yd[:, i].mean()) ** 2) for i in range(2)]
    f = m.get_formula(var_names=feature_names, round_places=5, lib=["x", "0"],
                      r2_threshold=0.80, weight_simple=0.0)
    fr = m.score_formula(f, th, yd, var_names=feature_names)
    results[tag] = (m, net, f, fr)
    print(f"[TRAP-3] trained on {tag:10s}: network R^2 {np.round(net, 6)}   "
          f"formula R^2 {np.round(fr, 5)}")

print("[TRAP-3] Same lift, same exact coordinates — only the sampling differs.")
print("[TRAP-3] Collinear data does not stop least squares, but it does defeat")
print("[TRAP-3] edge-by-edge symbolic snapping.")

model, r2, formulas, formula_r2 = results["box"]
theta_t = torch.tensor(Theta_box[:2048], dtype=torch.float32)

# ---------------------------------------------------------------------------
# 5. Symbolic extraction — clean formulas, opaque coordinates
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] From the box-trained model:")
for i, expr in enumerate(formulas):
    print(f"  dx{i + 1}/dt = {expr}")
print(f"[SYMBOLIC] Formula R^2: {np.round(formula_r2, 6)}")
print("[SYMBOLIC] Every edge is linear, as the theory demands.  But the result")
print("[SYMBOLIC] is expressed in phi_k — correct, and unreadable as physics.")
print("[SYMBOLIC] Contrast mathbio/hodgkin_huxley_example.py, where the")
print("[SYMBOLIC] coefficients ARE the biophysics.  That is the trade.")

# ---------------------------------------------------------------------------
# 6. Where it stops working — Van der Pol has no finite invariant subspace
# ---------------------------------------------------------------------------
vdp = solve_ivp(lambda t, s: [s[1], 2.0 * (1 - s[0] ** 2) * s[1] - s[0]],
                [0, T_END], [2.0, 0.0], t_eval=t_grid,
                rtol=1e-10, atol=1e-12).y.T
lift_vdp = DMDLift(n_modes=5, dictionary=PolynomialLift(degree=2)).fit(vdp)
est_vdp = continuous_eigs(lift_vdp)
print(f"\n[LIMIT] Van der Pol spectrum: {np.round(np.sort_complex(est_vdp), 4)}")
print("[LIMIT] A limit cycle admits no finite polynomial Koopman invariant")
print("[LIMIT] subspace, so these are EFFECTIVE modes with nothing exact to")
print("[LIMIT] check them against.  DMDLift still gives a usable basis — but")
print("[LIMIT] the exactness above belongs to the system, not to the method.")
print("[LIMIT] For Van der Pol, odes/van_der_pol_rbf_example.py is the better")
print("[LIMIT] recipe: RadialBasisLift, validated by rollout.")

# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/KoopmanSlowManifold", exist_ok=True)

# 7a. Trajectory and the slow manifold x2 = c*x1^2
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
axes[0].plot(t_grid, traj[:, 0], lw=1.4, color="#1f77b4", label="$x_1$")
axes[0].plot(t_grid, traj[:, 1], lw=1.4, color="#d62728", label="$x_2$")
axes[0].set_xlabel("time"); axes[0].set_ylabel("state")
axes[0].set_title("Fast collapse onto the slow manifold")
axes[0].legend(fontsize=8)

grid_x = np.linspace(traj[:, 0].min(), traj[:, 0].max(), 200)
axes[1].plot(grid_x, C_MANIFOLD * grid_x ** 2, "k--", lw=1.3,
             label=fr"$x_2 = {C_MANIFOLD:.3f}\,x_1^2$")
axes[1].plot(traj[:, 0], traj[:, 1], lw=1.6, color="#1f77b4", label="trajectory")
axes[1].set_xlabel("$x_1$"); axes[1].set_ylabel("$x_2$")
axes[1].set_title("Slow manifold")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/KoopmanSlowManifold/trajectory.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)

# 7b. Recovered spectrum vs the analytic eigenvalues
fig, ax = plt.subplots(figsize=(6.2, 4))
ax.scatter(est.real, est.imag, s=90, facecolors="none", edgecolors="#1f77b4",
           lw=1.8, label="DMDLift, $\\log(\\lambda)/\\Delta t$")
ax.scatter(TRUE_EIGS, np.zeros_like(TRUE_EIGS), s=30, color="#d62728",
           marker="x", lw=2, label="exact Koopman eigenvalues")
for tv in TRUE_EIGS:
    ax.annotate(f"{tv:g}", (tv, 0), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=8, color="#d62728")
ax.axhline(0, color="k", lw=0.5, alpha=0.3)
ax.set_xlabel("Re"); ax.set_ylabel("Im")
ax.set_title("Recovered spectrum (2 spurious modes at left)")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig("results/KoopmanSlowManifold/spectrum.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)

# 7c. The concatenation trap
fig, ax = plt.subplots(figsize=(6.2, 4))
w = 0.35
idx = np.arange(3)
ax.bar(idx - w / 2, [float(np.min(np.abs(continuous_eigs(lift_single) - t)))
                     for t in TRUE_EIGS], w, color="#1f77b4",
       label="one trajectory")
ax.bar(idx + w / 2, [float(np.min(np.abs(continuous_eigs(lift_glued) - t)))
                     for t in TRUE_EIGS], w, color="#d62728",
       label="12 glued trajectories")
ax.set_yscale("log")
ax.set_xticks(idx)
ax.set_xticklabels([f"$\\mu$={MU}", f"$2\\mu$={2*MU}", f"$\\lambda$={LAMBDA}"])
ax.set_ylabel("eigenvalue error")
ax.set_title("Gluing trajectories injects fake transitions")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/KoopmanSlowManifold/concatenation_trap.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)

# 7d. Loss curves and edges
if getattr(model, "train_results_", None):
    fig, ax = plot_loss_curves(model.train_results_,
                               save="results/KoopmanSlowManifold/loss_curves")
    plt.close(fig)

fig, axes = plot_all_edges(model.model_, X=theta_t,
                           in_var_names=feature_names,
                           out_var_names=["dx1/dt", "dx2/dt"],
                           save="results/KoopmanSlowManifold/edge_activations")
plt.close(fig)

print("\n[FIGS]  Saved results/KoopmanSlowManifold/")
