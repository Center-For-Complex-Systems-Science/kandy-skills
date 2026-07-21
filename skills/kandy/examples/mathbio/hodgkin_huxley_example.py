#!/usr/bin/env python3
"""KANDy example: Hodgkin-Huxley — recovering biophysical constants.

The Hodgkin-Huxley model is the origin of computational neuroscience, and its
membrane-current balance is a rare discovery target: the coefficients are
MEASURED PHYSICAL CONSTANTS with published values, not parameters invented for
a benchmark.  Getting them back to a fraction of a percent is a real test.

    C dV/dt = I_ext - gNa*m^3*h*(V - ENa) - gK*n^4*(V - EK) - gL*(V - EL)
    dm/dt   = alpha_m(V)*(1 - m) - beta_m(V)*m          (and h, n likewise)

with C=1 uF/cm^2, gNa=120, gK=36, gL=0.3 mS/cm^2, ENa=50, EK=-77,
EL=-54.387 mV.

The lift for the voltage equation
---------------------------------
Expand the conductance terms and the right-hand side becomes exactly linear in
six features:

    theta = [V, m^3*h, m^3*h*V, n^4, n^4*V, 1]

    dV/dt = -(gL/C)*V + (gNa*ENa/C)*m^3h - (gNa/C)*m^3hV
            + (gK*EK/C)*n^4 - (gK/C)*n^4V + (I + gL*EL)/C

`m^3*h*V` is a product of THREE different states, which no separable KAN can
assemble from edges on m, h and V — the cross-term rule at its most extreme.
Once the products are supplied, `lib=["x", "0"]` reads the constants straight
off the edges.

Three things this example demonstrates
--------------------------------------

1. **The constants come back.**  gNa, gK, ENa and EK to ~0.01% or better;
   gL and EL to a few tenths of a percent.  The leak parameters are the
   loosest because the leak is the smallest current in the balance — the fit
   constrains what contributes to the data.  (LBFGS is not bit-reproducible
   here, so the last digits move between runs; the leak error has been seen
   up to ~0.7%.  The script prints what it actually got.)

2. **Box sampling beats trajectory sampling, and conditioning is why.**
   Sampling states along the spike train gives cond(Theta) ~ 2.2e4; sampling
   uniformly from a box over the observed ranges gives ~5.8e2, a 38x
   improvement, and about two orders of magnitude better coefficient accuracy
   through the network.  Least squares is exact either way on noiseless data —
   conditioning only bites once the fit goes through splines.  Same lesson as
   `mathbio/sir_example.py`.

3. **Use LBFGS here, not Adam.**  The targets span +-3000 and the true
   coefficients reach 6000.  Adam at lr=1e-2 for 500 steps does not merely
   underperform — it fails outright (measured R^2 below zero, with the m^3hV
   edge flat so gNa comes out ~0).  LBFGS converges in ~13 s.  This is the
   counterexample to "many features, reach for Adam": when the coefficients
   span four orders of magnitude, a first-order method never gets close.

The gating equations: a ceiling, not a wall
-------------------------------------------
dm/dt = alpha_m(V)*(1-m) - beta_m(V)*m = alpha_m(V) - (alpha_m+beta_m)(V)*m
is a product of an UNKNOWN function of V with the state m.  A separable KAN
computes sum_j psi_j(theta_j), which cannot represent alpha(V)*m exactly.

The interesting part is that it does not fail loudly.  With the naive lift
[V, m, V*m] the KAN reaches R^2 ~ 0.991 — good enough to look like success.
The tell is that REFINING THE GRID DOES NOT HELP: grid 5, 10, 20, 40 all land
within ~0.001 of each other, an 8x refinement buying nothing.  A resolution
problem improves with resolution; a structural one plateaus.  This script
measures that plateau, then shows that
supplying the products explicitly (an "oracle" lift built from the true rate
functions) makes the equation exactly linear and recoverable.

    Diagnostic worth remembering: if refining `grid` leaves R^2 flat, the lift
    is missing a term.  Do not spend more capacity — change the features.

Runtime: ~1-2 minutes.
KAN:  width = [6, 1],  base_fun='zero'
"""

import os
import time
import copy
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from kandy import KANDy, CustomLift
from kandy.plotting import plot_all_edges, plot_loss_curves, use_pub_style

# ---------------------------------------------------------------------------
# 0. Reproducibility and the classic squid-axon constants
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

C_M = 1.0
G_NA, G_K, G_L = 120.0, 36.0, 0.3
E_NA, E_K, E_L = 50.0, -77.0, -54.387
I_EXT = 10.0


def _lin_exp(num, den_arg):
    """(num)/(1 - exp(-den_arg)) with the removable singularity handled."""
    small = np.abs(den_arg) < 1e-7
    safe = np.where(small, 1.0, den_arg)
    return np.where(small, num / (safe / (1.0 - np.exp(-safe)) + 1e-30) * 0 + num,
                    num / (1.0 - np.exp(-safe)))


def alpha_m(V):
    a = (V + 40.0) / 10.0
    return np.where(np.abs(a) < 1e-7, 1.0, 0.1 * (V + 40.0) / (1.0 - np.exp(-a)))


def beta_m(V):
    return 4.0 * np.exp(-(V + 65.0) / 18.0)


def alpha_h(V):
    return 0.07 * np.exp(-(V + 65.0) / 20.0)


def beta_h(V):
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))


def alpha_n(V):
    a = (V + 55.0) / 10.0
    return np.where(np.abs(a) < 1e-7, 0.1, 0.01 * (V + 55.0) / (1.0 - np.exp(-a)))


def beta_n(V):
    return 0.125 * np.exp(-(V + 65.0) / 80.0)


def hh_rhs(state):
    """Exact RHS for an (N, 4) array of (V, m, h, n)."""
    V, m, h, n = state.T
    i_na = G_NA * m ** 3 * h * (V - E_NA)
    i_k = G_K * n ** 4 * (V - E_K)
    i_l = G_L * (V - E_L)
    return np.stack([
        (I_EXT - i_na - i_k - i_l) / C_M,
        alpha_m(V) * (1 - m) - beta_m(V) * m,
        alpha_h(V) * (1 - h) - beta_h(V) * h,
        alpha_n(V) * (1 - n) - beta_n(V) * n,
    ], axis=1)


# ---------------------------------------------------------------------------
# 1. Simulate repetitive firing
# ---------------------------------------------------------------------------
T_END, DT, BURN = 120.0, 0.01, 20.0
t_grid = np.arange(0, T_END, DT)
sol = solve_ivp(lambda t, s: hh_rhs(s[None, :])[0], [0, T_END],
                [-65.0, 0.05, 0.6, 0.32], t_eval=t_grid,
                method="LSODA", rtol=1e-9, atol=1e-11)
if not sol.success:
    raise RuntimeError(sol.message)
keep = t_grid >= BURN
traj = sol.y.T[keep]
t_keep = t_grid[keep]
n_spikes = int((np.diff((traj[:, 0] > 0).astype(int)) == 1).sum())
print(f"[SIM]  I_ext={I_EXT}, {len(traj)} samples after {BURN:.0f} ms burn-in")
print(f"[SIM]  {n_spikes} spikes in {T_END - BURN:.0f} ms "
      f"= {1000 * n_spikes / (T_END - BURN):.1f} Hz repetitive firing")
print(f"[SIM]  V in [{traj[:,0].min():.2f}, {traj[:,0].max():.2f}] mV")

# ---------------------------------------------------------------------------
# 2. The six-feature lift, and its true coefficients
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["V", "m3h", "m3hV", "n4", "n4V", "one"]
TRUE_COEF = np.array([-G_L / C_M, G_NA * E_NA / C_M, -G_NA / C_M,
                      G_K * E_K / C_M, -G_K / C_M, (I_EXT + G_L * E_L) / C_M])


def phi_np(Z):
    V, m, h, n = Z.T
    m3h, n4 = m ** 3 * h, n ** 4
    return np.stack([V, m3h, m3h * V, n4, n4 * V, np.ones_like(V)], axis=1)


def phi_torch(Z):
    V, m, h, n = Z[:, 0], Z[:, 1], Z[:, 2], Z[:, 3]
    m3h, n4 = m ** 3 * h, n ** 4
    return torch.stack([V, m3h, m3h * V, n4, n4 * V, torch.ones_like(V)], dim=1)


# ---------------------------------------------------------------------------
# 3. Sampling: along the trajectory vs from a box
# ---------------------------------------------------------------------------
N_TRAIN = 10_000
rng = np.random.default_rng(SEED)

idx = rng.choice(len(traj), N_TRAIN, replace=False)
traj_states = traj[idx]
traj_targets = hh_rhs(traj_states)[:, 0:1]

lo, hi = traj.min(axis=0), traj.max(axis=0)
box_states = rng.uniform(lo, hi, size=(N_TRAIN, 4))
box_targets = hh_rhs(box_states)[:, 0:1]

for name, S in [("trajectory", traj_states), ("box", box_states)]:
    print(f"[DATA]  cond(Theta), {name:10s} sampling = "
          f"{np.linalg.cond(phi_np(S)):.4g}")

# ---------------------------------------------------------------------------
# 4. Fit the voltage equation (LBFGS — see the docstring on Adam)
# ---------------------------------------------------------------------------
def fit_voltage(states, targets, steps=200, grid=3, lamb=1e-4):
    lift = CustomLift(fn=phi_np, output_dim=6, torch_fn=phi_torch, name="hh_v")
    m = KANDy(lift=lift, grid=grid, k=3, steps=steps, seed=SEED, base_fun="zero")
    m.fit(X=states, X_dot=targets, val_frac=0.15, test_frac=0.15, lamb=lamb,
          opt="LBFGS", lr=1.0, batch=-1, stop_grid_update_step=5, verbose=False)
    return m


results = {}
for name, S, Y in [("trajectory", traj_states, traj_targets),
                   ("box", box_states, box_targets)]:
    t0 = time.time()
    m = fit_voltage(S, Y)
    p = m.predict(S)
    r2 = 1.0 - np.sum((Y - p) ** 2) / np.sum((Y - Y.mean()) ** 2)
    results[name] = (m, r2, time.time() - t0)
    print(f"[FIT]   {name:10s} sampling: R^2 = {r2:.8f}  "
          f"({time.time() - t0:.0f}s)")

model, r2_box, _ = results["box"]

# ---------------------------------------------------------------------------
# 5. Symbolic extraction and the recovered biophysical constants
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Extracting the current balance ...")
formula = model.get_formula(var_names=FEATURE_NAMES, round_places=5,
                            lib=["x", "0"], r2_threshold=0.80,
                            weight_simple=0.0)[0]
print(f"  dV/dt = {formula}")

import sympy

syms = {n: sympy.Symbol(n) for n in FEATURE_NAMES}
poly = sympy.expand(formula)
rec = np.array([float(poly.coeff(syms[n])) for n in FEATURE_NAMES[:-1]]
               + [float(poly.as_coefficients_dict().get(1, 0.0)
                        + poly.coeff(syms["one"]))])

gna_r, gk_r, gl_r = -rec[2], -rec[4], -rec[0]
ena_r, ek_r = rec[1] / gna_r, rec[3] / gk_r
el_r = (rec[5] - I_EXT / C_M) * C_M / gl_r
print("\n[PARAMS] biophysical constants recovered from the edges:")
print(f"{'param':>6} {'true':>10} {'recovered':>12} {'rel. err':>10}")
for nm, tv, rv in [("gNa", G_NA, gna_r), ("gK", G_K, gk_r), ("gL", G_L, gl_r),
                   ("ENa", E_NA, ena_r), ("EK", E_K, ek_r), ("EL", E_L, el_r)]:
    print(f"{nm:>6} {tv:10.3f} {rv:12.4f} {100*abs(rv-tv)/abs(tv):9.3f}%")
print("[PARAMS] gL and EL are the loosest — the leak is the smallest current")
print("[PARAMS] in the balance, so the data constrains it least.")

# ---------------------------------------------------------------------------
# 6. The gating ceiling — refine the grid and watch R^2 not move
# ---------------------------------------------------------------------------
print("\n[GATING] dm/dt = alpha_m(V) - (alpha_m+beta_m)(V)*m — a product of an")
print("[GATING] UNKNOWN function of V with m.  Naive lift [V, m, V*m]:")

gate_states = box_states
gate_target = hh_rhs(gate_states)[:, 1:2]


def naive_np(Z):
    return np.stack([Z[:, 0], Z[:, 1], Z[:, 0] * Z[:, 1]], axis=1)


grid_scan = []
for g in [5, 10, 20, 40]:
    lift = CustomLift(fn=naive_np, output_dim=3, name="gate_naive")
    mg = KANDy(lift=lift, grid=g, k=3, steps=80, seed=SEED)
    mg.fit(X=gate_states, X_dot=gate_target, val_frac=0.15, test_frac=0.15,
           lamb=0.0, opt="LBFGS", lr=1.0, batch=-1, stop_grid_update_step=5,
           verbose=False)
    p = mg.predict(gate_states)
    r2g = 1.0 - np.sum((gate_target - p) ** 2) / np.sum(
        (gate_target - gate_target.mean()) ** 2)
    grid_scan.append((g, r2g))
    print(f"[GATING]   grid={g:3d}  R^2 = {r2g:.6f}")
spread = max(r for _, r in grid_scan) - min(r for _, r in grid_scan)
print(f"[GATING] spread across a 8x grid refinement: {spread:.6f} — FLAT.")
print("[GATING] That plateau is the signature of a missing lift term, not of")
print("[GATING] insufficient resolution.")


def oracle_np(Z):
    V, m = Z[:, 0], Z[:, 1]
    return np.stack([alpha_m(V), alpha_m(V) * m, beta_m(V) * m], axis=1)


lift = CustomLift(fn=oracle_np, output_dim=3, name="gate_oracle")
mo = KANDy(lift=lift, grid=3, k=3, steps=80, seed=SEED, base_fun="zero")
mo.fit(X=gate_states, X_dot=gate_target, val_frac=0.15, test_frac=0.15,
       lamb=0.0, opt="LBFGS", lr=1.0, batch=-1, stop_grid_update_step=5,
       verbose=False)
p = mo.predict(gate_states)
r2_oracle = 1.0 - np.sum((gate_target - p) ** 2) / np.sum(
    (gate_target - gate_target.mean()) ** 2)
print(f"[GATING] oracle lift [a_m(V), a_m(V)*m, b_m(V)*m]: R^2 = {r2_oracle:.6f}")
print("[GATING] Supplying the PRODUCTS makes it exactly linear.  Without the")
print("[GATING] rate functions in hand, a polynomial-in-V times m lift gets")
print("[GATING] close numerically, but its coefficients are a fit to alpha_m,")
print("[GATING] not interpretable rate constants.")

# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/HodgkinHuxley", exist_ok=True)

# 7a. The spike train
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
axes[0].plot(t_keep, traj[:, 0], lw=0.9, color="#1f77b4")
axes[0].set_ylabel("V (mV)")
axes[0].set_title(f"Hodgkin-Huxley, $I_{{ext}}$={I_EXT:.0f} — "
                  f"{1000*n_spikes/(T_END-BURN):.0f} Hz")
for lbl, col, c in [("m", 1, "#d62728"), ("h", 2, "#2ca02c"), ("n", 3, "#ff7f0e")]:
    axes[1].plot(t_keep, traj[:, col], lw=0.9, color=c, label=lbl)
axes[1].set_xlabel("time (ms)"); axes[1].set_ylabel("gating")
axes[1].legend(fontsize=8, ncol=3)
fig.tight_layout()
fig.savefig("results/HodgkinHuxley/spike_train.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7b. Recovered constants vs truth
fig, ax = plt.subplots(figsize=(6.4, 4))
names = ["gNa", "gK", "gL", "ENa", "EK", "EL"]
errs = [100 * abs(r - t) / abs(t) for r, t in
        [(gna_r, G_NA), (gk_r, G_K), (gl_r, G_L),
         (ena_r, E_NA), (ek_r, E_K), (el_r, E_L)]]
ax.bar(names, errs, color="#1f77b4")
ax.set_ylabel("relative error (%)")
ax.set_yscale("log")
ax.set_title("Biophysical constants recovered from data")
fig.tight_layout()
fig.savefig("results/HodgkinHuxley/constants.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7c. The gating plateau
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([g for g, _ in grid_scan], [r for _, r in grid_scan], "o-",
        color="#1f77b4", lw=1.4, ms=6, label="naive lift $[V, m, Vm]$")
ax.axhline(r2_oracle, color="#2ca02c", ls="--", lw=1.4,
           label=f"oracle lift ($R^2$={r2_oracle:.4f})")
ax.set_xscale("log", base=2)
ax.set_xlabel("spline grid size")
ax.set_ylabel(r"$R^2$ on $dm/dt$")
ax.set_title("A structural ceiling does not move with resolution")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("results/HodgkinHuxley/gating_ceiling.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7d. Loss curves and edges
if getattr(model, "train_results_", None):
    fig, ax = plot_loss_curves(model.train_results_,
                               save="results/HodgkinHuxley/loss_curves")
    plt.close(fig)

fig, axes = plot_all_edges(
    model.model_, X=torch.tensor(phi_np(box_states)[:2048], dtype=torch.float32),
    in_var_names=FEATURE_NAMES, out_var_names=["dV/dt"],
    save="results/HodgkinHuxley/edge_activations",
)
plt.close(fig)

print("\n[FIGS]  Saved results/HodgkinHuxley/")
