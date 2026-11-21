#!/usr/bin/env python3
"""KANDy example: fMRI — discovery through a nonlinear observation operator.

Every other example in this directory hands KANDy the state it is supposed to
learn about.  fMRI does not work that way, and that is the entire point of this
script.  What a scanner records is the BOLD signal: a delayed, saturating,
low-pass function of latent neural activity, sampled once per TR (2 s here).
The state you want is not the state you observe.

The standard generative model is the one Dynamic Causal Modelling uses
(Friston et al. 2000, 2003; Stephan et al. 2007).  It has two layers, and
keeping them apart is the whole discipline:

    NEURAL (latent, ~1 s)      dz/dt = A z + C u
    HEMODYNAMIC (~6 s)         ds/dt = z - kappa*s - gamma*(f - 1)
      per region, 4 states     df/dt = s
                               dv/dt = (f - v^(1/alpha)) / tau
                               dq/dt = (f*E(f)/E0 - v^(1/alpha)*q/v) / tau
                               E(f)  = 1 - (1 - E0)^(1/f)
    OBSERVED (per TR)          y = V0*(k1*(1-q) + k2*(1-q/v) + k3*(1-v))

`A` is the effective connectivity — who drives whom — and it is what any fMRI
study is ultimately after.  Standard DCM priors are used throughout:
kappa=0.64, gamma=0.32, tau=2.0, alpha=0.32, E0=0.4.

No simulation library is needed; the model above is the whole simulator.  If
you want real connectomes or other neural-mass models, `neurolib` is the
lightest option and `tvb-library` (The Virtual Brain) is the reference
platform — both ship Balloon-Windkessel BOLD.  Synthetic is used here because
the point is to check a recovered `A` against a known one, which no real
dataset can offer.

ACT 1 — the hemodynamic layer is recoverable, and the lift rule says how
------------------------------------------------------------------------
All five hemodynamic constants come back to ~0.01% with NOTHING supplied.  The
lift is the minimal one the repo's rule prescribes, and both halves of that
rule are load-bearing here:

  * `v^(1/alpha)` is a POWER OF ONE STATE, so it belongs on a spline edge, not
    in the lift.  Putting both `v` and `v^(1/alpha)` in would repeat the
    FitzHugh-Nagumo trap — v -> v^(1/alpha) is invertible, so the two edges
    become functionally dependent and the decomposition stops being unique.

  * `v^(1/alpha) * q/v` is a PRODUCT OF TWO DIFFERENT STATES, so no separable
    KAN can build it and it must be in the lift.

That leaves a chicken-and-egg problem — the cross-term needs alpha, which is
what we are trying to find — and the fix is to do it in two stages:

    stage 1   lift [z, s, f, v]          -> kappa, gamma, tau from edge SLOPES,
                                           and alpha from the edge SHAPE: the
                                           v -> dv/dt edge is -v^(1/alpha)/tau,
                                           so fitting c*v^p + d gives p = 1/alpha
    stage 2   alpha_hat builds the cross-term q*v^(1/alpha - 1)
                                        -> E0 from the f -> dq/dt edge shape

tau is recovered three independent ways and the three agree, which is a free
consistency check.  Naming a univariate edge by curve-fitting its shape is the
same recipe as `mathbio/holling_mult_node_example.py`.

ACT 2 — on BOLD alone, the connectivity comes out BACKWARDS
-----------------------------------------------------------
Three regions wired as a chain 0 -> 1 -> 2.  Region 0 is given a slow HRF and
region 1 a fast one — a ~1.4 s difference in time-to-peak, which is ordinary
physiological variation across cortex, not a pathological choice.

    neural lag  z0 -> z1 :  +0.70 s     region 1 lags, as it must — 0 drives 1
    BOLD lag    y0 -> y1 :  -0.40 s     region 1 appears to LEAD region 0

Fit a linear state-space model to the BOLD directly and it reports the absent
connection at A[0,1] = +3.15 while the real one comes out NEGATIVE at
A[1,0] = -0.31.  The sign of the answer is set by hemodynamics, not by neurons.

The falsification is the part worth copying: hold `A` fixed and change ONLY the
regional hemodynamic parameters.  The true connectivity has not moved, so any
honest estimate of it should not move either.  It moves a lot.

This is not a noise problem — everything in Act 2 is noiseless.  Nor is it a
capacity problem, so no lift and no amount of spline resolution fixes it.  The
observation operator has destroyed the ordering information that a
lag-based method needs, and no model fitted in the wrong space can recover it.

ACT 3 — put the observation operator inside the loss
-----------------------------------------------------
The obvious repair is to deconvolve the BOLD back to z and fit that.  It does
not work, and the script measures why: the slow-HRF region is the most heavily
low-passed, so its deconvolution is the worst conditioned, and differentiating
the result amplifies exactly that error.  Sweeping the regularisation does not
move the answer, which is the same "a structural problem does not respond to
tuning" signature as the grid plateau in `mathbio/hodgkin_huxley_example.py`.

What does work is to never leave BOLD space: simulate the composed system
(neural + KNOWN hemodynamics) and fit by matching predicted BOLD to observed
BOLD.  Note the change of shape — this is a GLOBAL trajectory fit over 15
parameters, not a pointwise regression of estimated derivatives, and that is
why it survives an observation operator that destroys the pointwise route.
It recovers the latent neural trajectory at R^2 ~ 1.0 and, at 5% BOLD noise,
A and C to about 5%, with the absent connection correctly near zero.

    fit on BOLD                    -> direction reversed
    deconvolve, then fit           -> still wrong, and tuning does not help
    observation operator in-loop   -> A and C to ~5%, direction correct

KANDy then enters as the CHECK on the one assumption the inversion had to
make — that the neural layer is linear.  Fitting dz/dt = f(z, u) on the
recovered trajectory with free univariate edges returns straight edges and the
same coefficients; a bilinear DCM `B` term would bend them.

    A precaution worth an order of magnitude: the stimulus is a boxcar, so
    dz/dt is DISCONTINUOUS at every block edge and a central difference
    straddling one is garbage.  Masking +-0.2 s around each transition takes
    max|A - A_true| from 0.47 to 0.054 here (on noiseless data it goes from
    0.27 to 0.0008 — the floor is set by the BOLD noise, not the masking).  Do
    not widen the mask much: the blocks are only 2 s long, so a 1 s mask deletes
    the stimulus entirely and C stops being identifiable (cond -> inf).

HONEST LIMITS.  The inversion assumes the hemodynamic model and its parameters
are known and the neural model is linear — which is what DCM assumes too, and
why DCM needs fixed hemodynamic priors.  Act 3 is therefore parameter
estimation in a known model, not free discovery.  Real fMRI
additionally brings unknown regional HRFs, physiological confounds, and far
more noise than the 5% used here; A degrades roughly linearly with it
(max|A - A_true| ~ 0.03 / 0.06 / 0.11 at 5 / 10 / 20% noise).

Runtime: ~3 minutes.
"""

import sys
from pathlib import Path
import time

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from _utils import ensure_outdir, print_check, save_metrics
from kandy import KANDy, CustomLift
from kandy.plotting import get_edge_activation, plot_all_edges, use_pub_style

# ---------------------------------------------------------------------------
# 0. Reproducibility, and the standard DCM / Balloon-Windkessel priors
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

KAPPA, GAMMA, TAU, ALPHA, E0 = 0.64, 0.32, 2.0, 0.32, 0.4
V0, TE, EPSI = 0.04, 0.04, 1.43
K1 = 4.3 * 40.3 * E0 * TE
K2 = EPSI * 25.0 * E0 * TE
K3 = 1.0 - EPSI

# neural layer: a chain 0 -> 1 -> 2, driven by two independent stimulus trains
A_TRUE = np.array([[-1.0, 0.0, 0.0],
                   [0.6, -1.0, 0.0],
                   [0.0, 0.6, -1.0]])
C_TRUE = np.array([[1.0, 0.0],
                   [0.0, 0.0],
                   [0.0, 0.8]])
R, N_U = 3, 2

# per-region hemodynamics: region 0 SLOW, region 1 FAST, region 2 typical.
# columns are (kappa, gamma, tau); a ~1.4 s spread in time-to-peak.
HEMO = np.array([[0.40, 0.22, 2.4],
                 [1.00, 0.48, 1.6],
                 [0.64, 0.32, 2.0]])

DT, T_END, TR = 0.1, 600.0, 2.0
STIM_AMP = 0.6
BOLD_NOISE = 0.05          # measurement noise, as a fraction of each region's SD

EQ_NAMES = ["ds/dt", "df/dt", "dv/dt", "dq/dt"]


# ---------------------------------------------------------------------------
# 1. The simulator — the two-layer generative model, and nothing else
# ---------------------------------------------------------------------------
def balloon_rhs(S, z, hemo):
    """Balloon-Windkessel RHS. S is (R, 4) of (s, f, v, q); z is (R,)."""
    s, f, v, q = S.T
    f = np.maximum(f, 1e-3)
    v = np.maximum(v, 1e-3)
    E = 1.0 - (1.0 - E0) ** (1.0 / f)
    va = v ** (1.0 / ALPHA)
    kap, gam, tau = hemo.T
    return np.stack([z - kap * s - gam * (f - 1.0),
                     s,
                     (f - va) / tau,
                     (f * E / E0 - va * q / v) / tau], axis=1)


def bold_of(S):
    """BOLD observation operator: (R, 4) hemodynamic states -> (R,) signal."""
    v, q = S[:, 2], S[:, 3]
    return V0 * (K1 * (1.0 - q) + K2 * (1.0 - q / v) + K3 * (1.0 - v))


def initial_states():
    return np.column_stack([np.zeros(R), np.ones(R), np.ones(R), np.ones(R)])


def simulate(A, C, U, hemo, dt, n_steps):
    """Roll the composed system. Returns (BOLD, z, hemodynamic states)."""
    z = np.zeros(R)
    S = initial_states()
    Y = np.empty((n_steps, R))
    Z = np.empty((n_steps, R))
    H = np.empty((n_steps, R, 4))
    for i in range(n_steps):
        Y[i], Z[i], H[i] = bold_of(S), z, S
        ui = U[i]

        def fz(zz):
            return A @ zz + C @ ui

        k1z, k1S = fz(z), balloon_rhs(S, z, hemo)
        k2z, k2S = fz(z + .5 * dt * k1z), balloon_rhs(S + .5 * dt * k1S,
                                                      z + .5 * dt * k1z, hemo)
        k3z, k3S = fz(z + .5 * dt * k2z), balloon_rhs(S + .5 * dt * k2S,
                                                      z + .5 * dt * k2z, hemo)
        k4z, k4S = fz(z + dt * k3z), balloon_rhs(S + dt * k3S,
                                                 z + dt * k3z, hemo)
        z = z + dt / 6 * (k1z + 2 * k2z + 2 * k3z + k4z)
        S = S + dt / 6 * (k1S + 2 * k2S + 2 * k3S + k4S)
        if not np.isfinite(z).all() or not np.isfinite(S).all():
            return None, None, None
    return Y, Z, H


def stimulus_train(t, seed):
    """2 s blocks with jittered 6-14 s gaps — an ordinary event-related design."""
    rng_s = np.random.default_rng(seed)
    on = np.zeros_like(t)
    tt = 8.0
    while tt < t[-1]:
        on[(t >= tt) & (t < tt + 2.0)] = STIM_AMP
        tt += 2.0 + rng_s.uniform(6.0, 14.0)
    return on


N_STEP = int(T_END / DT)
t_grid = np.arange(N_STEP) * DT
U_GRID = np.column_stack([stimulus_train(t_grid, 1), stimulus_train(t_grid, 2)])

Y_clean, Z_TRUE, H_TRUE = simulate(A_TRUE, C_TRUE, U_GRID, HEMO, DT, N_STEP)
obs_idx = np.arange(0, N_STEP, int(TR / DT))
rng = np.random.default_rng(SEED)
Y_OBS = Y_clean[obs_idx] + rng.normal(
    0.0, BOLD_NOISE * Y_clean[obs_idx].std(axis=0), (len(obs_idx), R))

print(f"[SIM]   {R} regions, {T_END:.0f} s at TR={TR} s "
      f"-> {len(obs_idx)} volumes, {N_U} stimulus regressors")
print(f"[SIM]   BOLD {100 * Y_clean.min():+.2f} .. {100 * Y_clean.max():+.2f} "
      f"% signal change, measurement noise {100 * BOLD_NOISE:.0f}% of SD")
print(f"[SIM]   latent neural z in [{Z_TRUE.min():.3f}, {Z_TRUE.max():.3f}]")


def hrf_peak_times(hemo, dt=0.1, horizon=30.0):
    """Time-to-peak of each region's BOLD impulse response."""
    n = int(horizon / dt)
    imp = np.zeros((n, R))
    imp[0] = 1.0 / dt
    # drive the Balloon directly with the impulse — no neural layer involved
    S = initial_states()
    y = np.empty((n, R))
    for i in range(n):
        y[i] = bold_of(S)
        zi = imp[i]
        k1 = balloon_rhs(S, zi, hemo)
        k2 = balloon_rhs(S + .5 * dt * k1, zi, hemo)
        k3 = balloon_rhs(S + .5 * dt * k2, zi, hemo)
        k4 = balloon_rhs(S + dt * k3, zi, hemo)
        S = S + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.arange(n) * dt * 1.0, y, (np.argmax(y, axis=0) * dt)


t_hrf, HRF, PEAK_T = hrf_peak_times(HEMO)
print(f"[SIM]   HRF time-to-peak per region: {np.round(PEAK_T, 2)} s  "
      f"(spread {PEAK_T.max() - PEAK_T.min():.2f} s)")


# ===========================================================================
# ACT 1 — recover the hemodynamic layer, supplying nothing
# ===========================================================================
print("\n" + "=" * 72)
print("ACT 1  the hemodynamic layer")
print("=" * 72)

# Box samples over the physiological range, not the trajectory: on a trajectory
# the lifted features are correlated and the constants come out wrong at
# R^2 = 1. Same identifiability argument as mathbio/sir_example.py.
N_BOX = 8000
box = np.random.default_rng(SEED)
z_b = box.uniform(-0.5, 1.2, N_BOX)
s_b = box.uniform(-0.35, 0.45, N_BOX)
f_b = box.uniform(0.70, 1.70, N_BOX)
v_b = box.uniform(0.80, 1.45, N_BOX)
q_b = box.uniform(0.60, 1.20, N_BOX)

E_b = 1.0 - (1.0 - E0) ** (1.0 / f_b)
va_b = v_b ** (1.0 / ALPHA)
d_s = z_b - KAPPA * s_b - GAMMA * (f_b - 1.0)
d_f = s_b
d_v = (f_b - va_b) / TAU
d_q = (f_b * E_b / E0 - va_b * q_b / v_b) / TAU
print(f"[ACT1]  {N_BOX} box samples over the physiological ranges")


def edge_xy(model, i, j):
    """Sorted (input, output) samples of the edge feeding output j from input i."""
    x, y = get_edge_activation(model.model_, 0, i, j)
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    order = np.argsort(x)
    return x[order], y[order]


def edge_slope(model, i, j):
    x, y = edge_xy(model, i, j)
    return float(np.polyfit(x, y, 1)[0])


def fit_kandy(X, Y, name, grid=8, steps=200):
    m = KANDy(lift=CustomLift(fn=lambda Z: Z, output_dim=X.shape[1], name=name),
              grid=grid, k=3, steps=steps, seed=SEED, base_fun="zero")
    m.fit(X=X, X_dot=Y, val_frac=0.15, test_frac=0.15, lamb=0.0,
          opt="LBFGS", lr=1.0, batch=-1, stop_grid_update_step=5, verbose=False)
    m.model_.save_act = True
    with torch.no_grad():
        m.model_(torch.tensor(X[:4000], dtype=torch.float32))
    return m


def r2_cols(Y, P):
    return [float(1.0 - np.sum((Y[:, i] - P[:, i]) ** 2)
                  / np.sum((Y[:, i] - Y[:, i].mean()) ** 2))
            for i in range(Y.shape[1])]


# --- stage 1: [z, s, f, v] -> (ds/dt, df/dt, dv/dt) ------------------------
X1 = np.column_stack([z_b, s_b, f_b, v_b])
Y1 = np.column_stack([d_s, d_f, d_v])
t0 = time.time()
m1 = fit_kandy(X1, Y1, "hemo_stage1")
r2_stage1 = r2_cols(Y1, m1.predict(X1))
print(f"[ACT1]  stage 1 lift [z, s, f, v]  R^2 = {np.round(r2_stage1, 8)}  "
      f"({time.time() - t0:.0f}s)")

kappa_r = -edge_slope(m1, 1, 0)          # s  -> ds/dt  slope is -kappa
gamma_r = -edge_slope(m1, 2, 0)          # f  -> ds/dt  slope is -gamma
tau_from_f = 1.0 / edge_slope(m1, 2, 2)  # f  -> dv/dt  slope is 1/tau


def power_law(x, c, p, d):
    return c * x ** p + d


# alpha comes from the SHAPE of the v -> dv/dt edge, which is -v^(1/alpha)/tau
xv, yv = edge_xy(m1, 3, 2)
p_v, _ = curve_fit(power_law, xv, yv, p0=[-0.5, 3.0, 0.5], maxfev=400_000)
r2_v_edge = float(1.0 - np.sum((yv - power_law(xv, *p_v)) ** 2)
                  / np.sum((yv - yv.mean()) ** 2))
alpha_r = 1.0 / float(p_v[1])
tau_from_v = -1.0 / float(p_v[0])

# --- stage 2: alpha_hat builds the cross-term, then dq/dt -----------------
qv_b = q_b * v_b ** (1.0 / alpha_r - 1.0)
X2 = np.column_stack([f_b, qv_b])
Y2 = d_q[:, None]
m2 = fit_kandy(X2, Y2, "hemo_stage2")
r2_stage2 = r2_cols(Y2, m2.predict(X2))[0]
print(f"[ACT1]  stage 2 lift [f, q*v^(1/alpha-1)]  R^2 = {r2_stage2:.8f}")

tau_from_q = -1.0 / edge_slope(m2, 1, 0)   # cross-term edge slope is -1/tau


def f_times_E(x, e0, tau, d):
    return (x * (1.0 - (1.0 - e0) ** (1.0 / x))) / (tau * e0) + d


xf, yf = edge_xy(m2, 0, 0)
p_f, _ = curve_fit(f_times_E, xf, yf, p0=[0.35, 2.0, 0.0],
                   bounds=([0.05, 0.5, -5.0], [0.95, 6.0, 5.0]), maxfev=400_000)
E0_r = float(p_f[0])
r2_f_edge = float(1.0 - np.sum((yf - f_times_E(xf, *p_f)) ** 2)
                  / np.sum((yf - yf.mean()) ** 2))

HEMO_REC = {"kappa": (KAPPA, kappa_r), "gamma": (GAMMA, gamma_r),
            "tau": (TAU, tau_from_f), "alpha": (ALPHA, alpha_r),
            "E0": (E0, E0_r)}
print(f"\n[ACT1]  {'param':>6} {'true':>9} {'recovered':>11} {'rel err':>9}")
for nm, (tv, rv) in HEMO_REC.items():
    print(f"[ACT1]  {nm:>6} {tv:9.4f} {rv:11.5f} {100 * abs(rv - tv) / abs(tv):8.3f}%")
hemo_rel_err = max(abs(rv - tv) / abs(tv) for tv, rv in HEMO_REC.values())
tau_spread = float(np.ptp([tau_from_f, tau_from_v, tau_from_q]))
print(f"[ACT1]  tau three independent ways: {tau_from_f:.5f} (f edge), "
      f"{tau_from_v:.5f} (v edge), {tau_from_q:.5f} (cross-term edge)")
print(f"[ACT1]  spread {tau_spread:.2e} — a free consistency check")


# ===========================================================================
# ACT 2 — on BOLD alone the connectivity comes out backwards
# ===========================================================================
print("\n" + "=" * 72)
print("ACT 2  the hemodynamic-lag confound")
print("=" * 72)


def peak_lag(a, b, dt, max_lag=8.0):
    """Cross-correlation lag; positive means b LAGS a."""
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    n = int(max_lag / dt)
    lags = np.arange(-n, n + 1)
    cc = [np.mean(a[max(0, -L):len(a) - max(0, L)] * b[max(0, L):len(b) - max(0, -L)])
          for L in lags]
    return float(lags[int(np.argmax(cc))] * dt)


lag_neural = peak_lag(Z_TRUE[:, 0], Z_TRUE[:, 1], DT)
lag_bold = peak_lag(Y_clean[:, 0], Y_clean[:, 1], DT)
print(f"[ACT2]  true wiring is 0 -> 1 -> 2, so region 1 must LAG region 0")
print(f"[ACT2]  neural lag  z0 -> z1 : {lag_neural:+.2f} s   (correct)")
print(f"[ACT2]  BOLD  lag   y0 -> y1 : {lag_bold:+.2f} s   "
      f"{'<- REVERSED' if lag_bold < 0 else ''}")


def naive_connectivity(Y_volumes, U_volumes):
    """What a linear state-space fit on the BOLD time series reports as `A`."""
    dY = np.gradient(Y_volumes, TR, axis=0)
    G = np.column_stack([Y_volumes, U_volumes, np.ones(len(Y_volumes))])
    sol, *_ = np.linalg.lstsq(G, dY, rcond=None)
    return sol[:R].T


U_OBS = U_GRID[obs_idx]
A_naive = naive_connectivity(Y_OBS, U_OBS)
np.set_printoptions(precision=3, suppress=True)
print(f"\n[ACT2]  naive A from BOLD:\n{A_naive}")
print(f"[ACT2]  true A:\n{A_TRUE}")
print(f"[ACT2]  the real link  0->1 : A[1,0] = {A_naive[1, 0]:+.3f}  (true +0.600)")
print(f"[ACT2]  the absent link 1->0: A[0,1] = {A_naive[0, 1]:+.3f}  (true  0.000)")
naive_reversed = bool(A_naive[0, 1] > A_naive[1, 0])
print(f"[ACT2]  strongest inferred direction is "
      f"{'1 -> 0  — BACKWARDS' if naive_reversed else '0 -> 1'}")

# The falsification: hold A fixed, move ONLY the hemodynamics.
print("\n[ACT2]  falsification — same A, different regional HRFs:")
HEMO_VARIANTS = {
    "as above (r0 slow, r1 fast)": HEMO,
    "uniform HRF everywhere": np.tile([KAPPA, GAMMA, TAU], (R, 1)),
    "reversed (r0 fast, r1 slow)": HEMO[[1, 0, 2]],
}
naive_sweep = {}
for label, hv in HEMO_VARIANTS.items():
    Yv, _, _ = simulate(A_TRUE, C_TRUE, U_GRID, hv, DT, N_STEP)
    Av = naive_connectivity(Yv[obs_idx], U_OBS)
    naive_sweep[label] = Av
    print(f"[ACT2]    {label:<30s} A[1,0]={Av[1, 0]:+.3f}  A[0,1]={Av[0, 1]:+.3f}")
sweep_span = float(np.ptp([Av[0, 1] for Av in naive_sweep.values()]))
print(f"[ACT2]  A[0,1] moves by {sweep_span:.3f} across HRF settings while the "
      f"true A never changes.")
print("[ACT2]  Noiseless throughout — this is structural, not a noise or")
print("[ACT2]  capacity problem, so no lift and no grid refinement repairs it.")


# ===========================================================================
# ACT 3 — put the observation operator inside the loss
# ===========================================================================
print("\n" + "=" * 72)
print("ACT 3  fitting in BOLD space")
print("=" * 72)

# --- the repair that does NOT work, measured ------------------------------
DZ = 0.25
n_dz = int(T_END / DZ)
t_dz = np.arange(n_dz) * DZ
hrf_dz_t, HRF_DZ, _ = hrf_peak_times(HEMO, dt=DZ)
HRF_LEN = len(HRF_DZ)
obs_dz = np.round(obs_idx * DT / DZ).astype(int)
obs_dz = obs_dz[obs_dz < n_dz]
D2 = np.eye(n_dz) * -2.0 + np.eye(n_dz, k=1) + np.eye(n_dz, k=-1)
ix_dz = np.clip(np.round(t_dz / DT).astype(int), 0, N_STEP - 1)


def deconvolve(lamb):
    """Tikhonov deconvolution of each region's BOLD through its own HRF."""
    out = np.zeros((n_dz, R))
    for r in range(R):
        M = np.zeros((n_dz, n_dz))
        h = HRF_DZ[:, r]
        for j in range(n_dz):
            L = min(HRF_LEN, n_dz - j)
            M[j:j + L, j] = h[:L] * DZ
        Ms = M[obs_dz]
        G = Ms.T @ Ms + lamb * (D2.T @ D2) * np.trace(Ms.T @ Ms) / n_dz
        out[:, r] = np.linalg.solve(G, Ms.T @ Y_OBS[:len(obs_dz), r])
    return out


print("[ACT3]  first, the obvious repair: deconvolve BOLD -> z, then fit.")
deconv_scan = []
for lamb in [1e-4, 1e-3, 1e-2]:
    zc = deconvolve(lamb)
    dz = np.gradient(zc, DZ, axis=0)
    keep = slice(int(20 / DZ), n_dz - int(5 / DZ))
    G = np.column_stack([zc, U_GRID[ix_dz], np.ones(n_dz)])[keep]
    sol, *_ = np.linalg.lstsq(G, dz[keep], rcond=None)
    Ad = sol[:R].T
    zr2 = min(r2_cols(Z_TRUE[ix_dz], zc))
    deconv_scan.append((lamb, zr2, float(np.abs(Ad - A_TRUE).max())))
    print(f"[ACT3]    lambda={lamb:.0e}  z R^2 (worst region) {zr2:6.3f}   "
          f"max |A-A_true| {np.abs(Ad - A_TRUE).max():7.3f}")
deconv_best = min(s[2] for s in deconv_scan)
deconv_span = float(np.ptp([s[2] for s in deconv_scan]))
print(f"[ACT3]  best max |A - A_true| over the sweep: {deconv_best:.3f}, and the")
print(f"[ACT3]  answer barely moves across 100x of regularisation "
      f"(span {deconv_span:.3f}).")
print("[ACT3]  The slow-HRF region is the most low-passed, so its deconvolution")
print("[ACT3]  is the worst conditioned — and differentiating amplifies exactly")
print("[ACT3]  that error.  Tuning does not fix a structural problem.")

# --- the repair that does work --------------------------------------------
print("\n[ACT3]  now fit WITHOUT leaving BOLD space: simulate the composed")
print("[ACT3]  system (neural + known hemodynamics) and match observed BOLD.")
Y_SD = Y_OBS.std(axis=0)


def bold_residual(p):
    A = p[:R * R].reshape(R, R)
    C = p[R * R:].reshape(R, N_U)
    Yh, _, _ = simulate(A, C, U_GRID, HEMO, DT, N_STEP)
    if Yh is None:                       # unstable A — reject, do not crash
        return np.full(len(obs_idx) * R, 1e3)
    # per-region scaling, for the reason in mathbio/epileptor_example.py:
    # regions differ in BOLD amplitude and an unweighted loss ignores the small
    return ((Yh[obs_idx] - Y_OBS) / Y_SD).ravel()


t0 = time.time()
p0 = np.concatenate([(-0.5 * np.eye(R)).ravel(), np.zeros(R * N_U)])
n_par = R * R + R * N_U
res = least_squares(bold_residual, p0,
                    bounds=(np.full(n_par, -3.0), np.full(n_par, 3.0)),
                    method="trf", diff_step=1e-4, x_scale="jac",
                    xtol=1e-12, ftol=1e-12, max_nfev=3000)
A_joint = res.x[:R * R].reshape(R, R)
C_joint = res.x[R * R:].reshape(R, N_U)
_, Z_REC, _ = simulate(A_joint, C_joint, U_GRID, HEMO, DT, N_STEP)
z_r2 = r2_cols(Z_TRUE, Z_REC)
A_joint_err = float(np.abs(A_joint - A_TRUE).max())
C_joint_err = float(np.abs(C_joint - C_TRUE).max())
joint_correct = bool(A_joint[1, 0] > A_joint[0, 1])
print(f"[ACT3]  joint inversion: {time.time() - t0:.0f}s, {res.nfev} evaluations")
print(f"[ACT3]  latent neural trajectory R^2 per region: {np.round(z_r2, 6)}")
print(f"[ACT3]  A recovered:\n{A_joint}")
print(f"[ACT3]  true A:\n{A_TRUE}")
print(f"[ACT3]  the real link  0->1 : A[1,0] = {A_joint[1, 0]:+.3f}  (true +0.600)")
print(f"[ACT3]  the absent link 1->0: A[0,1] = {A_joint[0, 1]:+.3f}  (true  0.000)")
print(f"[ACT3]  max |A - A_true| = {A_joint_err:.4f}  "
      f"max |C - C_true| = {C_joint_err:.4f}   at {100 * BOLD_NOISE:.0f}% BOLD noise")
print(f"[ACT3]  strongest inferred direction is "
      f"{'0 -> 1  — CORRECT' if joint_correct else '1 -> 0 — still backwards'}")

# --- KANDy on the recovered latent trajectory -----------------------------
# The inversion above ASSUMED a linear neural layer in order to recover z.
# KANDy is the check on that assumption, not a second estimate of it: fit
# dz/dt = f(z, u) with free univariate edges and see whether the edges come
# back straight.  A bilinear DCM `B` term or any neural nonlinearity would
# bend them.
#
# ONE PRECAUTION, worth an order of magnitude.  The stimulus is a BOXCAR, so
# dz/dt jumps discontinuously at every block edge, and a central difference
# straddling such an edge is simply wrong.  The contrast is printed below.  Do
# not widen the mask much: the blocks are only 2 s long, so a 1 s mask deletes
# the stimulus entirely and C stops being identifiable (cond -> inf).
MASK_SEC = 0.2
NEURAL_NAMES = ["z0", "z1", "z2", "u0", "u1"]
dZ_REC = np.gradient(Z_REC, DT, axis=0)
base_mask = np.zeros(N_STEP, bool)
base_mask[int(20 / DT):N_STEP - int(5 / DT)] = True
switch = np.zeros(N_STEP, bool)
w_mask = int(round(MASK_SEC / DT))
for i in np.flatnonzero(np.any(np.abs(np.diff(U_GRID, axis=0)) > 1e-9, axis=1)):
    switch[max(0, i - w_mask):i + w_mask + 1] = True

Theta_neural = np.column_stack([Z_REC, U_GRID])
mask_scan = {}
for label, msk in [("all samples", base_mask),
                   (f"transitions masked (+-{MASK_SEC} s)", base_mask & ~switch)]:
    G = np.column_stack([Theta_neural, np.ones(N_STEP)])[msk]
    sol, *_ = np.linalg.lstsq(G, dZ_REC[msk], rcond=None)
    mask_scan[label] = (float(np.abs(sol[:R].T - A_TRUE).max()),
                        float(np.abs(sol[R:R + N_U].T - C_TRUE).max()),
                        float(np.linalg.cond(G)))
    print(f"[ACT3]    {label:<34s} max|A-A*|={mask_scan[label][0]:.5f}  "
          f"max|C-C*|={mask_scan[label][1]:.5f}  cond={mask_scan[label][2]:.0f}")

fit_mask = base_mask & ~switch
X3 = Theta_neural[fit_mask]
Y3 = dZ_REC[fit_mask]
m3 = fit_kandy(X3, Y3, "neural", grid=5, steps=150)
r2_neural = r2_cols(Y3, m3.predict(X3))
print(f"[ACT3]  KANDy on the recovered z:  R^2 = {np.round(r2_neural, 6)}")

A_kandy = np.array([[edge_slope(m3, i, j) for i in range(R)] for j in range(R)])
C_kandy = np.array([[edge_slope(m3, R + k, j) for k in range(N_U)]
                    for j in range(R)])
A_err = float(np.abs(A_kandy - A_TRUE).max())
C_err = float(np.abs(C_kandy - C_TRUE).max())
print(f"[ACT3]  A from KANDy edge slopes:\n{A_kandy}")
print(f"[ACT3]  max |A - A_true| = {A_err:.4f}   max |C - C_true| = {C_err:.4f}")

# Edge linearity, over the z edges only.  The u edges are excluded because the
# stimulus is binary: an edge sampled at exactly two input values is not
# constrained between them, so "how straight is it" has no answer there.
#
# The residual is scaled by the spread of the EQUATION the edge feeds, not by
# the edge's own range.  Several true coefficients are exactly zero, so those
# edges are flat and their own range is ~0 — dividing by it reports any numerical
# wiggle as 100% non-linear.  Same normalisation as the edge-sum check in
# mathbio/fitzhugh_nagumo_example.py.
edge_nonlin = 0.0
for j in range(R):
    out_span = max(float(np.ptp(Y3[:, j])), 1e-12)
    for i in range(R):
        xe, ye = edge_xy(m3, i, j)
        resid = ye - np.polyval(np.polyfit(xe, ye, 1), xe)
        edge_nonlin = max(edge_nonlin, float(np.ptp(resid) / out_span))
print(f"[ACT3]  worst z-edge deviation from a straight line: "
      f"{100 * edge_nonlin:.2f}% of its range — the linear neural model holds.")


# ===========================================================================
# Verification
#
#   Three assertions, one per act, and the middle one is NEGATIVE.  Act 2 must
#   keep failing: if a naive fit on BOLD ever recovered the right direction,
#   the confound this example exists to demonstrate would not be demonstrated,
#   and the script should say so rather than quietly pass.  The pair
#   (naive reversed, joint correct) is what carries the lesson, so both halves
#   are gated.
#
#   Act 1 gates the recovered CONSTANTS, not R^2 — alpha and E0 come from the
#   shapes of two edges, and the three independent estimates of tau must agree.
#   Act 3 gates the latent trajectory as well as A, because recovering z is the
#   thing the observation operator buys and A follows from it.  The
#   deconvolution route is gated as a FAILURE for the same reason Act 2 is: it
#   is the obvious wrong turn, and the example should notice if it stops being
#   wrong.
# ===========================================================================
THRESHOLDS = {
    "hemo_rel_err_max": 0.02,          # all five Balloon constants, nothing given
    "tau_spread_max": 0.01,            # three independent readings must agree
    "act1_r2_min": 0.999,
    "naive_must_reverse_direction": True,
    "naive_hrf_sweep_span_min": 0.10,  # HRF alone must move the naive answer
    "deconvolution_A_err_min": 0.5,    # deconvolve-then-fit must still fail
    "latent_z_r2_min": 0.99,
    "joint_direction_correct": True,
    "A_abs_err_max": 0.15,             # from the inversion, at 5% BOLD noise
    "C_abs_err_max": 0.15,
    "kandy_A_abs_err_max": 0.15,       # KANDy on the recovered latent z
    # Neural edges are straight lines.  Held at 15% rather than something
    # tighter because LBFGS is not bit-reproducible here and this number moves
    # a lot between runs (0.6% and 4.6% observed on consecutive runs of the
    # same script) — the wiggle lands on the edges whose true coefficient is
    # zero.  A genuine neural nonlinearity, such as a bilinear DCM `B` term,
    # bends an edge far past this, so the gate still discriminates.
    "edge_nonlinearity_max": 0.15,
}
passed = print_check(
    hemo_rel_err <= THRESHOLDS["hemo_rel_err_max"]
    and tau_spread <= THRESHOLDS["tau_spread_max"]
    and min(r2_stage1 + [r2_stage2]) >= THRESHOLDS["act1_r2_min"]
    and naive_reversed
    and sweep_span >= THRESHOLDS["naive_hrf_sweep_span_min"]
    and deconv_best >= THRESHOLDS["deconvolution_A_err_min"]
    and min(z_r2) >= THRESHOLDS["latent_z_r2_min"]
    and joint_correct
    and A_joint_err <= THRESHOLDS["A_abs_err_max"]
    and C_joint_err <= THRESHOLDS["C_abs_err_max"]
    and A_err <= THRESHOLDS["kandy_A_abs_err_max"]
    and edge_nonlin <= THRESHOLDS["edge_nonlinearity_max"],
    "fMRI: Balloon constants recovered; BOLD-only reverses the connectivity; "
    "observation operator in the loss restores it",
)

RESULT_DIR = ensure_outdir("results/FMRIDCM")
save_metrics(RESULT_DIR / "metrics.json", {
    "task": "FMRIDCM",
    "category": "mathbio",
    "seed": SEED,
    "n_regions": R,
    "n_volumes": int(len(obs_idx)),
    "TR": TR,
    "bold_noise_frac": BOLD_NOISE,
    "sampling_act1": "uniform box + true RHS (oracle derivatives)",
    "hrf_peak_times": PEAK_T.tolist(),
    "act1_r2_stage1": r2_stage1,
    "act1_r2_stage2": r2_stage2,
    "hemodynamic_constants": {k: {"true": v[0], "recovered": float(v[1]),
                                  "rel_err": float(abs(v[1] - v[0]) / abs(v[0]))}
                              for k, v in HEMO_REC.items()},
    "hemo_rel_err_max": float(hemo_rel_err),
    "tau_three_ways": [float(tau_from_f), float(tau_from_v), float(tau_from_q)],
    "tau_spread": tau_spread,
    "v_edge_powerlaw_r2": r2_v_edge,
    "f_edge_shape_r2": r2_f_edge,
    "lag_neural_z0_z1": lag_neural,
    "lag_bold_y0_y1": lag_bold,
    "A_true": A_TRUE.tolist(),
    "A_naive_from_bold": A_naive.tolist(),
    "naive_reversed": naive_reversed,
    "naive_hrf_sweep": {k: v.tolist() for k, v in naive_sweep.items()},
    "naive_hrf_sweep_span": sweep_span,
    "deconvolution_scan": [[float(a), float(b), float(c)] for a, b, c in deconv_scan],
    "deconvolution_best_A_err": float(deconv_best),
    "latent_z_r2": z_r2,
    "A_joint_inversion": A_joint.tolist(),
    "C_joint_inversion": C_joint.tolist(),
    "A_joint_abs_err": A_joint_err,
    "C_joint_abs_err": C_joint_err,
    "joint_direction_correct": joint_correct,
    "C_true": C_TRUE.tolist(),
    "A_kandy": A_kandy.tolist(),
    "C_kandy": C_kandy.tolist(),
    "A_abs_err": A_err,
    "C_abs_err": C_err,
    "transition_mask_sec": MASK_SEC,
    "transition_mask_contrast": {k: {"A_abs_err": v[0], "C_abs_err": v[1],
                                     "cond": v[2]}
                                 for k, v in mask_scan.items()},
    "neural_fit_r2": r2_neural,
    "edge_nonlinearity": edge_nonlin,
    "thresholds": THRESHOLDS,
    "passed": passed,
})


# ===========================================================================
# Figures
# ===========================================================================
def plot_generative_model(outdir):
    """The two layers and what the scanner actually sees."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    show = slice(0, int(200 / DT))
    ts = t_grid[show]
    for r, c in zip(range(R), ["#1f77b4", "#d62728", "#2ca02c"]):
        axes[0].plot(ts, Z_TRUE[show, r], color=c, lw=1.1, label=f"region {r}")
        axes[1].plot(ts, 100 * Y_clean[show, r], color=c, lw=1.1)
    axes[2].plot(ts, U_GRID[show, 0], color="#7f7f7f", lw=1.0, label="$u_0$")
    axes[2].plot(ts, U_GRID[show, 1], color="k", lw=1.0, ls="--", label="$u_1$")
    axes[0].set_ylabel("neural $z$")
    axes[0].set_title("The two layers: latent neural activity, and what is measured")
    axes[0].legend(fontsize=8, ncol=3)
    axes[1].set_ylabel("BOLD (% change)")
    axes[2].set_ylabel("stimulus")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "generative_model.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_hemodynamic_constants(outdir):
    """Act 1: what came back, and the two edges that gave alpha and E0."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    names = list(HEMO_REC)
    errs = [100 * abs(rv - tv) / abs(tv) for tv, rv in HEMO_REC.values()]
    axes[0].bar(names, errs, color="#1f77b4")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("relative error (%)")
    axes[0].set_title("Balloon constants, nothing supplied", fontsize=10)

    axes[1].plot(xv, yv, lw=2.6, color="#1f77b4", alpha=0.5, label="KAN edge")
    axes[1].plot(xv, power_law(xv, *p_v), lw=1.2, ls="--", color="#d62728",
                 label=f"$c\\,v^{{p}}+d$, $1/p$={alpha_r:.4f}")
    axes[1].set_xlabel("$v$")
    axes[1].set_ylabel(r"edge $\to\ dv/dt$")
    axes[1].set_title(f"$\\alpha$ from an edge SHAPE\n(true {ALPHA})", fontsize=10)
    axes[1].legend(fontsize=8)

    axes[2].plot(xf, yf, lw=2.6, color="#1f77b4", alpha=0.5, label="KAN edge")
    axes[2].plot(xf, f_times_E(xf, *p_f), lw=1.2, ls="--", color="#2ca02c",
                 label=f"$f\\,E(f)/(\\tau E_0)$, $E_0$={E0_r:.4f}")
    axes[2].set_xlabel("$f$")
    axes[2].set_ylabel(r"edge $\to\ dq/dt$")
    axes[2].set_title(f"$E_0$ from an edge SHAPE\n(true {E0})", fontsize=10)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "hemodynamic_constants.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lag_confound(outdir):
    """Act 2: the HRFs, the reversal, and the naive answer moving with the HRF."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for r, c in zip(range(R), ["#1f77b4", "#d62728", "#2ca02c"]):
        axes[0].plot(t_hrf, HRF[:, r] / HRF[:, r].max(), color=c, lw=1.4,
                     label=f"region {r} (peak {PEAK_T[r]:.1f} s)")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("normalised HRF")
    axes[0].set_title("Regional HRFs differ by ~1.4 s", fontsize=10)
    axes[0].legend(fontsize=8)

    # Zoom on ONE isolated event, so the reversal is visible rather than
    # inferred: peaks are marked, and the BOLD peaks come in the other order.
    onsets = np.flatnonzero(np.diff(U_GRID[:, 0]) > 0)
    gaps = np.diff(np.r_[onsets, N_STEP])
    ev = int(onsets[int(np.argmax(gaps))])
    show = slice(max(0, ev - int(3 / DT)), ev + int(18 / DT))
    ts = t_grid[show]
    for ci, (lab, c) in enumerate(zip(["(drives)", "(driven)"],
                                      ["#1f77b4", "#d62728"])):
        zz = Z_TRUE[show, ci] / max(Z_TRUE[show, ci].max(), 1e-12)
        yy = Y_clean[show, ci] / max(Y_clean[show, ci].max(), 1e-12)
        axes[1].plot(ts, zz, color=c, lw=1.6, label=f"$z_{ci}$ {lab}")
        axes[1].plot(ts, yy, color=c, lw=1.2, ls="--", label=f"$y_{ci}$ BOLD")
        axes[1].axvline(ts[int(np.argmax(zz))], color=c, lw=0.8, ls=":", alpha=0.8)
        axes[1].axvline(ts[int(np.argmax(yy))], color=c, lw=1.4, ls="--", alpha=0.5)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("normalised (peaks marked)")
    axes[1].set_title(f"neural lag {lag_neural:+.2f} s, "
                      f"BOLD lag {lag_bold:+.2f} s — order flips", fontsize=10)
    axes[1].legend(fontsize=7, ncol=2, loc="lower right")

    labels = list(naive_sweep)
    x = np.arange(len(labels))
    axes[2].bar(x - 0.18, [naive_sweep[k][1, 0] for k in labels], width=0.36,
                color="#1f77b4", label="A[1,0]  (real link, true +0.6)")
    axes[2].bar(x + 0.18, [naive_sweep[k][0, 1] for k in labels], width=0.36,
                color="#d62728", label="A[0,1]  (absent link, true 0)")
    axes[2].axhline(0.0, color="k", lw=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(["slow/fast", "uniform", "reversed"], fontsize=8)
    axes[2].set_ylabel("naive inferred coupling")
    axes[2].set_title("Same A — only the HRFs changed", fontsize=10)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "lag_confound.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_recovery(outdir):
    """Act 3: the recovered latent trajectory and the connectivity matrices."""
    fig = plt.figure(figsize=(13, 4.2))
    ax0 = fig.add_subplot(1, 3, 1)
    show = slice(int(60 / DT), int(200 / DT))
    for r, c in zip(range(R), ["#1f77b4", "#d62728", "#2ca02c"]):
        ax0.plot(t_grid[show], Z_TRUE[show, r], color=c, lw=2.4, alpha=0.45)
        ax0.plot(t_grid[show], Z_REC[show, r], color=c, lw=1.0, ls="--")
    ax0.plot([], [], color="k", lw=2.4, alpha=0.45, label="true $z$")
    ax0.plot([], [], color="k", lw=1.0, ls="--", label="recovered $z$")
    ax0.set_xlabel("time (s)")
    ax0.set_ylabel("neural $z$")
    ax0.set_title(f"Latent trajectory, $R^2$={min(z_r2):.5f}", fontsize=10)
    ax0.legend(fontsize=8)

    # clipped to the true scale: the naive matrix runs to ~3 and would otherwise
    # flatten every real coefficient to the same colour.  Cells are annotated.
    vmax = float(np.abs(A_TRUE).max())
    for k, (M, ttl) in enumerate([(A_naive, "naive on BOLD"),
                                  (A_joint, "observation operator in-loop")]):
        ax = fig.add_subplot(1, 3, k + 2)
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for i in range(R):
            for j in range(R):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="k")
                if A_TRUE[i, j] != 0:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                               edgecolor="k", lw=2.0))
        ax.set_xticks(range(R)); ax.set_yticks(range(R))
        ax.set_xlabel("from region"); ax.set_ylabel("to region")
        ax.set_title(ttl, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Boxed cells are the true connections", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "connectivity_recovery.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_edges(outdir):
    fig, _ = plot_all_edges(
        m1.model_, X=torch.tensor(X1[:2048], dtype=torch.float32),
        in_var_names=["z", "s", "f", "v"],
        out_var_names=EQ_NAMES[:3],
        save=str(outdir / "edges_hemodynamic"),
    )
    plt.close(fig)
    fig, _ = plot_all_edges(
        m3.model_, X=torch.tensor(X3[:2048], dtype=torch.float32),
        in_var_names=NEURAL_NAMES,
        out_var_names=["dz0/dt", "dz1/dt", "dz2/dt"],
        save=str(outdir / "edges_neural"),
    )
    plt.close(fig)


use_pub_style()
plot_generative_model(RESULT_DIR)
plot_hemodynamic_constants(RESULT_DIR)
plot_lag_confound(RESULT_DIR)
plot_recovery(RESULT_DIR)
plot_edges(RESULT_DIR)
print(f"[FIGS]  Saved {RESULT_DIR}/")
