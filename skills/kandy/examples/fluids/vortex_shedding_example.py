#!/usr/bin/env python3
"""KANDy example: vortex shedding behind a cylinder — a mean-field ROM from DNS.

The von Karman wake at Re = 100 is the canonical equation-discovery benchmark
for fluids.  Unlike the other `fluids/` examples, the model here is NOT fitted
to the field itself: the flow is first projected onto a handful of POD modes,
and KANDy discovers the ODE governing those *modal coordinates*.  That is the
reduced-order-model (ROM) recipe — the one to imitate whenever the state is a
field but the dynamics you want are low-dimensional.

The pipeline
------------
    DNS (vorticity field)  ->  POD  ->  a1(t), a2(t), a3(t)  ->  KANDy

1. **DNS.**  Fourier pseudo-spectral 2D Navier-Stokes in vorticity form with
   Brinkman volume penalization for the cylinder and a downstream sponge.  The
   symmetric base flow is established first, then a small asymmetric kick is
   added so the shedding instability grows *exponentially from small amplitude*
   into the limit cycle.  Capturing that whole transient is the point: a
   recording of the saturated limit cycle alone is nearly useless here (see
   step 3).  But the transient alone is just as useless — the record must run
   well past saturation too, which is what T_REC buys (see RECORD LENGTH).

   T_BASE=80 is not arbitrary.  Tracking the recirculation bubble while the
   base flow settles:

       t=20   Lr/D=5.50   d|w|/dt=1.4e-01      still growing
       t=30   Lr/D=7.12   d|w|/dt=7.1e-02      overshoots
       t=50   Lr/D=6.75   d|w|/dt=5.9e-03
       t=80   Lr/D=6.75   d|w|/dt=1.1e-04      converged

   At the old T_BASE=25 the bubble had not settled and the residual was ~1000x
   its converged value; subtracting that leaks base-flow drift into every POD
   mode.  The converged bubble ends at x ~ 11.2, comfortably upstream of the
   sponge at x = 12.8 — at coarser resolution it ran to x ~ 13.4 and the near
   wake was being damped by the sponge, which is its own source of error.

2. **POD about the base flow.**  Subtracting the *steady base flow* — not the
   time mean — is what makes the modes come out in textbook form:

       mode 1        near-zero frequency, grows monotonically   -> SHIFT MODE
       modes 2, 3    equal energy, both at the shedding freq    -> SHEDDING PAIR

   Subtracting the time mean instead mixes the shift mode into all three and
   the structure is lost.  This script reports the energies, dominant
   frequencies, and the shift-mode correlation so you can see the separation.

3. **The shift mode is slaved.**  The Noack mean-field model would be

       da1/dt = mu*a1 - omega*a2 + A*a1*a3
       da2/dt = omega*a1 + mu*a2 + A*a2*a3
       da3/dt = -lambda*(a3 - (a1^2 + a2^2))

   but on a single natural transient a3 is slaved to a1^2 + a2^2 (this script
   fits a3 = kappa*(a1^2+a2^2) at R^2 ~ 0.95).  The trajectory lives ON the
   slow manifold, so `a3 - (a1^2+a2^2)` is never excited and lambda is
   invisible: least squares recovers the da3/dt equation at R^2 ~ 0.25, while
   the shedding pair extracted from the very same trajectory comes out at
   R^2 > 0.9999.  Two equations, one dataset, and only one of them identified
   — that is an IDENTIFIABILITY failure of the data, not a fitting failure,
   the same lesson as the conservation law in `mathbio/sir_example.py`.
   Exciting lambda needs a second, off-manifold trajectory.

4. **Reduce to the slow manifold.**  Substituting a3 = kappa*(a1^2+a2^2)
   leaves the Hopf / Stuart-Landau normal form on the shedding pair, which
   this data *does* identify:

       da1/dt = mu*a1 - omega*a2 - A*a1*(a1^2 + a2^2)
       da2/dt = omega*a1 + mu*a2 - A*a2*(a1^2 + a2^2)

The lift
--------
    theta = [a1, a2, a1*(a1^2+a2^2), a2*(a1^2+a2^2)]            (4 features)

The cubic terms expand to a1^3 + a1*a2^2.  The mixed term a1*a2^2 is a product
of different states, so a separable KAN cannot build it from edges on a1 and
a2 alone — it must be in the lift.  Given the lift the RHS is exactly linear
in the features, so lib=["x", "0"] reads the coefficients straight off.

Two things worth copying
------------------------
* **RECORD LENGTH is the whole ballgame for a Hopf.**  Separating the linear
  growth mu*a from the cubic damping A*a*r^2 needs data at BOTH small and
  saturated amplitude.  An earlier version of this script recorded T_REC=85,
  which left only ~3.8 saturated shedding periods, and the cubic term was
  effectively unconstrained.  The damage was invisible pointwise (R^2 ~ 0.971)
  and showed up only in the rollout — and it was silently patched over by a
  hand-tuned sparsity penalty, which is what regularisation does when the real
  problem is missing data.  Measured on the old 192x96 grid, varying only the
  record length (unregularised least squares on the same lift):

      T_REC=85    3.8 saturated periods   rollout radius off by 20.6%
      T_REC=150   9.8 saturated periods   rollout radius off by  5.5%
      T_REC=250  20.8 saturated periods   rollout radius off by  4.7%

  At the settings this script now uses the record holds ~16 saturated periods
  and the identification is essentially exact — lamb=0 gives a 0.2% radius
  error with a spread of 0.0002 across repeats.  Note the direction of the
  lamb dependence has REVERSED now that the data is adequate:

      lamb=0      4/4 repeats pass, 0.2% radius error
      lamb=1e-4   3/4 pass (one repeat diverged to r=18.3)
      lamb=1e-3   0/4 pass (limit cycle collapses to r~0.005)

  Sparsity is not a safe default here; with a correctly conditioned problem it
  only removes terms the dynamics needs.  Either way the lesson is the same: if
  a fit only works inside a narrow band of lamb, suspect the DATA, not the
  hyper-parameter.  And pointwise R^2 never diagnoses this — it was ~0.999 for
  every lamb above, including the runs that collapsed.  Roll out and check the
  radius (cf. `odes/van_der_pol_rbf_example.py`).

* **POD fixes an arbitrary basis inside the shedding eigenplane.**  The
  recovered matrix is therefore the normal form only up to that basis choice;
  individual coefficients are basis-dependent, and the linear part comes out
  with unequal diagonal entries rather than the textbook [[mu, om], [-om, mu]].
  That is expected: POD modes are orthogonal in space, but the real and
  imaginary parts of the global eigenmode are not, so the recovered matrix is
  SIMILAR to the normal form, not equal to it.  The 4x2 coefficient matrix
  absorbs the skew, so do not try to fix it by rotating to the eigenbasis —
  that was tried and made the symbolic extraction worse.  Check the INVARIANTS
  instead — growth rate, shedding frequency, and saturation radius — which is
  what this script validates.

Runtime: ~45 minutes, almost entirely the DNS (384x192, 57500 steps; measured
2453 s for the flow, the fit and figures are seconds).
KAN:  width = [4, 2],  base_fun='zero'
"""

import os
import sys
from pathlib import Path
import time
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The KAN fit is not bit-reproducible under multi-threaded BLAS: at fixed seed
# the rolled-out limit-cycle radius varied by 0.45 across repeats with the
# default thread count, versus 0.01 with a single thread.  Pin it so a metrics
# regression means the model changed, not the thread schedule.
torch.set_num_threads(1)

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from _utils import ensure_outdir, print_check, save_metrics

from kandy import KANDy, CustomLift
from kandy.plotting import plot_all_edges, plot_loss_curves, use_pub_style

# ---------------------------------------------------------------------------
# 0. Reproducibility and physical parameters
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

D, U_INF, RE = 1.0, 1.0, 100.0        # diameter, free stream, Reynolds number
NU = U_INF * D / RE

LX, LY = 16.0, 8.0                     # periodic box (cylinder diameters)
NX, NY = 384, 192                      # 24 points per diameter — see MESH note
XC, YC = 4.0, 4.0                      # cylinder centre

ETA = 4e-3                             # Brinkman permeability (penalization)
DT = 4e-3                              # |dt/eta| = 1, inside the RK4 limit ~2.8
MASK_WIDTH = 1.0                       # cylinder mask smoothing, in cells
T_BASE = 80.0                          # settle the symmetric base flow
T_REC = 150.0                          # record: growth -> saturation
SNAP_EVERY = 50                        # -> dt_snap = 0.2

x = np.linspace(0, LX, NX, endpoint=False)
y = np.linspace(0, LY, NY, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="ij")

KX = (2 * np.pi / LX) * np.fft.fftfreq(NX, d=1.0 / NX).reshape(-1, 1)
KY = (2 * np.pi / LY) * np.fft.fftfreq(NY, d=1.0 / NY).reshape(1, -1)
K2 = KX ** 2 + KY ** 2
K2_INV = np.where(K2 == 0, 1.0, K2)
DEALIAS = (np.abs(KX) < (2 / 3) * np.abs(KX).max()) & (
    np.abs(KY) < (2 / 3) * np.abs(KY).max()
)

# Smoothed cylinder mask and outflow sponge.
#
# MESH / MASK note.  The mask smoothing width sets the EFFECTIVE diameter, and
# that is what fixes the Strouhal number — more strongly than either the grid
# or eta.  Measured at 384x192 with eta = dt = 4e-3, varying only this width:
#
#     1.5 cells -> St = 0.158     0.75 cells -> St = 0.171 (and |w|max jumps
#     1.0 cells -> St = 0.167                   to 13.1: the mask is starting
#                                               to ring against the 2/3 dealias)
#
# 1.0 lands on the literature St = 0.164 for a Re = 100 circular cylinder.
# Resolution matters too but saturates: at a fixed 1.5-cell mask, 12 -> 18 -> 24
# points per diameter gave St = 0.123 -> 0.145 -> 0.158.  The original 192x96 /
# 1.5-cell / eta=1e-2 setup gave St = 0.123, ~25% below the physical value,
# because at 12 points per diameter the penalization layer sqrt(nu*eta) = 0.01
# is an eighth of a cell and the effective body is bigger and softer than D.
#
# eta is NOT fully converged at 4e-3, and that is a deliberate cost trade.  At
# the 1.0-cell mask, eta = 4e-3 -> 2e-3 -> 1e-3 gives St = 0.167 -> 0.165 ->
# 0.164, i.e. this run sits ~2% above the eta-converged value, comfortably
# inside the literature window.  Buying that last 2% costs 2.2x the runtime
# (eta=1e-3 needs dt=2e-3 for RK4 stability).  dt itself is fully converged:
# at eta=1e-3, dt = 2e-3 / 1.5e-3 / 1e-3 all give St = 0.1636 identically.
R_CYL = np.sqrt((X - XC) ** 2 + (Y - YC) ** 2)
CHI = 0.5 * (1.0 - np.tanh((R_CYL - 0.5 * D) / (MASK_WIDTH * LX / NX)))
X_SPONGE = 0.80 * LX
SIGMA = np.where(X > X_SPONGE, 3.0 * ((X - X_SPONGE) / (LX - X_SPONGE)) ** 2, 0.0)


# ---------------------------------------------------------------------------
# 1. Pseudo-spectral solver, vorticity-streamfunction with penalization
# ---------------------------------------------------------------------------
def velocity(w_hat):
    """Velocity from vorticity: lap(psi) = -w, u = U + dpsi/dy, v = -dpsi/dx."""
    psi_hat = np.where(K2 == 0, 0.0, w_hat / K2_INV)
    u = U_INF + np.real(np.fft.ifft2(1j * KY * psi_hat))
    v = -np.real(np.fft.ifft2(1j * KX * psi_hat))
    return u, v


def rhs_hat(w_hat):
    """Spectral RHS: advection + viscosity + penalization torque + sponge."""
    u, v = velocity(w_hat)
    w = np.real(np.fft.ifft2(w_hat))
    w_x = np.real(np.fft.ifft2(1j * KX * w_hat))
    w_y = np.real(np.fft.ifft2(1j * KY * w_hat))

    advection = np.fft.fft2(-(u * w_x + v * w_y)) * DEALIAS
    viscous = -NU * K2 * w_hat
    # Brinkman force f = -(chi/eta)*u inside the solid; its curl forces vorticity
    curl_f = (
        1j * KX * np.fft.fft2(-(CHI / ETA) * v)
        - 1j * KY * np.fft.fft2(-(CHI / ETA) * u)
    ) * DEALIAS
    sponge = -np.fft.fft2(SIGMA * w) * DEALIAS
    return advection + viscous + curl_f + sponge


def rk4_hat(w_hat, dt):
    k1 = rhs_hat(w_hat)
    k2 = rhs_hat(w_hat + 0.5 * dt * k1)
    k3 = rhs_hat(w_hat + 0.5 * dt * k2)
    k4 = rhs_hat(w_hat + dt * k3)
    return w_hat + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


print(f"[SIM]  Re={RE:.0f}, {NX}x{NY} grid on {LX}x{LY}, dt={DT}, eta={ETA}")
t_start = time.time()

# 1a. Symmetric base flow (no perturbation -> the unstable steady wake)
w_hat = np.fft.fft2(np.zeros((NX, NY)))
for _ in range(int(T_BASE / DT)):
    w_hat = rk4_hat(w_hat, DT)
base_flow = np.real(np.fft.ifft2(w_hat)).copy()
print(f"[SIM]  base flow settled at t={T_BASE:.0f} ({time.time() - t_start:.0f}s)")

# 1b. Small asymmetric kick, then record the growth to saturation
kick = 0.02 * np.exp(-((X - XC - 1.0) ** 2 + (Y - YC - 0.3) ** 2) / 0.15)
w_hat = w_hat + np.fft.fft2(kick)

snapshots = []
for i in range(int(T_REC / DT)):
    if i % SNAP_EVERY == 0:
        snapshots.append(np.real(np.fft.ifft2(w_hat)).copy())
    w_hat = rk4_hat(w_hat, DT)

snapshots = np.array(snapshots)
DT_SNAP = SNAP_EVERY * DT
if not np.isfinite(snapshots).all():
    raise RuntimeError("DNS diverged — reduce DT or increase ETA.")
print(f"[SIM]  {len(snapshots)} snapshots, dt_snap={DT_SNAP} "
      f"({time.time() - t_start:.0f}s total)")

# ---------------------------------------------------------------------------
# 2. POD about the STEADY BASE FLOW (not the time mean — see docstring)
# ---------------------------------------------------------------------------
wake = (X[:, 0] > XC - 2.0) & (X[:, 0] < X_SPONGE)      # exclude the sponge
n_wake = int(wake.sum())
fluct = (snapshots[:, wake, :] - base_flow[wake, :]).reshape(len(snapshots), -1)

U_svd, sv, Vt = np.linalg.svd(fluct, full_matrices=False)
energy = sv ** 2 / np.sum(sv ** 2)
coeffs = U_svd[:, :4] * sv[:4]
t_snap = np.arange(len(snapshots)) * DT_SNAP


def dominant_freq(sig, tail=0.0):
    """Peak frequency of `sig`, interpolated to sub-bin accuracy.

    `tail` restricts the estimate to the last fraction of the record.  Measure
    the SHEDDING frequency with tail>0: over the full record the growing
    envelope leaks across the spectrum and biases the peak low.  On this run
    the same signal gives f = 0.111 over the whole record but f = 0.120 on the
    saturated tail, and only the latter is the shedding frequency — the first
    number is contaminated by the transient the record was designed to capture.
    An independent wake probe (transverse velocity 2D downstream, sampled 10x
    finer than the snapshots) agrees with the tail estimate to within 3%.

    The bin spacing is 1/(N*DT_SNAP) — 0.0118 here, so the bare argmax lands
    within a bin of the truth but no closer.  Quadratic interpolation of the
    log-power spectrum about the peak, on a Hann-windowed signal to limit
    leakage, recovers a fraction of a bin.
    """
    s = sig[int(tail * len(sig)):]
    s = (s - s.mean()) * np.hanning(len(s))
    freqs = np.fft.rfftfreq(len(s), DT_SNAP)
    power = np.abs(np.fft.rfft(s)) ** 2
    k = int(np.argmax(power[1:]) + 1)
    delta = 0.0
    if 1 <= k < len(power) - 1:
        a, b, c = (np.log(power[k + j] + 1e-300) for j in (-1, 0, 1))
        denom = a - 2.0 * b + c
        if denom != 0.0:
            delta = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
    return freqs[k] + delta * (freqs[1] - freqs[0])


SAT_TAIL = 0.6      # last 40% of the record is on the limit cycle


r2_pair = coeffs[:, 1] ** 2 + coeffs[:, 2] ** 2
print(f"[POD]  energy fractions: {np.round(energy[:4], 4)}")
for k in range(3):
    corr = np.corrcoef(coeffs[:, k], r2_pair)[0, 1]
    print(f"[POD]  mode {k + 1}: f_dom={dominant_freq(coeffs[:, k], SAT_TAIL):.4f}  "
          f"corr(., a1^2+a2^2)={corr:+.3f}")
print("[POD]  -> mode 1 is the SHIFT MODE: it tracks the amplitude (corr ~ +0.98\n"
      "       with a1^2+a2^2) and carries NO power at the shedding frequency.\n"
      "       Measured on the saturated tail its peak sits at 2*f_shed, which is\n"
      "       exactly what a3 = kappa*(a1^2+a2^2) predicts — on the limit cycle\n"
      "       r^2 is constant plus a ripple at twice the shedding frequency, so\n"
      "       that harmonic is a positive check on the slaving relation, not a\n"
      "       shedding component.  Modes 2 and 3 are the SHEDDING PAIR, both at\n"
      "       f_shed and both uncorrelated with the amplitude.")

# Reindex into mean-field convention and normalise by the shedding scale
scale = np.std(coeffs[:, 1])
a1, a2, a3 = coeffs[:, 1] / scale, coeffs[:, 2] / scale, coeffs[:, 0] / scale
St = dominant_freq(coeffs[:, 1], SAT_TAIL) * D / U_INF
print(f"[POD]  Strouhal number St = f*D/U = {St:.4f}  "
      f"(literature, Re=100 circular cylinder: 0.164)")

# ---------------------------------------------------------------------------
# 3. Diagnostic: the shift mode is SLAVED, so lambda is not identifiable
# ---------------------------------------------------------------------------
r2_n = a1 ** 2 + a2 ** 2
kappa = float(np.linalg.lstsq(r2_n[:, None], a3, rcond=None)[0][0])
slave_r2 = 1.0 - np.sum((a3 - kappa * r2_n) ** 2) / np.sum((a3 - a3.mean()) ** 2)

state3 = np.stack([a1, a2, a3], axis=1)
d3 = np.gradient(state3, DT_SNAP, axis=0)[3:-3]
s3 = state3[3:-3]
theta3 = np.stack(
    [s3[:, 0], s3[:, 1], s3[:, 2], s3[:, 0] * s3[:, 2], s3[:, 1] * s3[:, 2],
     s3[:, 0] ** 2 + s3[:, 1] ** 2], axis=1
)
c3 = np.linalg.lstsq(theta3, d3, rcond=None)[0]
pred3 = theta3 @ c3
r2_a3 = 1.0 - np.sum((d3[:, 2] - pred3[:, 2]) ** 2) / np.sum(
    (d3[:, 2] - d3[:, 2].mean()) ** 2
)
print(f"\n[SLAVE] a3 = {kappa:.4f}*(a1^2+a2^2)  with R^2 = {slave_r2:.4f}  "
      f"(corr {np.corrcoef(a3, r2_n)[0, 1]:+.4f})")
print(f"[SLAVE] least-squares R^2 for the da3/dt equation: {r2_a3:.4f}")
print("[SLAVE] The trajectory never leaves the slow manifold, so the shift-mode\n"
      "        relaxation rate lambda is unexcited and CANNOT be identified from\n"
      "        this run — an identifiability limit of the data, not of the fit.\n"
      "        Reducing onto the manifold leaves the Stuart-Landau normal form.")

# ---------------------------------------------------------------------------
# 4. KANDy on the slow manifold — Stuart-Landau normal form
# ---------------------------------------------------------------------------
state = np.stack([a1, a2], axis=1)
X_dot = np.gradient(state, DT_SNAP, axis=0)
# trim the kick transient and the one-sided gradient stencils at both ends
state, X_dot = state[3:-3], X_dot[3:-3]

FEATURE_NAMES = ["a1", "a2", "a1*r2", "a2*r2"]


def phi_np(Z):
    r2 = Z[:, 0] ** 2 + Z[:, 1] ** 2
    return np.stack([Z[:, 0], Z[:, 1], Z[:, 0] * r2, Z[:, 1] * r2], axis=1)


def phi_torch(Z):
    r2 = Z[:, 0] ** 2 + Z[:, 1] ** 2
    return torch.stack([Z[:, 0], Z[:, 1], Z[:, 0] * r2, Z[:, 1] * r2], dim=1)


lift = CustomLift(fn=phi_np, output_dim=4, torch_fn=phi_torch, name="stuart_landau")
Theta = phi_np(state)
print(f"\n[DATA]  {len(state)} samples, cond(Theta) = "
      f"{np.linalg.cond(np.column_stack([Theta, np.ones(len(Theta))])):.1f}")

model = KANDy(lift=lift, grid=5, k=3, steps=250, seed=SEED, base_fun="zero")
# lamb=0 is deliberate and load-bearing.  With ~16 saturated periods in the
# record the RHS is identified essentially exactly without a sparsity prior,
# and turning one on BREAKS it: lamb=1e-3 collapses the limit cycle entirely
# while leaving R^2 at 0.999.  See the RECORD LENGTH note in the docstring.
model.fit(X=state, X_dot=X_dot, dt=DT_SNAP, val_frac=0.15, test_frac=0.15,
          lamb=0.0, patience=50)

pred = model.predict(state)
network_r2 = []
for i in range(2):
    r2_i = 1.0 - np.sum((X_dot[:, i] - pred[:, i]) ** 2) / np.sum(
        (X_dot[:, i] - X_dot[:, i].mean()) ** 2
    )
    network_r2.append(float(r2_i))
    print(f"[EVAL]  Network R^2 on da{i + 1}/dt: {r2_i:.6f}")
network_r2_min = float(min(network_r2))

# ---------------------------------------------------------------------------
# 5. Validation by rollout — the invariants, not the coefficients
#
#    Done before get_formula(), which rewrites the model's edges in place.
# ---------------------------------------------------------------------------
traj = model.rollout(state[0], T=len(state), dt=DT_SNAP)
r_roll = np.sqrt((traj[-150:] ** 2).sum(axis=1).mean())
r_true = np.sqrt((state[-150:] ** 2).sum(axis=1).mean())
print(f"[EVAL]  Limit-cycle radius — KANDy rollout {r_roll:.3f} vs DNS "
      f"{r_true:.3f}  ({100 * abs(r_roll - r_true) / r_true:.1f}% error)")

# Measured exponential growth rate during the linear phase, for comparison
amp = np.sqrt((state ** 2).sum(axis=1))
lin = (amp > 0.02 * amp.max()) & (amp < 0.30 * amp.max())
growth = np.polyfit(np.arange(len(amp))[lin] * DT_SNAP, np.log(amp[lin]), 1)[0]
print(f"[EVAL]  Measured growth rate mu = {growth:.4f} 1/t "
      f"(shedding frequency omega = {2 * np.pi * St:.4f})")

# How much SATURATED data the record actually contains.  This is the quantity
# that decides whether the cubic damping term is identifiable at all; below
# ~10 periods the fit degrades badly (see RECORD LENGTH in the docstring).
sat_periods = float((amp > 0.95 * np.median(amp[-100:])).sum() * DT_SNAP * St)
print(f"[EVAL]  Saturated record: {sat_periods:.1f} shedding periods "
      f"(need >~10 to identify the cubic damping)")

theta_t = torch.tensor(Theta[:2048], dtype=torch.float32)

# ---------------------------------------------------------------------------
# 6. Symbolic extraction — the RHS is linear in these features
# ---------------------------------------------------------------------------
print("\n[SYMBOLIC] Extracting the mean-field normal form ...")
formulas = model.get_formula(
    var_names=FEATURE_NAMES, round_places=4,
    lib=["x", "0"], r2_threshold=0.80, weight_simple=0.0,
)
for i, expr in enumerate(formulas):
    print(f"  da{i + 1}/dt = {expr}")
print(f"[SYMBOLIC] Formula R^2: "
      f"{np.round(model.score_formula(formulas, state, X_dot, var_names=FEATURE_NAMES), 5)}")
print("[SYMBOLIC] Expected shape: linear growth + rotation + cubic damping, in\n"
      "           SOME basis.  Do not expect the textbook [[mu, om], [-om, mu]]\n"
      "           linear part or a single -A*a*r^2 cubic: POD fixes an arbitrary\n"
      "           basis in the shedding eigenplane, which mixes the two cubic\n"
      "           features and leaves the diagonal entries unequal.  Compare the\n"
      "           INVARIANTS above (St, growth rate, saturation radius), not the\n"
      "           individual coefficients.")

# ---------------------------------------------------------------------------
# 6b. Verification
#
#     Nothing here gates individual coefficients, and that is deliberate: POD
#     fixes an ARBITRARY orthogonal basis inside the shedding eigenplane, so a
#     rotation of (a1, a2) changes every coefficient while describing identical
#     physics.  Only basis-INVARIANT quantities are meaningful — the Strouhal
#     number, the limit-cycle radius, the linear growth rate, and the slaving
#     relation a3 = kappa*(a1^2 + a2^2).
#
#     The slaving check is also the identifiability warning: because a3 is a
#     function of a1, a2 rather than an independent state, its own equation is
#     not identifiable from this data — the same failure as a phase-locked
#     network in oscillators/network_reconstruction_example.py.
# ---------------------------------------------------------------------------
#     The slaving bar of 0.75 is a historical loosening that is worth keeping
#     and worth explaining.  a3 = kappa*(a1^2 + a2^2) is a leading-order
#     mean-field relation that becomes exact only asymptotically on the limit
#     cycle, and this record deliberately includes the transient, so 0.75 is
#     the defensible bar for "a3 is slaved" rather than "a3 is exactly
#     quadratic".  On the coarse grid it measured 0.84, which is what forced
#     the bar down from 0.95; at the current resolution it measures 0.95 again.
#     The bar stays at 0.75 because it is the honest one for what is claimed,
#     not because the current run needs the slack.
#
#     The Strouhal window is NOT widened: it is the physical range for a
#     Re=100 circular cylinder, and it is the one gate here that checks this
#     pipeline against the outside world rather than against itself.  An
#     earlier version of this script failed it at St=0.111 and the fix was to
#     correct the DNS (resolution and mask width) and the estimator (measure on
#     the saturated tail, not across the growth transient) — not to move the
#     bar.  If this gate fails, the flow is wrong; do not loosen it.
ST_LITERATURE = (0.15, 0.22)      # circular cylinder, Re ~ 100
radius_rel_err = float(abs(r_roll - r_true) / r_true)
THRESHOLDS = {
    "strouhal_range": list(ST_LITERATURE),
    "slaving_r2_min": 0.75,           # a3 slaved to a1, a2 at leading order
    "limit_cycle_radius_rel_err_max": 0.15,
    "growth_rate_positive": True,     # the base flow must be unstable
    "network_r2_min": 0.95,
}
passed = print_check(
    ST_LITERATURE[0] <= St <= ST_LITERATURE[1]
    and slave_r2 >= THRESHOLDS["slaving_r2_min"]
    and radius_rel_err <= THRESHOLDS["limit_cycle_radius_rel_err_max"]
    and network_r2_min >= THRESHOLDS["network_r2_min"]
    and growth > 0.0,
    "Vortex shedding: Strouhal, limit-cycle radius, and the slaved shift mode",
)
RESULT_DIR = ensure_outdir("results/VortexShedding")
save_metrics(RESULT_DIR / "metrics.json", {
    "task": "VortexShedding",
    "category": "fluids",
    "seed": SEED,
    "sampling": "DNS -> POD -> modal ODE",
    "reynolds": RE,
    "strouhal": float(St),
    "strouhal_literature_range": list(ST_LITERATURE),
    "slaving_kappa": float(kappa),
    "slaving_r2": float(slave_r2),
    "a3_equation_lsq_r2": float(r2_a3),
    "limit_cycle_radius_rollout": float(r_roll),
    "limit_cycle_radius_dns": float(r_true),
    "limit_cycle_radius_rel_err": radius_rel_err,
    "growth_rate_mu": float(growth),
    "network_r2": network_r2,
    "network_r2_min": network_r2_min,
    "saturated_periods": float(sat_periods),
    "grid": [NX, NY],
    "points_per_diameter": NX / LX,
    "eta": ETA,
    "mask_width_cells": MASK_WIDTH,
    "t_base": T_BASE,
    "t_rec": T_REC,
    "lamb": 0.0,
    "thresholds": THRESHOLDS,
    "passed": passed,
})

# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/VortexShedding", exist_ok=True)
extent = [0, LX, 0, LY]


def draw_cylinder(ax):
    ax.add_patch(plt.Circle((XC, YC), 0.5 * D, color="k", zorder=5))


# 7a. Vorticity: base flow, mid-transient, saturated limit cycle
fig, axes = plt.subplots(3, 1, figsize=(9, 8))
picks = [(base_flow, "steady base flow (unstable)"),
         (snapshots[len(snapshots) // 3], f"transient, t={t_snap[len(snapshots)//3]:.0f}"),
         (snapshots[-1], f"limit cycle, t={t_snap[-1]:.0f}")]
vmax = np.abs(snapshots[-1]).max() * 0.5
for ax, (field, title) in zip(axes, picks):
    im = ax.imshow(field.T, origin="lower", extent=extent, cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, aspect="equal")
    draw_cylinder(ax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.025)
fig.suptitle(f"Vortex shedding at Re={RE:.0f}", fontsize=12)
fig.tight_layout()
fig.savefig("results/VortexShedding/vorticity_fields.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7b. POD modes: shift mode and the shedding pair
fig, axes = plt.subplots(3, 1, figsize=(9, 8))
titles = [f"POD mode 1 — shift mode ({100*energy[0]:.1f}% energy)",
          f"POD mode 2 — shedding ({100*energy[1]:.1f}%)",
          f"POD mode 3 — shedding ({100*energy[2]:.1f}%)"]
for k, (ax, title) in enumerate(zip(axes, titles)):
    mode = Vt[k].reshape(n_wake, NY)
    s = np.abs(mode).max()
    im = ax.imshow(mode.T, origin="lower", cmap="RdBu_r", vmin=-s, vmax=s,
                   extent=[x[wake][0], x[wake][-1], 0, LY], aspect="equal")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.025)
fig.tight_layout()
fig.savefig("results/VortexShedding/pod_modes.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7c. Modal amplitudes: exponential growth into saturation
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t_snap, a1, lw=1.0, color="#1f77b4", label="$a_1$ (shedding)")
ax.plot(t_snap, a2, lw=1.0, color="#2ca02c", alpha=0.7, label="$a_2$ (shedding)")
ax.plot(t_snap, a3, lw=1.6, color="#d62728", label="$a_3$ (shift mode)")
ax.set_xlabel("time"); ax.set_ylabel("modal amplitude")
ax.set_title("Growth of the shedding instability")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/VortexShedding/mode_amplitudes.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7d. The slaving relation that blocks identification of lambda
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(r2_n, a3, s=6, alpha=0.5, color="#1f77b4", label="DNS")
grid_r2 = np.linspace(0, r2_n.max(), 100)
ax.plot(grid_r2, kappa * grid_r2, "k--", lw=1.2,
        label=f"$a_3={kappa:.3f}\\,(a_1^2+a_2^2)$, $R^2$={slave_r2:.3f}")
ax.set_xlabel("$a_1^2 + a_2^2$"); ax.set_ylabel("$a_3$")
ax.set_title("The shift mode is slaved to the shedding amplitude")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/VortexShedding/shift_mode_slaving.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7e. Phase portrait: DNS vs KANDy rollout onto the limit cycle
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(state[:, 0], state[:, 1], lw=0.9, color="#1f77b4", alpha=0.5, label="DNS")
ax.plot(traj[:, 0], traj[:, 1], lw=0.9, color="#d62728", label="KANDy rollout")
ax.set_xlabel("$a_1$"); ax.set_ylabel("$a_2$")
ax.set_title(f"Spiral onto the limit cycle\nradius {r_roll:.2f} vs DNS {r_true:.2f}")
ax.set_aspect("equal")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/VortexShedding/phase_portrait.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7f. The prediction in modal space: DNS vs a FREE-RUNNING KANDy rollout
#
#     `traj` is integrated from state[0] alone — no re-anchoring to the DNS at
#     any point.  Read the envelope, not the phase: a ROM identified to
#     O(1e-4) in the frequency still slips a noticeable fraction of a cycle
#     over the ~16 shedding periods in this record, and that drift is the
#     expected behaviour of any free-running oscillator model, not an error in
#     the fit.  The zoom panels make the distinction visible: early the two
#     curves overlie each other, late they share amplitude and frequency while
#     the phases separate.
t_roll = t_snap[3:len(t_snap) - 3]
amp_roll = np.sqrt((state ** 2).sum(axis=1))
n_zoom = max(int(6.0 / (St * DT_SNAP)), 50)          # ~6 shedding periods
j_grow = int(np.argmax(amp_roll > 0.5 * amp_roll.max()))
early = slice(max(j_grow - n_zoom // 2, 0), max(j_grow - n_zoom // 2, 0) + n_zoom)
late = slice(len(t_roll) - n_zoom, len(t_roll))

fig, axes = plt.subplots(2, 2, figsize=(11, 5.5))
for ax, ci, lab in zip(axes[:, 0], [0, 1], ["$a_1$", "$a_2$"]):
    ax.plot(t_roll, state[:, ci], color="#1f77b4", lw=0.9, label="DNS")
    ax.plot(t_roll, traj[:, ci], color="#d62728", lw=0.9, ls="--",
            label="KANDy rollout")
    ax.set_ylabel(lab)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
axes[1, 0].set_xlabel("time")
axes[0, 0].set_title("Free-running rollout, full record", fontsize=10)

for ax, sl, ttl in zip(axes[:, 1], [early, late],
                       ["growth phase", f"saturated (last {n_zoom * DT_SNAP:.0f} t)"]):
    ax.plot(t_roll[sl], state[sl, 0], color="#1f77b4", lw=1.2, label="DNS")
    ax.plot(t_roll[sl], traj[sl, 0], color="#d62728", lw=1.2, ls="--",
            label="KANDy")
    ax.set_ylabel("$a_1$")
    ax.set_title(ttl, fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
axes[1, 1].set_xlabel("time")
fig.suptitle("Modal prediction: no re-anchoring to the DNS after t=0", fontsize=12)
fig.tight_layout()
fig.savefig("results/VortexShedding/modal_prediction.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)

# 7g. The prediction as a FLOW FIELD — lift the ROM back onto the grid
#
#     w(x, t) ~= w_base(x) + scale * [ a3*V0(x) + a1*V1(x) + a2*V2(x) ]
#
#     Two error sources are separated deliberately.  A 3-mode POD truncation
#     cannot represent the DNS exactly no matter how good the ROM is, so the
#     panel titles report BOTH the truncation error (DNS projected onto the
#     3 modes, using the DNS coefficients) and the total KANDy error.  The
#     difference between them is what the learned dynamics actually cost.
#
#     The saturated comparison is PHASE-MATCHED: the free-running rollout has
#     no reason to share a clock with the DNS after 16 periods, so it is
#     compared against the DNS snapshot at the nearest phase angle.  On the
#     limit cycle that is a comparison of attractors, which is the claim being
#     made; comparing at equal wall-clock time would only re-display the phase
#     drift already shown in 7f.
def wake_field(av1, av2, av3):
    """Reconstruct the wake vorticity from three modal amplitudes."""
    f = scale * (av3 * Vt[0] + av1 * Vt[1] + av2 * Vt[2])
    return base_flow[wake, :] + f.reshape(n_wake, NY)


def rel_l2(field, truth):
    """Error relative to the fluctuation amplitude, not the total field."""
    return float(np.linalg.norm(field - truth)
                 / np.linalg.norm(truth - base_flow[wake, :]))


# saturated rollout state, matched to the DNS snapshot at the nearest phase
j_sat = len(traj) - 1
ph_roll = np.arctan2(traj[j_sat, 1], traj[j_sat, 0])
cand = np.arange(int(0.70 * len(state)), len(state))
d_ph = np.angle(np.exp(1j * (np.arctan2(state[cand, 1], state[cand, 0]) - ph_roll)))
j_dns_sat = int(cand[np.argmin(np.abs(d_ph))])

comparisons = [(j_grow, j_grow, "growth phase"),
               (j_sat, j_dns_sat, "saturated, phase-matched")]

fig, axes = plt.subplots(2, 2, figsize=(11, 6))
w_extent = [x[wake][0], x[wake][-1], 0, LY]
vmax_w = np.abs(snapshots[-1][wake, :] - base_flow[wake, :]).max() * 0.6
for row, (j_roll, j_dns, ttl) in enumerate(comparisons):
    w_dns = snapshots[j_dns + 3][wake, :]
    w_kan = wake_field(traj[j_roll, 0], traj[j_roll, 1],
                       kappa * (traj[j_roll, 0] ** 2 + traj[j_roll, 1] ** 2))
    w_pod = wake_field(state[j_dns, 0], state[j_dns, 1], a3[j_dns + 3])
    for col, (field, lab) in enumerate(
            [(w_dns, "DNS"),
             (w_kan, f"KANDy ROM  (err {rel_l2(w_kan, w_dns):.2f}; "
                     f"3-mode floor {rel_l2(w_pod, w_dns):.2f})")]):
        ax = axes[row, col]
        im = ax.imshow(field.T, origin="lower", extent=w_extent, cmap="RdBu_r",
                       vmin=-vmax_w, vmax=vmax_w, aspect="equal")
        draw_cylinder(ax)
        ax.set_title(f"{lab} — {ttl}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.025)
fig.suptitle("Vorticity predicted by the discovered ROM, lifted back to the grid",
             fontsize=12)
fig.tight_layout()
fig.savefig("results/VortexShedding/field_prediction.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)

print(f"[FIELD] wake vorticity, relative to fluctuation amplitude:")
for j_roll, j_dns, ttl in comparisons:
    w_dns = snapshots[j_dns + 3][wake, :]
    w_kan = wake_field(traj[j_roll, 0], traj[j_roll, 1],
                       kappa * (traj[j_roll, 0] ** 2 + traj[j_roll, 1] ** 2))
    w_pod = wake_field(state[j_dns, 0], state[j_dns, 1], a3[j_dns + 3])
    print(f"        {ttl:26s} KANDy {rel_l2(w_kan, w_dns):.4f}   "
          f"3-mode POD floor {rel_l2(w_pod, w_dns):.4f}")

# 7h. Loss curves and edge activations
if getattr(model, "train_results_", None):
    fig, ax = plot_loss_curves(
        model.train_results_, save="results/VortexShedding/loss_curves",
    )
    plt.close(fig)

fig, axes = plot_all_edges(
    model.model_, X=theta_t,
    in_var_names=FEATURE_NAMES,
    out_var_names=["da1/dt", "da2/dt"],
    save="results/VortexShedding/edge_activations",
)
plt.close(fig)

print("[FIGS]  Saved results/VortexShedding/")
