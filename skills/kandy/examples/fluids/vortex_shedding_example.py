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
   step 3).

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

   but on a single natural transient a3 tracks a1^2 + a2^2 almost exactly
   (this script measures corr ~ 0.96).  The trajectory lives ON the slow
   manifold, so `a3 - (a1^2+a2^2)` is never excited and lambda is invisible:
   least squares recovers the a3 equation with R^2 ~ 0.01.  That is an
   IDENTIFIABILITY failure of the data, not a fitting failure — the same
   lesson as the conservation law in `mathbio/sir_example.py`.  Exciting
   lambda needs a second, off-manifold trajectory.

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
* **lamb > 0 matters more than R^2 here.**  With lamb=0 the pointwise fit is
  just as good (R^2 ~ 0.988) but the rolled-out limit cycle is several times
  too large: on the limit cycle r is constant, so a1 and a1*r^2 are collinear
  and the split between linear growth and cubic damping is under-determined.
  Sparsity regularisation breaks the tie.  Pointwise R^2 does not validate a
  limit cycle — roll out and check the radius (cf. `odes/van_der_pol_rbf_example.py`).

* **POD fixes an arbitrary basis inside the shedding eigenplane.**  The
  recovered matrix is therefore the normal form only up to that basis choice;
  individual coefficients are basis-dependent.  Check the INVARIANTS instead —
  growth rate, shedding frequency, and saturation radius — which is what this
  script validates.

Runtime: ~4-5 minutes, dominated by the DNS.
KAN:  width = [4, 2],  base_fun='zero'
"""

import os
import time
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
NX, NY = 192, 96
XC, YC = 4.0, 4.0                      # cylinder centre

ETA = 1e-2                             # Brinkman permeability (penalization)
DT = 4e-3
T_BASE = 25.0                          # settle the symmetric base flow
T_REC = 85.0                           # record: growth -> saturation
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

# Smoothed cylinder mask and outflow sponge
R_CYL = np.sqrt((X - XC) ** 2 + (Y - YC) ** 2)
CHI = 0.5 * (1.0 - np.tanh((R_CYL - 0.5 * D) / (1.5 * LX / NX)))
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


def dominant_freq(sig):
    s = sig - sig.mean()
    freqs = np.fft.rfftfreq(len(s), DT_SNAP)
    power = np.abs(np.fft.rfft(s)) ** 2
    return freqs[np.argmax(power[1:]) + 1]


r2_pair = coeffs[:, 1] ** 2 + coeffs[:, 2] ** 2
print(f"[POD]  energy fractions: {np.round(energy[:4], 4)}")
for k in range(3):
    corr = np.corrcoef(coeffs[:, k], r2_pair)[0, 1]
    print(f"[POD]  mode {k + 1}: f_dom={dominant_freq(coeffs[:, k]):.4f}  "
          f"corr(., a1^2+a2^2)={corr:+.3f}")
print("[POD]  -> mode 1 is the SHIFT MODE (near-zero frequency, tracks the "
      "amplitude);\n       modes 2 and 3 are the SHEDDING PAIR at the "
      "shedding frequency.")

# Reindex into mean-field convention and normalise by the shedding scale
scale = np.std(coeffs[:, 1])
a1, a2, a3 = coeffs[:, 1] / scale, coeffs[:, 2] / scale, coeffs[:, 0] / scale
St = dominant_freq(coeffs[:, 1]) * D / U_INF
print(f"[POD]  Strouhal number St = f*D/U = {St:.4f}")

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
# lamb > 0 is essential — see the docstring; without it the rollout radius is
# several times too large even though the pointwise R^2 is unchanged.
model.fit(X=state, X_dot=X_dot, dt=DT_SNAP, val_frac=0.15, test_frac=0.15,
          lamb=1e-3, patience=50)

pred = model.predict(state)
for i in range(2):
    r2_i = 1.0 - np.sum((X_dot[:, i] - pred[:, i]) ** 2) / np.sum(
        (X_dot[:, i] - X_dot[:, i].mean()) ** 2
    )
    print(f"[EVAL]  Network R^2 on da{i + 1}/dt: {r2_i:.6f}")

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
print("[SYMBOLIC] Expected shape: linear growth + rotation + cubic damping.\n"
      "           The coefficients are basis-dependent (POD fixes an arbitrary\n"
      "           basis in the shedding eigenplane) — compare the invariants\n"
      "           above, not the individual numbers.")

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

# 7f. Loss curves and edge activations
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
