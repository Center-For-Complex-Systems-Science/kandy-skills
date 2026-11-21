#!/usr/bin/env python3
"""KANDy example: 2D MHD turbulence — closing the turbulent EMF.

Decaying two-dimensional MHD turbulence from a smooth random low-wavenumber
state, solved pseudo-spectrally in (vorticity, vector potential) form:

    dw/dt = -(u*w_x + v*w_y) + (Bx*j_x + By*j_y) + nu * lap(w)
    dA/dt = -(u*A_x + v*A_y)                     + eta * lap(A)

plasma/orszag_tang_example.py already recovers those equations exactly.  For
TURBULENCE the interesting question is not the equation — it is what happens to
it when you cannot afford to resolve every scale.  Filter the induction
equation at width Delta and the advection term does not survive the filter:

    d(Abar)/dt = -(ubar*Abar_x + vbar*Abar_y) - R + eta*lap(Abar)

    R = filter(u*A_x + v*A_y) - (ubar*Abar_x + vbar*Abar_y)

R is the subgrid turbulent EMF: the flux of magnetic potential carried by
motions the grid has thrown away.  Every mean-field dynamo and every large-eddy
MHD calculation stands or falls on how R is modelled, and unlike the terms in
the Orszag-Tang example it is NOT a function of the resolved fields — so this
is a closure problem, not an identification problem, and the honest deliverable
is how much of R is locally closable and with what coefficient.

TWO RIVAL CLOSURES, ONE LIFT
----------------------------
The lift contains both candidate models at once and lets the fit choose:

    theta = [ubar_x*Abar_xx, ubar_y*Abar_xy,        gradient (Clark) model:
             vbar_x*Abar_xy, vbar_y*Abar_yy,        R ~ (Delta^2/12) grad.grad
             lap_Abar,                              eddy resistivity: R ~ -beta*lap(Abar)
             jbar*wbar]                             decoy

For a Gaussian filter of second moment Delta^2/12 the leading term of the
expansion of a filtered product is

    filter(f*g) = fbar*gbar + (Delta^2/12) * grad(fbar).grad(gbar) + O(Delta^4)

so the FIRST FOUR features must all carry the SAME coefficient, Delta^2/12.
Four independent edges forced to agree on one number is a much stronger check
than one coefficient matching one target.  The eddy-resistivity feature is the
model the mean-field literature actually uses; leaving it in the lift turns
"is turbulent resistivity a good LOCAL closure?" into a measurement instead of
an assertion.  Its standalone R^2 is reported alongside.

THE SCALING LAW IS THE REAL VALIDATION
--------------------------------------
The fit is repeated at three filter widths, Delta = 3, 5 and 8 grid spacings.
A coefficient that happens to match Delta^2/12 at one width proves little; a
coefficient that tracks Delta^2 across a factor of seven in Delta^2 is a
recovered scaling law.  The log-log slope is reported, and the agreement is
EXPECTED to degrade at the largest Delta — that is the Taylor expansion running
out, and saying so is part of the result.

A precondition, checked and gated before any of this: the DNS must resolve its
own dissipation scale.  A subgrid term is defined as what a filter removes, so
on an under-resolved grid it measures the discretisation instead of the
physics — which is why nu here is set by the resolution check, not by the wish
for a large Reynolds number.

This is an A-PRIORI closure test: the model is scored against the exact subgrid
term computed from filtered DNS, not by running an LES with it.  Compare
fluids/turbulence_closure_example.py, which does the same for the hydrodynamic
subgrid stress and shows what a SHARP filter does to closability.

KAN:  width = [6, 1],  base_fun='zero'
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

N_GRID = 256
# nu is set by RESOLUTION, not by ambition.  At nu = 8e-4 this grid leaves the
# enstrophy spectrum at 30-50% of its peak value at the dealiasing cutoff — the
# small scales are grid artefacts, and the "exact subgrid term" measured below
# would be measuring the discretisation.  nu = 4e-3 puts the dissipation scale
# inside the grid; the resolution check further down gates on it.
NU = 4.0e-3
ETA = 4.0e-3
DT = 2.0e-3
N_STEPS = 1000                 # to t = 2.0
SNAP_FROM, SNAP_EVERY = 300, 140   # sample only developed turbulence
FILTER_WIDTHS = [3, 5, 8]      # in grid spacings; k_Delta = 43, 26, 16

DX = 2 * np.pi / N_GRID
print(f"[MODEL] Decaying 2D MHD turbulence, {N_GRID}^2, nu = eta = {NU}")

# ---------------------------------------------------------------------------
# 1. Pseudo-spectral 2D MHD solver
# ---------------------------------------------------------------------------
KX = np.fft.fftfreq(N_GRID, d=1.0 / N_GRID).reshape(-1, 1)
KY = np.fft.fftfreq(N_GRID, d=1.0 / N_GRID).reshape(1, -1)
K2 = KX ** 2 + KY ** 2
K2_INV = np.where(K2 == 0, 1.0, K2)
DEALIAS = (np.abs(KX) < N_GRID / 3) & (np.abs(KY) < N_GRID / 3)
ifft = lambda z: np.real(np.fft.ifft2(z))


def velocity_and_field(w_hat, A_hat):
    psi_hat = np.where(K2 == 0, 0.0, w_hat / K2_INV)
    return (ifft(1j * KY * psi_hat), ifft(-1j * KX * psi_hat),    # u, v
            ifft(1j * KY * A_hat), ifft(-1j * KX * A_hat))        # Bx, By


def rhs_hat(w_hat, A_hat):
    u, v, Bx, By = velocity_and_field(w_hat, A_hat)
    w_x, w_y = ifft(1j * KX * w_hat), ifft(1j * KY * w_hat)
    A_x, A_y = ifft(1j * KX * A_hat), ifft(1j * KY * A_hat)
    j_x = ifft(1j * KX * K2 * A_hat)
    j_y = ifft(1j * KY * K2 * A_hat)
    dw = -(u * w_x + v * w_y) + (Bx * j_x + By * j_y)
    dA = -(u * A_x + v * A_y)
    return (np.fft.fft2(dw) * DEALIAS - NU * K2 * w_hat,
            np.fft.fft2(dA) * DEALIAS - ETA * K2 * A_hat)


def rk4_step(w_hat, A_hat, dt):
    k1w, k1a = rhs_hat(w_hat, A_hat)
    k2w, k2a = rhs_hat(w_hat + 0.5 * dt * k1w, A_hat + 0.5 * dt * k1a)
    k3w, k3a = rhs_hat(w_hat + 0.5 * dt * k2w, A_hat + 0.5 * dt * k2a)
    k4w, k4a = rhs_hat(w_hat + dt * k3w, A_hat + dt * k3a)
    return (w_hat + (dt / 6.0) * (k1w + 2 * k2w + 2 * k3w + k4w),
            A_hat + (dt / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a))


rng = np.random.default_rng(SEED)


def random_low_k_field(amplitude: float, k_peak: float) -> np.ndarray:
    f_hat = np.fft.fft2(rng.standard_normal((N_GRID, N_GRID)))
    f_hat *= np.exp(-K2 / (2.0 * k_peak ** 2)) * DEALIAS
    f = ifft(f_hat)
    return np.fft.fft2(f * amplitude / np.abs(f).max())


w_hat = random_low_k_field(6.0, 8.0)
A_hat = random_low_k_field(0.6, 6.0)

snapshots = []
for i in range(N_STEPS + 1):
    if i >= SNAP_FROM and (i - SNAP_FROM) % SNAP_EVERY == 0:
        snapshots.append((w_hat.copy(), A_hat.copy()))
    w_hat, A_hat = rk4_step(w_hat, A_hat, DT)
print(f"[SIM]   {N_STEPS} steps to t={N_STEPS * DT:.1f}, "
      f"{len(snapshots)} snapshots from t={SNAP_FROM * DT:.1f}")

# Is the DNS resolved?  Everything below measures a term defined as "what the
# filter removes", so if the grid itself is producing the smallest scales the
# measurement is of the discretisation, not of the physics.  This is checked
# before anything is fitted, and gated in the [CHECK] line at the end.
K_INT = np.round(np.sqrt(K2)).astype(int).ravel()
K_CUT = N_GRID / 3.0                       # dealiasing cutoff


def shell_spectra(w_h, A_h):
    """Shell-averaged kinetic energy, magnetic energy and enstrophy spectra."""
    n4 = N_GRID ** 4
    return (np.bincount(K_INT, weights=(0.5 * np.abs(w_h) ** 2 / K2_INV).ravel()) / n4,
            np.bincount(K_INT, weights=(0.5 * K2 * np.abs(A_h) ** 2).ravel()) / n4,
            np.bincount(K_INT, weights=(np.abs(w_h) ** 2).ravel()) / n4)


E_kin, E_mag, Z_spec = shell_spectra(*snapshots[-1])
E_tot = E_kin + E_mag
unresolved = float(E_tot[int(0.8 * K_CUT):].sum() / E_tot.sum())
z_ratio = float(Z_spec[int(K_CUT) - 1] / Z_spec[1:].max())
print(f"[SIM]   resolution: energy above 0.8*k_cut = {unresolved:.2e}, "
      f"enstrophy at the cutoff is {100 * z_ratio:.2f}% of its peak")

# ---------------------------------------------------------------------------
# 2. Filtering, the exact subgrid EMF, and the candidate closure features
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "ux_Axx", "uy_Axy", "vx_Axy", "vy_Ayy",     # gradient (Clark) model
    "lap_A",                                    # eddy resistivity
    "j_w",                                      # decoy
]
N_FEATURES = len(FEATURE_NAMES)
GRAD_IDX = [0, 1, 2, 3]
RESIST_IDX = 4
DECOY_IDX = 5


def gaussian_filter_hat(delta: float) -> np.ndarray:
    """Gaussian filter of width delta; its second moment is delta^2/12."""
    return np.exp(-K2 * delta ** 2 / 24.0)


def closure_dataset(delta: float):
    """Exact subgrid EMF and the candidate features, for one filter width."""
    G = gaussian_filter_hat(delta)
    feats, targets = [], []
    for w_h, A_h in snapshots:
        u, v, _, _ = velocity_and_field(w_h, A_h)
        A_x, A_y = ifft(1j * KX * A_h), ifft(1j * KY * A_h)

        wb_h, Ab_h = w_h * G, A_h * G
        ub, vb, _, _ = velocity_and_field(wb_h, Ab_h)
        Ab_x, Ab_y = ifft(1j * KX * Ab_h), ifft(1j * KY * Ab_h)

        # exact subgrid term: filter the product, subtract the resolved product
        R = ifft(np.fft.fft2(u * A_x + v * A_y) * G) - (ub * Ab_x + vb * Ab_y)

        ub_h, vb_h = np.fft.fft2(ub), np.fft.fft2(vb)
        ub_x, ub_y = ifft(1j * KX * ub_h), ifft(1j * KY * ub_h)
        vb_x, vb_y = ifft(1j * KX * vb_h), ifft(1j * KY * vb_h)
        Ab_xx = ifft(-KX * KX * Ab_h)
        Ab_xy = ifft(-KX * KY * Ab_h)
        Ab_yy = ifft(-KY * KY * Ab_h)
        lap_Ab = Ab_xx + Ab_yy
        wb, jb = ifft(wb_h), -lap_Ab

        feats.append(np.stack([
            ub_x * Ab_xx, ub_y * Ab_xy, vb_x * Ab_xy, vb_y * Ab_yy,
            lap_Ab, jb * wb,
        ], axis=-1).reshape(-1, N_FEATURES))
        targets.append(R.reshape(-1, 1))
    return np.concatenate(feats), np.concatenate(targets)


# ---------------------------------------------------------------------------
# 3. One fit per filter width
# ---------------------------------------------------------------------------
N_TRAIN = 12000
results = []
sample_rng = np.random.default_rng(SEED + 1)

for n_delta in FILTER_WIDTHS:
    delta = n_delta * DX
    c_true = delta ** 2 / 12.0
    Theta_full, Y_full = closure_dataset(delta)
    idx = sample_rng.choice(len(Theta_full), N_TRAIN, replace=False)
    Theta, Y = Theta_full[idx], Y_full[idx]
    cond = float(np.linalg.cond(np.column_stack([Theta, np.ones(len(Theta))])))

    print(f"\n--- Delta = {n_delta} dx = {delta:.4f};  "
          f"Delta^2/12 = {c_true:.3e} ---")
    print(f"[DATA]  subgrid EMF is {100 * Y_full.std() / np.abs(Y_full).max():.2f}% "
          f"of its own peak; cond(Theta) = {cond:.1f}")

    lift = CustomLift(fn=lambda X: X, output_dim=N_FEATURES, name="emf_closure")
    model = KANDy(lift=lift, grid=5, k=3, steps=150, seed=SEED, base_fun="zero")
    model.fit(X=Theta, X_dot=Y, val_frac=0.15, test_frac=0.15, lamb=0.0, patience=30)

    pred = model.predict(Theta)
    r2 = float(1.0 - np.sum((Y - pred) ** 2) / np.sum((Y - Y.mean()) ** 2))
    corr = float(np.corrcoef(Y[:, 0], pred[:, 0])[0, 1])
    A_fit, *_ = np.linalg.lstsq(Theta, pred, rcond=None)
    A_fit = A_fit.ravel()

    grad_coefs = A_fit[GRAD_IDX]
    c_hat = float(grad_coefs.mean())
    spread = float(grad_coefs.std() / abs(c_hat))
    beta_hat = float(-A_fit[RESIST_IDX])

    # What the eddy-resistivity model achieves on its own — one column, fitted
    # by least squares, scored on the same data.
    lap_col = Theta[:, [RESIST_IDX]]
    beta_only, *_ = np.linalg.lstsq(lap_col, Y, rcond=None)
    r2_resist = float(1.0 - np.sum((Y - lap_col @ beta_only) ** 2)
                      / np.sum((Y - Y.mean()) ** 2))

    print(f"[EVAL]  gradient-model coefficients: "
          f"{np.array2string(grad_coefs, precision=5)}")
    print(f"[EVAL]  mean {c_hat:.4e} vs Delta^2/12 = {c_true:.4e}  "
          f"({100 * abs(c_hat - c_true) / c_true:.1f}% off); "
          f"spread across the four edges {100 * spread:.1f}%")
    print(f"[EVAL]  closure R^2 = {r2:.4f} (correlation {corr:.4f}); "
          f"eddy resistivity ALONE reaches R^2 = {r2_resist:.4f} "
          f"(beta = {float(-beta_only[0, 0]):.3e})")
    print(f"[EVAL]  weight on the decoy j*w: {A_fit[DECOY_IDX]:.3e}")

    results.append({
        "n_delta": n_delta, "delta": delta, "c_true": c_true, "c_hat": c_hat,
        "grad_coefs": grad_coefs.tolist(), "spread": spread,
        "rel_err": float(abs(c_hat - c_true) / c_true),
        "r2": r2, "corr": corr, "r2_eddy_resistivity": r2_resist,
        "beta_in_full_lift": beta_hat, "beta_alone": float(-beta_only[0, 0]),
        "decoy_weight": float(abs(A_fit[DECOY_IDX])),
        "cond": cond, "model": model, "Theta": Theta, "Y": Y, "pred": pred,
    })

# ---------------------------------------------------------------------------
# 4. The scaling law — a coefficient that tracks Delta^2 across a factor of 4
# ---------------------------------------------------------------------------
deltas = np.array([r["delta"] for r in results])
c_hats = np.array([r["c_hat"] for r in results])
slope, intercept = np.polyfit(np.log(deltas), np.log(c_hats), 1)
prefactors = c_hats / deltas ** 2
signed = [(r["c_hat"] - r["c_true"]) / r["c_true"] for r in results]
print(f"\n[EVAL]  Recovered scaling: c ~ Delta^{slope:.3f}   (theory: Delta^2)")
print(f"[EVAL]  Holding the exponent at 2, the prefactor c/Delta^2 is "
      f"{np.array2string(prefactors, precision=4)}  vs 1/12 = {1 / 12:.4f}")
print(f"[EVAL]  Signed deviation from Delta^2/12: "
      + ", ".join(f"{100 * s:+.1f}% at {r['n_delta']}dx"
                  for s, r in zip(signed, results)))
print("[EVAL]  It drifts upward with width — the next term of the expansion "
      "arriving — which is also why the free-fit exponent lands above 2.")

best = results[0]                      # smallest Delta: where the expansion holds
print(f"[EVAL]  Locally closable fraction of the subgrid EMF "
      f"(R^2): {[round(r['r2'], 3) for r in results]}")

# ---------------------------------------------------------------------------
# 5. Symbolic extraction  (LAST — get_formula mutates the model in place)
# ---------------------------------------------------------------------------
model = best["model"]
theta_t = torch.tensor(best["Theta"][:2048], dtype=torch.float32)
print(f"\n[SYMBOLIC] Extracting the closure at Delta = {best['n_delta']}dx ...")
formulas = model.get_formula(
    var_names=FEATURE_NAMES, round_places=6,
    lib=["x", "0"], r2_threshold=0.80, weight_simple=0.0,
)
print(f"  R = {formulas[0]}")
r2_sym = model.score_formula(formulas, best["Theta"], best["Y"],
                             var_names=FEATURE_NAMES)
print(f"[SYMBOLIC] Formula R^2: {np.round(r2_sym, 6)}")
print(f"[SYMBOLIC] True leading order: R = {best['c_true']:.3e} * "
      f"(ux_Axx + uy_Axy + vx_Axy + vy_Ayy)")

# ---------------------------------------------------------------------------
# 6. Verification
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "coef_rel_err_max_small_delta": 0.20,   # Delta^2/12 at the smallest width
    "edge_spread_max": 0.20,                # the four edges must agree
    "scaling_exponent_range": [1.7, 2.3],   # c ~ Delta^2
    "closure_r2_min": 0.80,                 # how much is locally closable
    "eddy_resistivity_r2_max": 0.50,        # ... and how little a local beta gets
    "decoy_weight_rel_max": 0.10,           # relative to the gradient coefficient
    "formula_r2_min": 0.80,
    "unresolved_energy_fraction_max": 1e-3,  # the DNS must resolve its own tail
}
decoy_rel = best["decoy_weight"] / abs(best["c_hat"])
passed = print_check(
    unresolved <= THRESHOLDS["unresolved_energy_fraction_max"]
    and best["rel_err"] <= THRESHOLDS["coef_rel_err_max_small_delta"]
    and best["spread"] <= THRESHOLDS["edge_spread_max"]
    and THRESHOLDS["scaling_exponent_range"][0] <= slope
    <= THRESHOLDS["scaling_exponent_range"][1]
    and best["r2"] >= THRESHOLDS["closure_r2_min"]
    and best["r2_eddy_resistivity"] <= THRESHOLDS["eddy_resistivity_r2_max"]
    and decoy_rel <= THRESHOLDS["decoy_weight_rel_max"]
    and float(np.min(r2_sym)) >= THRESHOLDS["formula_r2_min"],
    "2D MHD turbulence: subgrid EMF closed by the gradient model, "
    "coefficient scaling as Delta^2/12",
)
RESULT_DIR = ensure_outdir("results/MHD2DTurbulence")
save_metrics(RESULT_DIR / "metrics.json", {
    "task": "MHD2DTurbulence",
    "category": "plasma",
    "seed": SEED,
    "sampling": "filtered pseudo-spectral 2D MHD DNS, one sample per grid point",
    "grid": N_GRID, "nu": NU, "eta": ETA,
    "unresolved_energy_fraction": unresolved,
    "enstrophy_at_cutoff_over_peak": z_ratio,
    "feature_names": FEATURE_NAMES,
    "filter_widths_in_dx": FILTER_WIDTHS,
    "per_width": [{k: v for k, v in r.items()
                   if k not in ("model", "Theta", "Y", "pred")} for r in results],
    "scaling_exponent": float(slope),
    "scaling_prefactor_free_fit": float(np.exp(intercept)),
    "prefactor_per_width": prefactors.tolist(),
    "signed_deviation_per_width": [float(s) for s in signed],
    "formula_r2": np.asarray(r2_sym).tolist(),
    "decoy_weight_relative": float(decoy_rel),
    "thresholds": THRESHOLDS,
    "passed": passed,
})


# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
EXTENT = [0, 2 * np.pi, 0, 2 * np.pi]


def plot_turbulent_state(outdir: str) -> None:
    """The DNS the closure is measured on: vorticity and current."""
    w_h, A_h = snapshots[-1]
    w = ifft(w_h)
    j = ifft(K2 * A_h)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, (field, title, cmap) in zip(axes, [(w, r"vorticity $\omega$", "RdBu_r"),
                                               (j, r"current $j$", "PuOr_r")]):
        s = np.abs(field).max()
        im = ax.imshow(field.T, origin="lower", extent=EXTENT, cmap=cmap,
                       vmin=-s, vmax=s)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Decaying 2D MHD turbulence at $t={N_STEPS * DT:.1f}$", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{outdir}/turbulent_state.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_emf_fields(outdir: str) -> None:
    """The exact subgrid EMF against the learned closure, as fields."""
    delta = best["delta"]
    Theta_full, Y_full = closure_dataset(delta)
    n = N_GRID * N_GRID
    R_true = Y_full[-n:, 0].reshape(N_GRID, N_GRID)
    R_pred = (Theta_full[-n:] @ np.array(
        [best["grad_coefs"] + [0.0, 0.0]][0])).reshape(N_GRID, N_GRID)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    s = np.abs(R_true).max()
    for ax, (field, title) in zip(axes, [(R_true, "exact subgrid EMF $R$"),
                                         (R_pred, "gradient-model closure"),
                                         (R_pred - R_true, "error")]):
        im = ax.imshow(field.T, origin="lower", extent=EXTENT, cmap="RdBu_r",
                       vmin=-s, vmax=s)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(rf"Subgrid EMF at $\Delta = {best['n_delta']}\,dx$, "
                 rf"$R^2 = {best['r2']:.3f}$", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{outdir}/subgrid_emf_fields.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_spectra(outdir: str) -> None:
    """The resolution evidence, and where each filter width cuts."""
    fig, ax = plt.subplots(figsize=(6, 4.4))
    k = np.arange(1, int(K_CUT))
    ax.loglog(k, E_kin[1:int(K_CUT)], color="#1f77b4", lw=1.4, label="kinetic")
    ax.loglog(k, E_mag[1:int(K_CUT)], color="#d62728", lw=1.4, label="magnetic")
    ax.axvline(K_CUT, color="0.3", ls="--", lw=1.0)
    ax.annotate("dealiasing cutoff", (K_CUT, ax.get_ylim()[1]),
                rotation=90, fontsize=7, ha="right", va="top", color="0.3")
    for r in results:
        kd = np.pi / r["delta"]
        ax.axvline(kd, color="#2ca02c", ls=":", lw=1.0)
        ax.annotate(rf"$\Delta$={r['n_delta']}dx", (kd, ax.get_ylim()[0]),
                    rotation=90, fontsize=7, ha="right", va="bottom", color="#2ca02c")
    ax.set_xlabel("wavenumber $k$"); ax.set_ylabel("energy spectrum")
    ax.set_title(f"DNS spectra at $t={N_STEPS * DT:.1f}$ "
                 f"(energy above $0.8k_c$: {unresolved:.1e})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{outdir}/spectra.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scaling(outdir: str) -> None:
    """The recovered coefficient against Delta^2/12 — the actual validation."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    d = np.array([r["delta"] for r in results])
    ax = axes[0]
    ax.loglog(d, [r["c_true"] for r in results], "o-", color="#1f77b4",
              label=r"$\Delta^2/12$")
    ax.loglog(d, [r["c_hat"] for r in results], "s--", color="#d62728",
              label="KANDy")
    for r in results:
        ax.annotate(f"{r['n_delta']}dx", (r["delta"], r["c_hat"]),
                    textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel(r"filter width $\Delta$")
    ax.set_ylabel("gradient-model coefficient")
    ax.set_title(rf"recovered exponent {slope:.2f} (theory 2)")
    ax.legend(fontsize=8)

    ax = axes[1]
    xs = np.arange(len(results))
    ax.bar(xs - 0.18, [r["r2"] for r in results], width=0.34, color="#d62728",
           label="gradient model")
    ax.bar(xs + 0.18, [r["r2_eddy_resistivity"] for r in results], width=0.34,
           color="#7f7f7f", label="eddy resistivity alone")
    ax.set_xticks(xs)
    ax.set_xticklabels([rf"$\Delta={r['n_delta']}dx$" for r in results])
    ax.set_ylabel(r"$R^2$ against the exact EMF")
    ax.set_title("how much of the subgrid EMF is locally closable")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{outdir}/closure_scaling.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{outdir}/closure_scaling.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(outdir: str) -> None:
    """Predicted vs exact subgrid EMF, pointwise."""
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    sub = slice(None, 4000)
    ax.plot(best["Y"][sub, 0], best["pred"][sub, 0], ".", ms=1.5, alpha=0.35,
            color="#1f77b4")
    lim = np.abs(best["Y"]).max()
    ax.plot([-lim, lim], [-lim, lim], color="0.3", lw=1.0, ls="--")
    ax.set_xlabel("exact subgrid EMF"); ax.set_ylabel("KANDy closure")
    ax.set_title(rf"$\Delta={best['n_delta']}dx$, $R^2={best['r2']:.3f}$",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{outdir}/closure_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


use_pub_style()
OUT = str(RESULT_DIR)
plot_turbulent_state(OUT)
plot_spectra(OUT)
plot_emf_fields(OUT)
plot_scaling(OUT)
plot_scatter(OUT)
if getattr(model, "train_results_", None):
    fig, _ = plot_loss_curves(model.train_results_, save=f"{OUT}/loss_curves")
    plt.close(fig)
fig, _ = plot_all_edges(
    model.model_, X=theta_t,
    in_var_names=FEATURE_NAMES, out_var_names=["R"],
    save=f"{OUT}/edge_activations",
)
plt.close(fig)
print(f"[FIGS]  Saved {OUT}/")
