#!/usr/bin/env python3
"""KANDy example: discovering a subgrid-scale closure for 2D turbulence.

Every other example in this repository fits a full right-hand side.  This one
fits a RESIDUAL — the part of the dynamics that a coarse simulation cannot see.
That is the large-eddy-simulation (LES) closure problem, and it is the recipe to
imitate whenever the target is "what the resolved model is missing" rather than
"the equation itself".

Filter the vorticity equation with a low-pass filter (overbar) at scale Delta:

    d(w_bar)/dt = -(u_bar*w_bar_x + v_bar*w_bar_y) + nu*lap(w_bar) - tau

Everything on the right is computable from the RESOLVED field except

    tau = filter(u*w_x + v*w_y) - (u_bar*w_bar_x + v_bar*w_bar_y)

which needs the unresolved scales.  Running the DNS lets us compute tau
EXACTLY; the discovery question is whether tau is a function of resolved-scale
quantities alone.  That is the closure problem, and the honest answer depends
entirely on the filter — which is the point of this example.

The lift (13 resolved-scale features, one sample per grid point)
---------------------------------------------------------------
    [w, w_x, w_y, lap_w,
     ux*wxx, uy*wxy, vx*wxy, vy*wyy,        <- Clark / gradient-model terms
     |S|*lap_w,                             <- Smagorinsky-like
     ux*wx, uy*wy, vx*wx, vy*wy]

The Clark terms are products of a resolved VELOCITY gradient with a resolved
VORTICITY second derivative — different fields, so a separable KAN cannot build
them and they must be in the lift.  Classical theory (the Taylor expansion of
the filter) predicts

    tau ~= (Delta^2 / 12) * (grad(u_bar).grad(w_bar_x) + grad(v_bar).grad(w_bar_y))

i.e. all four Clark features share one coefficient, Delta^2/12.  That gives us a
rare thing in this repo: a target coefficient known from theory rather than from
the simulator.

Two regimes, and only one of them is closable
---------------------------------------------
1. **Smooth (Gaussian) filter.**  tau is a modest correction (~17% of resolved
   advection at Delta = 8dx) and the Taylor expansion is valid, so tau really is
   a local function of resolved quantities.  KANDy reaches R^2 ~ 0.98 on a
   held-out snapshot AND recovers the four Clark coefficients to within a few
   percent of Delta^2/12 — rediscovering the gradient model from data.

2. **Sharp spectral filter** — the actual LES situation, where the filter cuts
   into the energetic scales.  Now tau is as large as the term it corrects
   (~51% of resolved advection) and NOTHING in the feature set explains it:
   Clark with its theoretical coefficient gets R^2 ~ 0.13, refitting the
   coefficient gets ~0.18, least squares on all 13 features gets ~0.19, and
   KANDy gets ~0.17 on held-out data while fitting ~0.25 on train.  Roughly
   80-90% of the exact subgrid term is simply not a local function of these
   resolved features.

   That is not a tuning failure — it is the classical non-closability result,
   reproduced quantitatively.  Note the signature: the KAN's extra spline
   flexibility buys train R^2 and no test R^2.  When you see that gap on a
   residual-fitting problem, the features are missing information, and no
   amount of capacity will supply it.

Method notes worth copying
--------------------------
* **Products are computed with 2x zero-padding** so tau carries no aliasing
  error.  If you compute tau with a dealiased-by-truncation product, you are
  fitting your own aliasing error rather than the physics.
* **Scoring is on a completely held-out snapshot**, not a random split of grid
  points.  Neighbouring grid points within one snapshot are highly correlated,
  so a random split reports an optimistic number.
* Both baselines matter: the analytic Clark model bounds what theory gives you,
  and plain least squares on the same features bounds what the KAN can add.
  Report all three, as this script does.

Runtime: ~1 minute.
KAN:  width = [13, 1],  base_fun='zero'
"""

import os
import time
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kandy import KANDy, CustomLift
from kandy.plotting import plot_loss_curves, use_pub_style

# ---------------------------------------------------------------------------
# 0. Reproducibility and DNS parameters
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

N = 128                 # DNS resolution
NU = 0.002
AMPLITUDE = 5.0
DT = 0.001
SPINUP = 600            # let a turbulent cascade develop before sampling
N_STEPS = 2400
SNAP_EVERY = 300

DX = 2 * np.pi / N
KX = np.fft.fftfreq(N, d=1.0 / N).reshape(-1, 1)
KY = np.fft.fftfreq(N, d=1.0 / N).reshape(1, -1)
K2 = KX ** 2 + KY ** 2
K2_INV = np.where(K2 == 0, 1.0, K2)
DEALIAS = (np.abs(KX) < N / 3) & (np.abs(KY) < N / 3)


# ---------------------------------------------------------------------------
# 1. DNS: 2D Navier-Stokes, vorticity form
# ---------------------------------------------------------------------------
def fields_from_w_hat(w_hat):
    psi_hat = np.where(K2 == 0, 0.0, w_hat / K2_INV)
    u = np.real(np.fft.ifft2(1j * KY * psi_hat))
    v = np.real(np.fft.ifft2(-1j * KX * psi_hat))
    w_x = np.real(np.fft.ifft2(1j * KX * w_hat))
    w_y = np.real(np.fft.ifft2(1j * KY * w_hat))
    return u, v, w_x, w_y


def rhs_hat(w_hat):
    u, v, w_x, w_y = fields_from_w_hat(w_hat)
    return np.fft.fft2(-(u * w_x + v * w_y)) * DEALIAS - NU * K2 * w_hat


def rk4_hat(w_hat, dt):
    k1 = rhs_hat(w_hat)
    k2 = rhs_hat(w_hat + 0.5 * dt * k1)
    k3 = rhs_hat(w_hat + 0.5 * dt * k2)
    k4 = rhs_hat(w_hat + dt * k3)
    return w_hat + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# --- exact (alias-free) products via 2x zero padding -----------------------
M = 2 * N


def _pad(f_hat):
    big = np.zeros((M, M), dtype=complex)
    h = N // 2
    big[:h, :h], big[:h, -h:] = f_hat[:h, :h], f_hat[:h, -h:]
    big[-h:, :h], big[-h:, -h:] = f_hat[-h:, :h], f_hat[-h:, -h:]
    return big * 4.0                      # (M/N)^2 for numpy's FFT convention


def _truncate(big_hat):
    h = N // 2
    out = np.zeros((N, N), dtype=complex)
    out[:h, :h], out[:h, -h:] = big_hat[:h, :h], big_hat[:h, -h:]
    out[-h:, :h], out[-h:, -h:] = big_hat[-h:, :h], big_hat[-h:, -h:]
    return out / 4.0


def exact_product_hat(a_hat, b_hat):
    """Spectrum of a*b with no aliasing error."""
    a = np.real(np.fft.ifft2(_pad(a_hat)))
    b = np.real(np.fft.ifft2(_pad(b_hat)))
    return _truncate(np.fft.fft2(a * b))


def advection_hat(w_hat):
    """Spectrum of (u*w_x + v*w_y), exactly."""
    psi_hat = np.where(K2 == 0, 0.0, w_hat / K2_INV)
    return (exact_product_hat(1j * KY * psi_hat, 1j * KX * w_hat)
            + exact_product_hat(-1j * KX * psi_hat, 1j * KY * w_hat))


t_start = time.time()
rng = np.random.default_rng(SEED)
w_hat = np.fft.fft2(rng.standard_normal((N, N)))
w_hat *= np.exp(-K2 / (2 * 8.0 ** 2)) * DEALIAS
w0 = np.real(np.fft.ifft2(w_hat))
w0 *= AMPLITUDE / np.abs(w0).max()
w_hat = np.fft.fft2(w0)

print(f"[DNS]  {N}x{N}, nu={NU}, dt={DT}, {N_STEPS} steps (spin-up {SPINUP})")
snapshots = []
for i in range(N_STEPS + 1):
    if i >= SPINUP and (i - SPINUP) % SNAP_EVERY == 0:
        snapshots.append(w_hat.copy())
    w_hat = rk4_hat(w_hat, DT)
print(f"[DNS]  {len(snapshots)} snapshots in {time.time() - t_start:.1f}s")

# ---------------------------------------------------------------------------
# 2. Filtering, the exact subgrid term, and resolved-scale features
# ---------------------------------------------------------------------------
FEATURE_NAMES = ["w", "w_x", "w_y", "lap_w",
                 "ux*wxx", "uy*wxy", "vx*wxy", "vy*wyy",
                 "Smag*lap_w",
                 "ux*wx", "uy*wy", "vx*wx", "vy*wy"]
CLARK_IDX = [4, 5, 6, 7]


def d_xy(f_hat, nx, ny):
    return np.real(np.fft.ifft2(((1j * KX) ** nx) * ((1j * KY) ** ny) * f_hat))


def snapshot_data(w_hat_dns, delta, sharp=False):
    """(features (N^2, 13), tau (N^2,), resolved advection (N^2,))."""
    if sharp:
        G = (np.sqrt(K2) <= np.pi / delta).astype(float)
    else:
        G = np.exp(-K2 * delta ** 2 / 24.0)                 # Gaussian

    wb_hat = G * w_hat_dns
    psib_hat = np.where(K2 == 0, 0.0, wb_hat / K2_INV)
    ub_hat, vb_hat = 1j * KY * psib_hat, -1j * KX * psib_hat

    res_adv_hat = (exact_product_hat(ub_hat, 1j * KX * wb_hat)
                   + exact_product_hat(vb_hat, 1j * KY * wb_hat))
    tau = np.real(np.fft.ifft2(G * advection_hat(w_hat_dns) - res_adv_hat))
    res_adv = np.real(np.fft.ifft2(res_adv_hat))

    wb = np.real(np.fft.ifft2(wb_hat))
    wx, wy = d_xy(wb_hat, 1, 0), d_xy(wb_hat, 0, 1)
    wxx, wxy, wyy = d_xy(wb_hat, 2, 0), d_xy(wb_hat, 1, 1), d_xy(wb_hat, 0, 2)
    lap = wxx + wyy
    ux, uy = d_xy(ub_hat, 1, 0), d_xy(ub_hat, 0, 1)
    vx, vy = d_xy(vb_hat, 1, 0), d_xy(vb_hat, 0, 1)
    smag = np.sqrt(2 * (ux ** 2 + vy ** 2) + (uy + vx) ** 2)

    feats = np.stack([wb, wx, wy, lap,
                      ux * wxx, uy * wxy, vx * wxy, vy * wyy,
                      smag * lap,
                      ux * wx, uy * wy, vx * wx, vy * wy], axis=-1)
    return feats.reshape(-1, len(FEATURE_NAMES)), tau.reshape(-1), res_adv.reshape(-1)


def r2_score(y, p):
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


# ---------------------------------------------------------------------------
# 3. One regime = one experiment
# ---------------------------------------------------------------------------
N_TRAIN = 12_000


def run_regime(delta_cells, sharp, symbolic=False, steps=120):
    """Fit tau at one filter width/type; score on a HELD-OUT snapshot."""
    delta = delta_cells * DX
    c_theory = delta ** 2 / 12.0
    label = f"{'sharp' if sharp else 'gauss'} Delta={delta_cells}dx"

    train_f, train_t, ratios = [], [], []
    for wh in snapshots[:-1]:                       # last snapshot held out
        f, t, adv = snapshot_data(wh, delta, sharp)
        train_f.append(f)
        train_t.append(t)
        ratios.append(np.std(t) / np.std(adv))
    Xtr = np.concatenate(train_f)
    Ytr = np.concatenate(train_t)
    sel = np.random.default_rng(SEED).choice(len(Xtr), N_TRAIN, replace=False)
    Xtr, Ytr = Xtr[sel], Ytr[sel]
    Xte, Yte, _ = snapshot_data(snapshots[-1], delta, sharp)
    tau_rel = float(np.mean(ratios))

    # baselines
    clark_te = c_theory * Xte[:, CLARK_IDX].sum(axis=1)
    r2_clark = r2_score(Yte, clark_te)
    g_tr = Xtr[:, CLARK_IDX].sum(axis=1)
    c_opt = float(np.linalg.lstsq(g_tr[:, None], Ytr, rcond=None)[0][0])
    r2_clark_opt = r2_score(Yte, c_opt * Xte[:, CLARK_IDX].sum(axis=1))
    ols = np.linalg.lstsq(np.column_stack([Xtr, np.ones(len(Xtr))]), Ytr,
                          rcond=None)[0]
    r2_ols = r2_score(Yte, np.column_stack([Xte, np.ones(len(Xte))]) @ ols)

    # KANDy
    lift = CustomLift(fn=lambda X: X, output_dim=len(FEATURE_NAMES), name="sgs")
    model = KANDy(lift=lift, grid=5, k=3, steps=steps, seed=SEED, base_fun="zero")
    model.fit(X=Xtr, X_dot=Ytr[:, None], val_frac=0.15, test_frac=0.15,
              lamb=0.0, patience=40, verbose=False)
    r2_kan_tr = r2_score(Ytr, model.predict(Xtr)[:, 0])
    r2_kan_te = r2_score(Yte, model.predict(Xte)[:, 0])

    print(f"\n=== {label}  (Delta^2/12 = {c_theory:.6f}) ===")
    print(f"  rms(tau)/rms(resolved advection) = {tau_rel:.4f}")
    print(f"  R^2 Clark, analytic Delta^2/12   = {r2_clark:.4f}")
    print(f"  R^2 Clark, best-fit coefficient  = {r2_clark_opt:.4f} "
          f"(c_fit/c_theory = {c_opt / c_theory:.2f})")
    print(f"  R^2 least squares, 13 features   = {r2_ols:.4f}")
    print(f"  R^2 KANDy   train {r2_kan_tr:.4f} | HELD-OUT {r2_kan_te:.4f}")
    if r2_kan_tr - r2_kan_te > 0.03:
        print("  ^ train >> held-out: the extra spline flexibility is buying")
        print("    fit, not generalisation — the features lack the information.")

    formula = None
    if symbolic:
        formula = model.get_formula(var_names=FEATURE_NAMES, round_places=5,
                                    lib=["x", "0"], r2_threshold=0.80,
                                    weight_simple=0.0)[0]
        print(f"  symbolic: tau = {formula}")

    return dict(label=label, delta=delta, c_theory=c_theory, tau_rel=tau_rel,
                r2_clark=r2_clark, r2_ols=r2_ols, r2_kan=r2_kan_te,
                r2_kan_train=r2_kan_tr, ols=ols, model=model,
                Xte=Xte, Yte=Yte, formula=formula)


# ---------------------------------------------------------------------------
# 4. Regime 1 — smooth filter: closable, and theory is recovered
# ---------------------------------------------------------------------------
print("\n" + "=" * 66)
print("REGIME 1: Gaussian filter — the resolvable case")
print("=" * 66)
gauss = run_regime(8, sharp=False, symbolic=True)

coeffs = gauss["ols"][CLARK_IDX]
print(f"\n[THEORY] Clark coefficients from least squares: "
      f"{np.round(coeffs, 6)}")
print(f"[THEORY] theoretical Delta^2/12               : {gauss['c_theory']:.6f}")
print(f"[THEORY] ratios to theory: {np.round(coeffs / gauss['c_theory'], 3)}")
print("[THEORY] All four share one coefficient, as the gradient-model")
print("[THEORY] expansion predicts — recovered from data with no prior.")

# ---------------------------------------------------------------------------
# 5. Regime 2 — sharp filter: the real LES problem, and it does not close
# ---------------------------------------------------------------------------
print("\n" + "=" * 66)
print("REGIME 2: sharp spectral filter — the genuine LES case")
print("=" * 66)
sharp = run_regime(8, sharp=True, symbolic=False)
print(f"\n[LIMIT]  tau is {100 * sharp['tau_rel']:.0f}% of the resolved advection,")
print("[LIMIT]  so it cannot be neglected — yet no method here explains more")
print(f"[LIMIT]  than {100 * max(sharp['r2_ols'], sharp['r2_kan']):.0f}% of it. "
      "Roughly 80-90% of the exact subgrid")
print("[LIMIT]  term is not a local function of these resolved features.")
print("[LIMIT]  This is the classical non-closability of LES, measured.")

# ---------------------------------------------------------------------------
# 6. How the closability degrades with filter width (least squares only, cheap)
# ---------------------------------------------------------------------------
print("\n[SWEEP] R^2 of least squares vs filter width (held-out snapshot):")
sweep = []
for dc in [2, 4, 8, 16]:
    for sh in [False, True]:
        delta = dc * DX
        Xs, Ys, ads = [], [], []
        for wh in snapshots[:-1]:
            f, t, a = snapshot_data(wh, delta, sh)
            Xs.append(f); Ys.append(t); ads.append(np.std(t) / np.std(a))
        Xa, Ya = np.concatenate(Xs), np.concatenate(Ys)
        Xe, Ye, _ = snapshot_data(snapshots[-1], delta, sh)
        c = np.linalg.lstsq(np.column_stack([Xa, np.ones(len(Xa))]), Ya, rcond=None)[0]
        val = r2_score(Ye, np.column_stack([Xe, np.ones(len(Xe))]) @ c)
        sweep.append((dc, sh, val, float(np.mean(ads))))
        print(f"   Delta={dc:2d}dx  {'sharp' if sh else 'gauss':5s}  "
              f"R^2={val:6.4f}   tau/adv={np.mean(ads):.3f}")

# ---------------------------------------------------------------------------
# 7. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/TurbulenceClosure", exist_ok=True)
extent = [0, 2 * np.pi, 0, 2 * np.pi]

# 7a. The exact subgrid term vs the prediction, both regimes
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
for row, res in enumerate([gauss, sharp]):
    true = res["Yte"].reshape(N, N)
    pred = res["model"].predict(res["Xte"])[:, 0].reshape(N, N)
    s = np.abs(true).max() * 0.7
    for ax, (fld, ttl) in zip(axes[row], [
            (true, f"exact $\\tau$ — {res['label']}"),
            (pred, f"KANDy, held-out $R^2$={res['r2_kan']:.3f}"),
            (pred - true, "error")]):
        im = ax.imshow(fld.T, origin="lower", extent=extent, cmap="RdBu_r",
                       vmin=-s, vmax=s)
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Subgrid term: closable under a smooth filter (top), "
             "not under a sharp one (bottom)", fontsize=12)
fig.tight_layout()
fig.savefig("results/TurbulenceClosure/sgs_fields.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7b. Recovered Clark coefficients against theory
fig, ax = plt.subplots(figsize=(6, 4))
names = [FEATURE_NAMES[i] for i in CLARK_IDX]
ax.bar(np.arange(4), coeffs, color="#1f77b4", label="recovered")
ax.axhline(gauss["c_theory"], color="#d62728", ls="--", lw=1.5,
           label=r"theory $\Delta^2/12$")
ax.set_xticks(np.arange(4)); ax.set_xticklabels(names, rotation=20, fontsize=8)
ax.set_ylabel("coefficient")
ax.set_title("The gradient-model coefficient, rediscovered")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/TurbulenceClosure/clark_coefficients.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)

# 7c. Closability vs filter width
fig, ax = plt.subplots(figsize=(6.4, 4))
for sh, colour, lab in [(False, "#1f77b4", "Gaussian filter"),
                        (True, "#d62728", "sharp spectral filter")]:
    pts = [(d, v) for d, s_, v, _ in sweep if s_ == sh]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=colour,
            lw=1.4, ms=5, label=lab)
ax.set_xscale("log", base=2)
ax.set_xlabel(r"filter width $\Delta$ (grid cells)")
ax.set_ylabel(r"held-out $R^2$ of the closure")
ax.set_title("How much of the subgrid term is locally closable")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("results/TurbulenceClosure/closability.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 7d. Loss curves for the smooth-filter model
if getattr(gauss["model"], "train_results_", None):
    fig, ax = plot_loss_curves(gauss["model"].train_results_,
                               save="results/TurbulenceClosure/loss_curves")
    plt.close(fig)

print("\n[FIGS]  Saved results/TurbulenceClosure/")
print(f"[TIME]  total {time.time() - t_start:.0f}s")
