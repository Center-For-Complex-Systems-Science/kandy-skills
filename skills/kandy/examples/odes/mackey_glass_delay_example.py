#!/usr/bin/env python3
"""KANDy example: Mackey-Glass — delay embedding of a scalar time series.

The Mackey-Glass equation is a delay-differential equation (DDE):

    dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)

with beta=0.2, gamma=0.1, n=10, tau=17 — the standard chaotic parameters.  A
DDE has an infinite-dimensional state (a whole function on [t-tau, t]), and all
we get to observe is one scalar channel.  This is the situation delay embedding
exists for, and it is the only example here that starts from a scalar series
rather than a fully observed state.

The recipe
----------
    scalar x(t)  ->  DelayEmbedding  ->  pick the informative lag  ->  KANDy

Sampling is chosen so the delay is an exact number of samples: with dt=1.0,
tau=17 is lag 17, and `DelayEmbedding(delays=18)` produces the coordinates
[x_t, x_{t-1}, ..., x_{t-17}].

(This file carries a temporary fallback copy of `DelayEmbedding` for builds
that do not yet export it; it is used only if `from kandy import
DelayEmbedding` fails, and the example otherwise uses the library API
unchanged.)

    GOTCHA: DelayEmbedding CHANGES THE ROW COUNT.  It maps (T, n) to
    (T - delays + 1, n*delays), so the targets must be trimmed to match:
    `X_dot = x_dot[delays - 1:]`.  Misalign this and the fit silently
    regresses against shifted data.

Three findings, in the order the script prints them
---------------------------------------------------

1. **In delay coordinates, a high R^2 is cheap and tells you nothing.**
   A plain LINEAR least-squares fit on all 18 delay coordinates reaches
   R^2 ~ 0.999999.  That is not a discovery — it is Takens/Koopman doing its
   job: a rich enough delay space linearises almost any smooth dynamics (the
   principle behind Hankel-DMD and HAVOK).  The price is that the 18 delay
   coordinates are massively collinear (cond ~ 3.5e6) and the resulting
   coefficients are an uninterpretable smear across all lags.  Nothing about
   the mechanism has been learned.

2. **The informative lag can be recovered from the data.**  Scanning candidate
   lags L and asking how well (x_t, x_{t-L}) determines dx/dt — with an
   assumption-free binned-mean predictor, no model at all — peaks at L=17,
   the true tau.  The landscape is not sharply peaked (L=16 is a close
   runner-up), so read this as "the scan finds the right lag", not "the scan
   is decisive".

3. **With the right lag, two coordinates beat eighteen — and are readable.**
   KANDy on just (x_t, x_{t-17}) reaches the same R^2 ~ 0.999999 with 2
   features instead of 18, and the two spline edges are the actual mechanism:

       edge on x_t     ->  -gamma * x           (linear decay)
       edge on x_{t-17} ->  beta*x/(1+x^10)     (the Hill production term)

   Both are recovered against ground truth at R^2 = 1.00000.  This is
   dictionary completion (cf. `odes/pendulum_dictionary_completion_example.py`):
   the Hill function is in no polynomial or trigonometric library, and the
   spline learns its shape anyway.

Why there is no symbolic extraction here
----------------------------------------
`get_formula` snaps each edge onto a symbolic library, and a rational Hill
function of degree 10 is in none of the built-in ones — forcing a snap would
replace a correct spline with a wrong closed form.  The validation is instead
the edge shape against ground truth, plus the rollout below.  See
`references/symbolic.md` for building a custom library if you do need a
closed form.

Runtime: ~2 minutes.
KAN:  width = [2, 1]
"""

import os
import time
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kandy import KANDy, CustomLift, Lift
from kandy.plotting import get_edge_activation, plot_loss_curves, use_pub_style

# ---------------------------------------------------------------------------
# TEMPORARY SHIM — remove once the released package exports DelayEmbedding.
#
# `DelayEmbedding` is part of the KANDy API (see references/api.md) but is not
# in the currently released build, so this example installs a verbatim copy of
# `kandy.lifts.DelayEmbedding` when the import fails.  Everything below uses the
# ordinary library API — delete this block and the plain import will work.
# ---------------------------------------------------------------------------
try:
    from kandy import DelayEmbedding
except ImportError:
    class DelayEmbedding(Lift):
        """Takens delay-coordinate embedding.

        Maps a trajectory (T, n) to (T - delays + 1, n*delays); row i is
        [x_{i+d-1}, ..., x_i], most-recent first.
        """

        def __init__(self, delays: int = 3):
            self.delays = delays
            self._input_dim = None

        def fit(self, X):
            self._input_dim = X.shape[1] if X.ndim > 1 else 1
            return self

        @property
        def output_dim(self):
            if self._input_dim is None:
                raise RuntimeError("Call lift.fit(X) before accessing output_dim.")
            return self._input_dim * self.delays

        def __call__(self, X):
            if X.ndim == 1:
                X = X[:, None]
            if self._input_dim is None:
                self.fit(X)
            T, _ = X.shape
            if T < self.delays:
                raise ValueError(
                    f"Trajectory length {T} is shorter than delays={self.delays}."
                )
            cols = [X[self.delays - 1 - lag: T - lag] for lag in range(self.delays)]
            return np.hstack(cols)

    import kandy
    kandy.DelayEmbedding = DelayEmbedding
    print("[SHIM]  using the bundled DelayEmbedding — the installed kandy does "
          "not export one")

# ---------------------------------------------------------------------------
# 0. Reproducibility and Mackey-Glass parameters
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BETA, GAMMA, N_HILL, TAU = 0.2, 0.1, 10.0, 17.0
DT_FINE = 0.01                       # integration step
N_LAG = int(round(TAU / DT_FINE))    # tau in fine steps
T_END = 4000.0
SAMPLE_DT = 1.0                      # tau = 17 samples exactly
DELAYS = 18                          # lags 0 .. 17


def mg_rhs(x, x_tau):
    """Mackey-Glass right-hand side."""
    return BETA * x_tau / (1.0 + x_tau ** N_HILL) - GAMMA * x


# ---------------------------------------------------------------------------
# 1. Integrate the DDE (RK4 with a history buffer)
# ---------------------------------------------------------------------------
print(f"[SIM]  Mackey-Glass, beta={BETA}, gamma={GAMMA}, n={N_HILL:.0f}, tau={TAU:.0f}")
n_steps = int(T_END / DT_FINE)
buf = np.empty(n_steps + N_LAG + 1)
rng = np.random.default_rng(SEED)
buf[: N_LAG + 1] = 1.2 + 0.01 * rng.standard_normal(N_LAG + 1)   # constant-ish history

t0 = time.time()
for i in range(N_LAG, N_LAG + n_steps):
    x, x_tau = buf[i], buf[i - N_LAG]
    x_mid = 0.5 * (buf[i - N_LAG] + buf[i - N_LAG + 1])   # delayed value at t+dt/2
    k1 = mg_rhs(x, x_tau)
    k2 = mg_rhs(x + 0.5 * DT_FINE * k1, x_mid)
    k3 = mg_rhs(x + 0.5 * DT_FINE * k2, x_mid)
    k4 = mg_rhs(x + DT_FINE * k3, buf[i - N_LAG + 1])
    buf[i + 1] = x + (DT_FINE / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

series = buf[N_LAG:]
print(f"[SIM]  {len(series)} fine steps in {time.time() - t0:.0f}s, "
      f"x in [{series.min():.4f}, {series.max():.4f}]")

# ---------------------------------------------------------------------------
# 2. Sample the scalar channel; exact derivatives from the DDE
# ---------------------------------------------------------------------------
stride = int(SAMPLE_DT / DT_FINE)
idx = np.arange(0, len(series), stride)
idx = idx[idx >= N_LAG]                       # need x(t-tau) to exist
x_samp = series[idx]
x_tau_samp = series[idx - N_LAG]
x_dot = mg_rhs(x_samp, x_tau_samp)
print(f"[DATA]  {len(x_samp)} samples at dt={SAMPLE_DT} "
      f"(tau = {int(TAU / SAMPLE_DT)} samples)")

# ---------------------------------------------------------------------------
# 3. Delay embedding — note the row-count trim
# ---------------------------------------------------------------------------
embed = DelayEmbedding(delays=DELAYS)
E = embed(x_samp[:, None])                    # (T - DELAYS + 1, DELAYS)
Y = x_dot[DELAYS - 1:][:, None]               # <- the trim the gotcha warns about
assert len(E) == len(Y), "delay embedding and targets are misaligned"
print(f"[DATA]  embedded {x_samp.shape} -> {E.shape}; targets trimmed to {Y.shape}")
print(f"[DATA]  column 0 is x_t, column {DELAYS - 1} is x_(t-{int(TAU)})")

# ---------------------------------------------------------------------------
# 4. Finding 1 — a linear fit on all 18 delays already scores ~1.0
# ---------------------------------------------------------------------------
design = np.column_stack([E, np.ones(len(E))])
cond_full = np.linalg.cond(design)
coef, *_ = np.linalg.lstsq(design, Y, rcond=None)
pred = design @ coef
r2_linear_all = 1.0 - np.sum((Y - pred) ** 2) / np.sum((Y - Y.mean()) ** 2)
print(f"\n[FIND-1] linear fit on all {DELAYS} delays: R^2 = {r2_linear_all:.7f}")
print(f"[FIND-1] cond(design) = {cond_full:.3g} — the delay coordinates are")
print("[FIND-1] massively collinear, so this R^2 is Takens linearisation, not")
print("[FIND-1] a discovery.  The coefficients are a smear across all 18 lags.")

# For contrast: the same linear model restricted to the two coordinates that
# actually carry the mechanism does badly — the Hill term is genuinely nonlinear.
pair = np.column_stack([E[:, 0], E[:, DELAYS - 1], np.ones(len(E))])
cp, *_ = np.linalg.lstsq(pair, Y, rcond=None)
r2_linear_pair = 1.0 - np.sum((Y - pair @ cp) ** 2) / np.sum((Y - Y.mean()) ** 2)
print(f"[FIND-1] the same LINEAR model on just (x_t, x_(t-{int(TAU)})): "
      f"R^2 = {r2_linear_pair:.4f}")

# ---------------------------------------------------------------------------
# 5. Finding 2 — recover the delay by scanning lags (no model assumed)
# ---------------------------------------------------------------------------
def binned_r2(a, b, y, n_bins=26):
    """Held-out R^2 of a 2D binned-mean predictor of y from (a, b)."""
    ia = np.clip(((a - a.min()) / (np.ptp(a) + 1e-12) * n_bins).astype(int), 0, n_bins - 1)
    ib = np.clip(((b - b.min()) / (np.ptp(b) + 1e-12) * n_bins).astype(int), 0, n_bins - 1)
    key = ia * n_bins + ib
    train = np.arange(len(y)) % 2 == 0
    test = ~train
    tot = np.bincount(key[train], weights=y[train], minlength=n_bins ** 2)
    cnt = np.bincount(key[train], minlength=n_bins ** 2)
    means = np.where(cnt > 0, tot / np.maximum(cnt, 1), y[train].mean())
    p = means[key[test]]
    return 1.0 - np.sum((y[test] - p) ** 2) / np.sum((y[test] - y[test].mean()) ** 2)


lags = np.arange(1, DELAYS)
scan = np.array([binned_r2(E[:, 0], E[:, L], Y[:, 0]) for L in lags])
best_lag = int(lags[np.argmax(scan)])
print(f"\n[FIND-2] lag scan peaks at L = {best_lag} "
      f"(R^2 = {scan.max():.4f}); true tau = {int(TAU / SAMPLE_DT)}")
runner = int(lags[np.argsort(scan)[-2]])
print(f"[FIND-2] runner-up L = {runner} (R^2 = {np.sort(scan)[-2]:.4f}) — the scan")
print("[FIND-2] identifies the right lag but is not sharply peaked.")

# ---------------------------------------------------------------------------
# 6. Finding 3 — KANDy on the two informative coordinates
# ---------------------------------------------------------------------------
state = np.column_stack([E[:, 0], E[:, best_lag]])
FEATURE_NAMES = ["x_t", f"x_(t-{best_lag})"]

# grid=8: the Hill function has a sharp knee near x=1, so the spline needs
# more knots than the default 5.
model = KANDy(lift=CustomLift(fn=lambda X: X, output_dim=2, name="delay_pair"),
              grid=8, k=3, steps=120, seed=SEED)
t0 = time.time()
model.fit(X=state, X_dot=Y, val_frac=0.15, test_frac=0.15, lamb=0.0,
          patience=40, verbose=False)
pred = model.predict(state)
r2_kan = 1.0 - np.sum((Y - pred) ** 2) / np.sum((Y - Y.mean()) ** 2)
print(f"\n[FIND-3] KANDy on 2 delay coordinates: R^2 = {r2_kan:.6f} "
      f"({time.time() - t0:.0f}s)")
print(f"[FIND-3] same accuracy as the {DELAYS}-feature linear model, with "
      f"{DELAYS - 2} fewer coordinates")

# ---------------------------------------------------------------------------
# 7. The real validation: do the spline edges match the true functions?
# ---------------------------------------------------------------------------
state_t = torch.tensor(state, dtype=torch.float32)
edge_checks = {}
for i, (name, truth) in enumerate([
    ("x_t", lambda v: -GAMMA * v),
    (f"x_(t-{best_lag})", lambda v: BETA * v / (1.0 + v ** N_HILL)),
]):
    xs, ys = get_edge_activation(model.model_, 0, i, 0, X=state_t)
    xs, ys = np.asarray(xs).ravel(), np.asarray(ys).ravel()
    ref = truth(xs)
    # edges are determined only up to an additive constant (absorbed by the bias)
    ref_c, ys_c = ref - ref.mean(), ys - ys.mean()
    r2_edge = 1.0 - np.sum((ys_c - ref_c) ** 2) / np.sum(ref_c ** 2)
    edge_checks[name] = (xs, ys_c, ref_c, r2_edge)
    print(f"[EDGE]   spline on {name:12s} vs ground truth: R^2 = {r2_edge:.5f}")
print("[EDGE]   the Hill term is in no built-in symbolic library — the spline")
print("[EDGE]   learned its shape from data alone.")

# ---------------------------------------------------------------------------
# 8. Rollout — integrate the LEARNED delay system and compare attractors
# ---------------------------------------------------------------------------
net = model.model_


def learned_rhs(x, x_tau):
    with torch.no_grad():
        return float(net(torch.tensor([[x, x_tau]], dtype=torch.float32))[0, 0])


DT_ROLL, T_ROLL = 0.1, 600.0
n_lag_roll = int(TAU / DT_ROLL)
n_roll = int(T_ROLL / DT_ROLL)
out = np.empty(n_roll + n_lag_roll + 1)
out[: n_lag_roll + 1] = series[: (n_lag_roll + 1) * int(DT_ROLL / DT_FINE):
                               int(DT_ROLL / DT_FINE)][: n_lag_roll + 1]

print(f"\n[EVAL]  rolling out the learned DDE for t={T_ROLL:.0f} ...")
t0 = time.time()
for i in range(n_lag_roll, n_lag_roll + n_roll):
    x, x_tau = out[i], out[i - n_lag_roll]
    x_mid = 0.5 * (out[i - n_lag_roll] + out[i - n_lag_roll + 1])
    k1 = learned_rhs(x, x_tau)
    k2 = learned_rhs(x + 0.5 * DT_ROLL * k1, x_mid)
    k3 = learned_rhs(x + 0.5 * DT_ROLL * k2, x_mid)
    k4 = learned_rhs(x + DT_ROLL * k3, out[i - n_lag_roll + 1])
    out[i + 1] = x + (DT_ROLL / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    if not np.isfinite(out[i + 1]):
        raise RuntimeError(f"rollout diverged at t={(i - n_lag_roll) * DT_ROLL:.1f}")

roll = out[n_lag_roll:]
print(f"[EVAL]  rollout {time.time() - t0:.0f}s")
print(f"[EVAL]  attractor range — learned [{roll.min():.4f}, {roll.max():.4f}]  "
      f"true [{series.min():.4f}, {series.max():.4f}]")
print(f"[EVAL]  attractor mean/std — learned {roll.mean():.4f}/{roll.std():.4f}  "
      f"true {series.mean():.4f}/{series.std():.4f}")
print("[EVAL]  (chaotic system: trajectories decorrelate, so compare the "
      "attractor,\n        not pointwise agreement)")

# ---------------------------------------------------------------------------
# 9. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/MackeyGlass", exist_ok=True)

# 9a. The scalar signal
fig, ax = plt.subplots(figsize=(10, 3.2))
t_show = np.arange(0, 1200, DT_FINE)
ax.plot(t_show, series[:len(t_show)], lw=0.7, color="#1f77b4")
ax.set_xlabel("time"); ax.set_ylabel("$x(t)$")
ax.set_title(r"Mackey-Glass, $\tau=17$ — the only observable")
fig.tight_layout()
fig.savefig("results/MackeyGlass/series.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 9b. Lag scan — the delay recovered from data
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(lags, scan, "o-", color="#1f77b4", ms=4, lw=1.2)
ax.axvline(TAU / SAMPLE_DT, color="#d62728", ls="--", lw=1.2,
           label=fr"true $\tau$ = {int(TAU/SAMPLE_DT)}")
ax.set_xlabel("candidate lag $L$")
ax.set_ylabel(r"$R^2$ of $(x_t, x_{t-L}) \to \dot{x}$")
ax.set_title("Recovering the delay from data")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/MackeyGlass/lag_scan.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 9c. THE money figure: learned spline edges vs the true functions
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
labels = [r"$-\gamma\,x_t$", r"$\beta\,x_{t-\tau}/(1+x_{t-\tau}^{10})$"]
for ax, (name, (xs, ys_c, ref_c, r2e)), lab in zip(axes, edge_checks.items(), labels):
    ax.plot(xs, ref_c, lw=2.6, color="#1f77b4", alpha=0.45, label=f"true {lab}")
    ax.plot(xs, ys_c, lw=1.2, color="#d62728", ls="--", label="KAN spline edge")
    ax.set_xlabel(name)
    ax.set_ylabel("edge output (mean-centred)")
    ax.set_title(f"$R^2$ = {r2e:.5f}", fontsize=10)
    ax.legend(fontsize=8)
fig.suptitle("The splines recover the mechanism — no dictionary used", fontsize=12)
fig.tight_layout()
fig.savefig("results/MackeyGlass/learned_edges.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 9d. Delay-coordinate attractor, true vs learned rollout
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), sharex=True, sharey=True)
lag_fine = int(TAU / DT_FINE)
axes[0].plot(series[lag_fine:], series[:-lag_fine], lw=0.25, color="#1f77b4")
axes[0].set_title("true")
lag_roll = int(TAU / DT_ROLL)
axes[1].plot(roll[lag_roll:], roll[:-lag_roll], lw=0.25, color="#d62728")
axes[1].set_title("KANDy rollout")
for ax in axes:
    ax.set_xlabel("$x(t)$"); ax.set_aspect("equal")
axes[0].set_ylabel(r"$x(t-\tau)$")
fig.suptitle("Delay-coordinate attractor", fontsize=12)
fig.tight_layout()
fig.savefig("results/MackeyGlass/attractor.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 9e. Loss curves
if getattr(model, "train_results_", None):
    fig, ax = plot_loss_curves(model.train_results_,
                               save="results/MackeyGlass/loss_curves")
    plt.close(fig)

print("[FIGS]  Saved results/MackeyGlass/")
