#!/usr/bin/env python3
"""KANDy example: reconstructing the network — who couples to whom.

Given only the time series of N coupled oscillators, recover the ADJACENCY
MATRIX.  The unknown here is not a coefficient in a known equation; it is the
wiring diagram itself.

    dtheta_i/dt = omega_i + K * sum_j A_ij * g(theta_j - theta_i)

with a sparse, unknown A, heterogeneous natural frequencies omega_i, and — the
part that matters — an unknown coupling function g.  This script uses a
deliberately non-sinusoidal one,

    g(d) = sin(d - alpha) + 0.3 * sin(2d)          alpha = 0.3

i.e. Kuramoto-Sakaguchi phase lag plus a second harmonic.

The recipe: one regression per node
-----------------------------------
A joint fit is the wrong shape.  The full pairwise lift is N*(N-1) features
(90 for N=10) shared across N outputs — 900 spline edges, and every output gets
access to every pair, including pairs it cannot physically depend on.  Instead
fit each node separately:

    for node i:  features = [wrap(theta_j - theta_i) for j != i]     (N-1)
                 target   = dtheta_i/dt                             (1 output)

That is N tiny KANs of N-1 edges each — 90 edges total instead of 900 — and it
builds the structural knowledge (node i's equation only involves differences
*to* node i) into the problem instead of hoping the fit discovers it.

Then the edge on feature `theta_j - theta_i` IS the coupling term
K*A_ij*g(.).  Its AMPLITUDE gives the coupling strength; a flat edge means no
link.  Reading the network off the splines is the whole method.

Why a KAN rather than least squares
-----------------------------------
With features `sin(theta_j - theta_i)`, plain least squares solves this
exactly — when you already know the coupling is sinusoidal.  The point is what
happens when you do not.  This script measures both:

    least squares assuming sin(.)      edges 0.88 +- 0.26, and non-edges pick
                                       up weight as large as 0.64 — larger
                                       than the weakest true edge, so the
                                       topology no longer separates at all
    KANDy on the raw phase difference  edges 1.13 +- 0.13, non-edges at most
                                       0.17 — separated by ~6x, and the
                                       coupling function itself is recovered
                                       at R^2 ~ 0.99

The lesson is sharper than "the fit is worse".  Misspecifying the coupling
function does not merely bias the weights, it INVENTS EDGES: the part of the
dynamics that sin(.) cannot represent gets redistributed onto whichever other
phase differences happen to correlate with it.  A basis error becomes a
topology error, which is the one thing you were trying to measure.

The identifiability limit: never reconstruct a locked network
-------------------------------------------------------------
If the oscillators are phase-locked, every difference theta_j - theta_i sits at
a constant, and the data says nothing about how the coupling would respond if
they moved.  This script measures the collapse using the TRUE coupling basis,
so the only thing failing is identifiability:

    transient included, r ~ 0.98      edges 4.000 +- 0.000, non-edges 0.0000
    locked state only,  r ~ 0.99      edges 2.29 +- 1.27,  non-edges up to 2.31

Same system, same estimator, same amount of data — only the observation window
differs, and the second one is worthless.  Reconstruct from transients, from
many initial conditions, or from a regime below full synchronisation.  Same
family of failure as the conservation law in `mathbio/sir_example.py` and the
slaved shift mode in `fluids/vortex_shedding_example.py`.

Runtime: ~1 minute.
KAN:  N separate models, width = [N-1, 1]
"""

import os
import time
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from kandy import KANDy, CustomLift
from kandy.plotting import get_edge_activation, use_pub_style

# ---------------------------------------------------------------------------
# 0. Reproducibility, the hidden network, and the unknown coupling
# ---------------------------------------------------------------------------
SEED = 7
np.random.seed(SEED)
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

N = 10
EDGE_PROB = 0.30
K_COUPLING = 1.0
ALPHA = 0.3

ADJ = (rng.random((N, N)) < EDGE_PROB).astype(float)
ADJ = np.triu(ADJ, 1)
ADJ = ADJ + ADJ.T                       # undirected
np.fill_diagonal(ADJ, 0.0)
OMEGA = rng.uniform(-1.0, 1.0, N)

n_edges = int(ADJ.sum() // 2)
print(f"[NET]  {N} oscillators, {n_edges} edges of {N*(N-1)//2} possible "
      f"(mean degree {ADJ.sum(0).mean():.1f})")


def wrap(x):
    """Wrap phase differences to (-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def coupling(d):
    """The UNKNOWN coupling function: phase lag + second harmonic."""
    return np.sin(d - ALPHA) + 0.3 * np.sin(2 * d)


def rhs(theta, K=K_COUPLING):
    D = theta[None, :] - theta[:, None]          # D[i, j] = theta_j - theta_i
    return OMEGA + K * (ADJ * coupling(D)).sum(axis=1)


def order_parameter(theta_series):
    return float(np.abs(np.exp(1j * theta_series).mean(axis=1)).mean())


def simulate(K=K_COUPLING, n_runs=20, t_end=25.0, dt=0.05, burn=0.0):
    """Many short runs from random initial conditions."""
    states, targets = [], []
    t_eval = np.arange(0, t_end, dt)
    for _ in range(n_runs):
        theta0 = rng.uniform(-np.pi, np.pi, N)
        sol = solve_ivp(lambda t, s: rhs(s, K), [0, t_end], theta0,
                        t_eval=t_eval, rtol=1e-9, atol=1e-11)
        th = sol.y.T[t_eval >= burn]
        states.append(th)
        targets.append(np.stack([rhs(s, K) for s in th]))
    return np.concatenate(states), np.concatenate(targets)


states, targets = simulate()
print(f"[DATA] {len(states)} samples from 20 runs, "
      f"order parameter r = {order_parameter(states):.3f}")


def separation(W):
    """Edge vs non-edge weight statistics; the topology is recovered iff
    the smallest true-edge weight exceeds the largest non-edge weight."""
    on = np.abs(W[ADJ > 0])
    off = np.abs(W[(ADJ == 0) & ~np.eye(N, dtype=bool)])
    return on, off, off.max() < on.min()


# ---------------------------------------------------------------------------
# 1. Baseline — least squares that ASSUMES sinusoidal coupling
# ---------------------------------------------------------------------------
def lsq_network(S, Y, basis):
    W = np.zeros((N, N))
    for i in range(N):
        cols = [j for j in range(N) if j != i]
        F = np.column_stack([basis(wrap(S[:, cols] - S[:, [i]])),
                             np.ones(len(S))])
        c, *_ = np.linalg.lstsq(F, Y[:, i], rcond=None)
        W[i, cols] = c[:-1]
    return W


W_sin = lsq_network(states, targets, np.sin)
on, off, sep = separation(W_sin)
print(f"\n[LSQ]  assuming sin(.): edges {on.mean():.3f} +- {on.std():.3f}, "
      f"non-edges up to {off.max():.4f}")
print(f"[LSQ]  topology separated: {sep}  "
      f"(needs max non-edge < min edge = {on.min():.4f})")
print("[LSQ]  The mismatch between sin(.) and the true coupling is absorbed by")
print("[LSQ]  OTHER phase differences — a basis error becomes false edges.")

# ---------------------------------------------------------------------------
# 2. KANDy — one model per node, on the RAW phase differences
# ---------------------------------------------------------------------------
N_FIT = 1500
sub = np.linspace(0, len(states) - 1, N_FIT).astype(int)
amp = np.zeros((N, N))
edge_shape = None

print(f"\n[KAN]  fitting {N} per-node models "
      f"(width [{N-1}, 1], {N_FIT} samples each) ...")
t0 = time.time()
for i in range(N):
    cols = [j for j in range(N) if j != i]
    F = wrap(states[sub][:, cols] - states[sub][:, [i]])
    # grid=8: the coupling carries a second harmonic, so the spline needs
    # enough knots to resolve two humps over (-pi, pi].
    model = KANDy(lift=CustomLift(fn=lambda Z: Z, output_dim=N - 1,
                                  name="phase_diffs"),
                  grid=8, k=3, steps=40, seed=SEED)
    model.fit(X=F, X_dot=targets[sub][:, i:i + 1], val_frac=0.15,
              test_frac=0.15, lamb=0.0, patience=30, verbose=False)
    F_t = torch.tensor(F[:2048], dtype=torch.float32)
    for c, j in enumerate(cols):
        x, y = get_edge_activation(model.model_, 0, c, 0, X=F_t)
        y = np.asarray(y).ravel()
        amp[i, j] = 0.5 * (y.max() - y.min())        # edge amplitude
        if edge_shape is None and ADJ[i, j] > 0:
            edge_shape = (np.asarray(x).ravel(), y, i, j)

on_k, off_k, sep_k = separation(amp)
print(f"[KAN]  done in {time.time() - t0:.0f}s")
print(f"[KAN]  edge amplitude     {on_k.mean():.4f} +- {on_k.std():.4f} "
      f"(min {on_k.min():.4f})")
print(f"[KAN]  non-edge amplitude max {off_k.max():.4f} "
      f"(mean {off_k.mean():.4f})")
print(f"[KAN]  topology separated: {sep_k}   "
      f"separation ratio {on_k.min() / off_k.max():.2f}x")

# Did the spline also recover the coupling FUNCTION?
xs, ys, ei, ej = edge_shape
true_g = K_COUPLING * coupling(xs)
true_c, ys_c = true_g - true_g.mean(), ys - ys.mean()
r2_shape = 1.0 - np.sum((ys_c - true_c) ** 2) / np.sum(true_c ** 2)
print(f"[KAN]  edge ({ei},{ej}) spline vs the true coupling g: "
      f"R^2 = {r2_shape:.5f}")
print("[KAN]  Neither the wiring nor the coupling function was assumed.")

# ---------------------------------------------------------------------------
# 3. The identifiability limit — a locked network tells you nothing
#
#    Uses the TRUE coupling basis, so misspecification is ruled out and the
#    only thing that can fail is identifiability.
# ---------------------------------------------------------------------------
print("\n[SYNC] Same estimator, TRUE coupling basis, two observation windows:")
sync_rows = []
for K, burn, tag in [(4.0, 0.0, "transient included"),
                     (4.0, 18.0, "locked state only ")]:
    S_s, Y_s = simulate(K=K, burn=burn)
    W = lsq_network(S_s, Y_s, coupling)
    o_on, o_off, o_sep = separation(W)
    r = order_parameter(S_s)
    sync_rows.append((tag, r, o_on, o_off, o_sep))
    print(f"[SYNC]   {tag}  r={r:.3f}  edges {o_on.mean():.3f} +- "
          f"{o_on.std():.3f} (true {K})  non-edges up to {o_off.max():.4f}  "
          f"separated={o_sep}")
print("[SYNC] Locked phases are constants: the data never probes how the")
print("[SYNC] coupling responds, so the network is simply not identifiable.")
print("[SYNC] Reconstruct from transients or below full synchronisation.")

# ---------------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------------
use_pub_style()
os.makedirs("results/NetworkReconstruction", exist_ok=True)

# 4a. True vs recovered adjacency
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
panels = [(ADJ, "true adjacency"),
          (np.abs(W_sin), "least squares assuming sin"),
          (amp, "KANDy edge amplitudes")]
for ax, (M, title) in zip(axes, panels):
    im = ax.imshow(M, cmap="viridis")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("j"); ax.set_ylabel("i")
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Recovering the wiring diagram", fontsize=12)
fig.tight_layout()
fig.savefig("results/NetworkReconstruction/adjacency.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)

# 4b. Separation of edge vs non-edge weights
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (W, title) in zip(axes, [(np.abs(W_sin), "assuming sin(.)"),
                                 (amp, "KANDy, raw phase differences")]):
    o_on, o_off, o_sep = separation(W)
    ax.hist(o_off, bins=18, color="#1f77b4", alpha=0.75, label="non-edges")
    ax.hist(o_on, bins=18, color="#d62728", alpha=0.75, label="true edges")
    ax.set_xlabel("recovered weight")
    ax.set_title(f"{title}\nseparated: {o_sep}", fontsize=10)
    ax.legend(fontsize=8)
axes[0].set_ylabel("count")
fig.suptitle("A basis error turns into false edges", fontsize=12)
fig.tight_layout()
fig.savefig("results/NetworkReconstruction/separation.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)

# 4c. The recovered coupling function
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(xs, true_c, lw=2.6, color="#1f77b4", alpha=0.45,
        label=r"true $g(d)=\sin(d-\alpha)+0.3\sin 2d$")
ax.plot(xs, ys_c, lw=1.3, color="#d62728", ls="--", label="KAN spline edge")
ax.set_xlabel(r"$\theta_j-\theta_i$")
ax.set_ylabel("edge output (mean-centred)")
ax.set_title(f"Coupling function recovered, $R^2$={r2_shape:.4f}")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/NetworkReconstruction/coupling_function.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)

# 4d. The synchronisation limit
fig, ax = plt.subplots(figsize=(6.4, 4))
labels = [t for t, *_ in sync_rows]
xpos = np.arange(len(sync_rows))
ax.bar(xpos - 0.18, [r[2].mean() for r in sync_rows], 0.36,
       yerr=[r[2].std() for r in sync_rows], capsize=4,
       color="#d62728", label="true edges")
ax.bar(xpos + 0.18, [r[3].max() for r in sync_rows], 0.36,
       color="#1f77b4", label="largest non-edge")
ax.axhline(4.0, color="k", ls="--", lw=1.0, label="true coupling K=4")
ax.set_xticks(xpos)
ax.set_xticklabels([f"{t}\n$r$={r:.3f}" for t, r, *_ in sync_rows], fontsize=8)
ax.set_ylabel("recovered weight")
ax.set_title("A phase-locked network is unidentifiable")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/NetworkReconstruction/sync_limit.png", dpi=300,
            bbox_inches="tight")
plt.close(fig)

print("\n[FIGS]  Saved results/NetworkReconstruction/")
