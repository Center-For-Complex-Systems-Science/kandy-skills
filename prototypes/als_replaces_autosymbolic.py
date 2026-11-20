#!/usr/bin/env python3
"""ALS as a drop-in replacement for auto_symbolic on ONE term.

Train the multiplication-node model normally.  auto_symbolic handles the
additive edges fine — it is only the product that it cannot read back (it
collapses the model from R^2 0.99997 to 0.754).  So do not run it there.
Instead read the product term off the TRAINED MODEL by alternating least
squares, and name the resulting univariate factor in 1-D.

    train KAN  ->  evaluate on a grid  ->  ALS factors the product
                                       ->  name u(N) with a 1-D fit

Evaluating the trained model on a regular grid is what makes this easy: the
network is smooth and can be sampled anywhere, so the interaction becomes a
dense matrix that is exactly rank 1 once the additive part is removed.

Ambiguity in the split between additive and product parts is removed the same
way as in als_edge_recovery.py: require v(0) = 0 and allow no standalone
function of P.  The leftover scale (u -> c*u, v -> v/c) is fixed by
normalising v to unit slope in P.
"""
import numpy as np
import torch
from scipy.optimize import curve_fit

from kan import MultKAN

SEED = 42
R, K, A, H = 1.0, 50.0, 1.2, 0.1
AH = A * H

rng = np.random.default_rng(0)
N_ = rng.uniform(0.05, 50.0, 8000)
P_ = rng.uniform(0.05, 30.0, 8000)
target = (R * N_ * (1 - N_ / K)
          - (A * N_ * P_) / (1 + AH * N_)).astype(np.float32)
Xn = np.column_stack([N_, P_]).astype(np.float32)
X_MU, X_SD = Xn.mean(0, keepdims=True), Xn.std(0, keepdims=True)
XTn = torch.tensor((Xn - X_MU) / X_SD)
YT = torch.tensor(target[:, None])
Y_SD = float(target.std())


def train_mult_kan(steps=120):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    m = MultKAN(width=[2, [2, 1], 1], mult_arity=2, grid=8, k=3, seed=SEED)
    m.update_grid_from_samples(XTn)
    opt = torch.optim.LBFGS(m.parameters(), lr=1.0, max_iter=20,
                            history_size=10, line_search_fn="strong_wolfe")
    for s in range(steps):
        if s in (5, 15, 30):
            m.update_grid_from_samples(XTn)

        def closure():
            opt.zero_grad()
            loss = torch.mean((m(XTn) - YT / Y_SD) ** 2)
            loss.backward()
            return loss
        opt.step(closure)
    with torch.no_grad():
        p = m(XTn).numpy().ravel() * Y_SD
    return m, 1 - np.sum((target - p) ** 2) / np.sum((target - target.mean()) ** 2)


def model_on_grid(m, Ng, Pg):
    NN, PP = np.meshgrid(Ng, Pg, indexing="ij")
    raw = np.column_stack([NN.ravel(), PP.ravel()]).astype(np.float32)
    with torch.no_grad():
        out = m(torch.tensor((raw - X_MU) / X_SD)).numpy().ravel() * Y_SD
    return out.reshape(NN.shape)


def hats(x, n, lo, hi):
    kn = np.linspace(lo, hi, n)
    B = np.zeros((len(x), n))
    for i, k in enumerate(kn):
        a = kn[i - 1] if i > 0 else kn[0] - (kn[1] - kn[0])
        b = kn[i + 1] if i < n - 1 else kn[-1] + (kn[-1] - kn[-2])
        B[:, i] = np.minimum(np.clip((x - a) / (k - a), 0, 1),
                             np.clip((b - x) / (b - k), 0, 1))
    return B


def als_readout(M, Ng, Pg, n_basis=40, iters=30):
    """Factor M(N,P) = f(N) + u(N)*v(P) off the grid of model outputs."""
    NN, PP = np.meshgrid(Ng, Pg, indexing="ij")
    n, p, y = NN.ravel(), PP.ravel(), M.ravel()
    BN = hats(n, n_basis, Ng.min(), Ng.max())
    BP = hats(p, n_basis, 0.0, Pg.max())[:, 1:]        # v(0) = 0
    nb = BN.shape[1]
    v = np.linalg.lstsq(BP, p, rcond=None)[0]
    for _ in range(iters):
        w = BP @ v
        sol, *_ = np.linalg.lstsq(np.hstack([BN, BN * w[:, None]]), y, rcond=None)
        a, u = sol[:nb], sol[nb:]
        z = BN @ u
        sol, *_ = np.linalg.lstsq(np.hstack([BN, BP * z[:, None]]), y, rcond=None)
        a, v = sol[:nb], sol[nb:]
    resid = y - (BN @ a + (BN @ u) * (BP @ v))
    return a, u, v, float(np.sqrt(np.mean(resid ** 2)))


if __name__ == "__main__":
    m, r2 = train_mult_kan()
    print(f"trained mult-node KAN: fit R^2 = {r2:.8f}")
    print("auto_symbolic on this model collapses it to R^2 0.754 — so skip it "
          "for the product\nand read that term off with ALS instead.\n")

    Ng = np.linspace(1.0, 48.0, 120)
    Pg = np.linspace(0.5, 29.0, 120)
    M = model_on_grid(m, Ng, Pg)
    a, u, v, rmse = als_readout(M, Ng, Pg)
    print(f"ALS readout residual rmse = {rmse:.4e}  "
          f"(model output range {M.min():.1f}..{M.max():.1f})")

    # scale: normalise v to unit slope in P
    BPg = hats(Pg, 40, 0.0, Pg.max())[:, 1:]
    vP = BPg @ v
    slope = np.polyfit(Pg, vP, 1)[0]
    r2_v = 1 - np.sum((Pg - vP / slope) ** 2) / np.sum((Pg - Pg.mean()) ** 2)
    print(f"  recovered v(P) vs P:     R^2 = {r2_v:.8f}")

    BNg = hats(Ng, 40, Ng.min(), Ng.max())
    uN = (BNg @ u) * slope
    g_true = -A * Ng / (1 + AH * Ng)
    r2_u = 1 - np.sum((g_true - uN) ** 2) / np.sum((g_true - g_true.mean()) ** 2)
    print(f"  recovered u(N) vs -g(N): R^2 = {r2_u:.8f}\n")

    form = lambda x, c, alpha, d: d + c / (alpha * x + 1.0)
    popt, _ = curve_fit(form, Ng, -uN, p0=[-10.0, 0.1, 10.0], maxfev=400000)
    c_, alpha_, d_ = popt
    h_rec, a_rec = 1.0 / d_, alpha_ / d_
    a_rec = alpha_ * d_
    print(f"  named d + c/(alpha*N+1):  c={c_:.4f} alpha={alpha_:.5f} d={d_:.4f}")
    print(f"    h = 1/d     = {h_rec:.5f}  (true {H})   err {abs(h_rec-H)/H*100:.2f}%")
    print(f"    a = alpha*d = {a_rec:.5f}  (true {A})   err {abs(a_rec-A)/A*100:.2f}%")

    fN = BNg @ a
    cf = np.polyfit(Ng, fN, 2)
    print(f"\n  additive f(N) ~ {cf[0]:+.5f}*N^2 {cf[1]:+.5f}*N")
    print(f"           true  {-R/K:+.5f}*N^2 {R:+.5f}*N")
    print(f"\n  RECOVERED dN/dt = {cf[1]:.4f}*N {cf[0]:+.4f}*N^2 "
          f"- {a_rec:.4f}*N*P/(1 + {alpha_:.4f}*N)")
    print(f"  TRUE            = 1.0000*N -0.0200*N^2 - 1.2000*N*P/(1 + 0.1200*N)")
