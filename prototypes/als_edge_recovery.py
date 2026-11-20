#!/usr/bin/env python3
"""Recover the Holling functional response by ALTERNATING LEAST SQUARES.

The interaction term is a product of two univariate functions,

    dN/dt = f(N) + u(N) * v(P)          with u(N)*v(P) = -a*N*P/(1+a*h*N)

which is bilinear in the two factors.  Fix one and solving for the other is an
ordinary linear least squares; alternate and each half-step is convex.  This
sidesteps everything that made direct symbolic extraction fail on a
multiplication-node KAN: no gauge ambiguity left at the end (we fix it), no
two-layer composition, no output edge snapping to 1/x.

Represent every univariate factor in a hat-function basis, so each is a free
shape exactly as a spline edge would be:

    f = B_N a  +  B_P b  +  (B_N u) * (B_P v)

    given v:  w = B_P v         ->  linear in [a, b, u]
    given u:  z = B_N u         ->  linear in [a, b, v]

Gauge: the product is invariant under u -> c*u, v -> v/c, so at the end rescale
so that v has unit slope in P.  Then u(N) IS -g(N) and can be named in 1-D:

    g(N) = a*N/(1+a*h*N) = 1/h - (1/h)/(1 + a*h*N) = d + c/(alpha*N + 1)
    =>  h = 1/d,   a = alpha*d
"""
import numpy as np
from scipy.optimize import curve_fit

R, K, A, H, E, M = 1.0, 50.0, 1.2, 0.1, 0.6, 0.4
AH = A * H


def hats(x, n=40, lo=None, hi=None):
    kn = np.linspace(lo if lo is not None else x.min(),
                     hi if hi is not None else x.max(), n)
    B = np.zeros((len(x), n))
    for i, k in enumerate(kn):
        a = kn[i - 1] if i > 0 else kn[0] - (kn[1] - kn[0])
        b = kn[i + 1] if i < n - 1 else kn[-1] + (kn[-1] - kn[-2])
        B[:, i] = np.minimum(np.clip((x - a) / (k - a), 0, 1),
                             np.clip((b - x) / (b - k), 0, 1))
    return B, kn


def als(N_, P_, target, n_basis=40, iters=40, seed=0):
    """Alternating least squares for  target = f(N) + u(N)*v(P).

    GAUGE.  The decomposition f(N) + b(P) + u(N)v(P) is invariant under
        u -> u + delta,  b -> b - delta*v        and
        v -> v + eps,    f -> f - eps*u
    so u and v are each defined only up to an additive constant, which leaks
    into the additive parts and wrecks f(N).  Two constraints remove it:

      (1) v(0) = 0   — no interaction when P = 0 (no predation without
          predators).  Imposed by dropping the P-basis hat centred at P = 0,
          the only one non-zero there.
      (2) no standalone b(P) term — dN/dt has no pure-predator contribution.
          Without this, u -> u + delta is absorbable and delta stays free.

    What remains is the scale u -> c*u, v -> v/c, fixed after the fact by
    normalising v to unit slope in P.
    """
    BN, knN = hats(N_, n_basis)
    BP, knP = hats(P_, n_basis, lo=0.0)
    BP = BP[:, 1:]                                    # (1) v(0) = 0
    nb, nbp = BN.shape[1], BP.shape[1]
    v = np.linalg.lstsq(BP, P_, rcond=None)[0]        # start from v(P) = P
    hist = []
    for _ in range(iters):
        # --- solve [a, u] with v fixed  (2) no b(P) term -----------------
        w = BP @ v
        G = np.hstack([BN, BN * w[:, None]])
        sol, *_ = np.linalg.lstsq(G, target, rcond=None)
        a, u = sol[:nb], sol[nb:]
        # --- solve [a, v] with u fixed -----------------------------------
        z = BN @ u
        G = np.hstack([BN, BP * z[:, None]])
        sol, *_ = np.linalg.lstsq(G, target, rcond=None)
        a, v = sol[:nb], sol[nb:]
        r = target - (BN @ a + (BN @ u) * (BP @ v))
        hist.append(float(np.sqrt(np.mean(r ** 2))))
    return a, None, u, v, BN, BP, knN, knP, hist


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N_ = rng.uniform(0.05, 50.0, 30000)
    P_ = rng.uniform(0.05, 30.0, 30000)
    pred = (A * N_ * P_) / (1 + AH * N_)
    target = R * N_ * (1 - N_ / K) - pred

    print("TARGET  dN/dt = N - 0.02*N^2 - a*N*P/(1+a*h*N),  a=1.2, h=0.1")
    print("Nothing about the Holling form is supplied — only that the")
    print("interaction is a product of a function of N and a function of P.\n")

    a, b, u, v, BN, BP, knN, knP, hist = als(N_, P_, target)
    fit = BN @ a + (BN @ u) * (BP @ v)
    r2 = 1 - np.sum((target - fit) ** 2) / np.sum((target - target.mean()) ** 2)
    print(f"ALS converged: rmse {hist[0]:.4f} -> {hist[-1]:.2e} over {len(hist)} sweeps")
    print(f"  overall R^2 = {r2:.10f}\n")

    # --- gauge fix: rescale so v(P) has unit slope in P ------------------
    vP = BP @ v
    slope = np.polyfit(P_, vP, 1)[0]
    u_fixed = u * slope                       # u -> u*slope, v -> v/slope
    v_fixed = vP / slope

    # how close is the P-factor to being exactly P?
    r2_v = 1 - np.sum((P_ - v_fixed) ** 2) / np.sum((P_ - P_.mean()) ** 2)
    print(f"  recovered v(P) vs P:      R^2 = {r2_v:.8f}  (gauge-fixed)")

    # --- the N-factor should be -g(N) ------------------------------------
    grid = np.linspace(1.0, 48.0, 200)
    BG, _ = hats(grid, len(knN), knN[0], knN[-1])
    uN = BG @ u_fixed
    g_true = -A * grid / (1 + AH * grid)
    r2_u = 1 - np.sum((g_true - uN) ** 2) / np.sum((g_true - g_true.mean()) ** 2)
    print(f"  recovered u(N) vs -g(N):  R^2 = {r2_u:.8f}\n")

    # --- name the univariate factor in 1-D -------------------------------
    form = lambda n, c, alpha, d: d + c / (alpha * n + 1.0)
    popt, _ = curve_fit(form, grid, -uN, p0=[-10.0, 0.1, 10.0], maxfev=400000)
    c_, alpha_, d_ = popt
    h_rec = 1.0 / d_                    # g -> 1/h as N -> inf
    a_rec = alpha_ * d_                 # alpha = a*h, so a = alpha/h = alpha*d
    print(f"  named as d + c/(alpha*N+1):  c={c_:.4f} alpha={alpha_:.5f} d={d_:.4f}")
    print(f"    h = 1/d     = {h_rec:.5f}   (true {H})    err {abs(h_rec-H)/H*100:.2f}%")
    print(f"    a = alpha*d = {a_rec:.5f}   (true {A})    err {abs(a_rec-A)/A*100:.2f}%")

    # --- additive part ----------------------------------------------------
    fN = BG @ a
    cf = np.polyfit(grid, fN, 2)
    print(f"\n  additive f(N) ~ {cf[0]:+.5f}*N^2 {cf[1]:+.5f}*N")
    print(f"           true  {-R/K:+.5f}*N^2 {R:+.5f}*N")
    print(f"\n  RECOVERED  dN/dt = {cf[1]:.4f}*N {cf[0]:+.4f}*N^2 "
          f"- {a_rec:.4f}*N*P/(1 + {alpha_:.4f}*N)")
    print(f"  TRUE             = 1.0000*N -0.0200*N^2 "
          f"- 1.2000*N*P/(1 + 0.1200*N)")
