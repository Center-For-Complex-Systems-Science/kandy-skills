#!/usr/bin/env python3
"""The two-stage recipe: recover the interaction SHAPE, then name it in 1-D.

Direct symbolic extraction from a multiplication-node KAN fails (see
mult_node_equation.py): gauge freedom in the product, the output edge snapping
to 1/x, and two-layer composition together produce an unreadable rational mess
even at R^2 = 0.9995.

But the equation is recoverable, because the interaction is a PRODUCT:

    dN/dt = f(N) - P * g(N)      =>      d(dN/dt)/dP = -g(N)

So probe the trained model for the P-slope on a grid of N.  That is a purely
UNIVARIATE curve, and univariate symbolic naming is the one thing splines and
auto_symbolic do reliably (cf. odes/pendulum_dictionary_completion_example.py
and odes/mackey_glass_delay_example.py).

Then the naming step is exact, because the Holling response is a single
affine-composed primitive:

    g(N) = a*N/(1 + a*h*N) = 1/h - (1/h)/(1 + a*h*N)
         = d + c/(alpha*N + 1)     with d = 1/h, c = -1/h, alpha = a*h

Fitting that 3-parameter form recovers the ECOLOGICAL constants:

    h = 1/d          a = alpha/h = alpha*d

Neither a nor h is supplied anywhere.
"""
import numpy as np
import torch
from scipy.optimize import curve_fit

import kandy.core as kc
from kan import MultKAN
from kandy import KANDy, CustomLift

SEED = 42
R, K, A, H = 1.0, 50.0, 1.2, 0.1
AH = A * H

rng = np.random.default_rng(0)
N_ = rng.uniform(0.05, 50.0, 20000)
P_ = rng.uniform(0.05, 30.0, 20000)
pred = (A * N_ * P_) / (1 + AH * N_)
X = np.column_stack([N_, P_]).astype(np.float32)
Y = (R * N_ * (1 - N_ / K) - pred).astype(np.float32)[:, None]

_ORIG_KAN = kc.KAN


def fit_mult(n_sum=2, n_mult=1, steps=400):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    def factory(**kw):
        m, n = kw.pop("width")
        kw["width"] = [m, [n_sum, n_mult], n]
        kw["mult_arity"] = 2
        return MultKAN(**kw)

    kc.KAN = factory
    try:
        m = KANDy(lift=CustomLift(fn=lambda Z: Z, output_dim=2, name="identity"),
                  grid=8, k=3, steps=steps, seed=SEED)
        m.fit(X=X, X_dot=Y, val_frac=0.15, test_frac=0.15, lamb=0.0,
              patience=100, verbose=False)
        return m
    finally:
        kc.KAN = _ORIG_KAN


if __name__ == "__main__":
    print("TARGET: dN/dt = N - 0.02*N^2 - a*N*P/(1+a*h*N),  a = 1.2, h = 0.1\n")
    model = fit_mult()
    p = model.predict(X)
    print(f"stage 0  width=[2,[2,1],1] fit R^2 = "
          f"{1 - np.sum((Y-p)**2)/np.sum((Y-Y.mean())**2):.8f}\n")

    # ---- stage 1: probe the P-slope => g(N) as a numerical curve -----------
    Ns = np.linspace(0.5, 48.0, 60)
    Ps = np.linspace(1.0, 28.0, 25)
    g_hat = []
    for n in Ns:
        grid = np.column_stack([np.full_like(Ps, n), Ps]).astype(np.float32)
        f = model.predict(grid)[:, 0]
        g_hat.append(-np.polyfit(Ps, f, 1)[0])          # -d(dN/dt)/dP
    g_hat = np.array(g_hat)
    g_true = A * Ns / (1 + AH * Ns)
    r2g = 1 - np.sum((g_true - g_hat) ** 2) / np.sum((g_true - g_true.mean()) ** 2)
    print(f"stage 1  recovered g(N) as a curve: R^2 vs truth = {r2g:.6f}")

    # ---- stage 2: name the univariate curve -------------------------------
    def form(n, c, alpha, d):
        return d + c / (alpha * n + 1.0)

    popt, _ = curve_fit(form, Ns, g_hat, p0=[-10.0, 0.1, 10.0], maxfev=200000)
    c_, alpha_, d_ = popt
    fit_r2 = 1 - np.sum((g_hat - form(Ns, *popt)) ** 2) / np.sum((g_hat - g_hat.mean()) ** 2)
    h_rec = 1.0 / d_
    a_rec = alpha_ * d_
    print(f"stage 2  named as d + c/(alpha*N + 1):  R^2 = {fit_r2:.8f}")
    print(f"           c     = {c_:9.4f}   (true {-1/H:9.4f})")
    print(f"           alpha = {alpha_:9.4f}   (true {AH:9.4f})")
    print(f"           d     = {d_:9.4f}   (true {1/H:9.4f})")
    print(f"\n  => h = 1/d     = {h_rec:.5f}   (true {H})   err {abs(h_rec-H)/H*100:.2f}%")
    print(f"  => a = alpha*d = {a_rec:.5f}   (true {A})   err {abs(a_rec-A)/A*100:.2f}%")

    # ---- the additive part, for the full equation --------------------------
    # Subtract the recovered interaction INSIDE the training range rather than
    # evaluating at P -> 0, which extrapolates outside it and biased the linear
    # coefficient by ~3% with a spurious constant.
    g_named = form(Ns, *popt)
    f_hat = []
    for n, gv in zip(Ns, g_named):
        grid = np.column_stack([np.full_like(Ps, n), Ps]).astype(np.float32)
        f = model.predict(grid)[:, 0]
        f_hat.append(np.mean(f + Ps * gv))              # f(N) = model + P*g(N)
    cf = np.polyfit(Ns, np.array(f_hat), 2)
    print(f"\n  additive part f(N) ~ {cf[0]:+.5f}*N^2 {cf[1]:+.5f}*N {cf[2]:+.5f}")
    print(f"                 true {-R/K:+.5f}*N^2 {R:+.5f}*N {0.0:+.5f}")
    print(f"\n  RECOVERED EQUATION:")
    print(f"    dN/dt = {cf[1]:.4f}*N {cf[0]:+.4f}*N^2 - {a_rec:.4f}*N*P/(1 + {alpha_:.4f}*N)")
    print(f"    true   1.0000*N -0.0200*N^2 - 1.2000*N*P/(1 + 0.1200*N)")
