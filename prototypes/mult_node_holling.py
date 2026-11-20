#!/usr/bin/env python3
"""Prototype: does a multiplication node let KANDy DISCOVER the Holling term?

Separable KANDy cannot represent  pred = a*N*P/(1+a*h*N)  from the raw state:
it is a product of two different variables, and a sum of univariate functions
has a hard ceiling of R^2 = 0.9376 there regardless of spline resolution.  A
KAN multiplication node computes psi_a(.)*psi_b(.), factoring the term as
P * g(N) with g(N) = a*N/(1+a*h*N) univariate — which needs no knowledge of a
or h.

PyKAN's MultKAN supports this via width=[m, [n_sum, n_mult], n] and mult_arity.
KANDy hardcodes width=[m, n] ("depth fixed at 1 — never expose as param"), so
this patches the KAN constructor kandy.core uses instead of editing the
installed package.

Measured against:
    additive ceiling, identity lift [N, P]   R^2 = 0.937630
    convex mult-node fit (P-factor fixed)    R^2 = 0.99999987
"""
import sys
import numpy as np
import torch

import kandy.core as kc
from kan import MultKAN
from kandy import KANDy, CustomLift

SEED = 42
R, K, A, H, E, M = 1.0, 50.0, 1.2, 0.1, 0.6, 0.4
AH = A * H

rng = np.random.default_rng(0)
N_ = rng.uniform(0.05, 50.0, 40000)
P_ = rng.uniform(0.05, 30.0, 40000)
pred = (A * N_ * P_) / (1 + AH * N_)
X = np.column_stack([N_, P_]).astype(np.float32)
Y = np.column_stack([R * N_ * (1 - N_ / K) - pred,
                     E * pred - M * P_]).astype(np.float32)
Y_SCALE = Y.std(axis=0, keepdims=True)

_ORIG_KAN = kc.KAN


def r2(p, t):
    return [float(1 - np.sum((t[:, i] - p[:, i]) ** 2)
                  / np.sum((t[:, i] - t[:, i].mean()) ** 2))
            for i in range(t.shape[1])]


def make_mult_kan(n_sum, n_mult):
    def factory(**kw):
        m, n = kw.pop("width")
        kw["width"] = [m, [n_sum, n_mult], n]
        kw["mult_arity"] = 2
        return MultKAN(**kw)
    return factory


def run(tag, n_sum, n_mult, steps=300, lamb=0.0):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    kc.KAN = make_mult_kan(n_sum, n_mult) if n_mult else _ORIG_KAN
    try:
        model = KANDy(lift=CustomLift(fn=lambda Z: Z, output_dim=2, name="identity"),
                      grid=8, k=3, steps=steps, seed=SEED)
        model.fit(X=X, X_dot=Y / Y_SCALE, val_frac=0.15, test_frac=0.15,
                  lamb=lamb, patience=100, verbose=False)
        out = model.predict(X) * Y_SCALE
        print(f"  {tag:44s} R^2 = {np.round(r2(out, Y), 6)}", flush=True)
        return model
    except Exception as exc:
        print(f"  {tag:44s} FAILED: {type(exc).__name__}: {exc}", flush=True)
        return None
    finally:
        kc.KAN = _ORIG_KAN


def structure_probe(model):
    """The true system has d(dN/dt)/dP = -g(N), independent of P.  A genuine
    product P*g(N) reproduces that; a merely flexible surface need not."""
    print("\n" + "=" * 74)
    print("STRUCTURE PROBE: true d(dN/dt)/dP = -g(N), constant in P")
    print("=" * 74)
    g = lambda n: A * n / (1 + AH * n)
    Ps = np.linspace(1.0, 28.0, 40)
    print(f"  {'N':>6} {'true -g(N)':>12} {'learned slope':>14} {'slope drift over P':>20}")
    errs = []
    for n in (1.0, 5.0, 10.0, 25.0, 40.0):
        grid = np.column_stack([np.full_like(Ps, n), Ps]).astype(np.float32)
        f = model.predict(grid)[:, 0] * Y_SCALE[0, 0]
        slope = np.polyfit(Ps, f, 1)[0]
        s_lo = np.polyfit(Ps[:20], f[:20], 1)[0]
        s_hi = np.polyfit(Ps[20:], f[20:], 1)[0]
        errs.append(abs(slope - (-g(n))))
        print(f"  {n:6.1f} {-g(n):12.4f} {slope:14.4f} {abs(s_hi - s_lo):20.4f}")
    print(f"\n  max |learned slope - (-g(N))| = {max(errs):.4f}"
          f"   (|g| spans {g(1.0):.2f}..{g(40.0):.2f})")


if __name__ == "__main__":
    print("Identity lift [N, P] throughout — no cross-term, no oracle feature.\n")
    print("  reference: additive ceiling      R^2 = 0.937630")
    print("  reference: convex mult-node fit  R^2 = 0.99999987\n")
    run("additive KANDy (width=[2,2], current)", 0, 0)
    best = run("mult node  width=[2,[4,2],2]", 4, 2)
    if best is not None:
        structure_probe(best)
