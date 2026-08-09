#!/usr/bin/env python3
"""Does width=[2,[2,1],1] give us the EQUATION for Holling Type II?

Not the R^2 — the closed form.  The target is

    dN/dt = r*N - (r/K)*N^2 - a*N*P/(1 + a*h*N)

and the obstruction for a purely additive KAN is the last term: a product of
two different variables.  A multiplication node factors it as P * g(N).

The reason this should be SYMBOLICALLY recoverable, not merely fittable, is
that g is a single affine-composed primitive:

    g(N) = a*N/(1+a*h*N) = 1/h  -  (1/h) / (1 + a*h*N)

i.e. c*f(a'N + b') + d with f = 1/x, c = -1/h = -10, a' = a*h = 0.12, b' = 1,
d = 1/h = 10.  That is exactly the form PyKAN's auto_symbolic fits per edge, so
'1/x' in the library should snap it exactly — no knowledge of a or h required.

Minimal sufficient architecture:
    input  [N, P]
    hidden 2 sum nodes + 1 multiplication node (arity 2)
    output 1                                   -> width=[2, [2,1], 1]
"""
import numpy as np
import torch
import sympy as sp

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
Y = (R * N_ * (1 - N_ / K) - pred).astype(np.float32)[:, None]   # dN/dt only

_ORIG_KAN = kc.KAN


def mult_factory(n_sum, n_mult):
    def factory(**kw):
        m, n = kw.pop("width")
        kw["width"] = [m, [n_sum, n_mult], n]
        kw["mult_arity"] = 2
        return MultKAN(**kw)
    return factory


def fit(n_sum, n_mult, steps=400):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    kc.KAN = mult_factory(n_sum, n_mult)
    try:
        m = KANDy(lift=CustomLift(fn=lambda Z: Z, output_dim=2, name="identity"),
                  grid=8, k=3, steps=steps, seed=SEED)
        m.fit(X=X, X_dot=Y, val_frac=0.15, test_frac=0.15, lamb=0.0,
              patience=100, verbose=False)
        return m
    finally:
        kc.KAN = _ORIG_KAN


def r2_of(model):
    p = model.predict(X)
    return float(1 - np.sum((Y - p) ** 2) / np.sum((Y - Y.mean()) ** 2))


if __name__ == "__main__":
    print("TARGET:  dN/dt = 1.0*N - 0.02*N^2 - 1.2*N*P/(1 + 0.12*N)")
    print("         equivalently  ... - P*g(N),  g(N) = 10 - 10/(1 + 0.12*N)\n")

    model = fit(2, 1)
    print(f"width=[2,[2,1],1]   fit R^2 = {r2_of(model):.8f}\n")

    kan = model.model_
    kan.save_act = True
    with torch.no_grad():
        kan(torch.tensor(X[:4096], dtype=torch.float32))

    for lib in (["x", "x^2", "1/x", "0"],
                ["x", "x^2", "x^3", "1/x", "0"]):
        m2 = fit(2, 1)
        k2 = m2.model_
        k2.save_act = True
        with torch.no_grad():
            k2(torch.tensor(X[:4096], dtype=torch.float32))
        print(f"--- auto_symbolic(lib={lib}) ---")
        try:
            k2.auto_symbolic(lib=lib)
            exprs, vars_ = k2.symbolic_formula()
            e = sp.expand(sp.sympify(exprs[0]))
            sub = {v: s for v, s in zip(vars_, sp.symbols("N P"))}
            e = e.xreplace(sub)
            e = e.xreplace({n: sp.Float(round(float(n), 4))
                            for n in e.atoms(sp.Float)})
            print(f"  dN/dt = {sp.simplify(e)}\n")
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}\n")
