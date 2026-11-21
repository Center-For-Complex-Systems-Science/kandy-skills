#!/usr/bin/env python3
"""ALS as the TRAINING SCHEME for a multiplication-node KANDy model.

Architecture is what KANDy builds once the width passthrough exists:
width=[2,[2,1],1], mult_arity=2.  In MultKAN that lays out as

    layer 0:  in_dim 2 (N, P)  ->  4 subnodes
              subnodes 0,1 = the two SUM nodes
              subnodes 2,3 = the two FACTORS of the multiplication node
    layer 1:  3 nodes -> 1 output

Joint gradient descent on both factors at once is non-convex, leaves the split
between the additive and product parts undetermined, and produced an
unreadable rational expression from auto_symbolic (see mult_node_equation.py).

ALS instead freezes one factor while training the other, alternating.  Each
half-step is a well-posed problem in the free factor.  Implemented by zeroing
the gradients on the frozen factor's slice of layer 0 after backward().

Measured against:
    joint training           fit R^2 0.9995, equation unreadable
    standalone ALS + basis   a to 0.07%, h to 0.01%, equation exact
"""
import numpy as np
import torch
import sympy as sp

import kandy.core as kc
from kan import MultKAN

SEED = 42
R, K, A, H = 1.0, 50.0, 1.2, 0.1
AH = A * H
FACTOR_SUBNODES = (2, 3)          # the two branches of the multiplication node

rng = np.random.default_rng(0)
N_ = rng.uniform(0.05, 50.0, 8000)
P_ = rng.uniform(0.05, 30.0, 8000)
pred = (A * N_ * P_) / (1 + AH * N_)
target = (R * N_ * (1 - N_ / K) - pred).astype(np.float32)

Xn = np.column_stack([N_, P_]).astype(np.float32)
X_MU, X_SD = Xn.mean(0, keepdims=True), Xn.std(0, keepdims=True)
XT = torch.tensor(Xn)
XTn = torch.tensor((Xn - X_MU) / X_SD)          # normalised inputs for the KAN
YT = torch.tensor(target[:, None])
Y_SD = float(target.std())


def build():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return MultKAN(width=[2, [2, 1], 1], mult_arity=2, grid=8, k=3, seed=SEED)


def freeze_factor(model, which):
    """Zero grads on one factor branch of the multiplication node."""
    l0 = model.act_fun[0]
    for t, idx in ((l0.coef, (slice(None), which, slice(None))),
                   (l0.scale_base, (slice(None), which)),
                   (l0.scale_sp, (slice(None), which))):
        if t.grad is not None:
            t.grad[idx] = 0.0
    for name in ("subnode_bias_0", "subnode_scale_0"):
        p = getattr(model, name, None)
        if p is not None and p.grad is not None:
            p.grad[which] = 0.0


def pin_output_linear(model):
    """Force every layer-1 edge to the identity.

    Otherwise the output is chi(mult_node) with chi a free spline, and a
    perfect product gets reshaped on its way out — which is why the factor
    branches did not match -g(N) and auto_symbolic returned a degenerate
    expression.  With the output linear, the mult node enters the sum with a
    fixed coefficient and its factors become readable up to a single scale.
    """
    for i in range(model.act_fun[1].in_dim):
        model.fix_symbolic(1, i, 0, "x", fit_params_bool=False)


def train(model, mode, steps=120, swap_every=10, pin=False):
    """LBFGS with grid updates — mirrors what KANDy's own fit() does.

    A plain Adam loop on raw inputs scores R^2 = -1.5 here: PyKAN initialises
    its spline grids on [-1,1] and the data spans 0..50, so without
    update_grid_from_samples the splines never see their knots.  The joint
    baseline must reach the ~0.999 that KANDy's fit achieves before any
    comparison between joint and ALS means anything.
    """
    model.update_grid_from_samples(XTn)
    if pin:
        pin_output_linear(model)
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20,
                            history_size=10, line_search_fn="strong_wolfe")
    for s in range(steps):
        if s in (5, 15, 30):
            model.update_grid_from_samples(XTn)
        frozen = FACTOR_SUBNODES[(s // swap_every) % 2] if mode == "als" else None

        def closure():
            opt.zero_grad()
            loss = torch.mean((model(XTn) - YT / Y_SD) ** 2)
            loss.backward()
            if frozen is not None:
                freeze_factor(model, frozen)
            return loss

        opt.step(closure)
    with torch.no_grad():
        p = model(XTn).numpy().ravel() * Y_SD
    return 1 - np.sum((target - p) ** 2) / np.sum((target - target.mean()) ** 2)


def factor_curve(model, which, grid_N):
    """Evaluate one factor branch of the mult node as a function of N."""
    raw = np.column_stack([grid_N, np.full_like(grid_N, 15.0)]).astype(np.float32)
    g = torch.tensor((raw - X_MU) / X_SD)
    model.save_act = True
    with torch.no_grad():
        model(g)
    # spline_postacts[0] : (batch, out_subnode, in_dim)
    post = model.spline_postacts[0].numpy()
    return post[:, which, 0]          # contribution of input N to that subnode


if __name__ == "__main__":
    grid_N = np.linspace(1.0, 48.0, 200)
    g_true = -A * grid_N / (1 + AH * grid_N)

    for mode, pin in (("joint", False), ("als", False),
                      ("joint", True), ("als", True)):
        model = build()
        r2 = train(model, mode, pin=pin)
        print(f"\n=== {mode.upper()} training"
              f"{', output pinned linear' if pin else ''} ===")
        print(f"  fit R^2 = {r2:.8f}")
        best = None
        for w in FACTOR_SUBNODES:
            c = factor_curve(model, w, grid_N)
            if np.ptp(c) < 1e-9:
                continue
            s = np.polyfit(c, g_true, 1)[0]          # allow the free scale
            rr = 1 - np.sum((g_true - s * c) ** 2) / np.sum((g_true - g_true.mean()) ** 2)
            best = rr if best is None else max(best, rr)
            print(f"  factor subnode {w}: R^2 vs -g(N) after rescaling = {rr:.6f}")
        try:
            model.auto_symbolic(lib=["x", "x^2", "1/x", "0"])
            e = sp.expand(sp.sympify(model.symbolic_formula()[0][0]))
            e = e.xreplace({n: sp.Float(round(float(n), 3)) for n in e.atoms(sp.Float)})
            txt = str(e)
            print(f"  auto_symbolic -> {txt[:200]}{' ...' if len(txt) > 200 else ''}")
            print(f"  expression length: {len(txt)} chars "
                  f"(target 'N - 0.02*N**2 - 1.2*N*P/(1+0.12*N)' is 38)")
        except Exception as exc:
            print(f"  auto_symbolic FAILED: {type(exc).__name__}: {exc}")
