#!/usr/bin/env python3
"""KANDy: make auto_symbolic actually output sin for x_dot = x + sin(x).

Goal: phi(x) = (x, x), recover  x_dot = x + sin(x)  with the SINE explicitly
named by auto_symbolic (not pruned, not turned into a mixed spline).

Two ingredients are required:

  1. CLEAN disambiguation — each channel must be a *single* primitive, so one
     channel ~ pure x and the other ~ pure sin(x).  A single primitive per edge
     is what auto_symbolic can match; a mixture (0.5x + 0.876 sin) gets pruned.
     We break the channel symmetry with a per-channel heterogeneous base:
         base_fun(x)[:,0] = sin(x)      # channel a is sine-biased
         base_fun(x)[:,1] = x           # channel b is linear-biased

  2. A RESTRICTED symbolic library — lib=['x','sin'] forces pykan to pick
     between exactly x and sin, so the sine channel is named 'sin' rather than
     spelled '-cos(x + pi/2)' or pruned.
"""

import os
import numpy as np
import torch
import sympy as sp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kandy import KANDy, CustomLift

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

OUTDIR = "results/SinX"
os.makedirs(OUTDIR, exist_ok=True)

DOMAIN = (-6.0, 6.0)
N = 4000


def rhs(x):
    return x + np.sin(x)


def lift_two_copies(X):
    x = X[:, 0]
    return np.column_stack([x, x])


# per-channel heterogeneous base: sin on channel a, identity on channel b
def hetero_base(x):
    return torch.stack([torch.sin(x[:, 0]), x[:, 1]], dim=1)


# --- data & fit ------------------------------------------------------------
xs = np.random.uniform(*DOMAIN, size=N)
X = xs[:, None]
Xd = rhs(xs)[:, None]

lift = CustomLift(fn=lift_two_copies, output_dim=2, name="two")
model = KANDy(lift=lift, grid=10, k=3, steps=200, seed=SEED, base_fun=hetero_base)
model.fit(X, Xd, val_frac=0.15, test_frac=0.15, lamb=0.0, verbose=False)

# --- channel decomposition (BEFORE auto_symbolic mutates the model) --------
xg = np.linspace(*DOMAIN, 1000)
true_dot = rhs(xg)
with torch.no_grad():
    t = torch.tensor(lift_two_copies(xg[:, None]), dtype=torch.float32)
    model.model_.save_act = True
    model.model_(t)
    acts = model.model_.spline_postacts[0].detach().cpu().numpy()
chA, chB = acts[:, 0, 0], acts[:, 0, 1]
B = np.column_stack([xg, np.sin(xg), np.ones_like(xg)])
cA, *_ = np.linalg.lstsq(B, chA, rcond=None)
cB, *_ = np.linalg.lstsq(B, chB, rcond=None)
sin_a = abs(cA[1]) / (abs(cA[1]) + abs(cB[1]) + 1e-12)
print(f"  channel a:  {cA[0]:+.3f}*x  {cA[1]:+.3f}*sin(x)")
print(f"  channel b:  {cB[0]:+.3f}*x  {cB[1]:+.3f}*sin(x)")
print(f"  --> sin(x) localised in channel a: {sin_a*100:.1f}%\n")


# --- symbolic extraction with a RESTRICTED library -------------------------
def formula_with_lib(model, var_names, lib):
    """Replicate KANDy.get_formula but pass an explicit auto_symbolic library."""
    m = model.model_
    m.save_act = True
    with torch.no_grad():
        m(model._train_input)
    m.auto_symbolic(lib=lib)                       # <-- force the library
    exprs, inputs = m.symbolic_formula()
    sub = {sp.Symbol(str(i)): sp.Symbol(n) for i, n in zip(inputs, var_names)}
    out = []
    for e in exprs:
        s = sp.sympify(e).xreplace(sub)
        s = s.xreplace({a: round(float(a), 3) for a in s.atoms(sp.Number)})
        out.append(sp.expand(s))
    return out


f = formula_with_lib(model, ["x_a", "x_b"], lib=["x", "sin"])[0]
print(f"  auto_symbolic (lib=['x','sin']):  x_dot = {f}")

# --- figure ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(xg, chA, color="#2ca02c", lw=2.5, label="channel a (sin-base)")
ax.plot(xg, chB, color="#9467bd", lw=2.5, label="channel b (x-base)")
ax.plot(xg, np.sin(xg), "k:", alpha=0.5, label=r"$\sin x$ (ref)")
ax.plot(xg, xg, "k--", alpha=0.5, label=r"$x$ (ref)")
ax.set_xlabel("x"); ax.set_ylabel("channel output")
ax.set_title(f"Clean split -> auto_symbolic names sin\nsin in ch.a = {sin_a*100:.0f}%")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/sinx_recover_sine.png", dpi=300)
plt.close(fig)
print(f"\n[FIGS]  saved {OUTDIR}/sinx_recover_sine.png")
