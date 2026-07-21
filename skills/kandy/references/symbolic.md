# Symbolic Extraction

After fitting, KANDy converts each learned spline edge to a closed-form
expression via PyKAN's symbolic snapping, then folds in the mixing matrix A to
produce one SymPy expression per state equation.

## Basic extraction

```python
formulas = model.get_formula(
    var_names=FEATURE_NAMES,     # lift feature names — NOT the state variables
    round_places=3,
    simplify=False,              # True: factor → together → nsimplify pipeline
    lib=["x", "0"],              # match the EDGE shapes (see below)
    r2_threshold=0.80,           # edges fitting worse than this are zeroed
    weight_simple=0.0,           # simplicity pressure; default 0.8 is aggressive
)
# → list of SymPy expressions, one per state dimension
```

> **`get_formula` mutates `model_` in place.** Each edge is replaced by its
> snapped surrogate, so any later `predict`, `rollout` or edge plot reflects
> the surrogate rather than the trained network — and a bad snap can drop the
> model from R² 1.00 to 0.50 without raising anything. Do numeric validation
> first, or `copy.deepcopy(model)` before each extraction attempt (see
> `odes/pendulum_dictionary_completion_example.py`, which compares two
> libraries on the same trained model that way).

## Debugging a formula that came back too short

This is the most common failure, and it is almost never the fit — check
`model.predict` R² first. If the network is accurate but the formula is not,
work through these in order:

1. **Is `lib` rich enough for the EDGE SHAPES?** The library must describe
   each edge as a function of its own input, which is not the same as the term
   structure of the final equation. If the RHS is linear in the lifted
   features, `lib=["x", "0"]` is right and the default library will fit
   spurious quadratics to near-zero edges. If an edge is genuinely a parabola
   (common when the lift omits `x²` on purpose), you need `"x^2"`. Plot the
   edges — `plot_all_edges` — and look.

2. **Is `weight_simple` too high?** The default `0.8` biases hard toward the
   simplest primitive, and `'0'` is the simplest of all: real curved edges get
   snapped to zero and vanish from the formula. Setting `weight_simple=0.0`
   recovers them. Raising it above ~0.9 typically zeroes *everything* and
   returns a bare constant.

3. **Are the lift coordinates functionally independent?** If φ contains both
   `x` and `x²` (or `x` and `x³`), an edge on one can mimic a function of the
   other. The fit is perfect and the decomposition is meaningless — you get
   quartics like `(a - b·x²)²`. Drop the redundant coordinate and let the edge
   learn the power.

4. **Can the edge be snapped at all?** PyKAN fits ONE primitive under an affine
   composition, `c·f(a·x + b) + d`. This cannot represent a sum of two powers:
   for `v - v³/3`, matching the absent `v²` term forces `b = 0`, which also
   kills the linear term. A quadratic edge `αx + βx²` *is* representable, which
   is why quadratics work and cubics-with-linear-parts do not. When snapping
   is structurally impossible, fit the edge as a polynomial instead (below).

5. **Is the data degenerate?** A conserved quantity (`S+I+R` const, fixed
   energy) puts trajectory data on a manifold where the lifted features are
   linearly dependent, so coefficients are non-identifiable — the fit is
   perfect and the constants are wrong. Sample states independently over a box
   and check `np.linalg.cond` of the lifted design matrix.

## Edge-wise polynomial reconstruction

When per-edge snapping cannot work (case 4 above), exploit the fact that a
single-layer KAN output is exactly the sum of its edges:

```
output_j = Σ_i  edge_ij(theta_i)
```

Fit each edge with `np.polyfit` and add them up:

```python
from kandy.plotting import get_edge_activation

expr = 0
for i in range(n_features):
    x_e, y_e = get_edge_activation(model.model_, 0, i, j)
    coeffs = np.polyfit(np.asarray(x_e).ravel(), np.asarray(y_e).ravel(), 3)
    expr += sum(float(c) * syms[i] ** (3 - k) for k, c in enumerate(coeffs))
```

Always verify the reconstruction against the network (`Σ edges` vs
`model_(theta)`) before trusting it. On FitzHugh–Nagumo this recovers
`dv/dt = -v³/3 + v - w + 1/2` exactly, where `auto_symbolic` returns
`0.49 - w`.

## Reading coefficients out of a snapped formula

PyKAN returns each edge in the composed form `c*f(a*x + b) + d`, e.g.
`-0.008*(6.13 - 8.17*N1)**2.0`, which hides the coefficients. Expand it —
noting that SymPy will not expand a *float* exponent, so rationalise first:

```python
e = sp.expand(expr)
e = e.replace(lambda x: x.is_Pow and x.exp.is_Float,
              lambda x: sp.Pow(x.base, sp.Integer(round(float(x.exp)))))
e = sp.expand(e)
# then drop terms whose coefficient is below a tolerance
```

That turns the expression above into `0.80*N1 - 0.534*N1**2`, matching the
true coefficients. See `mathbio/lotka_volterra_competition_example.py`.

## Physics-informed extraction

`auto_symbolic_with_costs` assigns different complexity libraries per KAN edge
depending on whether its input feature is a known-physics term. Preferred
features get cheap costs, so the fit-vs-complexity solver selects them when fit
quality is comparable.

```python
import torch
from kandy import auto_symbolic_with_costs, TRIG_LIB_CHEAP, TRIG_LIB

# Populate activations first
model.model_.save_act = True
with torch.no_grad():
    model.model_(train_features)

# Example (Ikeda): the first 4 features are physics-informed trig products
auto_symbolic_with_costs(
    model.model_,
    preferred_idx=set(range(4)),
    preferred_lib=TRIG_LIB_CHEAP,   # sin/cos at cost 2
    other_lib=TRIG_LIB,             # sin/cos at cost 4
    weight_simple=0.1,
    r2_threshold=0.80,
    verbose=1,
)
```

`robust_auto_symbolic` is a fallback wrapper that retries snapping with
progressively relaxed thresholds.

## Symbol libraries

| Name | Contents |
|---|---|
| `POLY_LIB_CHEAP` | x, x², x³, 0 at costs 1–3 |
| `POLY_LIB` | x, x², x³, 0 at costs 3–5 |
| `TRIG_LIB_CHEAP` | Polynomial + sin, cos at costs 1–2 |
| `TRIG_LIB` | Polynomial + sin, cos at costs 3–4 |

Custom libraries — each entry is `(torch_fn, sympy_fn, cost)`:

```python
from kandy import make_symbolic_lib
import torch, sympy as sp

my_lib = make_symbolic_lib({
    "x":     (lambda x: x,                   lambda x: x,               1),
    "exp":   (torch.exp,                     sp.exp,                    3),
    "sech2": (lambda x: 1/torch.cosh(x)**2,  lambda x: 1/sp.cosh(x)**2, 3),
})
```

## Scoring and export

```python
from kandy import score_formula, formulas_to_latex, substitute_params

# R² of each formula on held-out (lifted) data
r2 = score_formula(formulas, Theta_test, Y_test, var_names=FEATURE_NAMES)

# Convenience wrapper — applies the lift automatically from raw states
r2 = model.score_formula(formulas, X_test, Y_test)

# LaTeX
tex = formulas_to_latex(formulas, lhs_names=[r"\dot{x}", r"\dot{y}", r"\dot{z}"])
# \begin{align*}
#   \dot{x} &= 10.0 y - 10.0 x \\ ...

# Substitute known parameters
sub = substitute_params(formulas, {"sigma": 10.0, "rho": 28.0, "beta": 2.667})
```

## Tips

- Always score formulas on **held-out** data; a snapped formula can fit train
  data well but drift from the spline it replaced.
- If extraction returns messy high-order polynomials, retrain with sparsity
  (`lamb > 0`) or shrink the lift, then re-extract.
- Report R² per equation — one bad equation usually points at one bad edge, and
  `plot_edge` / `plot_all_edges` will show which.
