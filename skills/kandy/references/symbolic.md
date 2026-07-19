# Symbolic Extraction

After fitting, KANDy converts each learned spline edge to a closed-form
expression via PyKAN's symbolic snapping, then folds in the mixing matrix A to
produce one SymPy expression per state equation.

## Basic extraction

```python
formulas = model.get_formula(
    var_names=["x", "y", "z"],   # lift feature names
    round_places=3,
    simplify=False,              # True: factor → together → nsimplify pipeline
)
# → list of SymPy expressions, one per state dimension
```

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
