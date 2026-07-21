# Koopman Lifts

The lift φ : ℝⁿ → ℝᵐ is the most critical design choice in KANDy. The KAN that
follows it is **separable** — each spline ψᵢ depends on exactly one lifted
coordinate θᵢ — so the lift must encode **every** cross-interaction term in the
target system's right-hand side. Missing cross-terms make the algorithm
structurally incorrect, not merely inaccurate (bilinear obstruction theorem:
x·y ≠ h(u(x) + v(y)) for any continuous h, u, v).

All lifts inherit from `Lift` (ABC) and implement `__call__(X)` → lifted array,
`output_dim`, `feature_names` (an attribute, not a method), and optionally
`fit(X)` for data-dependent lifts (`RadialBasisLift`, `DMDLift`).
`KANDy.fit` calls `lift.fit(X)` automatically.

## Two design rules

**1. Include products of DIFFERENT variables. Exclude powers of a SINGLE
variable.**

The first half is the bilinear obstruction above: `x·y` cannot be built from
separable functions, so it must be a lift coordinate. The second half is the
part that gets written backwards. A spline edge already *is* an arbitrary
univariate function, so `x²`, `x³`, `sin x`, `tanh x` need not appear in φ —
the edge learns them from data with no dictionary
(`odes/pendulum_dictionary_completion_example.py` recovers `sin θ` this way).

**2. Lift coordinates must be functionally independent of one another.**

Adding `x²` alongside `x` is not merely redundant, it is harmful. Because an
edge on `x²` can represent any function of `x` (the map is invertible on the
sampled range), the two edges become interchangeable: the decomposition is
non-unique, and symbolic extraction returns meaningless coefficients — often
quartics like `(a - b·x²)²` — even at R² = 1. Same for `x` and `x³`. Drop the
power and let the edge learn it.

Consequence for the sections below: `PolynomialLift(degree=d)` supplies every
monomial, including the single-variable powers rule 2 warns about. It fits
fine, but for **interpretable** results prefer a `CustomLift` carrying the raw
states plus only the genuine cross-products. Compare
`mathbio/lotka_volterra_competition_example.py` (minimal lift → exact
coefficients) against the same system fitted with `PolynomialLift`.

Diagnostic: if the fit is perfect but the recovered constants are wrong, check
`np.linalg.cond(np.column_stack([Theta, np.ones(len(Theta))]))`. A large value
means the lifted features are near-dependent — either from redundant
coordinates (this section) or from a conserved quantity in the data
(`mathbio/sir_example.py`).

**But a healthy condition number does not clear you.** `cond` sees only
*linear* dependence among lift coordinates, and a KAN edge is an arbitrary
univariate function — so any *nonlinear* relation `F(θᵢ, θⱼ, …) = 0` on the
sampled data breaks uniqueness just as badly, invisibly. In
`geometry/spherical_pendulum_example.py`, data confined to the constraint
manifold S² satisfies `θ4² + θ5² − (θ6² − θ6⁴) ≡ 0` to 4e-16 while `cond` reads
a perfectly benign 6.76; the fit reaches R² = 0.9985 and returns the wrong
equation. Sampling off the manifold (a shell `0.7 ≤ |q| ≤ 1.3`) leaves `cond`
essentially unchanged at 6.29, kills the nonlinear relation (residual 1.12),
and recovers every coefficient exactly.

So the real rule is about the *data*, not just the lift: if your states obey a
constraint or conservation law, the lifted features inherit it. Fit on samples
that violate it — independent draws over a box or shell, not trajectories —
and keep the on-manifold data for rollout validation.

## PolynomialLift

```python
PolynomialLift(degree=2, include_bias=False)
```

All monomials up to `degree`. `output_dim = C(n+d, d)` (minus 1 without bias).
Convenient for polynomial ODEs and maps when you only need predictive
accuracy.

```python
# Lorenz (RHS has xy and xz):  φ(x,y,z) = (x, y, z, x², xy, xz, y², yz, z²)
lift = PolynomialLift(degree=2)
```

For interpretable extraction, prefer the minimal hand-built version — states
plus cross-products only, dropping the `x²`, `y²`, `z²` coordinates that
duplicate what the edges can learn (this is what `odes/lorenz_example.py`
does):

```python
lift = CustomLift(
    fn=lambda X: np.column_stack([
        X[:, 0], X[:, 1], X[:, 2],
        X[:, 0] * X[:, 1], X[:, 0] * X[:, 2], X[:, 1] * X[:, 2],
    ]),
    output_dim=6, name="lorenz_lift",
)
```

## FourierLift

```python
FourierLift(n_modes)     # output_dim = 1 + 2·n_modes
```

DC component plus real/imaginary parts of the leading Fourier modes. For
periodic PDE fields u ∈ ℝᴺ (Burgers, KS).

## RadialBasisLift

```python
RadialBasisLift(n_centers, sigma=None, center_method="random")  # or "kmeans"
```

Gaussian RBF dictionary; `output_dim = n_centers`. σ defaults to the
median-distance heuristic. Use when structure is unknown but smooth.

## DMDLift

```python
DMDLift(n_modes, dictionary=None, sort_by="magnitude")
```

EDMD-based Koopman eigenfunctions computed from the trajectory data itself.
Separates real modes and complex-conjugate pairs;
`output_dim = n_real + 2·n_complex`. Compose with a dictionary:

```python
lift = DMDLift(n_modes=10, dictionary=PolynomialLift(degree=2))
```

## DelayEmbedding

```python
DelayEmbedding(delays=3)
```

Takens-style delay embedding for scalar time series.

## CustomLift

```python
CustomLift(fn, output_dim, name="custom")
```

Wrap any hand-crafted feature function `fn: (N, n) → (N, m)`. Use for
physics-informed lifts:

```python
import numpy as np
from kandy import CustomLift

# Ikeda optical-cavity map — 4 trig-product features
def ikeda_features(X):
    x, y = X[:, 0], X[:, 1]
    t  = 0.4 - 6.0 / (1.0 + x**2 + y**2)
    ct, st = np.cos(t), np.sin(t)
    return np.column_stack([0.9*x*ct, 0.9*y*ct, 0.9*x*st, 0.9*y*st])

lift = CustomLift(fn=ikeda_features, output_dim=4)
```

For Kuramoto-type systems, put the pairwise coupling terms sin(θⱼ−θᵢ) in the
lift. For gated/switched dynamics (e.g. iEEG ReLU-gated oscillators), put the
gate functions in the lift.

**Rational lifts.** When the RHS has a denominator coupling several variables,
the reciprocal itself is the feature a separable KAN cannot build — see
`geometry/mobius_riemann_sphere_example.py` (`u = 1/|cz+d|²`) and
`mathbio/holling_type_ii_example.py`. Two cautions specific to these:

- **Bound the feature range.** Near a pole the reciprocal diverges and the
  spline grid cannot cover it. Sample away from the pole (the Möbius example
  excludes a disc around it, 3.85 % rejection) and print the lifted-feature
  ranges to confirm.
- **Watch for a hidden partition of unity.** Reciprocal families often sum back
  to a constant: expanding `|cz+d|²·u = 1` makes `[u, xu, yu, r²u]` exactly
  linearly dependent with the bias, `cond = 1.3e16`. Drop one feature
  analytically (`cond` → 9.7) rather than hoping the fit sorts it out.

## Choosing degree / size

- Start with the smallest lift that can express the suspected RHS; oversized
  lifts slow training and blur symbolic extraction.
- If formulas come out with spurious high-order terms, reduce degree or add
  sparsity (`lamb > 0` in `fit`).
- If rollout diverges no matter the training budget, a required cross-term is
  probably missing from the lift.
