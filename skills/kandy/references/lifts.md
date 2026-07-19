# Koopman Lifts

The lift φ : ℝⁿ → ℝᵐ is the most critical design choice in KANDy. The KAN that
follows it is **separable** — each spline ψᵢ depends on exactly one lifted
coordinate θᵢ — so the lift must encode **every** cross-interaction term in the
target system's right-hand side. Missing cross-terms make the algorithm
structurally incorrect, not merely inaccurate (bilinear obstruction theorem:
x·y ≠ h(u(x) + v(y)) for any continuous h, u, v).

All lifts inherit from `Lift` (ABC) and implement `__call__(X)` → lifted array,
`output_dim`, `feature_names()`, and optionally `fit(X)` for data-dependent
lifts (`RadialBasisLift`, `DMDLift`, `KANELift`). `KANDy.fit` calls
`lift.fit(X)` automatically.

## PolynomialLift

```python
PolynomialLift(degree=2, include_bias=False)
```

All monomials up to `degree`. `output_dim = C(n+d, d)` (minus 1 without bias).
The workhorse for polynomial ODEs and maps.

```python
# Lorenz (RHS has xy and xz):  φ(x,y,z) = (x, y, z, x², xy, xz, y², yz, z²)
lift = PolynomialLift(degree=2)
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

## KANELift (EXPERIMENTAL)

```python
KANELift(latent_dim, hidden_dim=None, grid=5, k=3)
```

Learns φ itself as a KAN autoencoder; the encoder is symbolically extractable
after training (`lift.get_formula()`). Train with `lift.train_koopman(...)`.
Prefer explicit lifts whenever the physics is even partially known.

## Choosing degree / size

- Start with the smallest lift that can express the suspected RHS; oversized
  lifts slow training and blur symbolic extraction.
- If formulas come out with spurious high-order terms, reduce degree or add
  sparsity (`lamb > 0` in `fit`).
- If rollout diverges no matter the training budget, a required cross-term is
  probably missing from the lift.
