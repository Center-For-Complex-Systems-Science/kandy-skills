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

The edge is not limited to shapes you could have named in advance, and it is
worth reading its shape back afterwards. In `mathbio/fmri_dcm_example.py` the
Balloon–Windkessel term `v^(1/α)` is left on an edge, and curve-fitting that
one edge to `c·v^p + d` returns the Grubb exponent as `α = 1/p` to 0.01% — a
*fractional* power nobody supplied. Same recipe as naming the univariate factor
in `mathbio/holling_mult_node_example.py`.

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

## Nonlocal operators — a conditioning wall, not an obstruction

A nonlocal term (Hilbert transform, fractional Laplacian `|k|^α`, a convolution
kernel) goes in the lift as **one coordinate**, evaluated spectrally:

```python
def hilbert(u):                       # multiplier -i sgn(k)
    return np.fft.irfft(-1j * np.sign(K) * np.fft.rfft(u), n=NX)
```

The tempting justification — "a local dictionary can never represent a nonlocal
operator" — is **false on a periodic band-limited grid**, and it is worth
knowing why. For a real field only `k > 0` matters, and there the symbol of
`H[u_xx]` is `i·k²`, perfectly smooth. Odd derivatives supply `i·k`, `−i·k³`,
`i·k⁵`, …, so a local surrogate is just polynomial approximation of `k²` over
the band the data occupies; with `N_x/2` derivative orders it is exact in exact
arithmetic (the order→symbol map is an invertible Vandermonde).

What actually stops it is that the Vandermonde is as ill-conditioned as
Vandermondes are. Measured in `pdes/benjamin_ono_nonlocal_example.py`:

| dictionary | features | in-band R² | out-of-band R² | cond |
|---|---|---|---|---|
| local `p=3` | 4 | 0.9797 | 0.9445 | 1.6e3 |
| local `p=9` | 10 | 0.9985 | 0.9960 | 4.1e11 |
| local `p=11` | 12 | 0.4690 | 0.3288 | 5.5e14 |
| **nonlocal** | **2** | **1.000000** | **1.0000** | **1.5e2** |

Accuracy improves with derivative order until the design matrix dies, and it
dies long before exactness. Three consequences worth carrying:

- **The surrogate need not be a well-posed PDE.** Even-order derivatives have
  real symbols, so a positive fitted coefficient is an exponential instability.
  The `p=7` surrogate above scores R² = 0.998 pointwise, has max `Re(L̂) = +134`,
  and blows up 11 frames into a rollout. The nonlocal equation is exactly
  dispersive and rolls out to 0.001 relative error on a bandwidth it never saw.
- **The local fit is band-limited by construction** — it is a polynomial fit to
  the symbol over the training band, so it degrades outside it.
- **Watch the spline support, not just the lift.** Out of band the nonlocal
  *coordinate* is exact (least-squares R² = 1.000000) while the fitted
  *network* scores 0.665, because a KAN edge is a spline on the grid range it
  was shown and 8% of the test rows fall outside it. Extract the equation, then
  use the equation.

Two mechanical traps in this setting: spectral features on a `k²` operator reach
~1e3 against an O(1e2) target and LBFGS stalls — divide both sides by one scalar,
which leaves the coefficients unchanged. And if you integrate with ETDRK4, a
purely dispersive `L̂ = −i k²` has **complex** contour-integral coefficients;
taking `.real` (correct for KS, where `L̂` is real) silently costs 57% of a
conserved norm.

## Log chart — when the RHS is multiplicative

If every equation carries a factor of its own state, `ẋᵢ = xᵢ·fᵢ(x)` on a
strictly positive state (Lotka–Volterra, mass-action kinetics, epidemics,
chemostats), then `d(log xᵢ)/dt = fᵢ(x)` and the multiplicative couplings
become sums. For generalized Lotka–Volterra this turns
`ẋᵢ = xᵢ(rᵢ + Σⱼ Aᵢⱼxⱼ)` into something **exactly linear in the raw state**:

```python
model.fit(X=X, X_dot=X_dot / X)      # d(log x)/dt; identity lift, linear edges
```

This is a change of chart, not a target hack — `u = log x` is a diffeomorphism
`ℝⁿ₊ → ℝⁿ` and the induced field is its exact pushforward — but it only counts
as one if you **declare the inverse** and use it: integrate `u̇ = f(eᵘ)` and
report `x = eᵘ`, which makes positivity structural rather than hoped for.

What it buys, measured in `mathbio/glv_log_lift_example.py` at n=5:

- **Lift size `n(n+1)/2` → `n`** — 15 coordinates down to 5 here, 210 → 20 at
  n=20. All the `xᵢxⱼ` cross-products disappear and every edge is a line.
- **Conditioning 70× better** (5.3e4 → 7.5e2): squaring a 400× abundance spread
  puts `x₄x₅ ~ 4e-5` in the same design matrix as `x₁x₂ ~ 3e-1`.
- **Multiplicative noise stays additive.** Proportional error `x_obs = x·e^ε` is
  the natural model for abundances, and the raw lift *squares* it because its
  features are products. At σ = 2% the raw arm's worst interaction coefficient
  is off by ~1.2 against true entries of magnitude 0.1–2.0 — the reconstructed
  network is destroyed — while the log chart stays near 0.18.

Two things it does **not** buy, both worth knowing:

- **It is not needed to fix per-equation scale.** The obvious argument — that
  `ẋᵢ ∝ xᵢ` makes rare species invisible to an unweighted MSE — does not hold
  in KANDy. `width=[m, n]` gives every output row its own edges, so equations
  never compete for capacity; a 6355× spread in `|ẋ|` still let the unscaled
  raw fit recover all 25 coefficients.
- **It dies at extinction.** `log x → −∞` as `x → 0`, so the chart exists only
  on the open orthant. A community where a species actually dies out cannot be
  fitted this way at all.

Finally, a scoring warning this example makes concrete: `get_formula` dropped
the true term `A₁₅ = −0.1` while every formula still scored R² = 1.000, because
the rarest species contributes 0.04 % of the variance of the equation it
appears in. Read the coefficients, not the score.

### The log chart on STOCHASTIC data carries an Itô term

Everything above assumes a deterministic flow. For `dX = X(r − aX)dt + σX dW`,
Itô's lemma gives

```
d(log X) = (r − σ²/2 − a X) dt + σ dW
```

so the chart still linearizes the drift and the crowding coefficient `a` still
comes back correctly — but the intercept is `r − σ²/2`, **not** `r`. With
σ = 0.8 that is 1.18 in place of 1.50, a 21% error, and **no goodness-of-fit
measure detects it**, because `r − σ²/2` genuinely is the drift of `log X`. The
chart is not wrong; reading its intercept as a growth rate is.

Deterministic changes of variable commute with fitting. Stochastic ones do not.
Recovering `r` therefore requires estimating the diffusion as well — see
`mathbio/stochastic_logistic_sde_example.py`, which fits `σ²` from the quadratic
variation with a second model and then undoes the correction.

## Implicit form — when the denominator cannot be lifted at all

A rational lift works only when you can *write* the reciprocal. Sometimes you
cannot: for a mechanical system `M(q) q̈ = f(q, q̇)`, the explicit field is
`q̈ = M(q)⁻¹ f`, and the feature you would have to supply is every numerator
term divided by `det M`. Building that lift means inverting `M` symbolically
first — you would have to have already solved the problem. Enriching the
dictionary does not help, because the missing operation is division, and a
separable KAN sums terms.

**Fit the implicit equation instead.** Clear the denominator, then solve each
row for its own derivative while keeping the *other* derivatives as inputs
(SINDy-PI's move). This works in KANDy because `fit(X, X_dot)` is just a
regression from `lift(X)` to a target — **nothing requires the target to be the
time derivative of `X`**. So pass an augmented vector, accelerations included:

```python
Z = np.column_stack([q, q_dot, q_ddot])       # derivatives carried as INPUTS
model.fit(X=Z, X_dot=q_ddot[:, k:k+1])        # one scalar row as the target
```

Four rules make it work:

- **One model per row.** Never one model with all the derivatives as outputs:
  the feature `q̈₁·cos(Δ)` is ≈ `q̈₁` wherever `cos Δ ≈ 1`, so a row that
  outputs `q̈₁` will explain it with itself. Splitting keeps every regressor
  independent of that row's target.
- **Seed each lift with decoys** — the other row's terms, whose true
  coefficient is zero. Structural recovery means driving those to zero, not
  scoring R² = 1. Check `cond(Θ | 1)` to confirm the zeros are identifiable.
- **The model still integrates.** A single-layer KAN is additive and its
  derivative edges are linear, so at any state it is affine in `q̈`:
  `q̈ = k + J q̈`. Probe it at `q̈ = 0, e₁, e₂, …` to read off `k` and `J`, then
  solve `(I − J) q̈ = k`. That small solve *is* the rational function — the
  division done numerically at integration time, where it is exact and cheap,
  instead of approximated in a dictionary where it is impossible.
- **The price is errors-in-variables.** The derivatives are now regressors, so
  they must be measured, and their noise is not output noise. Use Savitzky–Golay
  derivatives, tune the window (too wide over-smooths and biases the
  coefficients), and use several trajectories at different energies — one
  trajectory confines every sample to a level set.

`odes/double_pendulum_implicit_example.py` measures both arms. Handed the exact
numerator dictionary — everything but the division — the explicit fit caps at
R² = 0.957 on `θ̈₁` and **0.133** on `θ̈₂`, and its rollout leaves the true
trajectory immediately (RMSE 0.99, energy drift 4.7 J). The implicit fit on the
same data reaches R² = 1 − 2e-8, recovers all six coefficients to 4 decimals
with the decoys at zero, and rolls out at RMSE 0.0018 with energy drift 7e-4.
Note the trap in the 0.957: read pointwise on the easier equation, a structural
impossibility looks like a near-miss that more training would close.

## Choosing degree / size

- Start with the smallest lift that can express the suspected RHS; oversized
  lifts slow training and blur symbolic extraction.
- If formulas come out with spurious high-order terms, reduce degree or add
  sparsity (`lamb > 0` in `fit`).
- If rollout diverges no matter the training budget, a required cross-term is
  probably missing from the lift.
