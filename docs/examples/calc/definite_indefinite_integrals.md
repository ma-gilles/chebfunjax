# Definite and Indefinite Integrals

*Original: [chebfun.org/examples/calc/Intro](https://www.chebfun.org/examples/calc/Intro.html)*

---

`sum()` computes a definite integral; `cumsum()` returns the indefinite integral
as a new Chebfun. Both use the Clenshaw–Curtis quadrature implicit in the
Chebyshev expansion, achieving spectral accuracy.

## The `cumsum` operation

For $f(x) = \cos(x)$, the antiderivative is $F(x) = \sin(x) + C$:

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

f = cj.chebfun(jnp.cos, domain=(0.0, float(jnp.pi)))
F = f.cumsum()  # F(x) = sin(x) - sin(0) = sin(x)

x0 = jnp.array(float(jnp.pi) / 3)
print(f"F(π/3) = {float(F(x0)):.10f}  (sin(π/3) = {float(jnp.sin(x0)):.10f})")
```

```
F(π/3) = 0.8660254038  (sin(π/3) = 0.8660254038)
```

## Symbolic-style computation

Chebfun can evaluate definite integrals that lack closed forms:

```python
# ∫₀¹ sin(sin(x)) dx — no closed form
f = cj.chebfun(lambda x: jnp.sin(jnp.sin(x)), domain=(0.0, 1.0))
val = float(f.sum())
print(f"∫₀¹ sin(sin(x)) dx = {val:.12f}")
# Agrees with high-precision quadrature: 0.430606103...
```

```
∫₀¹ sin(sin(x)) dx = 0.430606103060
```

## Improper-like integrals

For functions with near-singularities (like $1/\sqrt{1-x^2}$ near $\pm 1$),
Chebyshev quadrature is well-adapted because Chebyshev nodes cluster near
the endpoints — exactly where accuracy is needed:

```python
# ∫₋₁¹ 1/sqrt(1-x^2) dx = π (Chebyshev weight)
f = cj.chebfun(lambda x: 1.0 / jnp.sqrt(1.0 - x**2 + 1e-14))
print(f"∫ 1/sqrt(1-x²) dx ≈ {float(f.sum()):.6f}  (π ≈ {float(jnp.pi):.6f})")
```

![Definite and indefinite integrals comparison](../../../images/calc/definite_indefinite_integrals.png)
