# Definite and indefinite integrals

**Nick Trefethen, October 2012**

---

Chebfun computes both definite integrals (via `sum`) and indefinite integrals
(via `cumsum`) to machine precision.

## Definite integrals

```python
import jax.numpy as jnp
import chebfunjax as cj

# A simple example on [0, 10]
f = cj.chebfun(lambda x: jnp.sin(x)**2, domain=(0.0, 10.0))
print("Definite integral:", f.sum())   # = 5 - sin(20)/2

# Exponential
g = cj.chebfun(jnp.exp)
print("int exp(x) dx =", g.sum(), "  exact:", float(jnp.e) - 1.0)
```

## Indefinite integrals

`cumsum` returns a new chebfun $F(x) = \int_{a}^{x} f(t)\, dt$:

```python
# Fundamental theorem: d/dx of cumsum(f) recovers f
h    = cj.chebfun(lambda x: 3*x**2 - 1)
H    = h.cumsum()        # indefinite integral
dH   = H.diff()          # derivative of the antiderivative
err  = (dH - h).norm()
print("Round-trip error:", err)   # < 1e-14
```

## Combining sum and cumsum

The total integral can be extracted from a cumsum object by evaluating at the
right endpoint, which matches `f.sum()`:

```python
import numpy as np
x_vals = np.linspace(-1, 1, 500)
p = cj.chebfun(lambda x: jnp.exp(-x**2))
P = p.cumsum()
print("P(1) =", float(P(1.0)), "  sum =", float(p.sum()))
```

## Gallery

![Definite and indefinite integrals](../../../examples/calc/definite_indefinite_integrals.png)

Top: the integrand $f$. Bottom: its cumulative integral $F = \text{cumsum}(f)$.
