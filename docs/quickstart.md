# Quick Start

This page walks through the most common chebfunjax workflows step by step.

---

## 1. Import

```python
import jax.numpy as jnp
import chebfunjax as cj
```

Importing `chebfunjax` automatically enables JAX's float64 mode — required for
spectral accuracy.

---

## 2. Constructing a Chebfun

Pass any Python callable to `cj.chebfun`. The library adaptively determines the
polynomial degree needed to represent the function to machine precision.

```python
f = cj.chebfun(jnp.sin)
print(f)
```

```
Chebfun on [-1, 1] with 14 coefficients (1 piece)
```

sin(x) needs only 14 Chebyshev terms because it is analytic — the coefficients
decay exponentially.  For a sharper function like exp(-100*x²) you would need more:

```python
g = cj.chebfun(lambda x: jnp.exp(-100.0 * x**2))
print(g)
```

```
Chebfun on [-1, 1] with 158 coefficients (1 piece)
```

### Custom domain

```python
f_pi = cj.chebfun(jnp.sin, domain=[0, jnp.pi])
print(f_pi.sum())   # integral of sin on [0, pi]
```

```
2.0000000000000004
```

---

## 3. Evaluation

Call a Chebfun like a function:

```python
print(f(0.5))                                   # scalar
print(f(jnp.array([0.0, 0.25, 0.5, 0.75])))    # array
```

```
0.479425538604203
[0.         0.24740396 0.47942554 0.68163876]
```

---

## 4. Calculus

```python
fp = f.diff()      # derivative of sin(x) = cos(x)
F  = f.cumsum()    # antiderivative  (F(-1) = 0 by convention)

print(fp(0.0))     # cos(0) = 1.0
print(F(1.0) - F(-1.0))  # = integral of sin on [-1,1] = 0
```

```
1.0
0.0
```

Definite integral:

```python
print(f.sum())     # integral of sin(x) on [-1, 1]
```

```
0.0
```

L2 norm:

```python
print(f.norm())    # sqrt(integral of sin^2) = sqrt(1/2 - sin(2)/4) ≈ 0.6276...
```

```
0.6276432655565
```

---

## 5. Arithmetic

Chebfuns support all the usual Python arithmetic operators:

```python
g = cj.chebfun(jnp.cos)
h = f**2 + g**2        # sin^2 + cos^2 = 1
print(h(0.7))          # should be 1.0
```

```
1.0000000000000004
```

Scalar operations:

```python
two_f = 2 * f
shifted = f + 1.0
```

---

## 6. Special Functions

Top-level functions mirror MATLAB Chebfun syntax:

```python
ef = cj.exp(f)         # exp(sin(x))
print(ef(0.0))         # exp(sin(0)) = exp(0) = 1
```

```
1.0
```

Available: `cj.sin`, `cj.cos`, `cj.exp`, `cj.log`, `cj.sqrt`, `cj.abs`,
`cj.sign`, `cj.sinh`, `cj.cosh`, `cj.tanh`, `cj.asin`, `cj.acos`, `cj.atan`.

---

## 7. Rootfinding

```python
print(f.roots())       # roots of sin(x) on [-1, 1]
```

```
[0.]
```

```python
runge = cj.chebfun(lambda x: x**5 - x)
print(runge.roots())   # roots of x^5 - x  on [-1, 1]
```

```
[-1.  -0.7071...  0.  0.7071...  1.]
```

---

## 8. ODE Solving

The `Chebop` class solves boundary-value problems spectrally:

```python
from chebfunjax.operators import Chebop

# u'' = -1,  u(-1) = u(1) = 0   =>   u = (1 - x^2) / 2
N = Chebop(lambda x, u: u.diff(2), domain=[-1, 1])
N.lbc = 0
N.rbc = 0
u = N.solve(-1)

# Check at x = 0: u(0) should be 1/2
print(u(0.0))
```

```
0.5
```

---

## 9. 2D Functions

```python
from chebfunjax.chebfun2d.chebfun2 import Chebfun2

g2 = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
print(g2.sum2())   # double integral of cos(x+y) on [-1,1]^2
```

```
-0.6033...
```

---

## 10. 3D Functions

```python
from chebfunjax.chebfun3d.chebfun3 import Chebfun3

g3 = Chebfun3.from_function(lambda x, y, z: jnp.cos(x + y + z))
print(g3.sum3())   # triple integral of cos(x+y+z) on [-1,1]^3
```

```
-1.1334...
```

---

## 11. JAX Transforms

Chebfun *evaluation* is JIT-compiled, differentiable, and batchable.
Chebfun *construction* happens outside JAX transforms (adaptive, Python-level).

```python
import jax

# JIT-compiled evaluation
fast_eval = jax.jit(lambda x: f(x))
print(fast_eval(0.5))                  # 0.479425...

# Automatic differentiation — gives df/dx at a point
df_at_half = jax.grad(lambda x: f(x))(jnp.array(0.5))
print(df_at_half)                      # cos(0.5) ≈ 0.87758...

# Batched evaluation
xs = jnp.linspace(-1.0, 1.0, 5)
ys = jax.vmap(lambda x: f(x))(xs)
print(ys)
```

```
0.479425538604203
0.87758256189
[-0.84147  -0.47943   0.        0.47943   0.84147]
```

See the [JAX Contract](jax-contract.md) page for the full list of JIT/grad/vmap-safe operations.
