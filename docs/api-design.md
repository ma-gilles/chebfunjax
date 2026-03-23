# API Design

How chebfunjax should feel to the user.

## Guiding Principle

Chebfun's genius is making hard math feel easy. Every API decision should
optimize for: **"can a first-year grad student use this without reading the source?"**

## Construction

```python
import jax.numpy as jnp
import chebfunjax as cj

# From a callable (adaptive — the default and most common)
f = cj.chebfun(jnp.sin)                        # on [-1, 1]
f = cj.chebfun(jnp.sin, domain=[0, jnp.pi])    # custom domain
f = cj.chebfun(lambda x: jnp.exp(-x**2))       # lambda
f = cj.chebfun(jnp.sin, n=20)                   # fixed degree (not adaptive)

# From data
f = cj.chebfun.from_coeffs(coeffs)              # from Chebyshev coefficients
f = cj.chebfun.from_values(values)              # from values at Chebyshev points

# Constants and special
f = cj.chebfun(1.0)                             # constant function
x = cj.chebfun.identity()                       # the function f(x) = x
```

## Evaluation

```python
y = f(0.5)                          # scalar → scalar
y = f(jnp.linspace(-1, 1, 100))    # array → array (vectorized)
```

Evaluation is JIT-compiled, vmap-friendly, and differentiable.
See `docs/jax-contract.md` for what is and isn't JIT/grad/vmap-safe.

## Arithmetic

Natural Python operators. These return new Chebfun objects:
```python
g = f + 1                   # scalar
h = f * f                   # pointwise
k = f ** 2 - cj.chebfun(jnp.cos)
r = 1 / f                   # pointwise reciprocal
```

## Calculus

Methods, not standalone functions:
```python
fp = f.diff()               # first derivative
fpp = f.diff(2)             # second derivative
F = f.cumsum()              # antiderivative with F(a) = 0
I = f.sum()                 # definite integral over domain → scalar
n = f.norm()                # L2 norm
ip = f.inner(g)             # inner product <f, g>
m = f.mean()                # mean value
```

## Rootfinding & Extrema

```python
r = f.roots()               # all roots in domain
xmax, fmax = f.max()        # global maximum (location, value)
xmin, fmin = f.min()        # global minimum
```

## Information

```python
len(f)                      # number of Chebyshev coefficients
f.domain                    # (a, b) tuple
f.coeffs                    # jnp array of Chebyshev coefficients
f.values                    # jnp array of values at Chebyshev points
f.vscale                    # vertical scale (max abs value)
f.ishappy                   # True if adaptively resolved
```

## Display

`repr(f)` must be informative and compact, like Chebfun:

```
>>> f = cj.chebfun(jnp.sin)
>>> f
Chebfun column (1 smooth piece)
       interval       length     endpoint values
[      -1,       1]       14      -0.84      0.84
vscale = 8.41e-01
```

For piecewise:
```
>>> g = cj.chebfun(jnp.abs)
>>> g
Chebfun column (2 smooth pieces)
       interval       length     endpoint values
[      -1,       0]       15       1.00      0.00
[       0,       1]       15       0.00      1.00
vscale = 1.00e+00    total length = 30
```

`str(f)` is a one-liner:
```
<Chebfun [−1, 1], length 14>
```

## Plotting

```python
f.plot()                    # quick plot using matplotlib
f.plot(n=500)               # control resolution
cj.plot(f, g)               # overlay multiple chebfuns
```

Plotting calls `matplotlib.pyplot` directly. Returns `(fig, ax)` for customization.
Not required for correctness — a nice-to-have that agents can defer.

## Error Messages

Errors should be:
- **Specific**: "Cannot add Chebfun on [0, 1] to Chebfun on [-1, 1]: domains do not match"
- **Actionable**: "... use f.restrict(0, 1) to restrict the domain first"
- **Consistent**: always `raise ValueError` for bad inputs, `raise TypeError` for wrong types

Never:
- "Invalid input" (what's invalid?)
- "Error in computation" (what error?)
- Bare assertions without messages

## JAX Interop

Evaluation, differentiation, integration, and arithmetic are JIT/grad/vmap-safe.
Adaptive construction, rootfinding, and extrema are NOT (they use Python control flow).
See `docs/jax-contract.md` for the full contract.

```python
# JIT evaluation (fast, compiled)
fast_eval = jax.jit(f)
y = fast_eval(x)

# Gradient through evaluation
df_dx = jax.grad(lambda x: f(x))(0.5)  # should match f.diff()(0.5)

# Batched evaluation
ys = jax.vmap(f)(xs)  # evaluate at many points efficiently

# Custom loss function over coefficient space
def loss(coeffs):
    g = cj.Chebtech2.from_coeffs(coeffs)
    return g.norm()**2

jax.grad(loss)(f.coeffs)  # gradient of a functional w.r.t. coefficients

# NOT JIT-safe (call outside JIT, pass result in):
f = cj.chebfun(jnp.sin)  # adaptive construction — Python loop
r = f.roots()             # eigenvalue problem — variable output size
```
