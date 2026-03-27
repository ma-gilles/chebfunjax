# Chapter 1: Getting Started with chebfunjax

*Based on [Chebfun Guide Chapter 1](https://www.chebfun.org/docs/guide/guide01.html) by Lloyd N. Trefethen*

## 1.1 What is a Chebfun?

The chebfunjax library brings the power of the MATLAB Chebfun system to Python,
powered by JAX.  The central idea is that **a chebfun is a function**, not a
vector of numbers.  You can create one, evaluate it, differentiate it,
integrate it, find its roots, and compose it with other functions -- all with a
syntax that feels like working with ordinary Python objects, yet under the hood
every operation runs at the speed of numerics, not symbolics.

The implementation represents smooth functions by polynomial interpolants at
Chebyshev points (equivalently, by truncated Chebyshev series).  An adaptive
algorithm automatically determines how many points are needed to resolve the
function to roughly machine precision (about 15-16 digits in float64).
Simple functions such as $\sin(x)$ need only about 14 terms, while sharper or
more oscillatory functions may need hundreds or even millions.

Because chebfunjax is built on JAX, every Chebfun evaluation is
JIT-compilable, differentiable with `jax.grad`, and composable with
`jax.vmap`.  Construction (the adaptive step) runs in ordinary Python, but
once built, the resulting object is a first-class JAX pytree.

## 1.2 Constructing Simple Chebfuns

The main entry point is the `chebfun` factory function (or equivalently
`cj.chebfun`).  Pass any callable and chebfunjax will adaptively determine the
polynomial degree needed:

```python
import jax.numpy as jnp
import chebfunjax as cj

f = cj.chebfun(lambda x: jnp.cos(20 * x))
print(f)
```

This creates a chebfun representing $\cos(20x)$ on the default interval
$[-1, 1]$.  The `len(f)` tells you how many Chebyshev coefficients were
needed -- typically around 51 for this function, meaning it is represented by
a polynomial of degree 50.

### Chebyshev points

The $N+1$ Chebyshev points of the second kind on $[-1, 1]$ are

$$x_j = -\cos\!\left(\frac{j\pi}{N}\right), \qquad j = 0, 1, \ldots, N.$$

These points cluster near the endpoints of the interval.  Roughly 5 points per
wavelength in the interior suffice for 15-digit accuracy.  You can obtain them
directly with:

```python
from chebfunjax.utils.quadrature import chebpts

x = chebpts(6)           # 6 Chebyshev points on [-1, 1]
print(x)
# [-1.  -0.80901699 -0.30901699  0.30901699  0.80901699  1. ]
```

### Evaluation and integration

A chebfun is callable -- just pass a point or an array:

```python
print(f(0.5))                          # evaluate at a single point
print(f(jnp.array([0.0, 0.25, 0.5]))) # evaluate at multiple points
```

To compute the definite integral $\int_{-1}^{1} \cos(20x)\,dx$:

```python
print(float(f.sum()))
# 0.09129452507276277

# Compare with the exact answer sin(20)/10:
print(float(jnp.sin(20.0) / 10.0))
# 0.09129452507276277
```

### Bessel function example

You can construct a chebfun from any function on any interval:

```python
import scipy.special as sp

g = cj.chebfun(lambda t: sp.jv(0, t), domain=[0, 100])
print(len(g))   # about 89 coefficients

# Find the zeros of the Bessel function J_0 on [0, 100]:
r = g.roots()
print(r[:5])    # first 5 zeros
```

### The Runge function

The identity function $x$ on $[-1, 1]$ is obtained with `Chebfun.identity()`:

```python
x = cj.chebfun(lambda x: x)
f = 1 / (1 + 25 * x**2)
print(len(f))   # about 187 coefficients
```

This is the famous Runge function $f(x) = 1/(1 + 25x^2)$, which cannot be
accurately approximated by equispaced polynomial interpolation but is handled
effortlessly by Chebyshev interpolation.

## 1.3 Operations on Chebfuns

Chebfunjax overloads hundreds of operations on Chebfun objects.  The key
categories are:

| Category | Methods / Functions |
|---|---|
| **Arithmetic** | `+`, `-`, `*`, `/`, `**`, unary `-` |
| **Calculus** | `diff`, `cumsum`, `sum`, `inner`, `norm`, `mean` |
| **Rootfinding** | `roots`, `min`, `max`, `minandmax` |
| **Composition** | `sin`, `cos`, `exp`, `log`, `sqrt`, `abs`, `sign`, ... |
| **Special functions** | `besselj`, `bessely`, `airy`, `erf`, `erfc`, ... |
| **Linear algebra** | `qr`, `svd`, `inner` |
| **Inspection** | `len`, `coeffs`, `values`, `vscale`, `ishappy` |

### Example: sum, diff, and roots

```python
f = cj.chebfun(jnp.sin)

# Definite integral of sin(x) on [-1, 1]
print(float(f.sum()))
# 0.0  (by symmetry)

# Derivative: cos(x)
fp = f.diff()
print(float(fp(0.0)))
# 1.0  (cos(0) = 1)

# Antiderivative satisfying F(-1) = 0
F = f.cumsum()
print(float(F(1.0)))
# should match -cos(1) - (-cos(-1)) = 0

# Roots of sin(x) on [-1, 1]
r = f.roots()
print(r)
# [0.]
```

### Evaluation at a point

Call the chebfun like a function:

```python
f = cj.chebfun(jnp.exp)
print(float(f(0.5)))    # e^0.5
```

The evaluation uses the Clenshaw algorithm for Chebyshev series, which is
numerically stable even for very high-degree polynomials.

## 1.4 Piecewise Smooth Chebfuns

Many interesting functions have kinks, jumps, or other non-smooth features.
Chebfunjax handles these by representing the function as multiple smooth
*pieces*, each on its own sub-interval.

### Explicit breakpoints

You can specify breakpoints via the `domain` argument:

```python
import jax.numpy as jnp
import chebfunjax as cj

# Three pieces: x^2 on [-1, 1], constant 1 on [1, 2], 4-x on [2, 4]
def f_pw(x):
    return jnp.where(x < 1, x**2,
           jnp.where(x < 2, jnp.ones_like(x), 4.0 - x))

f = cj.chebfun(f_pw, domain=[-1, 1, 2, 4])
print(f)
```

This creates a 3-piece chebfun.  Each piece is represented independently by
its own Chebyshev polynomial; the global chebfun knows the breakpoints and
routes evaluation queries to the appropriate piece.

### Automatic breakpoints from abs, sign, max

Operations like `abs` and `sign` automatically introduce breakpoints at the
zeros of the function:

```python
x = cj.chebfun(lambda x: x)

g = x.abs()     # |x| -- two linear pieces meeting at x = 0
print(g)        # 2 smooth pieces

h = x.sign()    # sign(x) -- piecewise constant +/-1
print(h)
```

The `abs()` method finds the roots of the original function and inserts them
as breakpoints, so that each piece of the result is smooth.

### Calculus with piecewise functions

All calculus operations work seamlessly on piecewise chebfuns:

```python
f = cj.chebfun(lambda x: jnp.abs(jnp.sin(x)), domain=[0, 2 * jnp.pi])
# This is |sin(x)| on [0, 2*pi] -- smooth except at x = pi

print(float(f.sum()))        # integral of |sin(x)| over [0, 2*pi] = 4
print(float(f.norm(2)))      # L2 norm
```

## 1.5 Custom Domains

The default domain is $[-1, 1]$, but you can use any finite interval:

```python
# sin(x) on [0, pi]
f = cj.chebfun(jnp.sin, domain=[0, jnp.pi])
print(float(f.sum()))   # integral = 2.0
print(float(f(jnp.pi / 2)))  # sin(pi/2) = 1.0
```

Internally the function is mapped to the reference interval $[-1, 1]$ via an
affine transformation.  All operations -- differentiation, integration,
rootfinding -- account for this mapping automatically.

## 1.6 How Does Chebfunjax Work Internally?

A `Chebfun` object consists of:

- **`funs`**: a list of `_Piece` objects, one per smooth sub-interval.
- **`domain`**: a `Domain` object recording the breakpoints.

Each `_Piece` wraps a `Chebtech2` object (defined on the reference interval
$[-1, 1]$) together with the affine map from $[-1, 1]$ to its physical
sub-interval $[a, b]$.

The `Chebtech2` stores:

- **`coeffs`**: Chebyshev expansion coefficients $c_0, c_1, \ldots, c_{n-1}$
  so that
  $$f(x) \approx \sum_{k=0}^{n-1} c_k\, T_k(x),$$
  where $T_k$ is the $k$-th Chebyshev polynomial of the first kind.
- **`values`**: function values at the $n$ Chebyshev points of the second kind.

Conversion between coefficients and values is done via the FFT (the discrete
cosine transform), so it takes $O(n \log n)$ time.

### Adaptive construction

When you call `cj.chebfun(f)`, the library evaluates `f` on successively finer
Chebyshev grids ($n = 17, 33, 65, \ldots$) until the Chebyshev coefficients
have decayed to machine epsilon.  A *chopping* algorithm (`standard_chop`)
decides where to truncate the series.  This adaptive loop runs in Python and
is **not** JIT-safe, but the resulting Chebfun (with fixed coefficient arrays)
is fully JIT/grad/vmap compatible.

## 1.7 Fixed-Degree Construction

If you know in advance how many points you want, pass `n`:

```python
f = cj.chebfun(jnp.sin, n=20)
print(len(f))   # exactly 20
```

This skips adaptive refinement and uses exactly 20 Chebyshev points.  It is
useful when you want uniform cost across many function constructions (e.g.,
inside a `vmap`).

## 1.8 Construction from Data

You can also build a chebfun from Chebyshev coefficients or values at
Chebyshev points:

```python
import jax.numpy as jnp
import chebfunjax as cj

# From Chebyshev coefficients: f(x) = 1 + 2*T_1(x) + 3*T_2(x)
f = cj.Chebfun.from_coeffs(jnp.array([1.0, 2.0, 3.0]))
print(float(f(0.0)))   # 1 + 0 + 3*(-1) = 1 - 3 = -2? No: T_2(0) = -1, so 1 + 0 + 3*(-1) = -2

# From values at Chebyshev points
from chebfunjax.utils.quadrature import chebpts
pts = chebpts(5)
vals = jnp.sin(pts)
g = cj.Chebfun.from_values(vals)
```

### Polynomial interpolation of data

For arbitrary (non-Chebyshev) data points, use `interp1`:

```python
x_data = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])
y_data = jnp.sin(x_data)
f = cj.Chebfun.interp1(x_data, y_data, domain=(0.0, 2.0))
```

Or for piecewise cubic spline interpolation:

```python
f = cj.Chebfun.spline(x_data, y_data)
```

## 1.9 Printing and Display

Chebfuns have informative `repr` and `str` representations:

```python
f = cj.chebfun(jnp.sin)
print(repr(f))
```

This prints something like:

```
Chebfun column (1 smooth piece)
       interval       length     endpoint values
[      -1,       1]       14      -0.84      0.84
vscale = 8.41e-01
```

The display shows:

- The number of smooth pieces.
- For each piece: the interval, the number of Chebyshev coefficients (length),
  and the function values at the two endpoints.
- The vertical scale (`vscale`), which is the maximum absolute value across all
  pieces.

## 1.10 Composition with Standard Functions

Chebfunjax provides both method syntax and module-level function syntax for
composing chebfuns with standard functions:

```python
import chebfunjax as cj
import jax.numpy as jnp

x = cj.chebfun(lambda x: x)

# Method syntax
f1 = (3 * x).sin()           # sin(3x)
f2 = (x**2).exp()            # exp(x^2)

# Module-level syntax
f3 = cj.sin(3 * x)           # same as f1
f4 = cj.exp(x**2)            # same as f2

# Available functions:
# cj.sin, cj.cos, cj.exp, cj.log, cj.sqrt, cj.abs, cj.sign
# cj.sinh, cj.cosh, cj.tanh, cj.asin, cj.acos, cj.atan
```

These all work by adaptively constructing a new Chebfun that approximates the
composed function.

## 1.11 JAX Integration

Because chebfunjax is built on JAX and Equinox, Chebfun objects are pytrees.
For single-piece chebfuns, evaluation is fully JIT-safe:

```python
import jax

f = cj.chebfun(jnp.sin)

# JIT-compiled evaluation
f_jit = jax.jit(f)
print(float(f_jit(0.5)))

# Automatic differentiation of evaluation
df_dx = jax.grad(lambda x: f(x))
print(float(df_dx(0.5)))   # cos(0.5)

# Vectorized evaluation
f_batch = jax.vmap(f)
xs = jnp.linspace(-1.0, 1.0, 100)
ys = f_batch(xs)
```

Note that *construction* (the adaptive loop) is not JIT-safe -- only
*evaluation* of an already-constructed Chebfun is.

## 1.12 Plotting

Chebfunjax provides a `plot` function modeled on MATLAB Chebfun's plotting:

```python
import chebfunjax as cj
import jax.numpy as jnp

f = cj.chebfun(lambda x: jnp.sin(10 * x))
cj.plot(f)   # creates a matplotlib figure

# Plot Chebyshev coefficients (log scale)
cj.plotcoeffs(f)
```

The `plotcoeffs` function is particularly useful for understanding convergence:
the coefficients of an analytic function decay geometrically, so their
semilog plot is approximately a straight line.

## 1.13 Summary of Key Functions

| Function / Method | Description |
|---|---|
| `cj.chebfun(f)` | Construct from callable (adaptive) |
| `cj.chebfun(f, domain=[a,b])` | Construct on interval $[a, b]$ |
| `cj.chebfun(f, n=k)` | Fixed degree $k$ |
| `cj.Chebfun.from_coeffs(c)` | Construct from Chebyshev coefficients |
| `cj.Chebfun.from_values(v)` | Construct from values at Chebyshev points |
| `f(x)` | Evaluate at point(s) |
| `f.sum()` | Definite integral $\int_a^b f(x)\,dx$ |
| `f.diff()` | First derivative |
| `f.cumsum()` | Antiderivative with $F(a) = 0$ |
| `f.roots()` | All zeros in $[a, b]$ |
| `f.min()`, `f.max()` | Global min/max |
| `f.norm(p)` | $L^p$ norm |
| `len(f)` | Total number of Chebyshev coefficients |
| `f.coeffs` | Chebyshev coefficient array |
| `f.domain` | The Domain object |
| `f.vscale` | Maximum absolute value |

## 1.14 References

- Z. Battles and L. N. Trefethen, "An extension of MATLAB to continuous
  functions and operators," *SIAM J. Sci. Comp.* 25 (2004), 1743-1770.
- J.-P. Berrut and L. N. Trefethen, "Barycentric Lagrange interpolation,"
  *SIAM Review* 46 (2004), 501-517.
- L. N. Trefethen, *Approximation Theory and Approximation Practice*,
  SIAM, 2013.
- The original Chebfun project: [https://www.chebfun.org/](https://www.chebfun.org/)
