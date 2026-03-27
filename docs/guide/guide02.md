# Chapter 2: Integration and Differentiation

*Based on [Chebfun Guide Chapter 2](https://www.chebfun.org/docs/guide/guide02.html) by Lloyd N. Trefethen*

## 2.1 The `sum` Method

The workhorse for definite integration in chebfunjax is `f.sum()`, which
computes

$$\int_a^b f(x)\,dx$$

over the chebfun's domain $[a, b]$.  Under the hood it uses an FFT-based
version of Clenshaw-Curtis quadrature, applied independently to each smooth
piece and then summed.

```python
import jax.numpy as jnp
import chebfunjax as cj

# Integral of log(1 + tan(x)) from 0 to pi/4
f = cj.chebfun(lambda x: jnp.log(1 + jnp.tan(x)), domain=[0, jnp.pi / 4])
I = float(f.sum())
print(f"Computed:  {I:.15f}")
print(f"Exact:     {float(jnp.pi * jnp.log(2.0) / 8):.15f}")
# Should match pi * log(2) / 8
```

Here is another example -- the integral of $\sin(\sin(x))$ from 0 to 1:

```python
f = cj.chebfun(lambda x: jnp.sin(jnp.sin(x)), domain=[0, 1])
print(float(f.sum()))
# 0.430606103120691
```

### The error function via integration

The error function $\mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt$
can be computed by building a chebfun of the integrand and calling `sum`:

```python
f = cj.chebfun(lambda t: 2.0 / jnp.sqrt(jnp.pi) * jnp.exp(-t**2), domain=[0, 1])
erf1 = float(f.sum())
print(f"erf(1) via integration: {erf1:.15f}")

import jax
erf1_exact = float(jax.scipy.special.erf(jnp.float64(1.0)))
print(f"erf(1) exact:           {erf1_exact:.15f}")
```

### Integrals of piecewise functions

For piecewise-smooth chebfuns, `sum()` adds the integrals over all pieces.
For example, the integral of $|J_0(x)|$ over $[0, 20]$:

```python
import scipy.special as sp

g = cj.chebfun(lambda t: sp.jv(0, t), domain=[0, 20])
g_abs = g.abs()   # piecewise -- breakpoints at the zeros
print(float(g_abs.sum()))
```

The `abs()` method automatically detects the zeros of $J_0$ and inserts
breakpoints so that each piece of $|J_0(x)|$ is smooth.

## 2.2 Norm, Mean, and Standard Deviation

### The norm method

The `norm` method computes $L^p$ norms over the domain:

```python
f = cj.chebfun(jnp.sin, domain=[0, jnp.pi])

# L2 norm: ||f||_2 = sqrt(int |f|^2 dx)
print(float(f.norm(2)))

# L-infinity norm: ||f||_inf = max |f(x)|
print(float(f.norm(jnp.inf)))   # should be 1.0

# L1 norm: ||f||_1 = int |f(x)| dx
print(float(f.norm(1)))         # should be 2.0
```

The default is the $L^2$ norm (`p=2`):

$$\|f\|_2 = \sqrt{\int_a^b |f(x)|^2\,dx}$$

The infinity norm is computed by finding the global extrema of $|f|$:

$$\|f\|_\infty = \max_{x \in [a,b]} |f(x)|$$

### Mean value

The `mean` method computes the average value of the function:

$$\mathrm{mean}(f) = \frac{1}{b - a} \int_a^b f(x)\,dx$$

```python
f = cj.chebfun(jnp.sin, domain=[0, jnp.pi])
print(float(f.mean()))
# 2/pi = 0.6366...
```

### The inner product

The L2 inner product of two chebfuns is:

$$\langle f, g \rangle = \int_a^b f(x)\,g(x)\,dx$$

```python
f = cj.chebfun(jnp.sin)
g = cj.chebfun(jnp.cos)

ip = float(f.inner(g))
print(f"<sin, cos> = {ip:.15f}")
# Should be 0 by symmetry (sin is odd, cos is even on [-1,1])

# Or use the module-level function:
ip2 = float(cj.innerProduct(f, g))
```

The inner product is the building block for the L2 norm: $\|f\|_2 = \sqrt{\langle f, f \rangle}$.

## 2.3 The `cumsum` Method

The `cumsum` method computes the indefinite integral (antiderivative) of a
chebfun, with the convention that $F(a) = 0$ at the left endpoint:

$$F(x) = \int_a^x f(t)\,dt$$

The result is itself a chebfun, with polynomial degree one higher than the
input (integration raises the degree by 1).

```python
import jax.numpy as jnp
import chebfunjax as cj

f = cj.chebfun(jnp.sin)
F = f.cumsum()

# F(x) should be -cos(x) + cos(-1)
print(float(F(-1.0)))    # 0.0 by construction
print(float(F(0.0)))     # -cos(0) + cos(-1) = -1 + cos(1)
print(float(F(1.0)))     # -cos(1) + cos(-1) = 0 (by symmetry)
```

### The error function as an antiderivative

A natural application of `cumsum` is to compute the error function:

```python
g = cj.chebfun(lambda t: 2.0 / jnp.sqrt(jnp.pi) * jnp.exp(-t**2), domain=[-5, 5])
erf_cheb = g.cumsum()

# Shift so erf(0) = 0:  erf_cheb has F(-5) = 0, but we want F(0) = 0
# Since erf(-5) is essentially -1, erf_cheb is shifted
# We can evaluate and compare:
import jax
x_test = jnp.float64(1.0)
val = float(erf_cheb(x_test)) - float(erf_cheb(jnp.float64(0.0)))
print(f"erf(1) via cumsum: {val:.15f}")
print(f"erf(1) exact:      {float(jax.scipy.special.erf(x_test)):.15f}")
```

### The logarithmic integral

The logarithmic integral $\mathrm{Li}(x) = \int_0^x \frac{dt}{\ln t}$ arises
in the Prime Number Theorem.  While the integrand has a singularity at $t = 1$,
we can compute it on sub-intervals:

```python
# Li(x) on [2, 100]  (avoiding the singularity at t=1)
h = cj.chebfun(lambda t: 1.0 / jnp.log(t), domain=[2, 100])
Li = h.cumsum()   # Li(x) - Li(2), starting from the left endpoint

# Li(100) - Li(2):
print(float(Li(100.0)))
```

### Relationship between cumsum and diff

`cumsum` and `diff` are essentially inverse operations.  If $f$ is a chebfun
and $F = f.\text{cumsum}()$, then $F.\text{diff}()$ should recover $f$:

```python
f = cj.chebfun(lambda x: jnp.exp(x) * jnp.sin(x))
F = f.cumsum()
f_recovered = F.diff()

# Check agreement at a test point
x0 = jnp.float64(0.5)
print(float(f(x0)))
print(float(f_recovered(x0)))
```

## 2.4 The `diff` Method

The `diff` method differentiates a chebfun:

```python
f = cj.chebfun(jnp.exp)
fp = f.diff()       # first derivative: exp(x) again
fpp = f.diff(2)     # second derivative: still exp(x)

print(float(fp(0.0)))   # 1.0
print(float(fpp(0.0)))  # 1.0
```

Higher derivatives are specified by passing the order:

```python
f = cj.chebfun(lambda x: jnp.sin(5 * x))
f5 = f.diff(5)     # fifth derivative of sin(5x) = 5^5 * cos(5x)
print(float(f5(0.0)))
# 5^5 = 3125.0
```

### Caution: repeated differentiation

Differentiation is a notoriously ill-posed operation.  Each derivative
amplifies high-frequency noise in the Chebyshev coefficients.  For a chebfun
of polynomial degree $n$, after $n$ differentiations you will have lost all
information:

```python
f = cj.chebfun(jnp.exp)
print(f"length of f: {len(f)}")

# Differentiate many times
g = f
for k in range(len(f)):
    g = g.diff()

print(f"length after {len(f)} differentiations: {len(g)}")
# The result is essentially the zero polynomial
```

The mechanism is that differentiation of a degree-$n$ polynomial gives a
degree-$(n-1)$ polynomial, so after $n$ differentiations, nothing is left.
But even before that, errors in the small coefficients accumulate and are
amplified with each differentiation.

### Derivatives of piecewise functions

When you differentiate a piecewise chebfun, each piece is differentiated
independently.  At breakpoints where the function has a jump discontinuity
in its derivative, the derivative chebfun will reflect this:

```python
x = cj.chebfun(lambda x: x)
g = x.abs()     # |x| on [-1, 1], with breakpoint at 0
gp = g.diff()   # should be sign(x), piecewise constant

print(float(gp(-0.5)))   # -1
print(float(gp(0.5)))    #  1
```

## 2.5 Integration on Custom Domains

All of the integration methods work seamlessly on custom domains.  The
chain rule and affine scaling are handled automatically:

```python
# Integral of sin(x) on [0, pi]
f = cj.chebfun(jnp.sin, domain=[0, jnp.pi])
print(float(f.sum()))   # 2.0

# Integral of exp(x) on [-2, 3]
g = cj.chebfun(jnp.exp, domain=[-2, 3])
print(float(g.sum()))   # e^3 - e^{-2}
exact = float(jnp.exp(3.0) - jnp.exp(-2.0))
print(exact)
```

## 2.6 Gauss and Clenshaw-Curtis Quadrature

Chebfunjax provides functions for computing classical quadrature nodes and
weights:

```python
from chebfunjax.utils.quadrature import chebpts, chebweights, legpts

# Chebyshev points and Clenshaw-Curtis weights
x_cc = chebpts(5)
w_cc = chebweights(5)
print("Clenshaw-Curtis points:", x_cc)
print("Clenshaw-Curtis weights:", w_cc)

# Gauss-Legendre points and weights
x_gl, w_gl = legpts(5)
print("Gauss-Legendre points:", x_gl)
print("Gauss-Legendre weights:", w_gl)
```

These can be used to compute integrals directly:

```python
# 4-point Gauss quadrature for int_{-1}^{1} exp(x) dx
x, w = legpts(4)
I = float(jnp.sum(w * jnp.exp(x)))
exact = float(jnp.exp(1.0) - jnp.exp(-1.0))
print(f"4-point Gauss:  {I:.15f}")
print(f"Exact:          {exact:.15f}")
print(f"Error:          {abs(I - exact):.2e}")
```

Gauss quadrature with $n$ points is exact for polynomials of degree up to
$2n - 1$, while Clenshaw-Curtis quadrature with $n$ points is exact for
polynomials of degree up to $n - 1$.  In practice, for smooth functions, both
converge extremely rapidly.

### Quadrature on general intervals

All quadrature functions support general intervals $[a, b]$:

```python
from chebfunjax.utils.quadrature import chebpts_ab

# Chebyshev points on [0, pi]
x = chebpts_ab(10, 0.0, float(jnp.pi))
print(x)
```

The `legpts` function also accepts an `interval` parameter:

```python
x_gl, w_gl = legpts(10, interval=(0.0, float(jnp.pi)))
I = float(jnp.sum(w_gl * jnp.sin(x_gl)))
print(f"Gauss quadrature of sin on [0, pi]: {I:.15f}")
print(f"Exact: 2.0")
```

## 2.7 Summary

| Method | Description | MATLAB equivalent |
|---|---|---|
| `f.sum()` | Definite integral $\int_a^b f\,dx$ | `sum(f)` |
| `f.cumsum()` | Antiderivative with $F(a)=0$ | `cumsum(f)` |
| `f.diff(k)` | $k$-th derivative | `diff(f, k)` |
| `f.norm(p)` | $L^p$ norm | `norm(f, p)` |
| `f.mean()` | Mean value $\frac{1}{b-a}\int f\,dx$ | `mean(f)` |
| `f.inner(g)` | $L^2$ inner product $\int f g\,dx$ | `f' * g` |
| `cj.innerProduct(f, g)` | Same as `f.inner(g)` | `innerProduct(f, g)` |

## 2.8 References

- L. N. Trefethen, "Is Gauss quadrature better than Clenshaw-Curtis?",
  *SIAM Review* 50 (2008), 67-87.
- L. N. Trefethen, *Approximation Theory and Approximation Practice*,
  SIAM, 2013.
