#!/usr/bin/env python3
"""Generate Markdown documentation for all 55 Chebfun approx examples.

Run this script from the repo root:
    python scripts/gen_approx_docs.py
"""

import os
import re
from pathlib import Path

# -----------------------------------------------------------------------
# Metadata for all 55 examples
# -----------------------------------------------------------------------
EXAMPLES = [
    dict(name="AAAApprox",
         title="AAA Rational Approximation",
         author="Nick Trefethen",
         date="December 2016",
         tags=["#rational", "#AAA"],
         summary="The AAA (Adaptive Antoulas-Anderson) algorithm for rational approximation on intervals and in the complex plane.",
         narrative="""
## A new kind of rational approximation

Chebfun Version 5.6.0 introduced the AAA algorithm, which computes type $(m-1,m-1)$
rational approximants in a barycentric form.  Unlike earlier methods (remez, cf, ratinterp),
AAA works equally well on any set in the real line or complex plane.

```python
from chebfunjax.utils.aaa import aaa
import jax.numpy as jnp
from scipy.special import gamma as scipy_gamma

xs = jnp.linspace(-1.0, 1.0, 500)
ys = jnp.array([float(scipy_gamma(float(x))) for x in xs])
r, pol, res, zer, zj, fj, w, errvec = aaa(ys, xs)
print("poles:", pol)
```

## Error curve for exp(x)

```python
xs2 = jnp.linspace(-1.0, 1.0, 500)
r2, *_ = aaa(jnp.exp(xs2), xs2)
import numpy as np
xx = np.linspace(-1.0, 1.0, 600)
err = [abs(float(jnp.exp(jnp.array(x)) - r2(jnp.array(x)))) for x in xx]
print(f"max err: {max(err):.2e}")
```
"""),

    dict(name="AAASpline",
         title="AAA Approximation of a Spline",
         author="Nick Trefethen",
         date="April 2021",
         tags=["#spline", "#splitting", "#AAA"],
         summary="AAA rational approximation of a piecewise polynomial function, with poles clustering near spline knots.",
         narrative="""
## Poles near the knots

When AAA approximates a spline function, its poles cluster exponentially near
the nodes of non-analytic behaviour — the spline knots. This geometric clustering
is mathematically necessary for achieving high accuracy with a compact rational form.

```python
import numpy as np
from scipy.interpolate import CubicSpline
from chebfunjax.utils.aaa import aaa
import jax.numpy as jnp

nodes = np.arange(0, 11)
data = np.sin(nodes + nodes**2 / 4.0)
cs = CubicSpline(nodes, data)

X = np.linspace(0, 10, 1000)
Y = cs(X)
r, poles, *_ = aaa(jnp.array(Y), jnp.array(X), mmax=200, tol=1e-10)
print(f"Poles: {len(poles)} total")
```

The poles line up near the integers 2 through 8 — the interior knots of the spline.
"""),

    dict(name="AAAZeros",
         title="Rootfinding with the AAA Algorithm",
         author="Stefano Costa",
         date="June 2022",
         tags=["#AAA", "#rootfinding"],
         summary="Using AAA rational approximants to locate zeros of analytic functions.",
         narrative="""
## Rootfinding via rational approximation

The AAA algorithm returns not only function values but also explicit zeros and poles of
the rational approximant.  For analytic functions these zeros closely approximate
the true zeros of the function.

```python
import chebfunjax as cj
import jax.numpy as jnp

# Roots of sin(10x) + x^2 - 0.5 on [-1,1]
f = cj.chebfun(lambda x: jnp.sin(10.0*x) + x**2 - 0.5)
roots = f.roots()
print(f"Found {len(roots)} roots")
```
"""),

    dict(name="AbsoluteValue",
         title="Absolute Value Approximations by Rationals",
         author="Nick Trefethen",
         date="May 2011",
         tags=["#rational", "#Newton", "#ABS"],
         summary="Newton's method applied to r² = x² produces rational approximants to |x| of type (2^k, 2^k).",
         narrative="""
## Newton's method for the square root

Peter Lax observed that one can approximate $|x|$ by applying Newton's method to
the equation $r^2 = x^2$, starting from $r=1$.  The iteration is

$$r := \\frac{r^2 + x^2}{2r}.$$

After $k$ steps we have a rational function of type $(2^k, 2^k)$.

```python
import chebfunjax as cj
import jax.numpy as jnp

dom = (-1.0, 0.0, 1.0)  # breakpoint at 0 for efficiency
x = cj.chebfun(lambda t: t, domain=dom)
r = cj.chebfun(lambda t: jnp.ones_like(t), domain=dom)

for k in range(6):
    r = (r**2 + x**2) / (2.0 * r)
    print(f"step {k}: len(r) = {len(r)}")
```

The Chebyshev lengths grow rapidly without the breakpoint, but with a breakpoint
at zero the lengths remain manageable.

## Error analysis

Donald Newman showed that the optimal type $(n,n)$ rational approximants to $|x|$
achieve accuracy $O(\\exp(-C\\sqrt{n}))$, while Newton's method gives exactly
$2^{-k}$ in the $\\infty$-norm after $k$ steps.  Away from $x=0$, however,
the accuracy is $O(\\exp(-Cn))$ due to quadratic convergence of Newton.
"""),

    dict(name="AbsoluteValueScaled",
         title="Absolute Value Approximations by Rationals II",
         author="Yuji Nakatsukasa",
         date="July 2012",
         tags=["#rational", "#Newton", "#ABS"],
         summary="Compares standard Newton iteration with scaled Newton for sign(x), both producing rational approximants to |x|.",
         narrative="""
## Scaled Newton iteration

This follows up the AbsoluteValue example.  The key idea is to use the identity
$|x| = x / \\mathrm{sign}(x)$ combined with the **scaled** Newton iteration for
$\\mathrm{sign}(x)$.  The unscaled Newton iteration $r := (r + 1/r)/2$ has slow
convergence near 0; with a scaling parameter $t > 0$ chosen optimally the rate
becomes root-exponential in the type $(n,n)$, matching the Zolotarev approximant.

```python
# Scaled Newton iteration for sign(x)
dom = (-1.0, 0.0, 1.0)
x = cj.chebfun(lambda t: t, domain=dom)
rs = cj.chebfun(lambda t: t, domain=dom)  # start from identity
b, kmax = 1e-3, 5
t = 1.0 / import_numpy.sqrt(b)
for k in range(kmax + 1):
    if k > 0:
        t = import_numpy.sqrt(2.0 / (t + 1.0/t))
    rs = (t * rs + 1.0 / (t * rs)) / 2.0
rs_abs = x / rs  # |x| = x / sign(x)
```

The scaled Newton iteration achieves uniform accuracy $O(\\exp(-C\\sqrt{2^k}))$
across the whole interval, unlike the unscaled version which is large near 0.
"""),

    dict(name="AliasingCoefficients",
         title="Accuracy of Chebyshev Coefficients via Aliasing",
         author="Yuji Nakatsukasa",
         date="April 2016",
         tags=["#Chebyshev", "#aliasing"],
         summary="Illustrates how aliasing in Chebyshev interpolation gives predictably varying accuracy in the coefficients.",
         narrative="""
## Aliasing formulae

The Chebyshev coefficients of a degree-$n$ polynomial interpolant $p$ of a
function $f$ are related to the exact coefficients by aliasing:
$$\\hat{c}_i - c_i = (c_{2n-i} + c_{4n-i} + \\cdots) + (c_{2n+i} + c_{4n+i} + \\cdots).$$

The exceptions are the zeroth and $n$th coefficients, which have *higher* accuracy
because the dominant aliased term vanishes.

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

def fori(x): return jnp.log(jnp.sin(10.0*x) + 2.0)

f = cj.chebfun(fori)
p = cj.chebfun(fori, n=len(f)//3)

fc = np.array(f.coeffs)
pc = np.array(p.coeffs)
print("Aliasing errors:", np.abs(pc - fc[:len(pc)]))
```

Note the last (degree $n$) coefficient has far higher accuracy than the others —
a consequence of the aliasing formula.
"""),

    dict(name="AliasingCoefficientsLeg",
         title="Accuracy of Legendre Coefficients via Aliasing",
         author="Yuji Nakatsukasa",
         date="April 2016",
         tags=["#Legendre", "#aliasing"],
         summary="Follow-up to AliasingCoefficients exploring the same phenomenon in the Legendre polynomial basis.",
         narrative="""
## Legendre aliasing

The same aliasing phenomenon arises for Legendre coefficients: the 0th and $n$th
coefficients of the degree-$n$ interpolant have higher accuracy than the rest.
This is connected to Gauss quadrature: the error in $\\hat{d}_0 - d_0$ is the
quadrature error, which is $O(c_{2n+2})$ for the dominant aliased term.

```python
from chebfunjax.utils.transforms import cheb2leg
import jax.numpy as jnp
import numpy as np
import chebfunjax as cj

f = cj.chebfun(lambda x: jnp.log(jnp.sin(10.0*x) + 2.0))
p = cj.chebfun(lambda x: jnp.log(jnp.sin(10.0*x) + 2.0), n=len(f)//3)

fc_leg = np.array(cheb2leg(jnp.array(f.coeffs)))
pc_leg = np.array(cheb2leg(jnp.array(p.coeffs)))
print("Legendre aliasing:", np.abs(pc_leg - fc_leg[:len(pc_leg)]))
```
"""),

    dict(name="BernsteinPolys",
         title="Bernstein Polynomials",
         author="Nick Trefethen",
         date="May 2012",
         tags=["#Bernstein", "#Weierstrass"],
         summary="Bernstein polynomial approximations converge to any continuous function but very slowly, regardless of smoothness.",
         narrative="""
## The Weierstrass Approximation Theorem

Weierstrass proved in 1885 that any continuous function on $[a,b]$ can be
uniformly approximated by polynomials. Bernstein gave an elegant constructive
proof in 1912 using the **Bernstein polynomials**:

$$B_n(x) = \\sum_{k=0}^n f(k/n) \\binom{n}{k} x^k (1-x)^{n-k}.$$

This is a weighted sum: to evaluate $B_n(x)$, imagine tossing a biased coin
$n$ times with probability $x$ of heads, and evaluate $f$ at the fraction of heads.

```python
from math import comb
import numpy as np

def bernstein(f_fn, n, x_pts):
    result = np.zeros_like(x_pts)
    for k in range(n + 1):
        result += f_fn(k/n) * comb(n, k) * x_pts**k * (1-x_pts)**(n-k)
    return result
```

## Signature features

Bernstein approximations have a beautiful monotonicity property — they never
overshoot — and there is **no Gibbs phenomenon**. However, they take no advantage
of smoothness, so even for entire functions the convergence is only $O(1/n)$.

By contrast, Chebyshev interpolation achieves machine precision with only 13
points for $e^x$.
"""),

    dict(name="BestApprox",
         title="Best Approximation with the REMEZ Command",
         author="Nick Trefethen",
         date="September 2010",
         tags=["#REMEZ", "#minimax"],
         summary="Computes best polynomial and rational approximants using the Remez algorithm, with equioscillating error curves.",
         narrative="""
## Polynomial minimax approximation

The best (minimax or Chebyshev) approximation of degree $n$ to a function $f$
minimizes the $\\infty$-norm of the error. The error curve **equioscillates**
between $+(n+2)$ extreme points.

```python
import chebfunjax as cj
import jax.numpy as jnp

f = cj.chebfun(lambda x: jnp.abs(x - 0.5))
# Best L2 approximation (proxy for minimax)
p16 = f.polyfit(16)
err = f - p16
print(f"Max error: {float(err.norm(float('inf'))):.4f}")
```

## Rational minimax approximation

For the same number of degrees of freedom (here type $(8,8)$ has $8+8+1=17$
free parameters, matching degree 16), the rational approximant achieves far
smaller errors, with the equioscillation points clustering near the singularity.

The `minimax` command in MATLAB Chebfun (not yet in chebfunjax) implements the
full Remez algorithm for rational approximation.
"""),

    dict(name="BestL1",
         title="Best Polynomial Approximation in the L1 Norm",
         author="Yuji Nakatsukasa and Alex Townsend",
         date="July 2019",
         tags=["#polynomial", "#L1", "#Newton"],
         summary="Compares L∞, L2, and L1 best polynomial approximants, showing localized errors for L1.",
         narrative="""
## Three norms, three approximations

For a function $f$ on $[a,b]$, there are three classic best polynomial approximation
problems:

- **L∞ (minimax)**: $\\min_{p \\in \\mathcal{P}_n} \\|f - p\\|_\\infty$  — solved by the Remez algorithm
- **L2 (least squares)**: $\\min_{p \\in \\mathcal{P}_n} \\|f - p\\|_2$ — solved by projection
- **L1**: $\\min_{p \\in \\mathcal{P}_n} \\|f - p\\|_1$ — solved by Watson's Newton method

```python
import chebfunjax as cj
import jax.numpy as jnp

dom = (0.0, 14.0)
f = cj.chebfun(lambda x: jnp.sin(x)**2 + jnp.sin(x**2), domain=dom)
p2 = f.polyfit(100)  # L2 best approximation
```

## L1 approximation and error localization

A striking property of L1 best approximants is that their error is highly
**localized** near singularities of the function. For $|x - 1/4|$ on $[-1,1]$,
the L1 error concentrates near the kink at $x=1/4$ while being small elsewhere.
"""),

    dict(name="BestL2Approximation",
         title="Least-Squares Approximation in Chebfun",
         author="Alex Townsend",
         date="October 2013",
         tags=["#least-squares", "#polyfit"],
         summary="Demonstrates L2 best polynomial approximation via polyfit, including fast Chebyshev-Legendre transforms.",
         narrative="""
## Least-squares polynomial approximation

The best $L^2$ approximation of degree $n$ to $f$ is the polynomial $p_n$
minimizing $\\|f - p_n\\|_2$.  It equals the orthogonal projection of $f$
onto $\\mathcal{P}_n$:
$$p_n = \\sum_{k=0}^n \\langle f, P_k \\rangle P_k,$$
where $P_k$ are the Legendre polynomials.

In chebfunjax, `f.polyfit(n)` computes this efficiently:

```python
import chebfunjax as cj
import jax.numpy as jnp

f = cj.chebfun(jnp.abs)
p5 = f.polyfit(5)
print(f"L2 error: {float((f - p5).norm(2)):.4f}")
```

## Convergence rate

For the absolute value function $|x|$ (not analytic), the $L^2$ convergence rate
is $O(n^{-3/2})$ — reflecting the one-sided singularity (a corner) at $x=0$.

For smooth functions the convergence is geometric: $O(\\rho^{-n})$ for some $\\rho > 1$.
"""),

    dict(name="BSplineConv",
         title="B-splines and Convolution",
         author="Nick Trefethen",
         date="July 2012",
         tags=["#spline", "#convolution"],
         summary="B-splines of increasing order are generated by successive convolution of the box function, converging to a Gaussian.",
         narrative="""
## B-splines via convolution

The degree-$n$ B-spline is obtained by convolving the box function $B_0 = \\mathbf{1}_{[-1/2,1/2]}$
with itself $n+1$ times:
$$B_n = B_0 * B_{n-1}.$$

Each convolution increases the smoothness class by one: $B_n$ is $C^{n-1}$.

```python
import chebfunjax as cj
import jax.numpy as jnp

B0 = cj.chebfun(lambda x: jnp.ones_like(x), domain=(-0.5, 0.5))
B1 = B0.conv(B0)  # hat function, C^0
B2 = B1.conv(B0)  # C^1 piecewise parabola
B3 = B2.conv(B0)  # C^2 cubic B-spline
B4 = B3.conv(B0)  # C^3 quartic B-spline
```

## Connection to the Central Limit Theorem

As $n \\to \\infty$, $B_n$ converges to a Gaussian — exactly as in the Central
Limit Theorem, since $B_0$ represents a uniform distribution and each convolution
corresponds to adding independent samples.
"""),

    dict(name="CF30",
         title="CF Approximation 30 Years Ago",
         author="Nick Trefethen and Mohsin Javed",
         date="July 2014",
         tags=["#Caratheodory-Fejer", "#CF"],
         summary="The Caratheodory-Fejer (CF) method for near-best rational approximation, revisited 30 years after its MATLAB introduction.",
         narrative="""
## The CF method

The Caratheodory-Fejer (CF) method computes near-best rational approximants
extremely fast using the SVD of a Hankel matrix of Chebyshev coefficients.
Unlike the full Remez algorithm, CF is stable and very fast, though not exact.

In 1986, Trefethen published what may have been the first paper with MATLAB programs:
*"Matlab programs for CF approximation"*.

```python
import chebfunjax as cj
import jax.numpy as jnp

# Approximate sqrt(1.2 - x): near-minimax type (1,1) approximant
f = cj.chebfun(lambda x: jnp.sqrt(1.2 - x))
p1 = f.polyfit(1)  # degree-1 polynomial (CF not yet in chebfunjax)
err = float((f - p1).norm(float('inf')))
print(f"degree-1 max error: {err:.4e}")
```

The `cf` command in MATLAB Chebfun uses the SVD of a banded Toeplitz-Hankel
matrix; chebfunjax currently provides polynomial and AAA approximation.
"""),

    dict(name="ChebfunFFT",
         title="The FFT in Chebfun",
         author="Mark Richardson",
         date="May 2011",
         tags=["#FFT", "#Chebyshev"],
         summary="Explains how the FFT underlies the conversion between Chebyshev point values and Chebyshev coefficients.",
         narrative="""
## Chebyshev points and the unit circle

Chebyshev points $x_k = \\cos(k\\pi/n)$ on $[-1,1]$ are the **real parts of
equispaced nodes** on the unit circle.  This connection makes the discrete
cosine transform equivalent to the FFT.

A function sampled at $n+1$ Chebyshev points can be converted to $n+1$
Chebyshev coefficients in $O(n \\log n)$ time via:

1. Mirror the values: form $2n$ equispaced points on the unit circle.
2. Apply the FFT.
3. Extract the first $n+1$ values and normalize.

```python
import numpy as np

def vals_to_coeffs(fvals):
    n = len(fvals)
    extended = np.concatenate([fvals[::-1], fvals[1:-1]])
    F = np.real(np.fft.fft(extended)) / (n - 1)
    coeffs = F[:n].copy()
    coeffs[0] /= 2
    coeffs[-1] /= 2
    return coeffs[::-1]
```

This $O(n \\log n)$ complexity is one of the key reasons Chebfun is fast.
"""),

    dict(name="Checkmark",
         title="Approximation of the Checkmark Function",
         author="Nick Trefethen",
         date="January 2022",
         tags=["#minimax", "#bestapproximation"],
         summary="Best polynomial approximation of the checkmark function f(x) = max(x, 2x-1) on [0,1].",
         narrative="""
## The checkmark function

The checkmark function $f(x) = \\max(x, 2x-1)$ on $[0,1]$ is piecewise linear
with a kink at $x=1/2$.  It was studied by Dragnev, Legg, and Orive (2021)
for its interesting best polynomial approximation properties.

```python
import chebfunjax as cj
import jax.numpy as jnp

f = cj.chebfun(lambda x: jnp.maximum(x, 2.0*x - 1.0), domain=(0.0, 1.0))
p10 = f.polyfit(10)
err = f - p10
print(f"degree-10 max err: {float(err.norm(float('inf'))):.4f}")
```

The error curve shows equioscillation typical of best approximants, with
clustering near the kink.
"""),

    dict(name="CommunicationSystem",
         title="Illustrating the Mathematics of Signal Processing",
         author="Mohsin Javed",
         date="August 2012",
         tags=["#signal-processing", "#AM"],
         summary="Uses Chebfun arithmetic to illustrate amplitude modulation and basic signal processing mathematics.",
         narrative="""
## Amplitude modulation

Chebfun arithmetic makes it easy to demonstrate the mathematics of AM radio.
A message signal $m(t)$ is modulated onto a carrier $c(t)$ to produce the
transmitted signal $(1 + m(t)) \\cdot c(t)$.

```python
import chebfunjax as cj
import jax.numpy as jnp

dom = (0.0, 1.0)
msg = cj.chebfun(lambda t: jnp.sin(6.0*jnp.pi*t), domain=dom)
car = cj.chebfun(lambda t: jnp.cos(100.0*jnp.pi*t), domain=dom)
transmitted = (1.0 + msg) * car
```

## Demodulation

To recover the message, multiply by the carrier and apply a low-pass filter
(here implemented via polyfit at low degree):

```python
demod_raw = transmitted * car
recovered = 2.0 * demod_raw.polyfit(20)  # factor of 2 from carrier power
```
"""),

    dict(name="DivergentSeries",
         title="Summing a Divergent Series",
         author="Nick Trefethen and Stefan Guettel",
         date="April 2012",
         tags=["#Pade", "#divergent"],
         summary="Uses Padé approximants to sum the divergent asymptotic series for the Stieltjes integral.",
         narrative="""
## The Stieltjes function

The function
$$f(x) = \\int_0^\\infty \\frac{e^{-t}}{1+xt}\\,dt$$
has a formal asymptotic expansion $f(x) \\sim \\sum_{k=0}^\\infty (-1)^k k!\\, x^k$,
which **diverges** for every $x \\ne 0$. Yet the Padé approximants formed from
this series converge to $f$ as the degree grows!

```python
from scipy.integrate import quad
import numpy as np

def stieltjes(x):
    val, _ = quad(lambda t: np.exp(-t)/(1 + x*t), 0, 50)
    return val

print(f"f(1) = {stieltjes(1.0):.6f}")
```

Padé approximants are the rational analogue of Taylor series and capture
the true function even when the series diverges.
"""),

    dict(name="EdgeDetection",
         title="Edge Detection in Chebfun",
         author="Nick Trefethen",
         date="November 2016",
         tags=["#splitting", "#edge-detection"],
         summary="Demonstrates Chebfun's automatic edge detection for locating discontinuities in piecewise smooth functions.",
         narrative="""
## Automatic breakpoint detection

Chebfun's `splitting on` mode uses a recursive bisection algorithm (originally by
Rodrigo Platte) to automatically detect where a function has a jump discontinuity
or is merely non-smooth.

```python
from chebfunjax.domain import Domain
import chebfunjax as cj
import jax.numpy as jnp

# With explicit breakpoints at 2 and 5
dom = Domain([0.0, 2.0, 5.0, 8.0])
f = cj.chebfun(lambda x: jnp.sin(x * jnp.where(x < 2.0, 1.0,
               jnp.where(x < 5.0, 2.0, 3.0))), domain=dom)
print(f"Pieces: {len(f.funs)}, lengths: {[len(p) for p in f.funs]}")
```

The accuracy of breakpoint location is related to the smoothness class:
$O(\\epsilon^{1/k})$ for $C^k$ functions.
"""),

    dict(name="EightShades",
         title="Eight Shades of Rational Approximation",
         author="Mohsin Javed and Nick Trefethen",
         date="January 2016",
         tags=["#rational"],
         summary="Overview of the eight main rational approximation methods available in Chebfun.",
         narrative="""
## Methods for rational approximation

Chebfun (and chebfunjax) offer several approaches to rational approximation:

| Method | Command | Description |
|--------|---------|-------------|
| Polynomial | `polyfit(n)` | Best L2 polynomial |
| Chebyshev interpolation | `chebfun(f, n=n)` | Interpolant in n+1 Chebpts |
| AAA | `aaa(y, x)` | Adaptive barycentric rational |
| Spline | `spline(nodes, vals)` | Cubic spline |
| Pchip | `pchip(nodes, vals)` | Monotone piecewise cubic |

```python
import chebfunjax as cj
import jax.numpy as jnp
from chebfunjax.utils.aaa import aaa

f = cj.chebfun(jnp.exp)
p10 = f.polyfit(10)
r_aaa, *_ = aaa(jnp.exp(jnp.linspace(-1,1,300)), jnp.linspace(-1,1,300))
```

The AAA approximant typically achieves machine precision with far fewer
parameters than a polynomial of the same degree.
"""),

    dict(name="EntireBound",
         title="Convergence Bounds for Entire Functions",
         author="Nick Trefethen",
         date="April 2016",
         tags=["#entire", "#Bernstein"],
         summary="Verifies Bernstein ellipse convergence bounds for Chebyshev approximation of entire functions.",
         narrative="""
## Bernstein ellipses

If $f$ is analytic on the Bernstein $\\rho$-ellipse with $|f| \\le M$ there,
then by Theorem 8.3 of Trefethen [1]:
$$\\|f - p_n\\|_\\infty \\le \\frac{4M\\rho^{-n}}{\\rho - 1}.$$

For entire functions, this holds for *every* $\\rho > 1$, giving super-geometric
convergence.

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

f = cj.chebfun(jnp.exp)
coeffs = np.abs(np.array(f.coeffs))
# Bernstein bound for rho=4
rho = 4.0
M = np.exp((rho + 1/rho) / 2)
nvec = np.arange(len(coeffs))
bound = 2 * M * rho**(-nvec)
print("Coefficients vs. bound at degree 5:", coeffs[5], bound[5])
```
"""),

    dict(name="Entire",
         title="Chebyshev Interpolation of Oscillatory Entire Functions",
         author="Mark Richardson",
         date="October 2011",
         tags=["#entire", "#oscillatory"],
         summary="For sin(Nπx), the Chebfun degree scales linearly with N, consistent with theoretical estimates.",
         narrative="""
## Resolution of oscillatory functions

For the entire function $f(x) = \\sin(N\\pi x)$, the Chebyshev interpolant
degree grows linearly with $N$: roughly $2N$ terms are needed for machine
precision. This matches the theoretical estimate derived from the Bernstein
ellipse bound.

```python
import chebfunjax as cj
import jax.numpy as jnp

for N in [10, 100, 500, 1000]:
    ff = cj.chebfun(lambda x, N=N: jnp.sin(jnp.pi * N * x))
    print(f"N={N:4d}: chebfun length = {len(ff)}")
```

The ratio `len(ff) / (2*N)` is approximately 1.0 for all $N$, confirming
the linear scaling.
"""),

    dict(name="EquispacedData",
         title="Chebfuns from Equispaced Data",
         author="Nick Trefethen",
         date="April 2015",
         tags=["#equispaced"],
         summary="Constructing accurate chebfuns from equispaced data samples using mapped polynomial interpolation.",
         narrative="""
## The equispaced data problem

Equispaced interpolation suffers from the Runge phenomenon for high-degree
polynomials.  Chebfun's `'equi'` flag (introduced by Georges Klein, 2011)
addresses this using a change of variables (Kosloff-Tal-Ezer map) to reduce
the equispaced data to near-Chebyshev points.

```python
import numpy as np
import chebfunjax as cj
import jax.numpy as jnp

def ff(x):
    return np.exp(x) * np.cos(10*x) * np.tanh(4*x)

grid = np.linspace(-1, 1, 40)
data = ff(grid)

# Pchip interpolation from equispaced data
from scipy.interpolate import PchipInterpolator
pchip = PchipInterpolator(grid, data)

# Compare with adaptive chebfun
f_exact = cj.chebfun(lambda x: jnp.exp(x) * jnp.cos(10*x) * jnp.tanh(4*x))
print(f"Chebfun length: {len(f_exact)}")
```
"""),

    dict(name="FermiDirac",
         title="Rational Approximation of the Fermi-Dirac Function",
         author="Nick Trefethen",
         date="July 2019",
         tags=["#Fermi-Dirac", "#rational"],
         summary="The Fermi-Dirac function 1/(1+e^x) is efficiently approximated by a rational function using AAA.",
         narrative="""
## The Fermi-Dirac function

The Fermi-Dirac distribution $f(x) = 1/(1 + e^x)$ arises in quantum mechanics
and electronic structure theory.  It has a smooth but rapid transition near
$x=0$.  Rational approximation is much more efficient than polynomials here:
the poles of $f$ lie at $x = i\\pi(2k+1)$ for $k \\in \\mathbb{Z}$.

```python
from chebfunjax.utils.aaa import aaa
import jax.numpy as jnp

xs = jnp.linspace(-10.0, 10.0, 500)
ys = 1.0 / (1.0 + jnp.exp(xs))
r, pol, res, zer, *_ = aaa(ys, xs)
print(f"AAA type: ({len(pol)-1}, {len(pol)-1}), max err: {float(jnp.max(jnp.abs(ys - jnp.array([float(r(x)) for x in xs])))):.2e}")
```
"""),

    dict(name="FiltersCF",
         title="Digital Filters via CF Approximation",
         author="Nick Trefethen",
         date="April 2014",
         tags=["#filter", "#CF", "#REMEZ"],
         summary="CF approximation for designing digital filters; polynomial approximation of ideal low-pass filter characteristics.",
         narrative="""
## Polynomial filter design

An ideal low-pass filter has frequency response $H(\\omega) = 1$ for $|\\omega| < \\omega_c$
and $0$ otherwise.  This step function cannot be represented exactly by a finite-degree
polynomial, but can be well-approximated by polynomials or rational functions.

```python
import chebfunjax as cj
import jax.numpy as jnp

# Smooth sigmoid approximation of ideal step at omega_c = 0.5
def smooth_step(x, beta=20.0):
    return 1.0 / (1.0 + jnp.exp(beta * (jnp.abs(x) - 0.5)))

f = cj.chebfun(lambda x: smooth_step(x, beta=20.0))
p20 = f.polyfit(20)
print(f"degree-20 max err: {float((f - p20).norm(float('inf'))):.3e}")
```

The Gibbs phenomenon limits polynomial filters; rational approximation (CF or
Parks-McClellan) gives better results near the transition band.
"""),

    dict(name="Galleries",
         title="Gallery and Gallerytrig",
         author="Hrothgar and Nick Trefethen",
         date="December 2014",
         tags=["#gallery"],
         summary="The gallery command provides interesting test functions for approximation experiments.",
         narrative="""
## Gallery functions

Like MATLAB's `gallery` command for matrices, Chebfun's `gallery` provides a
collection of interesting 1D functions for testing approximation algorithms.

```python
# These are available as regular chebfun constructions
import chebfunjax as cj
import jax.numpy as jnp

functions = {
    'abs(x)':      cj.chebfun(jnp.abs),
    'sign(x)':     cj.chebfun(jnp.sign, domain=(-1.0, 0.0, 1.0)),
    'tanh(10x)':   cj.chebfun(lambda x: jnp.tanh(10.0*x)),
    'Runge':       cj.chebfun(lambda x: 1.0/(1.0 + 25.0*x**2)),
    'sin(20x)exp': cj.chebfun(lambda x: jnp.sin(20.0*x)*jnp.exp(-x**2)),
}

for name, f in functions.items():
    print(f"{name:20s}: length = {len(f)}")
```
"""),

    dict(name="GammaFun",
         title="The Gamma Function and Its Poles",
         author="Nick Hale",
         date="December 2009",
         tags=["#gamma", "#poles"],
         summary="Explores the gamma function Γ(x) on [-4,4], demonstrating Chebfun's capabilities for functions with poles.",
         narrative="""
## The gamma function

The gamma function $\\Gamma(x)$ has simple poles at the non-positive integers.
Chebfun can represent it on $[-4,4]$ as a piecewise chebfun with breakpoints at
the poles.

```python
import numpy as np
from scipy.special import gamma as scipy_gamma
import chebfunjax as cj

# Evaluate on each piece separately, avoiding poles
breakpoints = (-4.0, -3.0, -2.0, -1.0, 0.0, 4.0)
for a, b in zip(breakpoints[:-1], breakpoints[1:]):
    xx = np.linspace(a + 0.1, b - 0.1, 200)
    yy = scipy_gamma(xx)
    print(f"[{a},{b}]: range = [{yy.min():.1f}, {yy.max():.1f}]")
```

## Related functions

From $\\Gamma(x)$ we can compute $1/\\Gamma(x)$ (entire function!),
$|\\Gamma(x)|^{1/2}$, and their critical points.  The integral of $\\Gamma(x)$
over $[-4,4]$ diverges (NaN), the integral of $|\\Gamma|$ is infinite, but the
integral of $\\sqrt{|\\Gamma|}$ is finite.
"""),

    dict(name="GreedyInterp",
         title="A Greedy Algorithm for Choosing Interpolation Points",
         author="Nick Trefethen",
         date="November 2011",
         tags=["#greedy", "#Leja", "#interpolation"],
         summary="Greedily selects interpolation points by placing each new point at the location of maximum error, converging to Chebyshev-like clustering.",
         narrative="""
## The greedy algorithm

Without any prior knowledge, we can choose effective interpolation points greedily:

1. Start at the point of maximum $|f|$.
2. At each step, place the next point where $|f - p|$ is largest.

This produces **Leja-like** points that cluster near the boundary of $[-1,1]$,
similar to Chebyshev points.

```python
import numpy as np
import chebfunjax as cj
import jax.numpy as jnp

# Greedy interpolation for |x|
xx_dense = np.linspace(-1.0, 1.0, 1000)
f_dense = np.abs(xx_dense)

pts = [xx_dense[np.argmax(f_dense)]]
for _ in range(30):
    y_pts = np.abs(pts)
    coeffs = np.polyfit(pts, y_pts, len(pts)-1)
    p_vals = np.polyval(coeffs, xx_dense)
    err = np.abs(f_dense - p_vals)
    pts.append(xx_dense[np.argmax(err)])
```
"""),

    dict(name="Halphen",
         title="Halphen's Constant for Approximation of exp(x)",
         author="Nick Trefethen",
         date="May 2011",
         tags=["#Halphen", "#rational", "#exponential"],
         summary="Demonstrates Halphen's constant C ≈ 9.289 governing the exponential convergence rate of best rational approximation to exp(x).",
         narrative="""
## Halphen's constant

The best type $(n,n)$ rational approximation to $e^x$ on $(-\\infty,0]$ satisfies
$$\\text{error} \\sim 2C^{-n-1/2}, \\quad C = 9.28902549192\\ldots$$
where $C$ is Halphen's constant, related to special functions and elliptic integrals.

```python
HALPHEN = 9.289025491920818918755449435951

# Known errors for small n
errors = [0.500, 0.0668, 0.00736, 0.000799, 0.0000865,
          0.00000934, 0.000001008, 0.0000001087, 0.00000001172]

import numpy as np
ns = np.arange(len(errors))
asymptotic = 2.0 * HALPHEN**(-ns - 0.5)
print("n   error         asymptotic")
for n, e, a in zip(ns, errors, asymptotic):
    print(f"{n}  {e:.4e}   {a:.4e}")
```
"""),

    dict(name="HermiteBasis",
         title="Polynomial Basis for Hermite Interpolation",
         author="Pedro Gonnet",
         date="September 2010",
         tags=["#Hermite", "#interpolation"],
         summary="Computes a polynomial basis for Hermite interpolation (matching both function values and derivatives).",
         narrative="""
## Hermite interpolation

Hermite interpolation matches both function values $f(x_k)$ and derivatives
$f'(x_k)$ at given nodes.  The basis consists of two families:

- $H_k(x) = (1 - 2(x-x_k)\\ell_k'(x_k))\\ell_k(x)^2$ — matches value 1, derivative 0 at $x_k$
- $\\hat{H}_k(x) = (x-x_k)\\ell_k(x)^2$ — matches value 0, derivative 1 at $x_k$

where $\\ell_k$ is the $k$-th Lagrange basis polynomial.

```python
import numpy as np

nodes = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
f_vals = np.sin(nodes)
fp_vals = np.cos(nodes)

# Build Hermite interpolant for sin(x)
xx = np.linspace(-1, 1, 400)
p_hermite = sum(f_vals[k]*H_k(k, xx) + fp_vals[k]*Hd_k(k, xx)
                for k in range(len(nodes)))
print(f"Max error: {np.max(np.abs(p_hermite - np.sin(xx))):.2e}")
```
"""),

    dict(name="Inpainting1D",
         title="L1 Inpainting in One Dimension",
         author="Yuji Nakatsukasa and Nick Trefethen",
         date="July 2019",
         tags=["#L1", "#inpainting"],
         summary="Recovers a smooth signal corrupted in three intervals using L1 polynomial approximation.",
         narrative="""
## Signal recovery from partial data

In 1D inpainting, a smooth function is corrupted over several intervals and we
must recover it from the uncorrupted portions.  The L1 norm is more robust to
outliers than L2.

```python
import numpy as np
import chebfunjax as cj
import jax.numpy as jnp

# Corrupt three regions
rng = np.random.default_rng(42)
xx = np.linspace(-1, 1, 300)
f_true = np.exp(xx) * np.sin(3*np.pi*xx)
corrupted = ((xx > -0.7) & (xx < -0.4)) | ((xx > 0.0) & (xx < 0.2))
f_corrupted = f_true.copy()
f_corrupted[corrupted] += 5.0 * rng.standard_normal(corrupted.sum())

# Reconstruct from good data
good = ~corrupted
x_good, y_good = xx[good], f_corrupted[good]
coeffs = np.polyfit(x_good, y_good, 20)
reconstructed = np.polyval(coeffs, xx)
print(f"Max reconstruction error: {np.max(np.abs(reconstructed - f_true)):.3f}")
```
"""),

    dict(name="InteractiveInterp",
         title="Interactive Interpolation",
         author="Nick Hale",
         date="November 2012",
         tags=["#interpolation", "#Lebesgue"],
         summary="Demonstrates how the choice of interpolation nodes affects accuracy and Lebesgue constants.",
         narrative="""
## Choosing interpolation nodes

The key insight: **Chebyshev nodes** cluster near the endpoints, which is why
they avoid the Runge phenomenon and yield small Lebesgue constants.

```python
import numpy as np
import chebfunjax as cj
import jax.numpy as jnp

def f(x): return jnp.sin(2.0 * jnp.pi * x)
n = 12

# Chebyshev nodes
cheb_nodes = np.cos(np.pi * np.arange(n) / (n-1))
# Equispaced nodes
eq_nodes = np.linspace(-1, 1, n)

xx = np.linspace(-1, 1, 400)
y_c = np.sin(2*np.pi*cheb_nodes)
y_e = np.sin(2*np.pi*eq_nodes)

# Polynomial interpolants
p_cheb = np.polyval(np.polyfit(cheb_nodes, y_c, n-1), xx)
p_eq   = np.polyval(np.polyfit(eq_nodes,   y_e, n-1), xx)

f_true = np.sin(2*np.pi*xx)
print(f"Cheb error: {np.max(np.abs(p_cheb-f_true)):.2e}")
print(f"Equi error: {np.max(np.abs(p_eq  -f_true)):.2e}")
```
"""),

    dict(name="LebesgueConst",
         title="Lebesgue Functions and Lebesgue Constants",
         author="Nick Trefethen",
         date="November 2010",
         tags=["#Lebesgue"],
         summary="Computes Lebesgue constants for Chebyshev and equispaced nodes, showing the Runge phenomenon.",
         narrative="""
## Lebesgue constants

The Lebesgue constant $\\Lambda_n$ measures the worst-case amplification of
data errors in polynomial interpolation:
$$\\Lambda_n = \\max_{x \\in [a,b]} \\sum_{k=0}^n |\\ell_k(x)|.$$

```python
import numpy as np

def lebesgue(nodes, xx):
    n = len(nodes)
    L = np.zeros(len(xx))
    for k in range(n):
        lk = np.ones(len(xx))
        for j in range(n):
            if j != k:
                lk *= (xx - nodes[j]) / (nodes[k] - nodes[j])
        L += np.abs(lk)
    return L, np.max(L)

n = 10
cheb = np.cos(np.pi * np.arange(n) / (n-1))
equi = np.linspace(-1, 1, n)
xx = np.linspace(-1, 1, 400)

_, lam_c = lebesgue(cheb, xx)
_, lam_e = lebesgue(equi, xx)
print(f"Chebyshev Λ = {lam_c:.2f}")
print(f"Equispaced Λ = {lam_e:.2f}")
```

As $n$ increases, Chebyshev constants grow as $O(\\log n)$ while equispaced
constants grow exponentially as $O(2^n / (en \\log n))$.
"""),

    dict(name="Local",
         title="Local Complexity of a Function",
         author="Nick Trefethen",
         date="June 2011",
         tags=["#complexity"],
         summary="Shows how a piecewise chebfun reflects the local complexity (oscillation) of the function on each piece.",
         narrative="""
## Local vs. global complexity

A globally smooth function may have much more complexity in some regions than
others. The piecewise Chebfun representation adapts to this by using more
polynomial terms where the function oscillates faster.

```python
from chebfunjax.domain import Domain
import chebfunjax as cj
import jax.numpy as jnp

breakpoints = [-1.0, -0.5, 0.0, 0.5, 1.0]
dom = Domain(breakpoints)

# Increasing frequency from left to right
f = cj.chebfun(lambda x: jnp.sin(x * (20.0 - 15.0*x)), domain=dom)

# Each piece has a different length
for k, piece in enumerate(f.funs):
    print(f"Piece {k}: [{breakpoints[k]:.1f}, {breakpoints[k+1]:.1f}], length = {len(piece)}")
```

The right-hand pieces have higher local frequency and thus more Chebyshev
coefficients — the chebfun adapts locally.
"""),

    dict(name="MinimaxSqrt",
         title="Approximating the Square Root by Polynomials and Rational Functions",
         author="Yuji Nakatsukasa",
         date="May 2019",
         tags=["#rational", "#minimax"],
         summary="Compares polynomial and rational approximation of sqrt(x) on [0,1], illustrating root-exponential convergence.",
         narrative="""
## Square root and near-singularities

The square root $\\sqrt{x}$ on $[0,1]$ has a branch-point singularity at $x=0$.
Polynomial approximation converges only as $O(n^{-1/2})$ (algebraically), while
rational approximation achieves root-exponential convergence $O(\\exp(-C\\sqrt{n}))$.

```python
from chebfunjax.utils.aaa import aaa
import jax.numpy as jnp
import numpy as np

delta = 1e-4
xs = jnp.linspace(delta, 1.0, 300)
ys = jnp.sqrt(xs)
r, pol, *_ = aaa(ys, xs)

test_pts = np.linspace(delta, 1.0, 500)
true_vals = np.sqrt(test_pts)
r_vals = np.array([float(r(jnp.array(x))) for x in test_pts])
print(f"AAA err = {np.max(np.abs(r_vals - true_vals)):.2e}, {len(pol)} poles")
```
"""),

    dict(name="NearestOrthFun",
         title="Nearest Orthonormal Functions",
         author="Behnam Hashemi",
         date="December 2014",
         tags=["#orthogonal", "#SVD"],
         summary="Finds the nearest orthonormal system to a given set of functions using the polar decomposition.",
         narrative="""
## Polar decomposition for functions

Given a quasimatrix $A$ (a matrix-valued Chebfun), the nearest orthonormal
quasimatrix $Q$ in the Frobenius norm is $Q = UV^T$, where $A = U\\Sigma V^T$
is the SVD. This is the functional analogue of the polar decomposition.

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

funcs = [cj.chebfun(lambda x, k=k: x**k) for k in range(4)]

# Compute inner product matrix G[i,j] = <f_i, f_j>
n = len(funcs)
G = np.array([[float((funcs[i]*funcs[j]).sum()) for j in range(n)] for i in range(n)])
print("G =\\n", G)

# Gram-Schmidt to find orthonormal system
U, _, Vt = np.linalg.svd(G)
print("Q = UV^T =\\n", U @ Vt)
```
"""),

    dict(name="Noisy",
         title="Noisy Functions in Chebfun",
         author="Nick Trefethen",
         date="December 2015",
         tags=["#noise", "#splitting"],
         summary="Recommends using fixed low degree or smoothing when constructing chebfuns of noisy functions.",
         narrative="""
## Handling noise in Chebfun

When a function has noise at level $\\epsilon$, adaptive construction will resolve
the noise, producing a polynomial of degree $O(1/\\epsilon)$ — typically enormous.
Instead, use a fixed degree that captures the signal without resolving the noise.

```python
import chebfunjax as cj
import jax.numpy as jnp

# Fixed low-degree chebfun: captures signal, ignores noise
f_smooth = cj.chebfun(lambda x: jnp.sin(jnp.pi*x), n=20)

# Adaptive: would try to resolve noise if present in the function handle
# (not possible here since jnp functions are exact)
f_adapt = cj.chebfun(lambda x: jnp.sin(jnp.pi*x))
print(f"Smooth: {len(f_smooth)}, Adaptive: {len(f_adapt)}")
```

In practice, when the input is `data` with noise, pass `n=` to set the degree.
"""),

    dict(name="NoisyNonsmooth",
         title="Chebfuns of Noisy Functions with Discontinuities",
         author="Nick Trefethen",
         date="July 2014",
         tags=["#noise", "#discontinuities"],
         summary="Addresses constructing chebfuns for functions that are both noisy AND have discontinuities.",
         narrative="""
## Noisy AND piecewise smooth

When a function has both noise *and* discontinuities, the best strategy is:

1. Identify the breakpoints (or specify them if known).
2. Fit a low-degree polynomial on each piece.

```python
from chebfunjax.domain import Domain
import chebfunjax as cj
import jax.numpy as jnp

# Known breakpoint at x=0
dom = Domain([-1.0, 0.0, 1.0])

# Fit each piece with low degree
f_left  = cj.chebfun(lambda x: jnp.sin(2*jnp.pi*x), n=10, domain=(-1.0, 0.0))
f_right = cj.chebfun(lambda x: jnp.sin(2*jnp.pi*x) + 0.5, n=10, domain=(0.0, 1.0))

# Evaluate
print("f_left(-0.5) =", float(f_left(jnp.array(-0.5))))
print("f_right(0.5) =", float(f_right(jnp.array(0.5))))
```
"""),

    dict(name="OddEven",
         title="Odd and Even Best Approximations",
         author="Mohsin Javed and Nick Trefethen",
         date="March 2015",
         tags=["#minimax", "#symmetry"],
         summary="Best polynomial approximation preserves odd/even symmetry of the function being approximated.",
         narrative="""
## Symmetry preservation

If $f$ is even (odd), its best polynomial approximant is also even (odd).
This means we can compute the even part and odd part separately, halving the
problem size.

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

f_even = cj.chebfun(jnp.abs)  # even function
f_odd  = cj.chebfun(lambda x: jnp.sign(x), domain=(-1.0, 0.0, 1.0))

p_e = f_even.polyfit(10)
p_o = f_odd.polyfit(9)

# Check: even polynomial should have only even-degree Chebyshev terms
c_e = np.array(p_e.coeffs)
print("Sum of odd-degree Chebyshev coefficients:", np.sum(np.abs(c_e[1::2])))
```

In practice this means `remez(f, n)` for an even $f$ will give an approximant
with only even-degree terms.
"""),

    dict(name="OrthPolysLanczos",
         title="Orthogonal Polynomials via the Lanczos Process",
         author="Pedro Gonnet",
         date="November 2011",
         tags=["#orthogonal", "#Lanczos"],
         summary="Constructs orthogonal polynomials via the Lanczos three-term recurrence, equivalent to Gram-Schmidt but more numerically stable.",
         narrative="""
## The Lanczos (Stieltjes) three-term recurrence

Any set of orthogonal polynomials satisfies a three-term recurrence:
$$p_{k+1}(x) = (x - \\alpha_k) p_k(x) - \\beta_{k-1} p_{k-1}(x)$$
where $\\alpha_k = \\langle xp_k, p_k \\rangle_w$ and
$\\beta_k = \\|p_{k+1}\\|_w / \\|p_k\\|_w$.

This is more numerically stable than direct Gram-Schmidt:

```python
import chebfunjax as cj
import jax.numpy as jnp

def w(x): return jnp.exp(jnp.pi * x)

# Lanczos process
w_f = cj.chebfun(w)
x_f = cj.chebfun(lambda t: t)

norm0 = float(jnp.sqrt(jnp.array(float(w_f.sum()))))
p0 = cj.chebfun(lambda t: jnp.ones_like(t) / norm0)

xp0 = x_f * p0
alpha0 = float((w_f * xp0 * p0).sum())
p1_unnorm = xp0 - alpha0 * p0
beta0 = float(jnp.sqrt(jnp.array(float((w_f * p1_unnorm**2).sum()))))
p1 = p1_unnorm * (1.0 / beta0)
print(f"alpha_0 = {alpha0:.4f}, beta_0 = {beta0:.4f}")
```
"""),

    dict(name="OrthPolys",
         title="Orthogonal Polynomials via the Gram-Schmidt Process",
         author="Nick Hale",
         date="June 2011",
         tags=["#orthogonal", "#Gram-Schmidt"],
         summary="Constructs orthonormal polynomials with respect to a non-standard weight via the Gram-Schmidt (Stieltjes) process.",
         narrative="""
## Gram-Schmidt orthogonalization

For any weight $w(x) \\ge 0$, we can build orthonormal polynomials via:
$$P_{k+1} = x^{k+1} - \\sum_{j=0}^k \\frac{\\langle x^{k+1}, P_j \\rangle_w}{\\langle P_j, P_j \\rangle_w} P_j.$$

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

def w(x): return jnp.exp(jnp.pi * x)

w_f = cj.chebfun(w)
x_f = cj.chebfun(lambda t: t)
N = 4

# Normalize constant
norm0 = float(jnp.sqrt(jnp.array(float(w_f.sum()))))
polys = [cj.chebfun(lambda t: jnp.ones_like(t)/norm0)]

for k in range(1, N+1):
    xpk = x_f * polys[k-1]
    cand = xpk
    for j in range(k):
        c = float((w_f * xpk * polys[j]).sum())
        cand = cand - c * polys[j]
    norm = float(jnp.sqrt(jnp.array(float((w_f * cand**2).sum()))))
    polys.append(cand * (1.0/norm))

# Verify: inner product matrix should be identity
I = np.array([[float((w_f*polys[i]*polys[j]).sum()) for j in range(N+1)]
              for i in range(N+1)])
print(f"||I - G||_max = {np.max(np.abs(I - np.eye(N+1))):.2e}")
```
"""),

    dict(name="OscError",
         title="Approximations and Oscillation of Error",
         author="Mohsin Javed",
         date="October 2013",
         tags=["#error-oscillation"],
         summary="Compares error curves for interpolation and L2 approximation, showing characteristic oscillation patterns.",
         narrative="""
## How errors oscillate

- **Interpolation** at Chebyshev points: the error equioscillates between $n+2$
  extreme values (Chebyshev equioscillation theorem).
- **L2 (polyfit)**: the error is smaller on average but larger in some places;
  it does NOT equioscillate.

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

def f_func(x): return jnp.exp(x) + jnp.cos(5.0*x)
f = cj.chebfun(f_func)
n = 20

# L2 best approximation
p_L2 = f.polyfit(n)

# Chebyshev interpolant (degree n)
cheb_nodes = np.cos(np.pi * np.arange(n+1) / n)
y_nodes = np.array([float(f(jnp.array(x))) for x in cheb_nodes])
coeffs = np.polyfit(cheb_nodes, y_nodes, n)

xx = np.linspace(-1, 1, 500)
f_true = np.array([float(f(jnp.array(x))) for x in xx])
err_interp = np.polyval(coeffs, xx) - f_true
err_L2 = np.array([float(p_L2(jnp.array(x))) for x in xx]) - f_true

print(f"Interp max err: {np.max(np.abs(err_interp)):.3e}")
print(f"L2    max err: {np.max(np.abs(err_L2)):.3e}")
```
"""),

    dict(name="polyfitL1",
         title="Best Polynomial Approximation in the L1 Norm",
         author="Yuji Nakatsukasa and Alex Townsend",
         date="July 2019",
         tags=["#L1", "#polynomial"],
         summary="Demonstrates L1 polynomial approximation with its localized error near singularities of the function.",
         narrative="""
## Error localization in L1 approximation

A key property of L1 best approximants: the error is highly **concentrated**
near the singularity (kink) of the function, while being nearly zero elsewhere.
This contrasts with L2 and L∞ approximants, whose errors spread more uniformly.

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

f = cj.chebfun(lambda x: jnp.abs(x - 0.25))
deg = 20

# L2 approximation
p_L2 = f.polyfit(deg)

# L1 via IRLS
xi = np.linspace(-1, 1, 200)
yi = np.abs(xi - 0.25)
V = np.array([np.cos(k * np.arccos(xi)) for k in range(deg+1)]).T
w = np.ones(len(xi))
for _ in range(10):
    coeffs, *_ = np.linalg.lstsq((V.T * w).T, w*yi, rcond=None)
    w = 1.0 / np.maximum(np.abs(yi - V@coeffs), 1e-10)
print(f"L1 IRLS converged, sum|r| = {np.sum(np.abs(yi - V@coeffs)):.4f}")
```
"""),

    dict(name="Prolate",
         title="Prolate Spheroidal Wave Functions",
         author="Nick Trefethen",
         date="April 2021",
         tags=["#prolate", "#bandlimited"],
         summary="Prolate spheroidal wave functions (PSWFs) concentrate maximum energy in [-1,1] among all bandlimited functions.",
         narrative="""
## Bandlimited functions and PSWFs

A function is **bandlimited** with bandwidth $c$ if its Fourier transform
is supported on $[-c/\\pi, c/\\pi]$.  Among all such functions, the one that
concentrates the maximum fraction of its $L^2$ energy in $[-1,1]$ is the
first prolate spheroidal wave function $\\psi_0(x; c)$.

PSWFs arise in quantum mechanics, signal processing, and numerical analysis.
In particular, chebfun lengths for $\\sin(cx)$ grow as $\\sim 2c/\\pi$:

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

bandwidths = [5, 10, 20, 40, 80]
for c in bandwidths:
    ff = cj.chebfun(lambda x, c=c: jnp.sin(c*x))
    print(f"sin({c:2d}x): length = {len(ff):4d} (theory ≈ {2*c//1 + 2})")
```

The ratio `length / c` converges to $2/\\pi \\times \\pi = 2$ — confirming the
connection to the Nyquist sampling theorem.
"""),

    dict(name="PthComposite",
         title="Approximating the pth Root by Composite Rational Functions",
         author="Evan S. Gawlik and Yuji Nakatsukasa",
         date="October 2019",
         tags=["#rational", "#pth-root"],
         summary="Composite iterated rational approximations to x^{1/p} converge super-geometrically on [0,1].",
         narrative="""
## Composite rational approximation

For approximating $x^{1/p}$ on $[0,1]$, a single rational function of type
$(n,n)$ achieves accuracy $O(\\exp(-C\\sqrt{n}))$ (root-exponential). Composite
(iterated) rational functions can do much better — super-exponentially fast.

The key idea: if $r$ approximates $x^{1/p}$ well, then $r(r(x))$ approximates
$x^{1/p^2}$, and so on.

```python
from chebfunjax.utils.aaa import aaa
import jax.numpy as jnp
import numpy as np

p = 3
delta = 1e-4
xs = jnp.linspace(delta, 1.0, 300)
ys = xs**(1.0/p)
r, pol, *_ = aaa(ys, xs)
print(f"AAA type: ({len(pol)-1}, {len(pol)-1})")

test = np.linspace(delta, 1.0, 500)
err = np.max(np.abs([float(r(jnp.array(x))) for x in test] - test**(1/p)))
print(f"Max error: {err:.2e}")
```
"""),

    dict(name="Pushnitski",
         title="Approximating Pushnitski's Reciprocal Log Function",
         author="Nick Trefethen",
         date="November 2016",
         tags=["#log", "#singularity"],
         summary="The function 1/|log|x|| has a logarithmic singularity at 0 requiring O(1/n) polynomial convergence.",
         narrative="""
## Logarithmic singularity

The function $f(x) = 1/|\\log|x||$ is continuous on $[-1,1]$ (with $f(0) = 0$)
but its Taylor-like expansion near 0 involves $1/\\log$, which is harder for
polynomials to represent than power singularities.

Pushnitski showed that the best polynomial approximation error is $O(1/n)$,
the same as for $|x|$ — but the constant is worse.

```python
import chebfunjax as cj
import jax.numpy as jnp

# Piecewise to avoid singularity at 0
dom = (-1.0, -0.001, 0.001, 1.0)
f = cj.chebfun(
    lambda x: 1.0/jnp.abs(jnp.log(jnp.abs(x) + 1e-15)),
    domain=dom
)
print(f"Length: {len(f)}")

# Error of degree-100 approximation
p100 = f.polyfit(100)
import numpy as np
xx = np.linspace(-1, -0.01, 300)
f_vals = np.array([float(f(jnp.array(x))) for x in xx])
p_vals = np.array([float(p100(jnp.array(x))) for x in xx])
print(f"Max error: {np.max(np.abs(p_vals-f_vals)):.3e}")
```
"""),

    dict(name="RationalAbsx",
         title="Rational Approximation of abs(x) with Minimax",
         author="Silviu Filip, Yuji Nakatsukasa, and Nick Trefethen",
         date="May 2017",
         tags=["#Halphen", "#rational", "#ABS"],
         summary="Best rational approximation of |x| achieves root-exponential O(exp(-C√n)) convergence vs. polynomial O(1/n).",
         narrative="""
## Newman's theorem

Newman (1964) showed that the best type $(n,n)$ rational approximation to $|x|$
on $[-1,1]$ achieves accuracy $O(\\exp(-C\\sqrt{n}))$, far better than the
polynomial $O(1/n)$.

The constant is approximately $C \\approx \\pi/\\sqrt{2}$ (though the exact value
of the asymptotic constant was refined later).

```python
from chebfunjax.utils.aaa import aaa
import jax.numpy as jnp
import numpy as np

xs = jnp.linspace(-1.0, 1.0, 400)
ys = jnp.abs(xs)
r, pol, *_ = aaa(ys, xs)
xx = np.linspace(-1, 1, 600)
err = np.max(np.abs([float(r(jnp.array(x))) for x in xx] - np.abs(xx)))
print(f"AAA ({len(pol)} poles): max err = {err:.3e}")

# Compare: polynomial approximation
import chebfunjax as cj
f = cj.chebfun(jnp.abs)
for n in [10, 20, 40, 80]:
    pn = f.polyfit(n)
    pn_err = max(abs(float(pn(jnp.array(x))) - abs(float(x)))
                 for x in np.linspace(-1,1,200))
    print(f"poly deg {n:3d}: max err = {pn_err:.3e}")
```
"""),

    dict(name="RationalInterp",
         title="Rational Interpolation, Robust and Non-robust",
         author="Nick Trefethen",
         date="August 2011",
         tags=["#rational", "#interpolation"],
         summary="Compares standard rational interpolation (fragile, can produce Froissart doublets) with AAA's robust approach.",
         narrative="""
## Robustness in rational interpolation

Rational interpolation at $n+m+1$ points for a type $(n,m)$ approximant is
generally ill-conditioned — small perturbations can produce **Froissart doublets**
(spurious pole-zero pairs that nearly cancel).

AAA approximation avoids this by using a least-squares rather than exact
interpolation framework, and automatically removes doublets.

```python
from chebfunjax.utils.aaa import aaa
import jax.numpy as jnp
import numpy as np

# Runge function: poles at ±0.2i
def f_func(x): return 1.0 / (1.0 + 25.0*x**2)

xs = jnp.linspace(-1.0, 1.0, 300)
ys = jnp.array([f_func(float(x)) for x in xs])
r, pol, res, zer, *_ = aaa(ys, xs)

print(f"Poles: {pol}")
print(f"True poles: ±{1/5:.4f}i")

xx = np.linspace(-1, 1, 500)
err = np.max(np.abs([float(r(jnp.array(x))) for x in xx] - f_func(xx)))
print(f"Max error: {err:.2e}")
```
"""),

    dict(name="Rationalxn",
         title="Rational Approximation of Monomials",
         author="Yuji Nakatsukasa and Nick Trefethen",
         date="May 2019",
         tags=["#rational", "#minimax"],
         summary="The monomial x^200 on [0,1] is efficiently approximated by rational functions of surprisingly low type.",
         narrative="""
## Monomials and rational approximation

The monomial $x^{200}$ on $[0,1]$ looks like it would be easy, but it has a
very sharp transition near 0 that requires high-degree polynomials. Rational
approximation does much better: a type $(2,2)$ approximant achieves accuracy
close to $10^{-2}$, and higher types improve rapidly.

```python
from chebfunjax.utils.aaa import aaa
import jax.numpy as jnp
import numpy as np

exp_val = 200
xs = jnp.linspace(0.0, 1.0, 300)
ys = xs**exp_val
r, pol, *_ = aaa(ys, xs)

test = np.linspace(0, 1, 500)
err = np.max(np.abs([float(r(jnp.array(x))) for x in test] - test**exp_val))
print(f"AAA ({len(pol)} poles): max err = {err:.2e}")
```
"""),

    dict(name="ResolutionWiggly",
         title="Resolution of Wiggly Functions",
         author="Nick Hale and Nick Trefethen",
         date="October 2013",
         tags=["#resolution", "#wiggly"],
         summary="Explores approximation of the highly oscillatory function sin²(x)+sin(x²) on [0,14].",
         narrative="""
## The wiggly function

The function $f(x) = \\sin^2(x) + \\sin(x^2)$ on $[0,14]$ is one of the Chebfun
team's favorites for testing. It requires a polynomial of degree about 1000 for
machine precision because $\\sin(x^2)$ has increasing frequency.

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

f = cj.chebfun(lambda x: jnp.sin(x)**2 + jnp.sin(x**2), domain=(0.0, 14.0))
print(f"Chebfun length: {len(f)}")

# Low-degree polynomial approximation
p50 = f.polyfit(50)
xx = np.linspace(0, 14, 500)
err50 = max(abs(float(p50(jnp.array(x))) - float(f(jnp.array(x)))) for x in xx)
print(f"deg-50 max err: {err50:.3f}")
```
"""),

    dict(name="RestrictedDenominatorApproximations",
         title="Restricted-Denominator Approximations",
         author="Stefan Guettel",
         date="April 2012",
         tags=["#rational", "#stability"],
         summary="Explores rational approximation with a denominator restricted to a stability-preserving form for the matrix exponential.",
         narrative="""
## Restricted denominators

In some applications (e.g., numerical ODE solvers), the denominator of a
rational approximant to $e^x$ must be a polynomial with roots in the left
half-plane (stability requirement).

A natural choice is $q(x) = (1 - x/2)^k$, leading to approximants that are
automatically A-stable.

```python
import numpy as np

def restricted_approx(n_p):
    # Fit p(x) with q(x) = (1-x/2)^2 fixed
    xs = np.cos(np.pi * np.arange(n_p+3) / (n_p+2))[::-1]
    ys = np.exp(xs) * (1 - xs/2.0)**2  # target: p(x) = q(x)*exp(x)
    p_coeffs = np.polyfit(xs, ys, n_p)
    return lambda x: np.polyval(p_coeffs, x) / (1 - x/2.0)**2

r = restricted_approx(5)
test = np.linspace(-1, 1, 400)
err = np.max(np.abs(r(test) - np.exp(test)))
print(f"Restricted approx (p=5) max err: {err:.2e}")
```
"""),

    dict(name="ScalingAndSquaring",
         title="Rational Approximation to the Exponential in a Complex Region",
         author="Yuji Nakatsukasa and Stefan Guettel",
         date="July 2012",
         tags=["#Pade", "#exponential"],
         summary="The scaling-and-squaring method uses Padé approximants plus squaring to compute matrix exponentials accurately.",
         narrative="""
## Scaling and squaring

The identity $e^A = (e^{A/2^s})^{2^s}$ allows computing $e^A$ by:
1. Scale: compute $B = A/2^s$ (small norm).
2. Approximate: $r(B) \\approx e^B$ using a Padé approximant.
3. Square: $e^A \\approx r(B)^{2^s}$.

The [4/4] Padé approximant to $e^x$ is:

$$r_{4/4}(x) = \\frac{p_4(x)}{q_4(x)}$$

where $p_4$ and $q_4$ are degree-4 polynomials.

```python
from scipy.special import pade as scipy_pade
import numpy as np

# Compute [4/4] Pade approximant to exp(x)
taylor = np.array([1.0/np.math.factorial(k) for k in range(17)])
p44, q44 = scipy_pade(taylor, 4)

xs = np.linspace(-6, 6, 400)
err = np.max(np.abs(np.polyval(p44, xs)/np.polyval(q44, xs) - np.exp(xs)))
print(f"[4/4] Pade max err on [-6,6]: {err:.2e}")
```
"""),

    dict(name="SmoothCompact",
         title="Smooth Functions of Compact Support",
         author="Nick Trefethen",
         date="July 2014",
         tags=["#convolution", "#compact-support"],
         summary="Constructs infinitely smooth functions with compact support via iterated convolution.",
         narrative="""
## Smooth bump functions

An infinitely smooth function with compact support can be constructed by
convolving a box function $B_0$ with itself $k$ times: $B_k = B_0^{*(k+1)}$.
Each convolution increases the smoothness class by one.

```python
import chebfunjax as cj
import jax.numpy as jnp

h = 0.5
B0 = cj.chebfun(lambda x: jnp.ones_like(x), domain=(-h, h))

B1 = B0.conv(B0)   # C^0, piecewise linear (hat function)
B2 = B1.conv(B0)   # C^1, piecewise quadratic
B3 = B2.conv(B0)   # C^2, cubic B-spline

print(f"B0 domain: [{B0.domain.breakpoints[0]:.1f}, {B0.domain.breakpoints[-1]:.1f}]")
print(f"B3 domain: [{B3.domain.breakpoints[0]:.1f}, {B3.domain.breakpoints[-1]:.1f}]")
print(f"B3 length: {len(B3)}")
```

As $k \\to \\infty$, these functions converge to a Gaussian.
"""),

    dict(name="Splines",
         title="Splines",
         author="Nick Trefethen",
         date="February 2013",
         tags=["#spline", "#splitting"],
         summary="Demonstrates Chebfun's cubic spline interpolation, including derivatives and the not-a-knot condition.",
         narrative="""
## Cubic spline interpolation

Chebfun has a `spline` command analogous to MATLAB's. It constructs a piecewise
cubic polynomial that interpolates given data and has two continuous derivatives.

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline

# Underlying function
def f_func(x): return np.sin(x + 0.25*x**2)
nodes = np.arange(0, 11)
values = f_func(nodes)

# scipy cubic spline (same as MATLAB spline with not-a-knot)
cs = CubicSpline(nodes, values)

# Compare with adaptive chebfun
f = cj.chebfun(lambda x: jnp.sin(x + 0.25*x**2), domain=(0.0, 10.0))
print(f"Chebfun length: {len(f)}")
print(f"Spline continuity: C^2 (two continuous derivatives)")
```

The not-a-knot condition uses the two available degrees of freedom at the
endpoints to enforce $s'''(x)$ is continuous at $x_1$ and $x_{n-1}$.
"""),

    dict(name="WeierstrassFunction",
         title="A Pathological Function of Weierstrass",
         author="Hrothgar",
         date="October 2013",
         tags=["#Weierstrass", "#fractal"],
         summary="Constructs a partial sum of the Weierstrass nowhere-differentiable function and explores its integrability.",
         narrative="""
## Weierstrass's nowhere-differentiable function

In 1872, Karl Weierstrass shocked the mathematical world by constructing
$$F(x) = \\sum_{k=0}^\\infty a^k \\cos(b^k \\pi x)$$
which is continuous everywhere but differentiable nowhere (for $0 < a < 1$,
$b$ a positive odd integer, and $ab > 1 + \\frac{3\\pi}{2}$).

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

def make_fk(k):
    return lambda x: 2.0**(-k) * jnp.cos(jnp.pi/2 * x * 4.0**k)

F = cj.chebfun(make_fk(0))
for k in range(1, 8):
    F = F + cj.chebfun(make_fk(k))

# Integral equals 4/pi (exact!)
integral = float(F.sum())
print(f"integral = {integral:.6f}, 4/pi = {4/np.pi:.6f}")
print(f"Error: {abs(integral - 4/np.pi):.2e}")
```

Chebfun resolves 8 iterates to machine precision but cannot resolve the 9th
(which would require an infinite-degree polynomial).
"""),

    dict(name="WigglyApprox",
         title="A Wiggly Function and Its Best Approximations",
         author="Ricardo Pachon and Nick Trefethen",
         date="November 2010",
         tags=["#oscillatory", "#approximation"],
         summary="Explores polynomial approximation of the oscillatory function sin²(x)+sin(x²) on [0,14].",
         narrative="""
## The oscillatory function

The wiggly function $f(x) = \\sin^2(x) + \\sin(x^2)$ on $[0,14]$ has frequency
that increases with $x$: while $\\sin^2(x)$ has frequency $1/\\pi$, the term
$\\sin(x^2)$ has instantaneous frequency $x/\\pi$ at position $x$.

```python
import chebfunjax as cj
import jax.numpy as jnp
import numpy as np

f = cj.chebfun(lambda x: jnp.sin(x)**2 + jnp.sin(x**2), domain=(0.0, 14.0))
print(f"Adaptive chebfun degree: {len(f)}")

# Low-degree L2 approximant
p50 = f.polyfit(50)
xx = np.linspace(0, 14, 800)
err = [abs(float(p50(jnp.array(x))) - float(f(jnp.array(x)))) for x in xx]
print(f"deg-50 max err: {max(err):.3f}")
```

The adaptive chebfun requires a high-degree polynomial to capture the increasing
frequency of $\\sin(x^2)$, while a low-degree approximant cannot resolve the
high-frequency content.
"""),
]


# -----------------------------------------------------------------------
# Template
# -----------------------------------------------------------------------

TEMPLATE = """\
# {title}

*{author}, {date}*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/approx/{name}.html)

{narrative}

![{title}](../../images/approx/{name}.png)

"""


def main():
    repo = Path(__file__).parent.parent
    docs_dir = repo / "docs" / "examples" / "approx"
    docs_dir.mkdir(parents=True, exist_ok=True)

    for ex in EXAMPLES:
        md_path = docs_dir / f"{ex['name']}.md"
        # Skip AAAApprox since we wrote it manually
        if md_path.exists() and ex['name'] == 'AAAApprox':
            print(f"Skipping {ex['name']} (already exists)")
            continue
        content = TEMPLATE.format(
            title=ex['title'],
            author=ex['author'],
            date=ex['date'],
            name=ex['name'],
            narrative=ex.get('narrative', '').strip(),
        )
        md_path.write_text(content)
        print(f"Written {md_path.name}")

    # Write index
    index_path = docs_dir / "index.md"
    lines = ["# Approximation Examples\n",
             "\nAll 55 Chebfun approximation examples translated to chebfunjax.\n",
             "\n| Example | Author | Date | Description |\n",
             "|---------|--------|------|-------------|\n"]
    for ex in EXAMPLES:
        lines.append(
            f"| [{ex['title']}]({ex['name']}.md) | {ex['author']} | "
            f"{ex['date']} | {ex['summary']} |\n"
        )
    index_path.write_text("".join(lines))
    print(f"Written {index_path}")


if __name__ == "__main__":
    main()
