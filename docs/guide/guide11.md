# Chapter 11: Periodic Chebfuns

*Based on [Chebfun Guide Chapter 11](https://www.chebfun.org/docs/guide/guide11.html)*

## 11.1 Introduction

Chebfunjax supports trigonometric (Fourier) representations for smooth periodic functions via the `Trigtech` class (`chebfunjax.tech.trigtech`). While the standard `Chebtech2` uses Chebyshev polynomial interpolation at Gauss-Lobatto points, `Trigtech` uses trigonometric interpolation at equally spaced points, which is the natural representation for periodic functions.

The theoretical and practical advantages of trigonometric representations for periodic functions include:

- Trigonometric interpolants require only about 2 points per wavelength, compared to approximately $\pi$ points per wavelength for Chebyshev interpolants.
- Derivatives of trigonometric interpolants remain trigonometric interpolants of the same degree (no degree growth).
- The exponential convergence rate is sharper for analytic periodic functions.

## 11.2 Trigonometric Series and Interpolants

### Fourier Series

The classical Fourier series of a periodic function $u$ on $[-\pi, \pi]$ is

$$\mathcal{F}[u](t) = \sum_{k=-\infty}^{\infty} a_k\, e^{ikt},$$

with coefficients

$$a_k = \frac{1}{2\pi} \int_{-\pi}^{\pi} u(t)\, e^{-ikt}\, dt.$$

Equivalently, using real-valued sine and cosine functions:

$$\mathcal{F}[u](t) = \frac{a_0}{2} + \sum_{k=1}^{\infty} \bigl(a_k \cos(kt) + b_k \sin(kt)\bigr),$$

with

$$a_k = \frac{1}{\pi} \int_{-\pi}^{\pi} u(t)\cos(kt)\,dt, \qquad b_k = \frac{1}{\pi} \int_{-\pi}^{\pi} u(t)\sin(kt)\,dt.$$

### Convergence

The rate of convergence of the Fourier series depends on the smoothness of $u$:

- If $u$ is $(\ell - 1)$-times continuously differentiable with bounded variation at order $\ell$, then $|a_k| = O(|k|^{-\ell-1})$, giving $\|u - q_N\| = O(N^{-\ell})$ where $q_N$ is the degree-$N$ partial sum.
- If $u$ is $C^\infty$ periodic, convergence is faster than any inverse power of $N$.
- If $u$ is analytic in a strip around the real axis, convergence is exponential: $\|u - q_N\| = O(\rho^{-N})$ for some $\rho > 1$.

### Discrete Fourier Transform

The trigonometric interpolant through $N$ equally spaced points

$$t_j = -\pi + \frac{2\pi j}{N}, \quad j = 0, 1, \ldots, N-1$$

has coefficients computed by the discrete Fourier transform (DFT):

$$c_k = \frac{1}{N} \sum_{j=0}^{N-1} u(t_j)\, e^{-ikt_j}.$$

The DFT coefficients relate to the exact Fourier coefficients via the Poisson summation formula (aliasing):

$$c_k = a_k + \sum_{m=1}^{\infty} \bigl(a_{k+mN} + a_{k-mN}\bigr).$$

## 11.3 The `Trigtech` Class

The `Trigtech` class in chebfunjax stores a function as its values at equally spaced trigonometric points on $[-1, 1)$ and the corresponding Fourier coefficients. The internal convention maps any physical domain $[a, b]$ to the reference interval $[-1, 1]$, with the Fourier basis $\{e^{i\pi k x}\}_{k}$ on $[-1, 1]$.

### Creating a Trigtech

```python
from chebfunjax.tech.trigtech import Trigtech

import jax.numpy as jnp

# Create from a function
f = Trigtech.from_function(
    lambda t: jnp.tanh(3 * jnp.sin(jnp.pi * t)) - jnp.sin(jnp.pi * (t + 0.5)),
)
print(f"Number of points: {f.n}")
print(f"Number of Fourier modes: {f.n // 2}")
```

### Construction from Values and Coefficients

```python
# From equally spaced values
import jax.numpy as jnp
N = 64
t = jnp.linspace(-1, 1, N, endpoint=False)
vals = jnp.sin(jnp.pi * t)
f = Trigtech.from_values(vals)

# From Fourier coefficients
from chebfunjax.tech.trigtech import trig_vals2coeffs, trig_coeffs2vals

coeffs = trig_vals2coeffs(vals)
f2 = Trigtech.from_coeffs(coeffs)
```

## 11.4 Fourier Coefficient Transforms

Chebfunjax provides FFT-based functions for converting between values and Fourier coefficients:

### `trig_vals2coeffs`: Values to Coefficients

```python
from chebfunjax.tech.trigtech import trig_vals2coeffs

# Example: sin(pi*t) should have coefficients at k = +/-1 only
N = 5
t = jnp.linspace(-1, 1, N, endpoint=False)
vals = jnp.sin(jnp.pi * t)
coeffs = trig_vals2coeffs(vals)
print("Fourier coefficients:", coeffs)
```

The coefficients are stored in ascending wavenumber order: for odd $N = 2M+1$, the array contains $c_{-M}, c_{-M+1}, \ldots, c_0, \ldots, c_M$.

### `trig_coeffs2vals`: Coefficients to Values

```python
from chebfunjax.tech.trigtech import trig_coeffs2vals

vals_reconstructed = trig_coeffs2vals(coeffs)
print("Max error:", float(jnp.max(jnp.abs(vals_reconstructed - vals))))
```

## 11.5 Basic Operations on Trigtech

The `Trigtech` class supports many of the same operations as `Chebtech2`:

### Evaluation

```python
# Evaluate at specific points
x = jnp.array([0.0, 0.25, 0.5])
print(f(x))
```

### Differentiation

Differentiation of trigonometric interpolants is exact -- the derivative of a trigonometric polynomial is another trigonometric polynomial of the same degree:

```python
# d/dx sin(pi*x) = pi*cos(pi*x)
f = Trigtech.from_function(lambda t: jnp.sin(jnp.pi * t))
df = f.diff()

# Check: df(0) should be pi
print(f"df(0) = {float(df(jnp.float64(0.0))):.15f}")
print(f"pi    = {float(jnp.pi):.15f}")
```

This exactness is a key advantage over Chebyshev representations for periodic functions, where repeated differentiation gradually degrades accuracy near the endpoints.

### Integration

```python
# Integral of sin(pi*x) over [-1, 1] should be 0
integral = float(f.sum())
print(f"integral = {integral:.2e}")
```

## 11.6 Convergence: Trigfuns vs. Chebfuns

For smooth periodic functions, trigonometric interpolants converge faster than Chebyshev interpolants. The sampling efficiency ratio is approximately $\pi/2 \approx 1.57$: a Chebyshev interpolant needs about $\pi/2$ times as many points as a trigonometric interpolant for the same accuracy.

### Example Comparison

```python
import chebfunjax as cj

ff = lambda t: jnp.cos(11 * jnp.sin(3 * (t - 1.0 / jnp.pi)))

# Chebyshev representation
f_cheb = cj.chebfun(ff, domain=(-float(jnp.pi), float(jnp.pi)))

# Trigonometric representation
f_trig = Trigtech.from_function(
    lambda t: ff(jnp.pi * t),  # map [-1,1] to [-pi, pi]
)

print(f"Chebyshev length: {len(f_cheb)}")
print(f"Trigtech length:  {f_trig.n}")
print(f"Ratio: {len(f_cheb) / f_trig.n:.2f}")
```

### Derivative Accuracy

The advantage of trigonometric representations is even more pronounced for derivatives. Chebyshev differentiation matrices have condition numbers that grow as $O(N^2)$, while trigonometric differentiation is exact.

```python
# Third derivative of cos(10*sin(t))
f_trig = Trigtech.from_function(
    lambda t: jnp.cos(10 * jnp.sin(jnp.pi * t)),
)
df3 = f_trig.diff(3)
# At t = 1 (corresponding to pi), the function is periodic,
# so the third derivative should match the analytic value
print(f"d^3f/dx^3 at boundary: {float(df3(jnp.float64(1.0))):.2e}")
```

## 11.7 Coefficient Analysis with `plotcoeffs`

The decay rate of Fourier coefficients reveals the smoothness class of the function. Chebfunjax's `plotcoeffs` function can visualize this:

### Entire Functions (Super-Exponential Decay)

For functions like $e^{\sin t}$ that are entire (analytic everywhere in the complex plane), the Fourier coefficients decay faster than any exponential:

```python
# Coefficients of exp(sin(pi*t))
f = Trigtech.from_function(lambda t: jnp.exp(jnp.sin(jnp.pi * t)))
print(f"Number of modes: {f.n}")
# Use cj.plotcoeffs to visualize the coefficient decay
```

### Analytic Functions (Geometric Decay)

For functions analytic in a strip of width $d$ around the real axis, coefficients decay as $|c_k| \sim e^{-d|k|}$ (geometric/exponential rate):

```python
# 1/(2 - cos(pi*t)) is analytic with poles off the real axis
f = Trigtech.from_function(lambda t: 1.0 / (2.0 - jnp.cos(jnp.pi * t)))
print(f"Number of modes: {f.n}")
```

### Functions with Finite Smoothness (Algebraic Decay)

For $C^\ell$ periodic functions that are not $C^{\ell+1}$, the coefficients decay as $|c_k| = O(|k|^{-\ell-1})$:

```python
# |sin(pi*t)|^5 has 5 continuous derivatives but not 6
f = Trigtech.from_function(lambda t: jnp.abs(jnp.sin(jnp.pi * t))**5)
print(f"Number of modes: {f.n}")
# Coefficients decay as |k|^{-6}
```

## 11.8 Non-Periodic Functions: A Warning

Attempting to represent a non-periodic function with a trigonometric interpolant is a common mistake. The Gibbs phenomenon causes oscillations near the boundaries, and convergence is at best first-order.

```python
# BAD: t^2 is not periodic on [-1, 1]
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    f = Trigtech.from_function(lambda t: t**2)
    print(f"Length: {f.n}")
    # May produce a warning about non-convergence
```

If you see unexpectedly large lengths or convergence warnings, check whether your function is truly periodic on the domain.

## 11.9 Domain Transplantation

Trigtech works internally on the reference interval $[-1, 1]$. When your function is naturally defined on $[-\pi, \pi]$ or $[0, 2\pi]$, you need to account for the affine mapping between the physical domain and $[-1, 1]$.

For a function $u(t)$ on $[a, b]$, the Trigtech represents

$$u\!\left(\frac{b-a}{2}\,s + \frac{a+b}{2}\right), \quad s \in [-1, 1],$$

with Fourier basis $\{e^{i\pi k s}\}$. In terms of the physical variable $t$, this corresponds to the basis $\{e^{i\alpha k t}\}$ with $\alpha = 2\pi/(b-a)$.

### Example

```python
# sin(t) on [-pi, pi]: natural period 2*pi matches the domain length
import math
half_period = math.pi

# Map to [-1, 1]: sin(pi*s)
f = Trigtech.from_function(lambda s: jnp.sin(jnp.pi * s))
coeffs = f.coeffs
print("Fourier coefficients of sin on [-pi, pi]:")
print(coeffs)
# Should show coefficients at k = +/-1 only
```

## 11.10 Periodic ODE Solutions

Periodic boundary conditions for ODEs can be specified in chebfunjax's Chebop framework. This is useful for problems where the solution is expected to be periodic, such as Floquet problems or equations with periodic coefficients.

```python
from chebfunjax.operators.chebop import Chebop
import jax.numpy as jnp

# u'' + u' + 600*(1 + sin(x))*u = 1 on [-pi, pi]
# with periodic BCs
N = Chebop(
    lambda x, u: u.diff(2) + u.diff() + 600 * (1 + cj.sin(x)) * u,
    domain=(-float(jnp.pi), float(jnp.pi)),
)
# Note: periodic BCs are not yet directly supported via a 'periodic' flag
# in chebfunjax. Instead, manually impose u(-pi) = u(pi) and u'(-pi) = u'(pi).
```

## 11.11 Truncated Fourier Series and the Gibbs Phenomenon

Computing Fourier coefficients of a non-smooth function and forming a truncated Fourier series displays the Gibbs phenomenon -- oscillations near discontinuities that do not diminish in amplitude (though they narrow in width) as the number of terms increases.

```python
from chebfunjax.tech.trigtech import trig_vals2coeffs, trig_coeffs2vals

# Square wave approximation
# The Fourier series of sign(sin(t)) = (4/pi) * sum_{k odd} sin(kt)/k
N = 201
t = jnp.linspace(-1, 1, N, endpoint=False)
vals = jnp.sign(jnp.sin(jnp.pi * t))
coeffs = trig_vals2coeffs(vals)

# Truncate to degree 15 (31 terms)
degree = 15
M = N // 2
trunc_coeffs = jnp.zeros(2 * degree + 1, dtype=jnp.complex128)
for k in range(-degree, degree + 1):
    trunc_coeffs = trunc_coeffs.at[k + degree].set(coeffs[k + M])

# The truncated series is the best degree-15 trigonometric approximation
# in the L^2 sense
vals_trunc = trig_coeffs2vals(trunc_coeffs)
```

The $L^2$-optimal truncated Fourier series minimizes the integrated squared error, but does not minimize the pointwise (maximum) error. Near discontinuities, the overshoot converges to approximately 8.95% of the jump (the Gibbs constant).

## 11.12 Circular Convolution

For periodic functions with period $T$ on $[t_0, t_0 + T]$, the circular convolution is

$$(f * g)(t) = \int_{t_0}^{t_0 + T} g(s)\,f(t - s)\,ds.$$

In the Fourier domain, convolution becomes pointwise multiplication of coefficients (up to a scaling factor). This makes trigonometric representations natural for filtering and smoothing operations.

## 11.13 References

- A. P. Austin, P. Kravanja, and L. N. Trefethen, "Numerical algorithms based on analytic function values at roots of unity," *SIAM J. Numer. Anal.* 52 (2014), 1795-1821.

- C. Canuto, M. Y. Hussaini, A. Quarteroni, and T. A. Zang, *Spectral Methods*, 2 vols., Springer, 2006-2007.

- P. Henrici, *Applied and Computational Complex Analysis*, vol. 3, Wiley, 1986.

- L. N. Trefethen, *Spectral Methods in MATLAB*, SIAM, 2000.

- L. N. Trefethen and J. A. C. Weideman, "The exponentially convergent trapezoidal rule," *SIAM Review* 56 (2014), 385-458.

- G. B. Wright, M. Javed, H. Montanelli, and L. N. Trefethen, "Extension of Chebfun to periodic functions," *SIAM J. Sci. Comp.* 37 (2015), C554-C573.

- A. Zygmund, *Trigonometric Series*, Cambridge University Press, 1959.
