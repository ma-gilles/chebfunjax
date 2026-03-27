# Chapter 5: Complex Chebfuns

*Based on [Chebfun Guide Chapter 5](https://www.chebfun.org/docs/guide/guide05.html) by Lloyd N. Trefethen*

## 5.1 Complex Functions of a Real Variable

A "complex chebfun" is a chebfun whose values are complex numbers -- that is,
a complex-valued function of a real variable.  Such functions arise naturally
when parametrizing curves in the complex plane.

In chebfunjax, a complex function $z(t) = x(t) + i\,y(t)$ is typically
represented as a pair of real chebfuns -- one for the real part and one for the
imaginary part.  This is because JAX's core dtype for spectral methods is
`float64`, and the Chebyshev machinery in chebfunjax is built around real
coefficients.

### Parametric curves

The standard approach is to construct the real and imaginary parts separately:

```python
import jax.numpy as jnp
import chebfunjax as cj

# The unit circle: z(t) = exp(it) = cos(t) + i*sin(t),  t in [0, 2*pi]
t = cj.chebfun(lambda t: t, domain=[0, 2 * jnp.pi])
x_part = cj.cos(t)   # real part
y_part = cj.sin(t)   # imaginary part
```

To plot such a curve, evaluate both parts and use matplotlib:

```python
import matplotlib.pyplot as plt

s = jnp.linspace(0, 2 * jnp.pi, 200)
plt.plot(x_part(s), y_part(s))
plt.axis('equal')
plt.title('Unit circle')
plt.show()
```

### Semicircle example

Points on the upper semicircle from $-1$ to $1$:

```python
# Upper semicircle: z(s) = exp(i*s),  s in [0, pi]
s = cj.chebfun(lambda s: s, domain=[0, jnp.pi])
x_semi = cj.cos(s)
y_semi = cj.sin(s)

# Length of curve: integral of |z'(s)| ds
# z'(s) = -sin(s) + i*cos(s), so |z'(s)| = 1
# Arc length = pi
dxds = x_semi.diff()
dyds = y_semi.diff()
speed = (dxds**2 + dyds**2).sqrt()
arc_length = float(speed.sum())
print(f"Arc length of semicircle: {arc_length:.15f}")
print(f"Expected (pi):            {float(jnp.pi):.15f}")
```

### Spirals and other curves

More elaborate curves can be built by combining trig and polynomial chebfuns:

```python
# A spiral: z(t) = t * exp(i*t),  t in [0, 4*pi]
t = cj.chebfun(lambda t: t, domain=[0, 4 * jnp.pi])
x_spiral = t * cj.cos(t)
y_spiral = t * cj.sin(t)

# Evaluate and plot
tt = jnp.linspace(0, 4 * jnp.pi, 500)
plt.plot(x_spiral(tt), y_spiral(tt))
plt.axis('equal')
plt.title('Archimedean spiral')
plt.show()
```

### Arc length of curves

The arc length of a parametric curve $(x(t), y(t))$ for $t \in [a, b]$ is

$$L = \int_a^b \sqrt{x'(t)^2 + y'(t)^2}\,dt.$$

In chebfunjax:

```python
# Arc length of one period of a sine wave: y = sin(x), x in [0, 2*pi]
# Parametrize as x(t) = t, y(t) = sin(t)
t = cj.chebfun(lambda t: t, domain=[0, 2 * jnp.pi])
x_curve = t
y_curve = cj.sin(t)

dx = x_curve.diff()   # = 1
dy = y_curve.diff()   # = cos(t)
speed = (dx**2 + dy**2).sqrt()   # sqrt(1 + cos^2(t))
L = float(speed.sum())
print(f"Arc length of sin(x) over [0, 2*pi]: {L:.10f}")
# This is a complete elliptic integral -- approximately 7.6404...
```

## 5.2 Analytic Functions and Conformal Maps

An analytic function $w = f(z)$ maps curves and regions in the $z$-plane to
curves and regions in the $w$-plane.  Away from critical points (where
$f'(z) = 0$), the mapping is *conformal* -- it preserves angles.

### Visualizing conformal maps

A standard technique is to map a grid of lines and see how they are
transformed:

```python
import jax.numpy as jnp
import chebfunjax as cj
import matplotlib.pyplot as plt

# Grid lines in the z-plane: horizontal and vertical
t = cj.chebfun(lambda t: t, domain=[-1, 1])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Map z -> z^2
# Horizontal lines: z = t + i*c  for various c
for c in jnp.linspace(-1, 1, 11):
    x_in = t
    y_in = cj.chebfun(float(c))

    # w = z^2 = (x + iy)^2 = x^2 - y^2 + 2ixy
    u = x_in**2 - float(c)**2
    v = 2 * float(c) * x_in

    tt = jnp.linspace(-1, 1, 200)
    axes[0].plot(x_in(tt), jnp.full_like(tt, c), 'b-', alpha=0.5)
    axes[1].plot(u(tt), v(tt), 'b-', alpha=0.5)

# Vertical lines: z = c + i*t  for various c
for c in jnp.linspace(-1, 1, 11):
    y_in = t

    u = float(c)**2 - t**2
    v = 2 * float(c) * t

    tt = jnp.linspace(-1, 1, 200)
    axes[0].plot(jnp.full_like(tt, c), y_in(tt), 'r-', alpha=0.5)
    axes[1].plot(u(tt), v(tt), 'r-', alpha=0.5)

axes[0].set_title('z-plane')
axes[0].set_aspect('equal')
axes[1].set_title('w = z^2')
axes[1].set_aspect('equal')
plt.tight_layout()
plt.show()
```

### Mobius transformations

A Mobius transformation $w = (az + b)/(cz + d)$ maps circles and lines to
circles and lines.  These are fundamental in complex analysis:

```python
# Mobius transformation: w = (z - 1) / (z + 1)
# Maps the right half-plane to the unit disk
t = cj.chebfun(lambda t: t, domain=[0, 2 * jnp.pi])
# A circle of radius r centered at c
r, c = 2.0, 3.0
x_circle = c + r * cj.cos(t)
y_circle = r * cj.sin(t)

# Apply Mobius: w = (z-1)/(z+1) where z = x + iy
# Real and imaginary parts of w:
# w = ((x-1) + iy) / ((x+1) + iy)
# Multiply by conjugate of denominator:
denom = (x_circle + 1)**2 + y_circle**2
u = ((x_circle - 1) * (x_circle + 1) + y_circle**2) / denom
v = (y_circle * (x_circle + 1) - (x_circle - 1) * y_circle) / denom
# Simplify: u = (x^2 + y^2 - 1) / denom, v = 2y / denom

tt = jnp.linspace(0, 2 * jnp.pi, 300)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x_circle(tt), y_circle(tt))
plt.title('z-plane')
plt.axis('equal')
plt.subplot(1, 2, 2)
plt.plot(u(tt), v(tt))
plt.title('w = (z-1)/(z+1)')
plt.axis('equal')
plt.tight_layout()
plt.show()
```

## 5.3 Contour Integrals

Contour integrals in complex analysis take the form

$$\oint_\gamma f(z)\,dz = \int_a^b f(z(t))\,z'(t)\,dt,$$

where $z(t)$ parametrizes the contour $\gamma$.  With chebfunjax, we split
this into real and imaginary parts.

### Cauchy's theorem

If $f(z)$ is analytic inside and on a closed contour $\gamma$, then
$\oint_\gamma f(z)\,dz = 0$.

```python
import jax.numpy as jnp
import chebfunjax as cj

# Integrate z^2 around the unit circle
# z(t) = cos(t) + i*sin(t),  z'(t) = -sin(t) + i*cos(t)
t = cj.chebfun(lambda t: t, domain=[0, 2 * jnp.pi])
x = cj.cos(t)
y = cj.sin(t)
dx = -cj.sin(t)
dy = cj.cos(t)

# f(z) = z^2 = (x + iy)^2 = (x^2 - y^2) + 2ixy
f_re = x**2 - y**2
f_im = 2 * x * y

# f(z) * z'(t) = (f_re + i*f_im) * (dx + i*dy)
# Real part: f_re*dx - f_im*dy
# Imag part: f_re*dy + f_im*dx
integrand_re = f_re * dx - f_im * dy
integrand_im = f_re * dy + f_im * dx

I_re = float(integrand_re.sum())
I_im = float(integrand_im.sum())
print(f"Integral of z^2 around unit circle: {I_re:.2e} + {I_im:.2e}i")
# Should be 0 + 0i (Cauchy's theorem)
```

### The residue theorem

For a function with a pole inside the contour, the integral equals $2\pi i$
times the residue.  For example, $f(z) = 1/z$ has a simple pole at $z = 0$
with residue 1:

```python
# Integrate 1/z around the unit circle
# 1/z = 1/(x + iy) = (x - iy)/(x^2 + y^2)
r_sq = x**2 + y**2
f_re = x / r_sq
f_im = -y / r_sq

integrand_re = f_re * dx - f_im * dy
integrand_im = f_re * dy + f_im * dx

I_re = float(integrand_re.sum())
I_im = float(integrand_im.sum())
print(f"Integral of 1/z around unit circle: {I_re:.6f} + {I_im:.6f}i")
print(f"Expected: 0 + {2 * float(jnp.pi):.6f}i")
# Should be 2*pi*i (residue = 1)
```

### Higher-order poles and the Cauchy integral formula

The Cauchy integral formula states that for $f$ analytic inside $\gamma$ and
$a$ inside $\gamma$:

$$f^{(n)}(a) = \frac{n!}{2\pi i} \oint_\gamma \frac{f(z)}{(z-a)^{n+1}}\,dz.$$

For example, to compute $e^0 = 1$ via the Cauchy integral formula with
$f(z) = e^z$, $a = 0$, $n = 0$:

```python
# f(z) = exp(z) / z  around the unit circle
# exp(z) = exp(x) * (cos(y) + i*sin(y))
exp_re = cj.exp(x) * cj.cos(y)
exp_im = cj.exp(x) * cj.sin(y)

# exp(z)/z = exp(z) * conj(z) / |z|^2
g_re = (exp_re * x + exp_im * y) / r_sq
g_im = (exp_im * x - exp_re * y) / r_sq

integrand_re = g_re * dx - g_im * dy
integrand_im = g_re * dy + g_im * dx

I_re = float(integrand_re.sum())
I_im = float(integrand_im.sum())
# Result should be 2*pi*i * exp(0) = 2*pi*i
print(f"Integral of exp(z)/z: {I_re:.6f} + {I_im:.6f}i")
print(f"Expected:             0.000000 + {2*float(jnp.pi):.6f}i")
# Dividing by 2*pi*i gives exp(0) = 1
result = I_im / (2 * float(jnp.pi))
print(f"exp(0) via Cauchy formula: {result:.15f}")
```

## 5.4 Winding Numbers and the Argument Principle

The *winding number* of a closed curve $\gamma$ around a point $a$ is

$$n(\gamma, a) = \frac{1}{2\pi i} \oint_\gamma \frac{dz}{z - a}.$$

The *argument principle* states that for a meromorphic function $f$ inside
$\gamma$:

$$\frac{1}{2\pi i} \oint_\gamma \frac{f'(z)}{f(z)}\,dz = N - P,$$

where $N$ is the number of zeros and $P$ the number of poles (counted with
multiplicity).

```python
import jax.numpy as jnp
import chebfunjax as cj

# Count zeros of sin(z)^3 + cos(z)^3 in the disk |z| < 2
# This function has no poles, so N - P = N

t = cj.chebfun(lambda t: t, domain=[0, 2 * jnp.pi])
R = 2.0
x = R * cj.cos(t)
y = R * cj.sin(t)
dx = -R * cj.sin(t)
dy = R * cj.cos(t)

# f(z) = sin(z)^3 + cos(z)^3
# sin(z) = sin(x)*cosh(y) + i*cos(x)*sinh(y)
# cos(z) = cos(x)*cosh(y) - i*sin(x)*sinh(y)
sin_re = cj.sin(x) * cj.cosh(y)
sin_im = cj.cos(x) * cj.sinh(y)
cos_re = cj.cos(x) * cj.cosh(y)
cos_im = -cj.sin(x) * cj.sinh(y)

# For the argument principle, we need f'/f integrated around the contour
# This is equivalent to computing the winding number of the image curve f(gamma)
# around the origin: N = (1/2*pi) * change in argument of f along gamma

# We can compute this numerically by evaluating f on a fine grid
tt = jnp.linspace(0, 2 * jnp.pi, 1000, endpoint=False)
x_vals = R * jnp.cos(tt)
y_vals = R * jnp.sin(tt)
sin_z_re = jnp.sin(x_vals) * jnp.cosh(y_vals)
sin_z_im = jnp.cos(x_vals) * jnp.sinh(y_vals)
cos_z_re = jnp.cos(x_vals) * jnp.cosh(y_vals)
cos_z_im = -jnp.sin(x_vals) * jnp.sinh(y_vals)

# f = sin^3 + cos^3 (complex cube)
# sin^3: use binomial expansion or repeated multiplication
# For simplicity, compute numerically:
z_vals = x_vals + 1j * y_vals
f_vals = jnp.sin(z_vals)**3 + jnp.cos(z_vals)**3

# Winding number = total change in argument / (2*pi)
angles = jnp.angle(f_vals)
dangle = jnp.diff(angles)
# Unwrap jumps
dangle = jnp.where(dangle > jnp.pi, dangle - 2*jnp.pi, dangle)
dangle = jnp.where(dangle < -jnp.pi, dangle + 2*jnp.pi, dangle)
N = jnp.sum(dangle) / (2 * jnp.pi)
print(f"Number of zeros of sin(z)^3 + cos(z)^3 in |z| < 2: {float(N):.1f}")
# Should be 3
```

## 5.5 Contour Integrals for Computing Special Quantities

### Bernoulli numbers via contour integration

The Bernoulli numbers can be computed via

$$B_n = \frac{n!}{2\pi i} \oint \frac{z}{e^z - 1} \cdot \frac{dz}{z^{n+1}},$$

where the contour encloses the origin but no other singularities of
$z/(e^z - 1)$ (which has poles at $z = 2\pi i k$ for nonzero integers $k$).

```python
import jax.numpy as jnp
import chebfunjax as cj

def bernoulli_via_contour(n, R=1.0):
    """Compute B_n via contour integration on a circle of radius R."""
    t = cj.chebfun(lambda t: t, domain=[0, 2 * jnp.pi])
    x = R * cj.cos(t)
    y = R * cj.sin(t)
    dx = -R * cj.sin(t)
    dy = R * cj.cos(t)

    # f(z) = z / (exp(z) - 1) / z^{n+1} = 1 / ((exp(z) - 1) * z^n)
    # But we need to be careful with complex arithmetic
    # Work with fine evaluation grid instead
    tt = jnp.linspace(0, 2 * jnp.pi, 2000, endpoint=False)
    z = R * jnp.exp(1j * tt)
    dz = 1j * z  # dz/dt = i*z for z = R*exp(it)

    integrand = z / (jnp.exp(z) - 1) / z**(n + 1) * dz
    dt = 2 * jnp.pi / 2000
    I = jnp.sum(integrand) * dt

    # B_n = n! / (2*pi*i) * I
    from scipy.special import factorial
    Bn = factorial(n, exact=True) / (2 * jnp.pi * 1j) * I
    return float(jnp.real(Bn))

# Compute B_10 = 5/66
B10 = bernoulli_via_contour(10, R=1.0)
print(f"B_10 via contour integration: {B10:.15f}")
print(f"B_10 exact (5/66):            {5/66:.15f}")
```

## 5.6 Parametric Curves and Their Properties

Chebfunjax provides natural tools for analyzing parametric curves through
chebfun calculus.

### Curvature

The curvature of a plane curve $(x(t), y(t))$ is

$$\kappa = \frac{|x'y'' - y'x''|}{(x'^2 + y'^2)^{3/2}}.$$

```python
import jax.numpy as jnp
import chebfunjax as cj

# Curvature of an ellipse: x = 2*cos(t), y = sin(t)
t = cj.chebfun(lambda t: t, domain=[0, 2 * jnp.pi])
x = 2 * cj.cos(t)
y = cj.sin(t)

xp = x.diff()
yp = y.diff()
xpp = x.diff(2)
ypp = y.diff(2)

kappa = (xp * ypp - yp * xpp).abs() / (xp**2 + yp**2).sqrt()**3

# Maximum and minimum curvature
k_min_loc, k_min_val = kappa.min()
k_max_loc, k_max_val = kappa.max()
print(f"Min curvature: {k_min_val:.6f} at t = {k_min_loc:.6f}")
print(f"Max curvature: {k_max_val:.6f} at t = {k_max_loc:.6f}")
# For ellipse with semi-axes a=2, b=1:
# kappa_min = b/a^2 = 1/4 = 0.25 (at ends of major axis)
# kappa_max = a/b^2 = 2 (at ends of minor axis)
```

### Enclosed area

The area enclosed by a closed curve $(x(t), y(t))$ is given by the shoelace
formula:

$$A = \frac{1}{2} \left|\oint (x\,dy - y\,dx)\right| = \frac{1}{2}\left|\int_a^b (x y' - y x')\,dt\right|.$$

```python
# Area of the ellipse x = 2*cos(t), y = sin(t)
area_integrand = x * yp - y * xp
area = float(area_integrand.sum()) / 2
print(f"Area of ellipse (a=2, b=1): {abs(area):.15f}")
print(f"Expected (pi*a*b = 2*pi):   {2*float(jnp.pi):.15f}")
```

## 5.7 Complex Arithmetic with Chebfun Pairs

Since chebfunjax stores real and imaginary parts as separate chebfuns, complex
arithmetic must be done manually.  Here is a helper pattern:

```python
import jax.numpy as jnp
import chebfunjax as cj

def complex_mul(a_re, a_im, b_re, b_im):
    """Multiply two complex chebfuns: (a_re + i*a_im) * (b_re + i*b_im)."""
    re = a_re * b_re - a_im * b_im
    im = a_re * b_im + a_im * b_re
    return re, im

def complex_div(a_re, a_im, b_re, b_im):
    """Divide two complex chebfuns: (a_re + i*a_im) / (b_re + i*b_im)."""
    denom = b_re**2 + b_im**2
    re = (a_re * b_re + a_im * b_im) / denom
    im = (a_im * b_re - a_re * b_im) / denom
    return re, im

def complex_abs(a_re, a_im):
    """Modulus |a_re + i*a_im|."""
    return (a_re**2 + a_im**2).sqrt()

# Example: compute |exp(z)| on the unit circle
t = cj.chebfun(lambda t: t, domain=[0, 2 * jnp.pi])
z_re = cj.cos(t)
z_im = cj.sin(t)

# exp(z) = exp(x)*(cos(y) + i*sin(y))
exp_re = cj.exp(z_re) * cj.cos(z_im)
exp_im = cj.exp(z_re) * cj.sin(z_im)

modulus = complex_abs(exp_re, exp_im)
# |exp(z)| = exp(Re(z)) = exp(cos(t))
expected = cj.exp(cj.cos(t))

# Check agreement
tt = jnp.linspace(0, 2 * jnp.pi, 100)
err = float(jnp.max(jnp.abs(modulus(tt) - expected(tt))))
print(f"Max error in |exp(z)| on unit circle: {err:.2e}")
```

## 5.8 The Phase Portrait

For visualizing complex functions, *phase portraits* color each point in the
domain according to the phase (argument) of $f(z)$.  Chebfunjax provides:

```python
import chebfunjax as cj

# Phase plot of a complex function (uses the phaseplot utility)
# cj.phaseplot(f, domain=[-2, 2, -2, 2])
```

The `phaseplot` function evaluates a complex function on a grid and colors
each point using a standard HSV color wheel based on $\arg f(z)$.

## 5.9 Summary

| Task | Approach in chebfunjax |
|---|---|
| Complex curve $z(t)$ | Two real chebfuns: `x = cj.cos(t)`, `y = cj.sin(t)` |
| Arc length | `speed = (dx**2 + dy**2).sqrt(); L = speed.sum()` |
| Contour integral | Split into real/imag parts, integrate each |
| Complex multiplication | `(a*c - b*d, a*d + b*c)` for `(a+ib)(c+id)` |
| Winding number | Numerically track argument change |
| Curvature | $(x'y'' - y'x'')/(x'^2 + y'^2)^{3/2}$ |
| Enclosed area | $(1/2)\int (x y' - y x')\,dt$ |

### Key differences from MATLAB Chebfun

In MATLAB Chebfun, a single chebfun object can hold complex values natively
(the coefficients are complex).  In chebfunjax, the standard storage is
`float64`, so complex functions are represented as pairs of real chebfuns.
This is a deliberate design choice that keeps the core Chebyshev machinery
simple and GPU-friendly, at the cost of requiring manual real/imaginary
bookkeeping for complex-valued problems.

For contour integrals in particular, it is often simpler to evaluate on a fine
grid using JAX's native complex arithmetic (via `jnp.complex128`) and perform
the integration numerically, rather than building separate chebfuns for every
real and imaginary part.

## 5.10 References

- L. N. Trefethen, "Numerical computation of the Schwarz-Christoffel
  transformation," *SIAM J. Sci. Stat. Comp.* 1 (1980), 82-102.
- J. A. C. Weideman, "Computing the Hilbert transform on the real line,"
  *Math. Comp.* 64 (1995), 745-762.
- F. Bornemann, "Accuracy and stability of computing high-order derivatives
  of analytic functions by Cauchy integrals," *Found. Comp. Math.* 11 (2011),
  1-63.
- E. Wegert, *Visual Complex Functions: An Introduction with Phase Portraits*,
  Birkhauser, 2012.
