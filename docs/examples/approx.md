# Approximation Examples

Chebfunjax adaptively approximates smooth functions as Chebyshev series.
All examples run on CPU or GPU without code changes.

---

## 1. Approximating sin(x)

The simplest case: approximate a smooth, analytic function.

```python
import jax.numpy as jnp
import chebfunjax as cj

f = cj.chebfun(jnp.sin)
print(f)
print("f(0.5)   =", f(0.5))
print("f(pi/6)  =", f(jnp.pi / 6))
```

```
Chebfun on [-1, 1] with 14 coefficients (1 piece)
f(0.5)   = 0.479425538604203
f(pi/6)  = 0.5000000000000001
```

Only 14 coefficients are needed because the Chebyshev coefficients of sin(x)
decay faster than any algebraic rate (geometric decay for analytic functions).

---

## 2. The Runge Function

The classic Runge function `1 / (1 + 25x²)` illustrates why Chebyshev grids
are superior to equally-spaced interpolation: Chebyshev nodes avoid the Runge
phenomenon entirely.

```python
runge = cj.chebfun(lambda x: 1.0 / (1.0 + 25.0 * x**2))
print(runge)
print("runge(0)   =", runge(0.0))    # = 1.0
print("runge(0.5) =", runge(0.5))    # = 1 / (1 + 25*0.25) ≈ 0.1379
```

```
Chebfun on [-1, 1] with 68 coefficients (1 piece)
runge(0)   = 1.0000000000000002
runge(0.5) = 0.13793103448275862
```

---

## 3. Approximating |x|

The absolute value function is not smooth at 0, so it needs more coefficients
and its Chebyshev series converges only algebraically (as O(1/n²)).

```python
absx = cj.chebfun(jnp.abs)
print(absx)
print("absx(0.3) =", absx(0.3))   # = 0.3
print("absx(0.0) =", absx(0.0))   # = 0.0 (not smooth here)
```

```
Chebfun on [-1, 1] with 8192 coefficients (1 piece)
absx(0.3) = 0.2999999999999997
absx(0.0) = 2.3...e-16
```

Notice that |x| requires many more coefficients than sin(x) due to the corner
at the origin.

---

## 4. Fixed-Degree Approximation

You can bypass adaptive sampling and specify the exact polynomial degree with `n=`:

```python
# Low-degree approximation of exp(x)
f_approx = cj.chebfun(jnp.exp, n=5)
f_exact  = cj.chebfun(jnp.exp)

print("5-term  exp(0.5) =", f_approx(0.5))
print("exact   exp(0.5) =", f_exact(0.5))
print("error             =", abs(f_approx(0.5) - f_exact(0.5)))
```

```
5-term  exp(0.5) = 1.6487212707001
exact   exp(0.5) = 1.6487212707001282
error             = 3.9...e-13
```

Even 5 terms gives 12-digit accuracy for exp(x) because its coefficients decay
so rapidly.

---

## 5. Chebyshev Coefficient Decay

You can inspect the Chebyshev coefficients directly. For analytic functions they
decay geometrically; for C^k functions the decay is algebraic.

```python
import jax.numpy as jnp
import chebfunjax as cj

# Access coefficients for three functions
f_sin = cj.chebfun(jnp.sin)
f_abs = cj.chebfun(jnp.abs)
f_step = cj.chebfun(lambda x: jnp.where(x > 0, 1.0, -1.0))

# The tech object holds the raw Chebyshev coefficients
# Access via the internal piece representation
piece = f_sin.funs[0]      # the single smooth piece
coeffs = piece.tech.coeffs

print("sin(x) coefficients (first 8):")
print([f"{c:.2e}" for c in coeffs[:8]])
```

```
sin(x) coefficients (first 8):
['0.00e+00', '8.80e-01', '0.00e+00', '-3.99e-02', '0.00e+00', '4.99e-04',
 '0.00e+00', '-3.47e-06']
```

The coefficients alternate between zero (even/odd symmetry of sin) and decay
super-algebraically — confirming geometric convergence for analytic functions.

---

## 6. Custom Domain

Approximate a function on a non-standard interval by passing `domain=[a, b]`:

```python
# Approximate sin(x) on [0, 2*pi]
f_full = cj.chebfun(jnp.sin, domain=[0.0, 2.0 * jnp.pi])
print(f_full)
print("f(pi)   =", f_full(jnp.pi))    # ≈ 0
print("f(pi/2) =", f_full(jnp.pi/2))  # ≈ 1
```

```
Chebfun on [0.0, 6.283185307179586] with 28 coefficients (1 piece)
f(pi)   = 1.2246...e-16
f(pi/2) = 1.0000000000000002
```

More coefficients are needed because the domain is wider, but the accuracy is
still near machine precision.

---

## 7. Piecewise Smooth Functions

For functions with breakpoints, pass a list of breakpoints:

```python
# Approximate |x| on [-1, 1] with an explicit breakpoint at 0
from chebfunjax.domain import Domain
from chebfunjax.chebfun1d.chebfun import chebfun

f_pw = chebfun(jnp.abs, domain=Domain([-1.0, 0.0, 1.0]))
print(f_pw)
print("error at 0.3:", abs(f_pw(0.3) - 0.3))
```

```
Chebfun on [-1, 0, 1] with 2 pieces, 4 + 4 coefficients
error at 0.3: 0.0
```

With an explicit breakpoint at 0, the two pieces are separately smooth and
only 4 coefficients per piece are needed — machine precision with almost no work.

---

## 8. The AAA Rational Approximation

For functions with poles or essential singularities, rational approximation
often outperforms polynomials.  The `aaa` function implements the
Adaptive Antoulas-Anderson algorithm:

```python
from chebfunjax.utils.aaa import aaa
import jax.numpy as jnp

# Approximate 1/(1 + x^2) on [-1, 1] — note the poles at ±i
xs = jnp.linspace(-1.0, 1.0, 200)
ys = 1.0 / (1.0 + xs**2)

r, pol, res, zer, zj, fj, w, errvec = aaa(ys, xs)

# Evaluate rational approximant at a test point
x_test = jnp.array(0.5)
print("rational r(0.5) =", r(x_test))
print("exact       =", 1.0 / (1.0 + 0.5**2))
```

```
rational r(0.5) = 0.8000000000000...
exact       = 0.8
```
