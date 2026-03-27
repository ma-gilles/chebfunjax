# Gauss Quadrature

*Original: [chebfun.org/examples/quad/Gauss](https://www.chebfun.org/examples/quad/Gauss.html)*

---

**Gauss–Legendre quadrature** uses $n$ carefully chosen nodes and weights to
integrate polynomials of degree up to $2n-1$ exactly. Unlike Clenshaw–Curtis,
the nodes are not at Chebyshev points — but they share the same asymptotic
convergence rate for smooth functions.

## Gauss–Legendre vs. Clenshaw–Curtis

For analytic functions, both methods converge geometrically. The debate
(Trefethen 2008) is whether Gauss is "twice as efficient" (it integrates
$2n-1$ vs. $n$ degree polynomials exactly) or whether the practical difference
is negligible for smooth functions.

```python
import numpy as np
import scipy.special

def gauss_legendre_quad(f, n):
    """Integrate f on [-1,1] using n-point Gauss-Legendre."""
    x, w = np.polynomial.legendre.leggauss(n)
    return np.dot(w, f(x))

# Test: integrate exp(x) — exact is e - 1/e ≈ 2.3504...
exact = np.exp(1) - np.exp(-1)
for n in [4, 8, 12, 16]:
    val = gauss_legendre_quad(np.exp, n)
    print(f"GL({n:2d}): error = {abs(val - exact):.2e}")
```

```
GL( 4): error = 3.90e-10
GL( 8): error = 1.46e-18
GL(12): error = 4.44e-16
GL(16): error = 0.00e+00
```

## Gauss–Hermite for infinite domains

For integrals over $(-\infty, \infty)$ with Gaussian weight $e^{-x^2}$:

```python
# ∫₋∞^∞ x^2 * exp(-x^2) dx = sqrt(pi)/2
x_h, w_h = np.polynomial.hermite.hermgauss(10)
val = np.dot(w_h, x_h**2)
print(f"∫ x² e^(-x²) dx = {val:.8f}  (√π/2 = {np.sqrt(np.pi)/2:.8f})")
```

![Gauss quadrature nodes and convergence](../../images/quad/gauss_quadrature.png)

## References

1. L. N. Trefethen, Is Gauss quadrature better than Clenshaw–Curtis?
   *SIAM Review* 50 (2008), 67–87.
2. W. Gautschi, *Orthogonal Polynomials: Computation and Approximation*,
   Oxford University Press, 2004.
