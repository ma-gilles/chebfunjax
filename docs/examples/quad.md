# Quadrature Examples

chebfunjax computes integrals by summing Chebyshev coefficients:
$\int_{-1}^{1} f(x)\, dx = 2 c_0 - 2 c_2/3 + 2 c_4/15 - \cdots$,
where $f(x) = \sum_k c_k T_k(x)$. This gives spectral accuracy for smooth
functions and handles integrable singularities at the endpoints gracefully.

*See also: [Chebfun quad examples](https://www.chebfun.org/examples/quad/)*

---

## 1. Clenshaw-Curtis quadrature

Clenshaw-Curtis quadrature uses Chebyshev nodes and gives the same exponential
convergence as Gauss-Legendre for analytic integrands:

```python
import jax.numpy as jnp
import chebfunjax as cj

# int_0^pi sin(x) dx = 2
f = cj.chebfun(jnp.sin, domain=(0.0, float(jnp.pi)))
print("Clenshaw-Curtis:", f.sum(), "  exact: 2.0")
```

---

## 2. Gauss-Legendre nodes

Gauss-Legendre nodes are the zeros of Legendre polynomials $P_n(x)$.
The $n$-point rule is exact for polynomials of degree $\le 2n-1$:

```python
# int_-1^1 P_n(x)^2 dx = 2/(2n+1)
import scipy.special
n = 8
x_gl, w_gl = scipy.special.roots_legendre(n)
P8  = scipy.special.legendre(n)(x_gl)
I   = float(w_gl @ P8**2)
print(f"int P_{n}^2 = {I:.10f}  exact: {2/(2*n+1):.10f}")
```

---

## Translated Chebfun Examples

| Example | Description |
|---------|-------------|
| [SpikeIntegral](quad/SpikeIntegral.md) | Spike function with Chebfun degree ~13000 |
| [HermiteQuad](quad/HermiteQuad.md) | Gauss-Hermite quadrature via Jacobi matrix eigenvalues |
| [SymbolicNumeric](quad/SymbolicNumeric.md) | Comparison of 8 symbolic and numerical integrals |

---

## Gallery

