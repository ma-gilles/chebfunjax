# Sturm–Liouville Theory

*Original: [chebfun.org/examples/ode-eig/SturmLiouville](https://www.chebfun.org/examples/ode-eig/SturmLiouville.html)*

---

The general **Sturm–Liouville** eigenvalue problem:

$$-(p(x)u')' + q(x)u = \lambda w(x)u, \quad x \in [a,b]$$

with appropriate boundary conditions has a countable spectrum $\lambda_1 < \lambda_2 < \cdots$
and orthogonal eigenfunctions with weight $w(x)$.

## Legendre's equation as Sturm-Liouville

Legendre's equation $(1-x^2)y'' - 2xy' + \lambda y = 0$ is a Sturm–Liouville
problem with $p = 1-x^2$, $q = 0$, $w = 1$, and eigenvalues $\lambda_n = n(n+1)$:

```python
import numpy as np

n = 100
x = np.linspace(-1+1e-6, 1-1e-6, n)
h = x[1] - x[0]
p = 1 - x**2

# -(p*u')' = λ*u: discretize using centered differences
# D_minus D_plus (p * D_plus u) + q*u = lambda * w * u
# Simplified: tridiagonal operator
A = np.zeros((n, n))
for i in range(1, n-1):
    pm = (p[i] + p[i-1]) / 2
    pp = (p[i] + p[i+1]) / 2
    A[i, i-1] = -pm / h**2
    A[i, i]   = (pm + pp) / h**2
    A[i, i+1] = -pp / h**2
A[0, 0] = A[-1, -1] = 1  # boundary conditions

eigvals = np.sort(np.linalg.eigvalsh(A[1:-1, 1:-1]))[:8]
exact = [n*(n+1) for n in range(8)]
print("Sturm-Liouville (Legendre) eigenvalues:")
for k, (ev, ex) in enumerate(zip(eigvals, exact)):
    print(f"  n={k}: λ = {ev:.4f}  (exact: {ex})")
```

![Sturm-Liouville eigenfunctions](../../../images/ode-eig/sturm_liouville.png)

## Orthogonality

The eigenfunctions $u_m$, $u_n$ are orthogonal with weight $w(x)$:

$$\int_a^b u_m(x) u_n(x) w(x)\,dx = 0, \quad m \neq n.$$

## References

1. B. Spain and M. G. Smith, *Functions of Mathematical Physics*, Van Nostrand, 1970.
