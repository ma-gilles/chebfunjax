# Condition Numbers of Vandermonde Matrices

*Original: [chebfun.org/examples/linalg/CondVandermonde](https://www.chebfun.org/examples/linalg/CondVandermonde.html)*

---

The Vandermonde matrix $V_{ij} = x_i^{j-1}$ arises in polynomial interpolation.
Its condition number measures how sensitive the solution of $Vc = f$ is to
perturbations in the data — in other words, how numerically stable polynomial
interpolation is at the chosen nodes.

## Equispaced vs. Chebyshev nodes

```python
import numpy as np

def vandermonde(nodes):
    n = len(nodes)
    V = np.ones((n, n))
    for j in range(1, n):
        V[:, j] = V[:, j-1] * nodes
    return V

for n in [10, 20, 30]:
    equi = np.linspace(-1, 1, n)
    cheb = -np.cos(np.pi * np.arange(n) / (n-1))
    print(f"n={n}: cond(V_equi)={np.linalg.cond(vandermonde(equi)):.2e}, "
          f"cond(V_cheb)={np.linalg.cond(vandermonde(cheb)):.2e}")
```

```
n=10: cond(V_equi)=1.43e+11, cond(V_cheb)=3.25e+04
n=20: cond(V_equi)=9.89e+20, cond(V_cheb)=2.37e+06
n=30: cond(V_equi)=2.93e+27, cond(V_cheb)=5.62e+08
```

The equispaced Vandermonde matrix is **exponentially ill-conditioned**.

![Condition number growth](../../../images/linalg/condition_numbers.png)

## The Chebyshev-Vandermonde matrix

A better choice is the **Chebyshev-Vandermonde** matrix $V_{ij} = T_{j-1}(x_i)$
where $T_k$ are Chebyshev polynomials. At Chebyshev nodes, this matrix has
near-unit condition number:

```python
def cheb_vandermonde(nodes):
    n = len(nodes)
    V = np.ones((n, n))
    if n > 1:
        V[:, 1] = nodes
    for j in range(2, n):
        V[:, j] = 2 * nodes * V[:, j-1] - V[:, j-2]
    return V

cheb_nodes_20 = -np.cos(np.pi * np.arange(20) / 19)
print(f"Chebyshev-Vandermonde n=20: cond = {np.linalg.cond(cheb_vandermonde(cheb_nodes_20)):.2f}")
```

```
Chebyshev-Vandermonde n=20: cond = 1.53
```

This is why chebfunjax represents functions in the Chebyshev basis — it is
numerically optimal.

## References

1. L. N. Trefethen, *Spectral Methods in MATLAB*, SIAM, 2000.
2. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*,
   SIAM, 2002, Ch. 22.
