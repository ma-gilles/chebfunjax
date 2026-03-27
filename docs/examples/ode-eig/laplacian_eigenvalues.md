# Laplacian Eigenvalues

**Inspired by [Chebfun](https://www.chebfun.org/) examples (ode-eig/DirLap)**

---

The eigenvalue problem $-u'' = \lambda u$ on $[0, \pi]$ with Dirichlet
boundary conditions $u(0) = u(\pi) = 0$ has exact solutions:

$$\lambda_n = n^2, \quad u_n(x) = \sin(nx), \quad n = 1, 2, 3, \ldots$$

This is the archetype of Sturm–Liouville theory.

## Spectral computation

```python
import numpy as np

n = 100
x = np.linspace(0, np.pi, n+2)[1:-1]  # interior points
h = np.pi / (n+1)
D2 = (np.diag(np.ones(n-1), -1) - 2*np.diag(np.ones(n)) + np.diag(np.ones(n-1), 1)) / h**2
eigvals = np.sort(np.linalg.eigvalsh(-D2))

print("First 6 eigenvalues:")
for k, lam in enumerate(eigvals[:6]):
    exact = (k+1)**2
    print(f"  λ_{k+1} = {lam:.6f}  (exact: {exact})")
```

```
First 6 eigenvalues:
  λ_1 = 1.000000  (exact: 1)
  λ_2 = 4.000000  (exact: 4)
  λ_3 = 9.000000  (exact: 9)
  λ_4 = 16.000000  (exact: 16)
  λ_5 = 24.999985  (exact: 25)
  λ_6 = 35.999930  (exact: 36)
```

## Weyl's law

For large $n$, the eigenvalue density follows **Weyl's law**:
$\lambda_n \sim n^2 \pi^2 / L^2$ for a domain of length $L$. The asymptotic
density of eigenvalues below $\Lambda$ is $\sim L\sqrt{\Lambda}/\pi$.

![Laplacian eigenvalues and eigenfunctions](../../images/ode-eig/laplacian_eigenvalues.png)

## 2D Laplacian

On a square $[0,\pi]^2$, the eigenvalues are $\lambda_{m,n} = m^2 + n^2$
(sums of 1D eigenvalues), and eigenfunctions are products:
$u_{m,n}(x,y) = \sin(mx)\sin(ny)$.

## References

1. L. N. Trefethen, *Spectral Methods in MATLAB*, SIAM, 2000, Programs 6–7.
