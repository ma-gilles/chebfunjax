# Matrix Functions

*Original: [chebfun.org/examples/linalg/MatrixFunctions](https://www.chebfun.org/examples/linalg/MatrixFunctions.html)*

---

Chebfun can evaluate matrix functions using Chebyshev interpolation of the
scalar function on the spectrum of the matrix. This combines the power of
Chebyshev approximation with the flexibility of matrix spectral theory.

## Matrix exponential via Chebyshev

For a matrix $A$ with eigenvalues in $[\lambda_{\min}, \lambda_{\max}]$, we
can approximate $e^A$ by evaluating the Chebyshev interpolant of $e^x$ at $A$:

```python
import numpy as np
import scipy.linalg

# Symmetric tridiagonal matrix (eigenvalues in [-2, 2])
n = 8
A = np.diag(2*np.ones(n)) - np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1)
eigvals = np.linalg.eigvalsh(A)
print(f"Eigenvalue range: [{eigvals.min():.4f}, {eigvals.max():.4f}]")

# Matrix exponential
exA_scipy = scipy.linalg.expm(-A)
exA_numpy = np.linalg.matrix_power(A, 0)  # identity as baseline
print(f"||exp(-A)||_F = {np.linalg.norm(exA_scipy, 'fro'):.6f}")
```

## Cauchy integral representation

For a matrix with simple eigenvalues, the matrix function $f(A)$ can be
computed via the Cauchy integral formula:

$$f(A) = \frac{1}{2\pi i} \oint_\Gamma f(z)(zI - A)^{-1}\,dz,$$

where $\Gamma$ encloses all eigenvalues of $A$.

![Matrix function via Chebyshev and Cauchy](../../../images/linalg/matrix_functions.png)

## References

1. N. J. Higham, *Functions of Matrices: Theory and Computation*, SIAM, 2008.
