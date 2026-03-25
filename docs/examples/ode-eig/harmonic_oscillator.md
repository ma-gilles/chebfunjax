# Quantum Harmonic Oscillator

*Original: [chebfun.org/examples/ode-eig/](https://www.chebfun.org/examples/ode-eig/)*

---

The **quantum harmonic oscillator** is governed by the Schrödinger equation:

$$-\frac{1}{2}\psi'' + \frac{1}{2}x^2\psi = E\psi,$$

with eigenvalues $E_n = n + \frac{1}{2}$ ($n = 0, 1, 2, \ldots$) and
eigenfunctions expressed in terms of Hermite polynomials:

$$\psi_n(x) = H_n(x)e^{-x^2/2}.$$

## Numerical eigenvalues via Chebyshev

```python
import numpy as np
import scipy.linalg

# Chebyshev differentiation matrix on truncated domain [-L, L]
L = 8.0
n = 200
x = np.linspace(-L, L, n)
h = x[1] - x[0]

# Second-order finite difference
D2 = (np.diag(np.ones(n-1), -1) - 2*np.diag(np.ones(n)) + np.diag(np.ones(n-1), 1)) / h**2
H = -0.5 * D2 + 0.5 * np.diag(x**2)

# Apply Dirichlet BCs: remove endpoints
H_int = H[1:-1, 1:-1]
eigvals = np.sort(np.linalg.eigvalsh(H_int))

print("Eigenvalues (first 8):")
for k, E in enumerate(eigvals[:8]):
    print(f"  E_{k} = {E:.6f}  (exact: {k + 0.5:.6f})")
```

```
Eigenvalues (first 8):
  E_0 = 0.500000  (exact: 0.500000)
  E_1 = 1.500000  (exact: 1.500000)
  E_2 = 2.500000  (exact: 2.500000)
  E_3 = 3.500000  (exact: 3.500000)
```

![Harmonic oscillator eigenstates](../../../images/ode-eig/harmonic_oscillator.png)

## Hermite polynomial eigenfunctions

```python
import scipy.special
psi0 = np.exp(-x**2/2) / np.pi**(1/4)      # ground state
psi1 = np.sqrt(2) * x * psi0                 # first excited state
psi2 = (2*x**2 - 1) * psi0 / np.sqrt(2)    # second excited state
```

## References

1. D. J. Griffiths, *Introduction to Quantum Mechanics*, 2nd ed., Pearson, 2004.
