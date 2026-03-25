# Double Well Potential

*Original: [chebfun.org/examples/ode-eig/](https://www.chebfun.org/examples/ode-eig/)*

---

The **double well potential** $V(x) = (x^2-1)^2$ has two minima at $x = \pm 1$.
In quantum mechanics, this leads to **tunnel splitting**: the lowest two
eigenvalues are nearly degenerate, with the splitting exponentially small
in the barrier height.

## Tunnel splitting

```python
import numpy as np

L = 5.0
n = 400
x = np.linspace(-L, L, n)
h = x[1] - x[0]

# -y'' + V(x)*y = E*y, V = (x^2-1)^2
D2 = (np.diag(np.ones(n-1), -1) - 2*np.diag(np.ones(n)) + np.diag(np.ones(n-1), 1)) / h**2
V = (x**2 - 1)**2
H = -D2 + np.diag(V)
H_int = H[1:-1, 1:-1]
eigvals = np.sort(np.linalg.eigvalsh(H_int))

splitting = eigvals[1] - eigvals[0]
print(f"E_0 = {eigvals[0]:.8f}")
print(f"E_1 = {eigvals[1]:.8f}")
print(f"Tunnel splitting: ΔE = {splitting:.2e}")
```

![Double well eigenstates and tunnel splitting](../../../images/ode-eig/double_well.png)

## Symmetric/antisymmetric states

The two lowest eigenstates are symmetric ($\psi_0$, even) and antisymmetric
($\psi_1$, odd) under $x \to -x$. Their superpositions are localized in
one well:

$$\psi_L = \frac{\psi_0 + \psi_1}{\sqrt{2}}, \quad \psi_R = \frac{\psi_0 - \psi_1}{\sqrt{2}}.$$

## References

1. R. Shankar, *Principles of Quantum Mechanics*, 2nd ed., Plenum, 1994.
