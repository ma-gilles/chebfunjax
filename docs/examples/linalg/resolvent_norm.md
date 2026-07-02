# Resolvent Norms and Pseudospectra

*Original: [chebfun.org/examples/linalg/ResolventNorm](https://www.chebfun.org/examples/linalg/ResolventNorm.html)*
**Author(s):** Nick Trefethen, May 2011

---

The **resolvent** of a matrix $A$ is $(zI - A)^{-1}$ for $z \notin \sigma(A)$.
Its norm $\|(zI-A)^{-1}\|$ measures how sensitive the eigenvalues are to
perturbations and characterizes the **pseudospectra** of $A$.

## Resolvent norm on the real axis

For a diagonal matrix, $\|(xI - A)^{-1}\|_2 = 1/\text{dist}(x, \sigma(A))$.
For a non-normal matrix, the resolvent can be large even far from eigenvalues:

```python
import numpy as np

# Upper triangular matrix: eigenvalues on diagonal but highly non-normal
n = 6
A = np.diag(np.arange(1, n+1, dtype=float)) + np.triu(np.ones((n,n)), k=1)
eigvals = np.linalg.eigvalsh(A + A.T) / 2  # for display

# Resolvent norm on the imaginary axis
y_vals = np.linspace(-10, 10, 200)
res_norms = np.array([
    np.linalg.norm(np.linalg.inv(1j*y * np.eye(n) - A))
    for y in y_vals
])
print(f"Max resolvent norm on imaginary axis: {np.max(res_norms):.4f}")
```

## Pseudospectra

The $\varepsilon$-pseudospectrum of $A$ is:

$$\Lambda_\varepsilon(A) = \{z \in \mathbb{C} : \|(zI - A)^{-1}\| \geq 1/\varepsilon\}.$$

For normal matrices, this is just an $\varepsilon$-neighborhood of the spectrum.
For non-normal matrices, it can be much larger.

![Resolvent norm and pseudospectrum](../../images/linalg/resolvent_norm.png)

## References

1. L. N. Trefethen and M. Embree, *Spectra and Pseudospectra*, Princeton
   University Press, 2005.