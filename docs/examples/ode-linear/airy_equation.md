# Airy Equation

*Original: [chebfun.org/examples/ode-linear/Airy](https://www.chebfun.org/examples/ode-linear/Airy.html)*

---

The **Airy equation** $y'' = xy$ is the simplest second-order ODE with a
turning point. Its two independent solutions — $\text{Ai}(x)$ and $\text{Bi}(x)$
— decay and grow, respectively, as $x \to +\infty$.

## Numerical solution via finite differences

```python
import numpy as np
import scipy.linalg
import scipy.special

n = 200
x = np.linspace(-5, 5, n)
h = x[1] - x[0]

# Second difference matrix: y''_i ≈ (y_{i-1} - 2y_i + y_{i+1})/h^2
D2 = (np.diag(np.ones(n-1), -1) - 2*np.diag(np.ones(n)) + np.diag(np.ones(n-1), 1)) / h**2
X = np.diag(x)

# Solve (D2 - X) y = 0 with boundary conditions y(-5) = Ai(-5), y(5) = Ai(5)
Ai_m5, _, _, _ = scipy.special.airy(-5)
Ai_5, _, _, _ = scipy.special.airy(5)

A = D2 - X
A[0, :] = 0; A[0, 0] = 1
A[-1, :] = 0; A[-1, -1] = 1
rhs = np.zeros(n); rhs[0] = Ai_m5; rhs[-1] = Ai_5
y = np.linalg.solve(A, rhs)

# Compare with exact
Ai_exact = scipy.special.airy(x)[0]
err = np.max(np.abs(y - Ai_exact))
print(f"Airy BVP error: {err:.2e}")
```

![Airy equation solution](../../../images/ode-linear/airy_equation.png)

## Turning point behavior

At $x=0$, the character of the solution changes: oscillatory for $x < 0$
(wave-like) and exponentially decaying for $x > 0$. The asymptotic forms are:

$$\text{Ai}(x) \approx \frac{e^{-2x^{3/2}/3}}{2\sqrt{\pi} x^{1/4}}, \quad x \to +\infty.$$

## References

1. F. W. J. Olver, *Asymptotics and Special Functions*, Academic Press, 1974.
