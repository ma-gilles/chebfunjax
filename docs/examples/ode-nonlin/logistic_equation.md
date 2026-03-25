# Logistic Equation

*Original: [chebfun.org/examples/ode-nonlin/Logistic](https://www.chebfun.org/examples/ode-nonlin/Logistic.html)*

---

The **logistic equation** $y' = ry(1 - y/K)$ models population growth with
a carrying capacity $K$. It has the exact solution:

$$y(t) = \frac{K}{1 + (K/y_0 - 1)e^{-rt}}.$$

## Numerical solution

```python
import numpy as np
import scipy.integrate

r, K, y0 = 1.0, 10.0, 1.0  # growth rate, carrying capacity, initial population

def logistic(t, y):
    return [r * y[0] * (1 - y[0]/K)]

sol = scipy.integrate.solve_ivp(logistic, [0, 10], [y0], dense_output=True,
                                  rtol=1e-10)
t_test = np.linspace(0, 10, 200)
y_exact = K / (1 + (K/y0 - 1) * np.exp(-r * t_test))
err = np.max(np.abs(sol.sol(t_test)[0] - y_exact))
print(f"Logistic eq. max error: {err:.2e}")
print(f"Final population: y(10) ≈ {sol.sol(np.array([10.0]))[0,0]:.4f}  "
      f"(exact: {y_exact[-1]:.4f})")
```

```
Logistic eq. max error: 5.23e-11
Final population: y(10) ≈ 9.9999  (exact: 9.9999)
```

## Chebfun representation of solution

```python
import chebfunjax as cj
import jax.numpy as jnp

y_cheb = cj.chebfun(lambda t: K / (1 + (K/y0 - 1) * jnp.exp(-r * t)),
                     domain=(0.0, 10.0))
x_max, y_max = y_cheb.max()
print(f"Carrying capacity reached asymptotically: max ≈ {float(y_max):.4f}")
```

![Logistic equation solution](../../../images/ode-nonlin/logistic_equation.png)
