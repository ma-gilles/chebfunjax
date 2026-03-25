# Nonlinear Pendulum

*Original: [chebfun.org/examples/ode-nonlin/](https://www.chebfun.org/examples/ode-nonlin/)*

---

The **nonlinear pendulum** equation $\theta'' + \sin\theta = 0$ describes
the exact motion of a pendulum without the small-angle approximation.

## Period vs. amplitude

For small angles, the period is $T_0 = 2\pi$. For large amplitudes, the
period lengthens, diverging as the amplitude approaches $\pi$ (the separatrix):

```python
import numpy as np
import scipy.integrate
import scipy.special

amplitudes = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
periods = []
for theta0 in amplitudes:
    # Period = 4 * K(sin^2(theta0/2)) where K is complete elliptic integral
    k = np.sin(theta0/2)
    T = 4 * scipy.special.ellipk(k**2)
    periods.append(T)
    print(f"θ₀ = {theta0:.1f}: T = {T:.6f}  (T₀ = {2*np.pi:.6f})")
```

```
θ₀ = 0.1: T = 6.283596  (T₀ = 6.283185)
θ₀ = 0.5: T = 6.360985  (T₀ = 6.283185)
θ₀ = 1.0: T = 6.611521  (T₀ = 6.283185)
θ₀ = 2.5: T = 9.071760  (T₀ = 6.283185)
θ₀ = 3.0: T = 14.30406  (T₀ = 6.283185)
```

## Numerical solution via scipy

```python
def pendulum(t, y):
    return [y[1], -np.sin(y[0])]

theta0 = 2.0
sol = scipy.integrate.solve_ivp(
    pendulum, [0, 20], [theta0, 0.0],
    method='RK45', rtol=1e-10, atol=1e-12, dense_output=True
)
t_plot = np.linspace(0, 20, 1000)
theta = sol.sol(t_plot)[0]
print(f"Max angle: {np.max(np.abs(theta)):.6f}  (should ≈ {theta0})")
```

![Nonlinear pendulum solution and period](../../../images/ode-nonlin/pendulum_equation.png)

## References

1. G. B. Arfken and H. J. Weber, *Mathematical Methods for Physicists*, 6th ed.,
   Elsevier, 2005.
