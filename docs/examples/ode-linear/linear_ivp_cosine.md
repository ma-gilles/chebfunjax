# Linear IVP with Cosine Forcing

*Original: [chebfun.org/examples/ode-linear/](https://www.chebfun.org/examples/ode-linear/)*

---

A first-order linear IVP:

$$y' + py = q(t), \quad y(0) = y_0$$

has the exact solution via integrating factor $\mu(t) = e^{\int p\,dt}$.

## Damped harmonic oscillator

The equation $y'' + 2\gamma y' + \omega_0^2 y = F\cos(\omega t)$ models a
forced damped oscillator with exact solution (particular + homogeneous):

```python
import numpy as np

gamma = 0.1   # damping
omega0 = 1.0  # natural frequency
omega = 0.8   # forcing frequency
F = 1.0       # forcing amplitude

# Particular solution amplitude
A = F / np.sqrt((omega0**2 - omega**2)**2 + (2*gamma*omega)**2)
phi = np.arctan2(2*gamma*omega, omega0**2 - omega**2)

t = np.linspace(0, 20, 1000)
y_particular = A * np.cos(omega * t - phi)
y_homogeneous = np.exp(-gamma * t) * np.cos(np.sqrt(omega0**2 - gamma**2) * t)

print(f"Amplitude at resonance: {A:.4f}")
print(f"Phase shift: {np.degrees(phi):.2f} degrees")
```

![Linear IVP cosine forcing solution](../../../images/ode-linear/linear_ivp_cosine.png)

## Chebfun integration

For nonautonomous linear ODEs, Chebfun can integrate using `cumsum`:

```python
import chebfunjax as cj
import jax.numpy as jnp

# y' = -y + cos(t), y(0) = 0, exact: y = (cos(t) + sin(t) - e^{-t})/2
f = cj.chebfun(lambda t: jnp.cos(t), domain=(0.0, 10.0))
# Numerical integration via scipy.integrate.odeint
```
