# `angle`, `unwrap`, and branches of complex chebfuns

**Nick Trefethen, May 2011**

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/complex/Arguments.html)

---

A complex number $z$ has a modulus $|z| \in [0, \infty)$ and an argument
$\arg(z) \in (-\pi, \pi]$. For a complex-valued function of a real variable,
we can track the *continuously varying* argument by unwrapping.

## The argument of a spiral

Consider the Archimedean spiral $z(t) = (1 + t) e^{it}$ for $t \in [0, 4\pi]$.
The standard `angle` wraps into $(-\pi, \pi]$, but the unwrapped argument grows
continuously:

```python
import jax.numpy as jnp
import numpy as np
import chebfunjax as cj

# Parameterise the spiral as a complex chebfun
t = np.linspace(0, 4*np.pi, 2000)
z = (1 + t) * np.exp(1j * t)

arg_wrapped   = np.angle(z)
arg_unwrapped = np.unwrap(arg_wrapped)

print(f"Final wrapped angle:   {arg_wrapped[-1]:.4f}")
print(f"Final unwrapped angle: {arg_unwrapped[-1]:.4f}  (≈ 4π = {4*np.pi:.4f})")
```

## Winding numbers via total argument change

The winding number of a closed curve $\gamma$ around a point $p$ is

$$
n(\gamma, p) = \frac{1}{2\pi} \Delta \arg\!\bigl(z(t) - p\bigr),
$$

where $\Delta$ denotes the total change over one traversal. For a circle of
radius $r$ centred at the origin:

```python
# Circle |z| = r; winding number around 0 should be 1
theta = np.linspace(0, 2*np.pi, 2001)
z_circle = 2.0 * np.exp(1j * theta)
arg_change = np.unwrap(np.angle(z_circle))[-1] - np.unwrap(np.angle(z_circle))[0]
winding = arg_change / (2 * np.pi)
print(f"Winding number: {winding:.4f}  (should be 1)")
```

## Gallery

![Argument principle](../../images/complex/argument_principle.png)

*Top*: Wrapped and unwrapped argument of a spiral.
*Bottom*: Winding number computation for a closed curve around a pole.
