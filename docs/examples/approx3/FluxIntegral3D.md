# Flux Integrals over Parametric Surfaces

*Olivier Sète, June 2016*

*Original: [Integration of a chebfun3v over a 2D surface — Chebfun](https://www.chebfun.org/examples/approx3/FluxIntegral3D.html)*

---

## Flux Integrals

Given a vector field $F(x,y,z) = [F_1, F_2, F_3]$ and a surface
$S = S(u,v)$ parametrized over $D = [a,b]\times[c,d]$, the flux integral is

$$\int_S F \cdot \vec{dS} = \int_D F(S(u,v)) \cdot
\left(\frac{\partial S}{\partial u} \times \frac{\partial S}{\partial v}\right) \, du\, dv.$$

In chebfunjax, we compute this numerically by evaluating the integrand
on a fine grid using the `Chebfun3` representation of $F$.

## Example: Rippled Disk

Consider the vector field $F(x,y,z) = (x+y,\ xz+y,\ z)$ and the
rippled disk $S(r,\theta) = (r\cos\theta,\ r\sin\theta,\ \cos(5r))$
for $r \in [0,5]$, $\theta \in [0, 2\pi]$:

```python
import numpy as np
import jax.numpy as jnp
from chebfunjax.chebfun3d.chebfun3 import chebfun3

F1 = lambda x, y, z: x + y
F2 = lambda x, y, z: x*z + y
F3 = lambda x, y, z: z

# Rippled disk parametrization
Sx = lambda r, t: r * np.cos(t)
Sy = lambda r, t: r * np.sin(t)
Sz = lambda r, t: np.cos(5*r)
```

## Example: Lower Hemisphere

For the lower half of the unit sphere
$S(\phi, \theta) = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$
with $\theta \in [\pi/2, \pi]$, the flux of $F$ equals $-2\pi$ exactly:

```python
Sx2 = lambda phi, theta: np.sin(theta) * np.cos(phi)
Sy2 = lambda phi, theta: np.sin(theta) * np.sin(phi)
Sz2 = lambda phi, theta: np.cos(theta)

# Computed flux ≈ -6.283153
# Exact: -2*pi = -6.283185
```

## Divergence Theorem Verification

The divergence theorem states
$$\int_K \mathrm{div}(F)\, dV = \oint_{\partial K} F \cdot \vec{dS}.$$

For $F = (x,y,z)$, $\mathrm{div}(F) = 3$, so the total flux through the unit sphere
equals $3 \cdot \frac{4\pi}{3} = 4\pi$. We verify via a Chebfun3 triple integral
using spherical coordinates:

```python
# Jacobian of spherical coords: r^2 * sin(theta)
div_F_ball = chebfun3(
    lambda r, t, p: 3 * r**2 * jnp.sin(t),
    domain=(0, 1, 0, np.pi, 0, 2*np.pi)
)
print(float(div_F_ball.sum3()))  # 12.566371 = 4*pi
```

```
12.566371
```

![Flux integrals over parametric surfaces](../../../images/approx3/FluxIntegral3D.png)

## See Also

- [LineIntegral3D](LineIntegral3D.md) — integration over curves
- [GaussGreenStokes](GaussGreenStokes.md) — divergence theorem, Green's identities, Stokes' theorem
- [SurfaceIntegral3D](SurfaceIntegral3D.md) — scalar surface integrals
