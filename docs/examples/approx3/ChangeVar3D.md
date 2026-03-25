# Triple Integrals via Coordinate Transformations

*Rodrigo Platte, November 2016*

*Original: [Triple integrals in spherical, cylindrical and other coordinate systems — Chebfun](https://www.chebfun.org/examples/approx3/ChangeVar3D.html)*

---

## Transformations

This example uses mappings to compute integrals over non-rectangular three-dimensional volumes.
We apply the change of variables

$$x = x(u,v,w), \quad y = y(u,v,w), \quad z = z(u,v,w),$$

where $u$, $v$, $w$ are defined as `Chebfun3` objects on a rectangular domain.

## Triple Integrals in Spherical Coordinates

We use spherical coordinates to compute the mass of an "ice-cream cone" region
with variable density. The region is defined using `Chebfun3` objects:

```python
import numpy as np
import jax.numpy as jnp
from chebfunjax.chebfun3d.chebfun3 import chebfun3

dom_sph = (0.0, 1.0, 0.0, 2*np.pi, np.pi/4, np.pi/2)

r_f = chebfun3(lambda r, t, p: r, domain=dom_sph)
t_f = chebfun3(lambda r, t, p: t, domain=dom_sph)
p_f = chebfun3(lambda r, t, p: p, domain=dom_sph)

x_sph = chebfun3(lambda r, t, p: r * jnp.cos(t) * jnp.cos(p), domain=dom_sph)
y_sph = chebfun3(lambda r, t, p: r * jnp.sin(t) * jnp.cos(p), domain=dom_sph)
z_sph = chebfun3(lambda r, t, p: r * jnp.sin(p), domain=dom_sph)
```

The mass integral with density $\rho = r^2$ can be evaluated exactly.
The Jacobian for spherical coordinates is $|J| = r^2 \cos\phi$:

```python
M_simple = chebfun3(
    lambda r, t, p: r**2 * r**2 * jnp.cos(p),
    domain=dom_sph
).sum3()
# Exact: pi*(2-sqrt(2))/5
exact = np.pi * (2 - np.sqrt(2)) / 5
print(f"Computed: {float(M_simple):.10f}")
print(f"Exact:    {exact:.10f}")
```

```
Computed: 0.3680604738
Exact:    0.3680604738
```

## Triple Integrals in Cylindrical Coordinates

We compute the center of mass of a sector of a cylinder. With cylindrical
coordinates $x = r\cos\theta$, $y = r\sin\theta$, the Jacobian is $|J| = r$.

For uniform density, the $z$-coordinate of the center of mass is:

$$z_c = \frac{\int_0^1\int_0^\pi\int_0^1 z \cdot r \, dz\, d\theta\, dr}
             {\int_0^1\int_0^\pi\int_0^1 r \, dz\, d\theta\, dr} = 0.5$$

## Triple Integrals over the Torus

The torus with major radius $R=4$ and minor radius $r=1$ is parametrized by

$$x = (R + r'\cos t)\cos\phi, \quad y = (R + r'\cos t)\sin\phi, \quad z = r'\sin t$$

The Jacobian is $|J| = r'(R + r'\cos t)$. The volume is

$$V = \int_0^1\int_0^{2\pi}\int_0^{2\pi} r(4 + r\cos t)\, dr\, dt\, d\phi = 8\pi^2$$

```python
vol_torus = chebfun3(
    lambda r, t, p: r * (4 + r * jnp.cos(t)),
    domain=(0.0, 1.0, 0.0, 2*np.pi, 0.0, 2*np.pi)
).sum3()
print(f"Torus volume: {float(vol_torus):.6f}")
print(f"Exact 8π²:    {8*np.pi**2:.6f}")
```

```
Torus volume: 78.956835
Exact 8π²:    78.956835
```

![Coordinate transformations for 3D integration](../../../images/approx3/ChangeVar3D.png)

## References

1. R. Platte, *Chebfun Examples*, 2016.
