# Integration over 2D Surfaces in 3D

*Behnam Hashemi, June 2016*

*Original: [Integration of functions over 2D surfaces in 3D — Chebfun](https://www.chebfun.org/examples/approx3/SurfaceIntegral3D.html)*

---

## Surface Integrals

For a scalar field $f(x,y,z)$ represented as a `Chebfun3` and a
parametric surface $S: (x(u,v), y(u,v), z(u,v))$, the surface integral is

$$\int_S f\, dS = \int_D f(S(u,v))\,
\left\|T_u \times T_v\right\|\, du\, dv,$$

where $T_u = \partial S/\partial u$ and $T_v = \partial S/\partial v$.

This is the scalar analogue of the [flux integral](FluxIntegral3D.md)
(which uses the dot product rather than the norm of the cross product).

## Example 1: $x^2$ over the Unit Sphere

The surface integral $\int_{\text{sphere}} x^2\, dS = 4\pi/3$:

```python
import numpy as np
import jax.numpy as jnp
from chebfunjax.chebfun3d.chebfun3 import chebfun3

f = chebfun3(lambda x, y, z: x**2)

# Sphere: S(u,v) = (sin(u)*cos(v), sin(u)*sin(v), cos(u))
Sx = lambda u, v: np.sin(u) * np.cos(v)
Sy = lambda u, v: np.sin(u) * np.sin(v)
Sz = lambda u, v: np.cos(u)

I = surface_integral(f, Sx, Sy, Sz, (0, np.pi), (0, 2*np.pi))
print(f"Computed: {I:.10f}")
print(f"Exact:    {4*np.pi/3:.10f}")
```

```
Computed: 4.1887902051
Exact:    4.1887902048
```

## Example 2: Conical Surface

For $f = \sqrt{1+x^2+y^2}$ over the cone $S(u,v) = (u\cos v, u\sin v, v)$,
$u \in [0,2]$, $v \in [0,\pi]$, the exact result is $14\pi/3$:

```python
f2 = chebfun3(lambda x, y, z: jnp.sqrt(1 + x**2 + y**2),
              domain=(-3, 3, -3, 3, -3, 3))
I2 = surface_integral(f2, lambda u,v: u*np.cos(v), lambda u,v: u*np.sin(v),
                      lambda u,v: v, (0, 2), (0, np.pi))
# Computed: 14.659..., Exact: 14*pi/3 = 14.660...
```

## Example 3: Seashell Surface

A seashell surface can be parametrized as:

$$S(u,v) = \left(\frac{5}{4}(1-\tfrac{v}{2\pi})\cos(2v)(1+\cos u) + \cos(2v),\ \ldots\right)$$

```python
f3 = chebfun3(lambda x, y, z: x+y+z, domain=(-6,6,-6,6,0,25))
I3 = surface_integral(f3, Sx_shell, Sy_shell, Sz_shell,
                      (0, 2*np.pi), (-2*np.pi, 2*np.pi))
```

## Example 4: Spring Surface

A toroidal spring with $r_1=r_2=0.5$, $t=1.5$:

```python
I4 = surface_integral(f4, Sx_spring, Sy_spring, Sz_spring,
                      (0, 10*np.pi), (0, 10*np.pi))
# Exact: 1878.4483...
```

![Surface integrals of 3D scalar fields](../../../images/approx3/SurfaceIntegral3D.png)

## References

1. J. Stewart, *Calculus: Early Transcendentals*, 6th Edition, Thomson
   Brooks/Cole, 2008.
