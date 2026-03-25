# Vector Calculus Examples

Chebfun2v and Chebfun3v represent 2D and 3D vector fields and support
all the classical vector calculus operations: gradient, divergence, curl,
and Laplacian.

---

## Verifying vector calculus identities

**Source:** `veccalc/CheckingVectorCalculus.m` — Trefethen, 2010

```python
import jax.numpy as jnp
import chebfunjax as cj

# Scalar potential
f = cj.chebfun2(lambda x, y: jnp.sin(jnp.pi*x) * jnp.cos(jnp.pi*y))

# Gradient field
F = f.grad()   # Returns Chebfun2v

# curl(grad(f)) = 0  exactly
curl_F = F.curl()
print(float(curl_F.norm()))   # < 1e-8

# div(grad(f)) = Laplacian(f)
lap_computed = F.div()
lap_exact = cj.chebfun2(
    lambda x, y: -2*jnp.pi**2 * jnp.sin(jnp.pi*x) * jnp.cos(jnp.pi*y)
)
print(float((lap_computed - lap_exact).norm()))   # < 1e-6
```

![Vector calculus](../../../examples/veccalc/vector_calculus.png)

---

## Key identities verified

| Identity | Formula |
|---|---|
| Curl of gradient | `curl(grad f) = 0` |
| Divergence of curl | `div(curl F) = 0` |
| Laplacian via grad | `div(grad f) = Δf` |
| Green's identities | `∫∫ u Δv dA = -∫∫ ∇u·∇v dA + ∮ u ∂v/∂n ds` |

---

## Other vector calculus examples

| MATLAB example | Description |
|---|---|
| `veccalc/AutonomousSystems.m` | Autonomous dynamical systems |
| `veccalc/EventHandling.m` | Event handling in ODEs |
| `veccalc/UndergraduateCalculus.m` | Undergraduate calculus review |
