# 3D Approximation Examples (Chebfun3)

Chebfunjax represents trivariate functions in Tucker format:
`f(x,y,z) ≈ Σ_{i,j,k} G_{ijk} u_i(x) v_j(y) w_k(z)`.

---

## Smooth 3D function approximation

**Source:** `approx3/` — Nick Trefethen

```python
import jax.numpy as jnp
import chebfunjax as cj

# exp(x+y+z) is rank-1 in Tucker format
f = cj.chebfun3(lambda x, y, z: jnp.exp(x + y + z))
print(f.tucker_rank)   # (1, 1, 1)

# Triple integral over [-1,1]^3
integral = f.sum3()
exact = (jnp.exp(1.0) - jnp.exp(-1.0))**3
print(abs(integral - exact))   # < 1e-8
```

![3D smooth functions](../../../examples/approx3/smooth_functions_3d.png)

---

## Flux integrals and vector calculus in 3D

**Source:** `approx3/FluxIntegral3D.m`, `approx3/GaussGreenStokes.m`

The divergence theorem: `∫∫∫ div(F) dV = ∮∮ F · dS`.

```python
# div(F) = 3 => integral over [-1,1]^3 = 24
div_g = cj.chebfun3(lambda x, y, z: 3 * jnp.ones_like(x))
print(div_g.sum3())   # 24.0
```

![Flux integrals](../../../examples/approx3/flux_integral_3d.png)
