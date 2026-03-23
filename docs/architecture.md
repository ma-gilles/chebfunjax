# Architecture

## Design Decisions

These are locked. Changes require explicit approval.

### JAX-only backend

`jax.numpy` everywhere. No numpy fallback layer. JAX runs on CPU transparently —
users switch via `JAX_PLATFORMS=cpu` or `jax.default_device(jax.devices("cpu")[0])`.
The library never manages device placement internally.

### Precision: float64

Spectral methods require double precision. All array creation uses
`dtype=jnp.float64` explicitly. `jax_enable_x64` is set at import time.

### Object model: Equinox Modules

All chebfunjax objects are frozen `eqx.Module` pytrees. Array fields are pytree
children (traced by JIT/vmap). Non-array metadata uses `eqx.field(static=True)`.
Operations return new objects — never mutate.

### JIT boundaries

| Code pattern | JIT? | How |
|-------------|------|-----|
| Adaptive construction (convergence loop) | No | Python loop; JIT inner FFT/eval |
| Evaluation at fixed points | Yes | `@jax.jit` |
| Differentiation (coefficient manipulation) | Yes | `@jax.jit` |
| Integration, inner products | Yes | `@jax.jit` |
| Rootfinding (colleague matrix eigenvalues) | Yes | `@jax.jit` |
| Operator construction | Yes | `@jax.jit` |

### GPU transparency

No `device=` arguments, no `.to_gpu()` methods, no backend selection.
JAX arrays live where they're created; operations follow the data.

```python
# CPU
with jax.default_device(jax.devices("cpu")[0]):
    f = cj.chebfun(jnp.sin)

# GPU (automatic if available)
f = cj.chebfun(jnp.sin)
```

## Class Hierarchy

```
Chebtech (abstract eqx.Module)
├── Chebtech1 (1st kind grid)
└── Chebtech2 (2nd kind grid, default)
Trigtech (periodic)

Classicfun
├── Bndfun (bounded [a,b])
└── Unbndfun (unbounded)
Singfun, Deltafun

Chebfun (piecewise, user-facing)

Chebmatrix, Linop, Chebop

SeparableApprox → Chebfun2, Diskfun, Spherefun
Chebfun3, Ballfun

Spinop family
```

## Dependency Graph

```
Utilities → Tech → Fun → Chebfun → {Operators, 2D/3D, Spin}
```

See [../PLAN.md](../PLAN.md) for the full module dependency graph and translation units.
