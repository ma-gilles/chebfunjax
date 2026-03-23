# chebfunjax

Chebfun in Python, powered by JAX.

A translation of the [MATLAB Chebfun](https://github.com/chebfun/chebfun) library
to Python using [JAX](https://github.com/jax-ml/jax) as the numerical backend,
enabling GPU acceleration, JIT compilation, automatic differentiation, and
vectorized operations (`vmap`) for spectral methods and function approximation.

## Installation

```bash
pixi install
pixi run smoke   # verify: "chebfunjax imported OK"
```

## Quick Start

```python
import chebfunjax as cj

f = cj.chebfun(jnp.sin)          # adaptive on [-1, 1]
f.diff()                          # differentiate
f.sum()                           # integrate
f.roots()                         # find zeros
f(0.5)                            # evaluate
```

## Status

Early development — see [STATUS.md](STATUS.md) for per-module progress.

## Architecture

- **JAX-only backend** — `jax.numpy` everywhere, GPU-transparent
- **Equinox Modules** — immutable pytree objects, JIT/vmap compatible
- **float64 always** — spectral methods need double precision
- **MATLAB-validated** — every function tested against MATLAB Chebfun at `rtol ≤ 1e-12`

See [docs/architecture.md](docs/architecture.md) for design decisions.

## Contributing

Each function tracks provenance back to the original MATLAB Chebfun source
(commit `7574c77`), preserving author credits and algorithm references.

This project is a derivative work of [Chebfun](https://www.chebfun.org/),
licensed under BSD-2-Clause. See [LICENSE](LICENSE).

## References

- [Chebfun Guide](https://www.chebfun.org/docs/guide/)
- Trefethen, *Approximation Theory and Approximation Practice* (2013)
- Driscoll, Hale, & Trefethen, "Chebfun Guide" (2014)
