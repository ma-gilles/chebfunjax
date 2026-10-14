"""Preserve individual source binary64 operations under CPU compilation.

The scan schedule prevents reciprocal rewrites and fused consumer operations.
Shared by source Chebdouble/PDE arithmetic, Diskfun rank-one updates and
Chebfun3 ACA. Operation codes: multiply0, add1, subtract2, divide3.
Other dtypes/backends and nonfinite operands retain ordinary JAX arithmetic.

Provenance: Chebfun pin7574c77680d7e82b79626300bf255498271a72df,
@chebdouble/chebdouble.m; @diskfun/constructor.m;
@chebfun3/chebfun3classic.m and chebfun3double.m completeACA2D.
"""

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("operation",))
def binary(a, b, operation):
    generic = (jnp.multiply, jnp.add, jnp.subtract, jnp.divide)[operation]
    a, b = jnp.broadcast_arrays(a, b)
    if a.dtype != jnp.float64 or b.dtype != jnp.float64:
        return generic(a, b)

    def ordered(a, b):
        operands = jnp.stack((b, jnp.ones_like(b)))
        opcodes = jnp.asarray([operation, 0], jnp.int32)

        def step(value, instruction):
            opcode, operand = instruction
            result = jax.lax.switch(
                opcode,
                (
                    lambda: value * operand,
                    lambda: value + operand,
                    lambda: value - operand,
                    lambda: value / operand,
                ),
            )
            return result, None

        # Multiplication by one preserves signed zeros; its separate iteration
        # also keeps this operation separate from its consumer under JIT.
        return jax.lax.scan(step, a, (opcodes, operands), unroll=False)[0]

    def cpu(a, b):
        finite = jnp.all(jnp.isfinite(a)) & jnp.all(jnp.isfinite(b))
        return jax.lax.cond(finite, ordered, generic, a, b)

    return jax.lax.platform_dependent(a, b, cpu=cpu, default=generic)
