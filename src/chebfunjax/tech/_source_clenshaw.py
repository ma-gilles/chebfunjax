"""Ordered real CPU Clenshaw evaluation, with ordinary JAX differentiation.

Provenance
----------
MATLAB source : @chebtech/clenshaw.m
Chebfun commit: 7574c77 (7574c77680d7e82b79626300bf255498271a72df).
The paired recurrence and final half-times-doubled argument follow that source.
Each recurrence operation occupies its own loop iteration, preventing contraction
without a finite-array reduction and nested loop for every arithmetic operation.
Setup and epilogue retain the existing CPU ordering primitive. No host numerical
provider, degree padding, or input-specific coefficients are used. Native rounding
qualification covers finite ordinary inputs; exceptional/subnormal behavior is
not claimed bitwise native from that corpus.
"""

import jax
import jax.numpy as jnp

from chebfunjax.utils._ordered_binary import binary


def source_clenshaw(coeffs, x):
    """Evaluate real binary64 arrays; caller handles empty/constant series."""
    if coeffs.ndim == 2:
        x = x[..., None]
    x2 = binary(jnp.asarray(2.0, x.dtype), x, 0)
    shape = jnp.broadcast_shapes(x.shape, coeffs.shape[1:])
    zero = jnp.zeros(shape, dtype=coeffs.dtype)
    degree = coeffs.shape[0] - 1

    def step(i, state):
        b1, b2 = state
        k = degree - 2 * i

        def instruction(registers, opcode):
            first, second, temporary = registers
            # A dynamic instruction selects exactly one source operation per
            # iteration. The loop carry separates multiply/add/subtract even
            # after compilation; an optimization barrier alone does not suffice.
            operations = (
                lambda: (first, second, x2 * first),
                lambda: (first, second, coeffs[k] + temporary),
                lambda: (first, temporary - second, temporary),
                lambda: (first, second, x2 * second),
                lambda: (first, second, coeffs[k - 1] + temporary),
                lambda: (temporary - first, second, temporary),
            )
            return jax.lax.switch(opcode, operations), None

        # The prior binary helper performed the operation followed by multiply
        # by one for finite operands, or the same operation for nonfinite ones.
        # The instruction boundary makes that identity step unnecessary here.
        # No finiteness assumption is made about inputs or intermediate values.
        (b1, b2, _), _ = jax.lax.scan(
            instruction, (b1, b2, zero), jnp.arange(6, dtype=jnp.int32),
            unroll=False,
        )
        return b1, b2

    b1, b2 = jax.lax.fori_loop(0, degree // 2, step, (zero, zero))
    if degree % 2:
        b1, b2 = binary(binary(coeffs[1], binary(x2, b1, 0), 1), b2, 2), b1
    half_x = binary(jnp.asarray(0.5, x.dtype), x2, 0)
    return binary(binary(coeffs[0], binary(half_x, b1, 0), 1), b2, 2)
