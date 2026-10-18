"""Ordered real CPU Clenshaw evaluation, with ordinary JAX differentiation.

Provenance
----------
MATLAB source : @chebtech/clenshaw.m
Chebfun commit: 7574c77 (7574c77680d7e82b79626300bf255498271a72df).
The paired recurrence and final half-times-doubled argument follow that source.
Multiplication occupies a separate loop iteration from its following add/subtract
pair, preventing contraction while retaining the source add-then-subtract order.
A packed register buffer and branch-local row reads avoid copying untouched rows.
Independent points are processed in chunks; their recurrence order is unchanged.
Setup and epilogue retain whole-array CPU ordering and finiteness decisions.
No host numerical provider, degree padding, or input-specific coefficients are used.
Native rounding qualification covers ordinary finite inputs; preserving baseline
exceptional/subnormal behavior does not establish native bit identity.
"""

import math

import jax
import jax.numpy as jnp

from chebfunjax.utils._ordered_binary import binary


def source_clenshaw(coeffs, x):
    """Evaluate real binary64 arrays; caller handles empty/constant series."""
    multi = coeffs.ndim == 2
    if multi:
        x = x[..., None]
    x2 = binary(jnp.asarray(2.0, x.dtype), x, 0)
    shape = jnp.broadcast_shapes(x.shape, coeffs.shape[1:])
    degree = coeffs.shape[0] - 1

    def recurrence(points):
        # The temporary row survives coefficient pairs but instruction zero
        # overwrites it before any read in the next pair.
        initial = jnp.zeros((3,) + points.shape, dtype=coeffs.dtype)
        destinations = jnp.asarray([2, 1, 2, 0], dtype=jnp.int32)

        def pair(i, registers):
            k = degree - 2 * i

            def instruction(registers, opcode):
                # Read rows inside each branch so XLA can fuse the required
                # loads with its arithmetic. Each add/subtract pair has no
                # product to contract; the scan carry separates its preceding
                # multiply. The written add-then-subtract order is retained.
                operations = (
                    lambda r: points * r[0],
                    lambda r: (coeffs[k] + r[2]) - r[1],
                    lambda r: points * r[1],
                    lambda r: (coeffs[k - 1] + r[2]) - r[0],
                )
                row = jax.lax.switch(opcode, operations, registers)
                registers = jax.lax.dynamic_update_slice_in_dim(
                    registers, row[None, ...], destinations[opcode], axis=0)
                return registers, None

            registers, _ = jax.lax.scan(
                instruction, registers, jnp.arange(4, dtype=jnp.int32),
                unroll=False)
            return registers

        registers = jax.lax.fori_loop(0, degree // 2, pair, initial)
        return registers[0], registers[1]

    # Only point axes are chunked: component columns stay together. The tail
    # is evaluated at its actual size, without padding. All operations using
    # global finiteness decisions remain outside this recurrence map.
    point_count = math.prod(shape[:-1] if multi else shape)
    columns = coeffs.shape[1:] if multi else ()
    flat = jnp.broadcast_to(x2, shape).reshape((point_count,) + columns)
    chunk = 1024
    full, tail = divmod(point_count, chunk)
    if full:
        blocks = flat[:full * chunk].reshape((full, chunk) + columns)
        b1, b2 = jax.lax.map(recurrence, blocks)
        b1 = b1.reshape((full * chunk,) + columns)
        b2 = b2.reshape((full * chunk,) + columns)
        if tail:
            t1, t2 = recurrence(flat[full * chunk:])
            b1, b2 = jnp.concatenate((b1, t1)), jnp.concatenate((b2, t2))
    else:
        b1, b2 = recurrence(flat)
    b1, b2 = b1.reshape(shape), b2.reshape(shape)
    if degree % 2:
        b1, b2 = binary(binary(coeffs[1], binary(x2, b1, 0), 1), b2, 2), b1
    half_x = binary(jnp.asarray(0.5, x.dtype), x2, 0)
    return binary(binary(coeffs[0], binary(half_x, b1, 0), 1), b2, 2)
