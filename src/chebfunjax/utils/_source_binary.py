"""Hermite-private materialization of source real binary operations.

Provenance: Chebfun hermpts.m, pinned commit
7574c77680d7e82b79626300bf255498271a72df. The scan schedule is derived
from chebfunjax.utils._ordered_binary.binary; see that module for its
shared source-operation materialization contract. This private version
selects the original arithmetic statically while preserving the finite
predicate, operands, two scan iterations and identity multiply. The shared
helper remains unchanged. Other dtypes/backends and nonfinite operands
retain the original ordinary JAX fallback.
"""

from functools import partial

import jax
import jax.numpy as j
from jax import lax


@partial(jax.jit, static_argnames=("operation",))
def binary(a, b, operation):
    generic = (j.multiply, j.add, j.subtract, j.divide)[operation]
    a, b = j.broadcast_arrays(a, b)
    if a.dtype != j.float64 or b.dtype != j.float64:
        return generic(a, b)

    def ordered(a, b):
        operands = j.stack((b, j.ones_like(b)))
        identities = j.asarray([False, True])

        def step(value, instruction):
            identity, operand = instruction
            result = lax.cond(identity, lambda: value * operand, lambda: generic(value, operand))
            return result, None

        return lax.scan(step, a, (identities, operands), unroll=False)[0]

    def cpu(a, b):
        finite = j.all(j.isfinite(a)) & j.all(j.isfinite(b))
        return lax.cond(finite, ordered, generic, a, b)

    return lax.platform_dependent(a, b, cpu=cpu, default=generic)
