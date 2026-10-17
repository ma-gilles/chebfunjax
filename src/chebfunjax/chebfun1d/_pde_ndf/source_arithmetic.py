"""Scoped CPU arithmetic for the native chebdouble/PDE value path.

JAX-only adapter of native binary operations and the qualified MATLAB R2025b
oneMKL 2024.1 default DGEMV path, Chebfun pin 7574c77.
The native chebdouble.diff source scales the result of D*values.
General Chebfun coefficient differentiation does
not use this module. Qualification is CPU binary64 finite matrices up to514rows/columns.

Provenance
----------
Chebfun pin 7574c77680d7e82b79626300bf255498271a72df:
@chebdouble/chebdouble.m (diff and binary value methods).
Native CPU accumulation qualification: MATLAB R2025b Update 2,
oneMKL 2024.1 build 20240215, mkl_blas_def_xdgemv. See
docs/pde15s_cpu_source_arithmetic_20261011.json for retained evidence.
"""

import jax
import jax.numpy as jnp

from chebfunjax.utils._ordered_binary import binary as binary  # compatibility re-export


def _serial_gemv(matrix, vector):
    zero = jnp.zeros(matrix.shape[0], dtype=matrix.dtype)

    def step(index, state):
        total, product = state
        column = index // 2
        return jax.lax.cond(
            index % 2 == 0,
            lambda: (total, matrix[:, column] * vector[column]),
            lambda: (total + product, product),
        )

    # Separate iterations prevent multiply-add contraction. A barrier around
    # the product inside one iteration does not preserve these native bits.
    return jax.lax.fori_loop(0, 2 * vector.size, step, (zero, zero))[0]


@jax.jit
def matvec(matrix, vector):
    generic = lambda a, x: a @ x  # noqa: E731 - preserve the traced generic fallback
    if (
        matrix.ndim != 2
        or vector.ndim != 1
        or matrix.shape[1] != vector.size
        or matrix.dtype != jnp.float64
        or vector.dtype != jnp.float64
        or not 2 <= matrix.shape[0] <= 514
        or not 2 <= vector.size <= 514
    ):
        return generic(matrix, vector)

    def cpu(a, x):
        finite = jnp.all(jnp.isfinite(a)) & jnp.all(jnp.isfinite(x))
        return jax.lax.cond(finite, _serial_gemv, generic, a, x)

    return jax.lax.platform_dependent(matrix, vector, cpu=cpu, default=generic)
