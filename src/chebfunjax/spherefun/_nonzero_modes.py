"""Composition of source sphere operators and pivoted JAX band LU.

Provenance: Chebfun 7574c77680d7e82b79626300bf255498271a72df @spherefun/{poisson,helmholtz}.m.
Zero longitude is deliberately returned separately for its native constraint.
Used by the public Poisson and Helmholtz coefficient solvers.
"""
from functools import partial

import jax
import jax.numpy as jnp

from ._banded_lu import solve_banded


@partial(jax.jit, static_argnames=('lower', 'upper', 'zero_longitude'))
def solve_nonzero_modes(ab, shifts, rhs, *, lower, upper, zero_longitude):
    """Solve independent modes in source order with bounded band workspace.

    Return (indices, solutions, singular, nonfinite), with solutions[:, t]
    corresponding to indices[t]. Callers must reject any flagged result.
    Finite input/output checks complement the LU helper's exact zero pivots;
    neither flag certifies accuracy, which requires independent residual tests.
    Sequential lax.map retains O(m*bandwidth) factorization workspace rather
    than batching all longitude factorizations. No magnitude clipping occurs.
    """
    ab, shifts, rhs = jnp.asarray(ab), jnp.asarray(shifts), jnp.asarray(rhs)
    if rhs.ndim != 2 or shifts.ndim != 1 or rhs.shape[1] != shifts.size:
        raise ValueError('RHS and longitudinal shifts must have matching columns')
    m, n = rhs.shape
    if not 0 <= zero_longitude < n:
        raise ValueError('zero longitude index outside coefficient matrix')
    if ab.shape != (m, 2*lower+upper+1):
        raise ValueError('invalid latitude band storage')
    indices = jnp.concatenate((jnp.arange(zero_longitude-1, -1, -1),
                               jnp.arange(zero_longitude+1, n)))
    dtype = jnp.result_type(ab, shifts, rhs, jnp.float64)
    ab, shifts, rhs = ab.astype(dtype), shifts.astype(dtype), rhs.astype(dtype)

    def solve_one(k):
        matrix = ab.at[:, lower].add(shifts[k])
        b = rhs[:, k]
        x, singular = solve_banded(matrix, b, lower=lower, upper=upper)
        nonfinite = (~jnp.all(jnp.isfinite(matrix)) |
                     ~jnp.all(jnp.isfinite(b)) | ~jnp.all(jnp.isfinite(x)))
        return x, singular, nonfinite

    if n == 1:
        return (indices, jnp.empty((m, 0), dtype=dtype),
                jnp.zeros(0, dtype=bool), jnp.zeros(0, dtype=bool))
    solutions, singular, nonfinite = jax.lax.map(solve_one, indices)
    return indices, solutions.T, singular, nonfinite
