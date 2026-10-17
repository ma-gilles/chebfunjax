"""Scoped normest left DTRSM alpha1 source schedule.

Algorithm provenance: https://www.netlib.org/blas/dtrsm.f lines269-332,
reference BLAS left/no-transpose and left/transpose substitution loops.
Chebfun consumer: @chebfun/pdeSolve.m, pin7574c77; MATLAB R2025b condest/normest1.
Existing JAX ordered binary helper preserves each multiply/subtract/divide.
This independent transcription does not claim vendor MKL bit parity.
"""

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl

from chebfunjax.utils._ordered_binary import binary


@partial(jax.jit, static_argnames=("lower", "transpose", "unit"))
def _source_triangular(a, b, *, lower, transpose=False, unit=False):
    n = a.shape[0]
    if not transpose:

        def step(q, x):
            k = q if lower else n - 1 - q
            active = x[k] != 0
            value = x[k] if unit else binary(x[k], a[k, k], 3)
            value = jnp.where(active, value, x[k])
            x = x.at[k].set(value)
            product = binary(a[:, k, None], value[None, :], 0)
            updated = binary(x, product, 2)
            rows = jnp.arange(n) > k if lower else jnp.arange(n) < k
            return jnp.where(rows[:, None] & active[None, :], updated, x)

        return jax.lax.fori_loop(0, n, step, b)

    def step(q, x):
        i = n - 1 - q if lower else q

        def prior(k, total):
            term = binary(a[k, i], x[k], 0)
            result = binary(total, term, 2)
            return jnp.where(k > i if lower else k < i, result, total)

        value = jax.lax.fori_loop(0, n, prior, x[i])
        if not unit:
            value = binary(value, a[i, i], 3)
        return x.at[i].set(value)

    return jax.lax.fori_loop(0, n, step, b)


@partial(jax.jit, static_argnames=("lower", "transpose", "unit"))
def triangular_solve(a, b, *, lower, transpose=False, unit=False):
    """Finite CPU binary64 n<=10 normest actions with implicit derivatives.

    Provenance: Netlib DTRSM left N/T substitution. Native R2025b normest
    fixtures select the scoped provider; larger/other types retain JAX solve.
    The operator masks the unused triangle and replaces the unit diagonal,
    so their tangents are excluded. Implicit AD preserves zero-RHS derivatives
    despite the source primal's zero-update shortcut.
    """

    def generic(a, b):
        return jl.solve_triangular(
            a, b, lower=lower, unit_diagonal=unit, trans="T" if transpose else "N"
        )

    if (
        a.ndim != 2
        or a.shape[0] != a.shape[1]
        or b.ndim not in (1, 2)
        or b.shape[0] != a.shape[0]
        or not 1 <= a.shape[0] <= 10
        or a.dtype != jnp.float64
        or b.dtype != jnp.float64
    ):
        return generic(a, b)

    def cpu(a, b):
        masked = jnp.tril(a) if lower else jnp.triu(a)
        if unit:
            masked = masked.at[jnp.diag_indices(a.shape[0])].set(1.0)
        operator = masked.T if transpose else masked

        def owned(a, b):
            def apply(rhs, trans):
                matrix_rhs = rhs[:, None] if rhs.ndim == 1 else rhs
                result = _source_triangular(a, matrix_rhs, lower=lower, transpose=trans, unit=unit)
                return result[:, 0] if rhs.ndim == 1 else result

            return jax.lax.custom_linear_solve(
                lambda x: operator @ x,
                b,
                solve=lambda _, rhs: apply(rhs, transpose),
                transpose_solve=lambda _, rhs: apply(rhs, not transpose),
            )

        finite = jnp.all(jnp.isfinite(masked)) & jnp.all(jnp.isfinite(b))
        return jax.lax.cond(finite, owned, generic, a, b)

    return jax.lax.platform_dependent(a, b, cpu=cpu, default=generic)
