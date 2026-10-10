"""Common-coefficient BMC-I projection; source array-valued factor semantics.

Provenance
----------
MATLAB source : @spherefun/projectOntoBMCI.m; @trigtech/{real,populate,
    prolong,simplify,extractColumns}.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
from functools import partial

import jax
import jax.numpy as jnp

from chebfunjax.spherefun._factor_assembly import (
    select_factor_columns,
    stack_factor_coefficients,
)
from chebfunjax.tech.trigtech import (
    Trigtech,
    _trig_coeffs2vals_impl,
    _trig_prolong_coeffs,
    _trig_source_pairs_for_chop,
    _trig_vals2coeffs_impl,
)
from chebfunjax.utils.misc import standard_chop


def _stack(techs):
    return stack_factor_coefficients(techs)


@partial(jax.jit, static_argnames=("even", "nonzero_poles"))
def _columns(coeffs, even, nonzero_poles):
    original_n = coeffs.shape[0]
    x = coeffs
    if original_n % 2 == 0:
        x = jnp.concatenate((x[:1] * .5, x[1:], x[:1] * .5), axis=0)
    m = x.shape[0]
    if even:
        x = x - jax.lax.optimization_barrier(.5 * (x - x[::-1]))
        start = 1 if nonzero_poles else 0
        if start < x.shape[1]:
            y = x[:, start:]
            y = y.at[::2].add(jax.lax.optimization_barrier(
                -(2.0 / (m + 1)) * jnp.sum(y[::2], axis=0)))
            if m > 1:
                y = y.at[1::2].add(jax.lax.optimization_barrier(
                    -(2.0 / (m - 1)) * jnp.sum(y[1::2], axis=0)))
            x = x.at[:, start:].set(y)
    else:
        x = x - jax.lax.optimization_barrier(.5 * (x + x[::-1]))
    if original_n % 2 == 0:
        x = x.at[0].set(x[0] + x[-1])[:-1]
    return x


@partial(jax.jit, static_argnames=("even",))
def _rows(coeffs, even):
    modes = jnp.arange(coeffs.shape[0]) - coeffs.shape[0] // 2
    remove = (modes % 2 != 0) if even else (modes % 2 == 0)
    return jnp.where(remove[:, None], 0, coeffs)


@jax.jit
def _real_source_values(coeffs):
    """Source populate values and aggregate real flag in one reusable kernel.

    MATLAB @trigtech/populate.m at 7574c77 uses 3*(eps*vscale).
    The barrier expresses the source multiplication boundary; qualification
    is limited to inspected/tested CPU behavior, not arbitrary compiler output.
    """
    values = _trig_coeffs2vals_impl(coeffs)
    eps = jnp.finfo(jnp.float64).eps
    scale = jnp.max(jnp.abs(values), axis=0)
    threshold = 3 * jax.lax.optimization_barrier(eps * scale)
    flags = jnp.max(jnp.abs(jnp.imag(values)), axis=0) <= threshold
    return values, jnp.all(flags)


def _real_source(coeffs):
    # populate({'',coeffs}) stores original coefficients while classifying
    # each value column with3eps. real() returns unchanged if ALL are real.
    values, all_real = _real_source_values(coeffs)
    if bool(all_real):
        return coeffs
    values = jnp.real(values)
    if not bool(jnp.any(values != 0)):
        return jnp.zeros((1, coeffs.shape[1]), dtype=coeffs.dtype)
    coeffs = _trig_vals2coeffs_impl(values)
    return _simplify_coefficients(coeffs, None)


def _simplify_coefficients(coeffs, tolerances):
    """Literal matrix trigtech simplify, including max-column cutoff."""
    nold = coeffs.shape[0]
    padded_n = max(17, int(nold * 1.25 + 5 + .5))
    padded = _trig_prolong_coeffs(coeffs, padded_n)
    noisy = _trig_vals2coeffs_impl(_trig_coeffs2vals_impl(jnp.abs(padded[::-1])))
    if tolerances is None:
        tolerances = jnp.full((coeffs.shape[1],), jnp.finfo(jnp.float64).eps)
    else:
        tolerances = jnp.ravel(jnp.asarray(tolerances))
        if tolerances.size != coeffs.shape[1]:
            tolerances = jnp.full((coeffs.shape[1],), jnp.max(tolerances))
    cutoff = max(standard_chop(_trig_source_pairs_for_chop(noisy[:, j]),
                               float(tolerances[j]))
                 for j in range(noisy.shape[1]))
    cutoff = min(cutoff, nold)
    keep = cutoff + 1 if cutoff % 2 == 0 else cutoff
    return _trig_prolong_coeffs(padded, keep)


def simplify_factors(techs, tol=None):
    """Array-valued globaltol simplify before splitting factor storage.

    Provenance
    ----------
    MATLAB source : @separableApprox/simplify.m, @chebfun/simplify.m,
        @trigtech/simplify.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    if not techs or not all(t.ishappy for t in techs):
        return list(techs)
    coefficients = _stack(techs)
    values = _trig_coeffs2vals_impl(coefficients)
    # Source prolong/populate keep real column values real for their vscale.
    real_flags = jnp.asarray([t.is_real for t in techs])
    values = jnp.where(real_flags[None, :], jnp.real(values), values)
    local = jnp.max(jnp.abs(values), axis=0)
    base = jnp.finfo(jnp.float64).eps if tol is None else tol
    tolerances = jnp.asarray(base) * jnp.max(local) / local
    result = _simplify_coefficients(coefficients, tolerances)
    return [Trigtech(coeffs=result[:, j], real_columns=t.real_columns, ishappy=True)
            for j, t in enumerate(techs)]


def factors_from_values(values):
    """Keep full PhaseTwo matrix coefficients until outer simplify.

    Provenance
    ----------
    MATLAB source : @spherefun/constructor.m (PhaseTwo outputs)
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    coefficients = _trig_vals2coeffs_impl(jnp.asarray(values, dtype=jnp.float64))
    return [Trigtech(coeffs=coefficients[:, j], is_real=True, ishappy=True)
            for j in range(coefficients.shape[1])]


def project(f):
    """Source projection with full-factor common lengths before extraction.

    Provenance
    ----------
    MATLAB source : @spherefun/projectOntoBMCI.m; @trigtech/extractColumns.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    from chebfunjax.spherefun.spherefun import Spherefun
    if not f.cols:
        return f
    col_coeffs, row_coeffs = _stack(f.cols), _stack(f.rows)
    cols, rows = list(f.cols), list(f.rows)
    for indices, even in ((f.idx_plus, True), (f.idx_minus, False)):
        if not indices:
            continue
        index = jnp.asarray(indices, dtype=jnp.int32)
        c = _real_source(_columns(select_factor_columns(col_coeffs, index), even, f.nonzero_poles and even))
        r = _real_source(_rows(select_factor_columns(row_coeffs, index), even))
        for j, i in enumerate(indices):
            cols[i] = Trigtech(coeffs=c[:, j], is_real=True, ishappy=True)
            rows[i] = Trigtech(coeffs=r[:, j], is_real=True, ishappy=True)
    return Spherefun(cols=cols, rows=rows, pivots=f.pivots,
                     idx_plus=f.idx_plus, idx_minus=f.idx_minus,
                     pivot_locations=f.pivot_locations, nonzero_poles=f.nonzero_poles)
