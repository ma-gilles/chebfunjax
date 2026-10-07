"""Literal source Spherefun numerical rank and fixed-shape predicate.

Provenance
----------
MATLAB source : @spherefun/rank.m -> @separableApprox/rank.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
import jax.numpy as jnp


def rank_decision(singular_values, tol=0.0):
    """Fixed-shape source result: integer value and whether MATLAB returns it.

    Empty arrays and a nonzero spectrum with no ratio above tol return
    has_result=False. An identically zero spectrum returns scalar zero.
    No epsilon cutoff, default numerical-rank heuristic, or spectrum cleanup.

    Provenance
    ----------
    MATLAB source : @separableApprox/rank.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    s = jnp.asarray(singular_values)
    if s.shape[0] == 0:
        return jnp.asarray(0, dtype=jnp.int64), jnp.asarray(False)
    zero = jnp.max(s) == 0
    # Safe inactive zero branch does not change source's zero result.
    denominator = jnp.where(zero, jnp.ones_like(s[0]), s[0])
    selected = s / denominator > jnp.asarray(tol)
    last = jnp.max(jnp.where(selected, jnp.arange(s.shape[0], dtype=jnp.int64) + 1, 0))
    return jnp.where(zero, 0, last), zero | jnp.any(selected)


def numerical_rank(f, tol=None):
    """Host wrapper preserving source scalar-or-empty output semantics.

    Scalar real tolerances include negative, NaN and infinite values,
    preserving literal source comparisons. Complex/non-scalar tolerances
    are outside this explicitly supported adapter and rejected after the
    source empty and zero-spectrum early returns.

    Provenance
    ----------
    MATLAB source : @spherefun/rank.m, @separableApprox/rank.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    # Source does not inspect tolerance for empty input or zero spectrum.
    if f.isempty():
        return jnp.empty((0,), dtype=jnp.float64)
    spectrum = f.svd()
    if bool(jnp.max(spectrum) == 0):
        return jnp.asarray(0, dtype=jnp.int64)
    tolerance = jnp.asarray(0.0 if tol is None else tol)
    if tolerance.ndim != 0 or jnp.iscomplexobj(tolerance):
        raise ValueError('numerical_rank requires a real scalar tolerance')
    value, present = rank_decision(spectrum, tolerance)
    # Source has data-dependent scalar/empty output shape. This wrapper
    # is a host API; rank_decision is the separately traceable primitive.
    if not bool(present):
        return jnp.empty((0,), dtype=jnp.float64)
    return value
