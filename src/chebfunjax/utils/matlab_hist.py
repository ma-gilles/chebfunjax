"""Bounded MATLAB legacy hist counts for finite real vector data.

Provenance
----------
MATLAB source : R2017a toolbox/matlab/datafun/hist.m, lines89-126
Chebfun example: stats/ResamplingRandomVariables.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
from __future__ import annotations

import operator

import jax
import jax.numpy as jnp


def _ordered_finite_keys(values):
    """Binary64 numeric order; canonicalize both zeros before comparison."""
    bits = jax.lax.bitcast_convert_type(values, jnp.uint64)
    sign = jnp.uint64(1 << 63)
    magnitude = bits & (sign - jnp.uint64(1))
    bits = jnp.where(magnitude == 0, jnp.uint64(0), bits)
    return jnp.where((bits & sign) != 0, ~bits, bits | sign)


def _shifted_edge_keys(edges):
    """Exact finite binary64 edge+eps(edge) keys, including gradual endpoints.

    Positive edges advance one representable step. Negative normal powers of
    two above the smallest normal advance two steps toward zero, since their
    positive-magnitude spacing is twice the lower-binade spacing. Other negative
    edges advance one step. Both signed zeros shift to the minimum subnormal.
    This is source eps(abs(edge)) addition, not a nextafter-toward-positive rule.
    """
    bits = jax.lax.bitcast_convert_type(edges, jnp.uint64)
    sign = jnp.uint64(1 << 63)
    magnitude = bits & (sign - jnp.uint64(1))
    exponent = magnitude >> jnp.uint64(52)
    fraction = magnitude & jnp.uint64((1 << 52) - 1)
    two_steps = (exponent > 1) & (fraction == 0)
    decrement = jnp.where(two_steps, jnp.uint64(2), jnp.uint64(1))
    lower = magnitude - decrement
    negative_result = jnp.where(lower == 0, jnp.uint64(0), sign | lower)
    shifted = jnp.where((bits & sign) != 0, negative_result, magnitude + jnp.uint64(1))
    shifted = jnp.where(magnitude == 0, jnp.uint64(1), shifted)
    # Convert keys directly: no floating operation may flush subnormal words.
    return jnp.where((shifted & sign) != 0, ~shifted, shifted | sign)


@jax.jit
def _source_divide(numerator, divisor):
    """Source binary64 division by a scalar within the histogram edge grid.

    XLA rewrites division by a broadcast scalar into multiplication by its
    rounded reciprocal, moving exact source histogram edges by one ulp.
    A barrier on the full broadcast operand retains vector division. This
    private eager-histogram adapter uses default compiler options and makes
    no general correctly-rounded arithmetic or AD promise.
    """
    numerator = jnp.asarray(numerator, dtype=jnp.float64)
    divisor = jnp.asarray(divisor, dtype=jnp.float64)
    if divisor.ndim != 0:
        raise ValueError('source histogram division requires a scalar divisor')
    denominator = jax.lax.optimization_barrier(
        jnp.broadcast_to(divisor, numerator.shape))
    return numerator / denominator


def _source_linspace(left, right, count):
    """R2017a linspace edge grid, retaining source operation order.

    Eager scalar branching matches this adapter's validation contract. Extreme
    subnormal grid arithmetic remains platform-dependent; bit-key classification
    below separately preserves already constructed subnormal edge/data words.
    """
    k = jnp.arange(count + 1, dtype=jnp.float64)
    delta = right - left
    check = delta * (count - 1)
    if bool(jnp.isinf(check)):
        if bool(jnp.isinf(delta)):
            values = left + _source_divide(right, count) * k - _source_divide(left, count) * k
        else:
            values = left + k * _source_divide(delta, count)
    else:
        values = left + _source_divide(k * delta, count)
    if bool(left == right):
        values = jnp.full((count + 1,), left, dtype=jnp.float64)
    else:
        values = values.at[0].set(left).at[-1].set(right)
    return values


def matlab_hist_counts(values, bins=10):
    """Return legacy scalar-bin hist counts and centers using JAX arithmetic.

    Supports nonempty finite real vector data and a positive integer bin count.
    Other MATLAB hist overloads are deliberately outside this adapter. The
    validation is eager; no whole-function JIT/AD contract is claimed. Interior
    edges are nudged by MATLAB eps(edge), assigning exact midpoint hits left.

    Provenance
    ----------
    MATLAB source : R2017a toolbox/matlab/datafun/hist.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    count = operator.index(bins)
    if isinstance(bins, bool) or count < 1:
        raise ValueError('bins must be a positive integer')
    data = jnp.asarray(values)
    if data.ndim != 1 or data.size == 0 or jnp.iscomplexobj(data):
        raise ValueError('hist adapter requires a nonempty real vector')
    data = data.astype(jnp.float64)
    if not bool(jnp.all(jnp.isfinite(data))):
        raise ValueError('hist adapter requires finite values')
    lo, hi = jnp.min(data), jnp.max(data)
    equal = lo == hi
    left = jnp.where(equal, lo - count//2 - .5, lo)
    right = jnp.where(equal, hi + (count+1)//2 - .5, hi)
    edges = _source_linspace(left, right, count)
    width = edges[1] - edges[0]
    centers = edges[:-1] + width/2
    interior = edges[1:-1]
    positions = jnp.searchsorted(_shifted_edge_keys(interior),
                                 _ordered_finite_keys(data), side='right')
    counts = jnp.bincount(positions, length=count)
    return counts, centers
