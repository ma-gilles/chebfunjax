"""Numeric disk PhaseOne arithmetic on JAX arrays.

Source: @diskfun/constructor.m, Chebfun7574c77680d7e82b79626300bf255498271a72df.
The numeric source rank cap and reciprocal/product/update order are literal.
Python inspects scalar branch metadata; this is not a whole-JIT or AD promise.
IEEE helpers transport binary64 exponents across CPU subnormal flushing.
NaN payloads and generic reduction-provider word equality are unqualified.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from chebfunjax.operators import _qz_ieee as ieee
from chebfunjax.utils._signed_gradual import flip_sign_bits

from ._numeric_extrema import source_abs_extrema


@jax.jit
def _exact_tiny_sum(row):
    """Return eligibility and an order-independent exact binary64 tiny sum.

    Words through2**53 encode units of2**-1074 directly. If the sum of
    absolute units is at most2**53, every partial sum is representable;
    IEEE addition cannot round, regardless of the sum provider's order.
    Saturating nonnegative reduction at2**53+1 cannot overflow uint64:
    each addition is at most2**54+2, for arbitrary static row length.
    """
    row = jnp.asarray(row, jnp.float64).reshape(-1)
    words = lax.bitcast_convert_type(row, jnp.uint64)
    signs = words >> jnp.uint64(63)
    units = words & jnp.uint64(0x7fffffffffffffff)
    limit = jnp.uint64(1 << 53)
    clipped = jnp.minimum(units, jnp.uint64((1 << 53) + 1))

    def saturating_total(values):
        return lax.reduce(
            values, jnp.uint64(0),
            lambda a, b: jnp.minimum(a + b, jnp.uint64((1 << 53) + 1)),
            (0,))

    eligible = (jnp.all(units <= limit)
                & (saturating_total(clipped) <= limit))
    positive = saturating_total(jnp.where(signs == 0, clipped, jnp.uint64(0)))
    negative = saturating_total(jnp.where(signs != 0, clipped, jnp.uint64(0)))
    negative_sum = negative > positive
    magnitude = jnp.where(negative_sum, negative - positive, positive - negative)
    # Native sum preserves -0 for a nonempty all-negative-zero row;
    # mixed zeros and exact nonzero cancellation give +0.
    negative_zeros = (row.size > 0) & jnp.all(words == jnp.uint64(1 << 63))
    sign = jnp.where(negative_sum | negative_zeros,
                     jnp.uint64(1 << 63), jnp.uint64(0))
    return eligible, lax.bitcast_convert_type(magnitude | sign, jnp.float64)


@jax.jit
def _origin_mean(row):
    """Source mean.m computes sum(row)/numel(row), not reciprocal scaling.

    Eligible tiny rows have a proved exact sum. Other rows use JAX's sum
    provider; generic native reduction-order equality remains unqualified.
    The sign of an exact zero mean does not affect PhaseOne's abs test.
    """
    row = jnp.asarray(row, jnp.float64).reshape(-1)
    eligible, exact = _exact_tiny_sum(row)
    total = lax.cond(eligible, lambda _: exact, lambda _: jnp.sum(row), None)
    return ieee.divide(total, jnp.float64(row.size))


class PhaseOneResult(NamedTuple):
    indices: jax.Array
    pivot_array: jax.Array
    remove_pole: bool
    happy: bool
    cols: jax.Array
    pivots: jax.Array
    rows: jax.Array
    plus: list[int]
    minus: list[int]


def _maximum(values):
    _, value, index = source_abs_extrema(values)
    column, row = divmod(int(index), values.shape[0])
    return value, (row, column)


def _pair_extrema(a, b):
    minimum, maximum, _ = source_abs_extrema(jnp.stack((a, b)))
    return minimum, maximum


def _subtract(a, b):
    return ieee.add(a, flip_sign_bits(b))


def _update(matrix, column, row, pivot):
    # Source GE: column * (row * (1/pivot)), then subtraction. Pole removal
    # separately uses row/pivot directly, as written in the source.
    reciprocal = ieee.divide(jnp.float64(1), pivot)
    scaled = ieee.multiply(row, reciprocal)
    outer = ieee.multiply(column[:, None], scaled[None, :])
    return _subtract(matrix, outer)


def phase_one_numeric(values, tol, alpha):
    """Literal factor0 PhaseOne; indices and plus/minus lists are zero based.

    The result includes the pole's source index/pivot-array row. The existing
    numeric caller adapter removes that metadata row before grid conversion.
    Scalar float transfers control comparisons without flushing subnormals;
    all numerical array and scalar arithmetic is JAX binary64.
    """
    values = jnp.asarray(values, jnp.float64)
    m, n = values.shape
    half = n // 2
    if m == 1:
        return PhaseOneResult(
            jnp.asarray([[0, 0]], jnp.int64), jnp.asarray([[1., 0.]]),
            True, True, values[:, :1], jnp.ones(1), jnp.ones((n, 1)), [0], [])

    c, b = values[:, :half], values[:, half:]
    fp = ieee.multiply(jnp.float64(.5), ieee.add(b, c))
    fm = ieee.multiply(jnp.float64(.5), _subtract(b, c))
    pole = _origin_mean(fp[0, :])
    remove_pole = float(ieee.magnitude(pole)) > float(tol)
    rank = 0
    plus, minus, indices, pivot_array = [], [], [], []
    columns, rows, pivots = [], [], []
    if remove_pole:
        _, row_value, pole_index = source_abs_extrema(fp)
        pole_column = int(pole_index) // fp.shape[0]
        pole_row = ieee.multiply(row_value, jnp.ones(half))
        pole_col = fp[:, pole_column]
        scaled = ieee.divide(pole_row, row_value)
        fp = _subtract(fp, ieee.multiply(pole_col[:, None], scaled[None, :]))
        columns.append(jnp.concatenate((pole_col[::-1], pole_col[1:])))
        rows.append(jnp.concatenate((pole_row, pole_row)))
        pivots.append(row_value)
        plus.append(0)
        indices.append((0, pole_column))
        pivot_array.append(jnp.stack((row_value, jnp.float64(0))))
        rank = 1

    fp, fm = fp[1:, :], fm[1:, :]
    maxp, ip = _maximum(fp)
    maxm, im = _maximum(fm)
    if float(maxp) == 0 and float(maxm) == 0 and not remove_pole:
        return PhaseOneResult(
            jnp.asarray([[0, 0]], jnp.int64), jnp.asarray([[0., 0.]]),
            False, True, jnp.zeros((m, 1)), jnp.asarray([jnp.inf]),
            jnp.zeros((n, 1)), [0], [])

    while float(_pair_extrema(maxp, maxm)[1]) > float(tol) and rank < min(m, n):
        row, column = ip if float(maxp) >= float(maxm) else im
        evp, evm = fp[row, column], fm[row, column]
        ap, am = ieee.magnitude(evp), ieee.magnitude(evm)
        indices.append((row + 1, column))
        smaller, larger = _pair_extrema(ap, am)
        coupled = float(larger) <= float(ieee.multiply(jnp.float64(alpha), smaller))
        plus_dominant = float(ap) > float(am)
        if coupled or plus_dominant:
            cp, rp = fp[:, column], fp[row, :]
            fp = _update(fp, cp, rp, evp)
            plus_col = jnp.concatenate((cp[::-1], jnp.zeros(1), cp))
            plus_row = jnp.concatenate((rp, rp))
        if coupled or not plus_dominant:
            cm, rm = fm[:, column], fm[row, :]
            fm = _update(fm, cm, rm, evm)
            minus_col = jnp.concatenate((flip_sign_bits(cm[::-1]), jnp.zeros(1), cm))
            minus_row = jnp.concatenate((flip_sign_bits(rm), rm))
        if coupled:
            if float(ap) >= float(am):
                plus.append(rank)
                minus.append(rank + 1)
                columns.extend((plus_col, minus_col))
                rows.extend((plus_row, minus_row))
                pivots.extend((evp, evm))
            else:
                minus.append(rank)
                plus.append(rank + 1)
                columns.extend((minus_col, plus_col))
                rows.extend((minus_row, plus_row))
                pivots.extend((evm, evp))
            rank += 2
            pivot_array.append(jnp.stack((evp, evm)))
            maxp, ip = _maximum(fp)
            maxm, im = _maximum(fm)
        elif plus_dominant:
            plus.append(rank)
            columns.append(plus_col)
            rows.append(plus_row)
            pivots.append(evp)
            rank += 1
            pivot_array.append(jnp.stack((evp, jnp.float64(0))))
            maxp, ip = _maximum(fp)
        else:
            minus.append(rank)
            columns.append(minus_col)
            rows.append(minus_row)
            pivots.append(evm)
            rank += 1
            pivot_array.append(jnp.stack((jnp.float64(0), evm)))
            maxm, im = _maximum(fm)

    return PhaseOneResult(
        jnp.asarray(indices, jnp.int64).reshape(-1, 2),
        jnp.stack(pivot_array) if pivot_array else jnp.empty((0, 2)),
        remove_pole, float(_pair_extrema(maxp, maxm)[1]) <= float(tol),
        jnp.stack(columns, axis=1) if columns else jnp.zeros((2 * m - 1, 0)),
        jnp.stack(pivots) if pivots else jnp.empty(0),
        jnp.stack(rows, axis=1) if rows else jnp.zeros((n, 0)), plus, minus)
