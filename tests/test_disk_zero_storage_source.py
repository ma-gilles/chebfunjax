"""Actual native public zero-result storage and observables, real finite factors.

Native rank-two scalar-zero retains stale parity indices; resulting sum2 or
partition errors are recorded source behavior, not silently sanitized here.
Python rank spells native one-output length; length() spells two-output length.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.diskfun.diskfun import Diskfun
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.tech.trigtech import Trigtech

_FIXTURES = Path(__file__).parent / 'fixtures'
_PLUS = json.loads((_FIXTURES / 'disk_zero_poles_native_20261018.json').read_text())['cases']
_SCALE = json.loads((_FIXTURES / 'disk_zero_scale_native_20261018.json').read_text())['cases']


def _array(record):
    def component(key):
        bits = jnp.asarray([int(w, 16) for w in record[key]], dtype=jnp.uint64)
        values = jax.lax.bitcast_convert_type(bits, jnp.float64)
        # MATLAB num2hex flattened column-major; all captured arrays are 2D.
        return values.reshape(tuple(record['shape'][::-1])).T

    real = component('real')
    return real if record['isreal'] else jax.lax.complex(real, component('imag'))


def _indices(values):
    values = values if isinstance(values, list) else [values]
    return tuple(v - 1 for v in values)


def _object(record):
    cols, rows = _array(record['cols']), _array(record['rows'])
    pivots = _array(record['pivot']).reshape(-1)
    return Diskfun(
        cols=[Chebtech2.from_coeffs(cols[:, i]) for i in range(cols.shape[1])],
        rows=[Trigtech.from_coeffs(rows[:, i], is_real=True) for i in range(rows.shape[1])],
        pivots=pivots,
        idx_plus=_indices(record['idxPlus']),
        idx_minus=_indices(record['idxMinus']),
        nonzero_poles=bool(record['nonZeroPoles']),
        pivot_locations=tuple((0., 0.) for _ in range(pivots.size)),
    )


def _assert_words(actual, expected):
    expected_values = _array(expected)
    actual = jnp.asarray(actual).reshape(expected_values.shape)
    assert actual.shape == expected_values.shape
    for part in [jnp.real, jnp.imag]:
        a = jax.lax.bitcast_convert_type(part(actual), jnp.uint64)
        b = jax.lax.bitcast_convert_type(part(expected_values), jnp.uint64)
        assert jnp.array_equal(a, b)


def _assert_storage(f, native):
    assert isinstance(f, Diskfun)
    assert len(f.cols) == len(f.rows) == native['term_count'] == 1
    assert isinstance(f.cols[0], Chebtech2) and isinstance(f.rows[0], Trigtech)
    assert f.cols[0].coeffs.shape[0] == native['cols']['shape'][0]
    assert f.rows[0].coeffs.shape[0] == native['rows']['shape'][0]
    _assert_words(f.cols[0].coeffs, native['cols'])
    _assert_words(f.rows[0].coeffs, native['rows'])
    _assert_words(f.pivots, native['pivot'])
    assert f.idx_plus == _indices(native['idxPlus'])
    assert f.idx_minus == _indices(native['idxMinus'])
    assert f.nonzero_poles == bool(native['nonZeroPoles'])
    assert f.rank == 1  # native one-output length, not numerical rank
    assert f.length() == (native['rows']['shape'][0], native['cols']['shape'][0])


def _assert_zero_observables(f):
    assert not f.isempty() and f.iszero()
    assert f.numerical_rank() == 0
    assert f.vscale() == 0 and f.norm() == 0
    x, y = jnp.asarray([-.2, 0., .3, .9]), jnp.asarray([.1, 0., -.2, 0.])
    assert jnp.array_equal(f.feval_cart(x, y), jnp.zeros(4))
    assert jnp.array_equal(f.coeffs2(), jnp.zeros_like(f.coeffs2()))


@pytest.mark.parametrize('case', range(4))
def test_actual_native_public_plus_zero_storage(case):
    native = _PLUS[case]
    f = _object(native['input_a']) + _object(native['input_b'])
    _assert_storage(f, native['output'])
    _assert_zero_observables(f)
    assert f.sum2() == 0


@pytest.mark.parametrize('case', range(4))
@pytest.mark.parametrize('operation', ['scale', 'negative_zero_scale', 'equality_plus'])
def test_actual_native_zero_scale_and_equality(case, operation):
    native = _SCALE[case]
    f = _object(native['input'])
    if operation == 'scale':
        actual, expected = 0. * f, native['scaled']
    elif operation == 'negative_zero_scale':
        actual, expected = -0. * f, native['negative_zero_scaled']
    else:
        actual, expected = f + (-f), native['cancelled']
    _assert_storage(actual, expected)
    _assert_zero_observables(actual)
    if expected['sum2']['ok']:
        assert actual.sum2() == 0
    else:
        # Native stale indices also raise (MATLAB badsubscript); no exception
        # text/identifier equality is claimed across the Python adapter.
        with pytest.raises(IndexError):
            actual.sum2()
    if expected['partition']['ok']:
        plus, minus = actual.partition()
        assert plus.isempty() == expected['partition']['plus_empty']
        assert minus.isempty() == expected['partition']['minus_empty']
    else:
        with pytest.raises(IndexError):
            actual.partition()


def test_nonempty_plus_discards_cancelled_pole_at_source_function_bound():
    native = _PLUS[4]
    h = _object(native['input_a']) + _object(native['input_b'])
    assert not h.nonzero_poles and len(h.cols) == 1
    theta, radius = jnp.asarray([-.7, 0., .3, 2.]), jnp.asarray([.1, .25, .7, 1.])
    expected = 1.5 * radius**2 * jnp.cos(2 * theta)
    # tests/diskfun/test_feval.m source bound; no vscale or fitted tolerance.
    assert jnp.linalg.norm(h(theta, radius) - expected) < 1000 * jnp.finfo(jnp.float64).eps


def test_public_cancelled_pole_polar_coordinate_derivatives():
    native = _PLUS[1]
    h = _object(native['input_a']) + _object(native['input_b'])

    def value(p):
        return h(p[0], p[1])

    point = jnp.asarray([.3, .4])
    actual, gradient, hessian = jax.jit(lambda p: (value(p), jax.grad(value)(p), jax.hessian(value)(p)))(point)
    assert actual == 0
    assert jnp.array_equal(gradient, jnp.zeros(2))
    assert jnp.array_equal(hessian, jnp.zeros((2, 2)))


@pytest.mark.parametrize('zero_side', ['left', 'right', 'both'])
def test_source_zero_remainder_aliases_preserve_other_block(zero_side):
    one = Trigtech.from_coeffs(jnp.ones(1), is_real=True)

    def operand(pole_value, zero, mode, pivot):
        coeffs = jnp.zeros(2 * mode + 1).at[0].set(.5).at[-1].set(.5)
        row = Trigtech.from_coeffs(coeffs, is_real=True)
        col = Chebtech2.from_coeffs(jnp.zeros(1) if zero else jnp.asarray([.5, 0., .5]))
        f = Diskfun(
            cols=[Chebtech2.from_coeffs(jnp.asarray([pole_value])), col],
            rows=[one, row], pivots=jnp.asarray([1., pivot]),
            idx_plus=(0, 1), idx_minus=(), nonzero_poles=True,
        )
        return f, col, row

    a, ac, ar = operand(1., zero_side != 'right', 2, 3.)
    b, bc, br = operand(2., zero_side != 'left', 4, 2.)
    h = a + b
    # @separableApprox/plus.m returns g when iszero(f), otherwise f when
    # iszero(g). No compression or re-pivoting occurs in these branches.
    col, row, pivot = (ac, ar, 3.) if zero_side == 'right' else (bc, br, 2.)
    assert len(h.cols) == 2 and h.nonzero_poles
    assert h.pivots[1] == pivot
    assert jnp.array_equal(h.cols[1].coeffs, col.coeffs)
    assert jnp.array_equal(h.rows[1].coeffs, row.coeffs)
    theta, radius = jnp.asarray([-.3, .1, .7]), jnp.asarray([.2, .5, .9])
    expected = 3. + col(radius) * row(theta / jnp.pi) / pivot
    assert jnp.linalg.norm(h(theta, radius) - expected) < 1000 * jnp.finfo(jnp.float64).eps
