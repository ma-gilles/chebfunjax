"""Source controls for mixed Quasimatrix/array-Chebfun arithmetic.

Pinned Chebfun source: commit 7574c77; @chebfun/plus.m lines 82-115,
@chebfun/minus.m (unary negation then plus), @chebfun/times.m lines 65-101,
and @chebfun/dimCheck.m. MATLAB turns Quasimatrix and array-valued Chebfun
operands into column cells, applies singleton-column expansion, and combines
corresponding scalar columns. Columns may have different interior breakpoints
but share the same outer domain. At stored breakpoints, the binary result uses
the operation on the operands' pointValues.

These independent polynomial controls use analytic formulas, not sampled
MATLAB exports. Each value bound is 100 float64 eps times
max(1, max(abs(expected))). The controls intentionally
asserts the scalar-column Quasimatrix output shape so a list of accidentally
nested 3-vector Chebfuns cannot appear numerically plausible.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, cell2quasi
from chebfunjax.domain import Domain

_EPS = float(jnp.finfo(jnp.float64).eps)
_TOL_FACTOR = 100.0
_X = jnp.asarray([-0.83, -0.22, 0.41, 0.92])


def _column(fn, breaks):
    return Chebfun.from_function(fn, domain=Domain(tuple(breaks)))


def _array_chebfun3():
    return Chebfun.from_function(
        lambda x: jnp.stack((2.0 + x, 1.0 + 2.0 * x, 3.0 - x), axis=-1),
        domain=Domain((-1.0, -0.7, 0.25, 0.4, 1.0)),
    )


def _quasi3():
    # Deliberately different interior partitions for all three scalar columns.
    cols = [
        _column(lambda x: 1.0 + x, (-1.0, -0.45, 0.25, 1.0)),
        _column(lambda x: 2.0 - x, (-1.0, -0.6, 0.25, 0.65, 1.0)),
        _column(lambda x: 1.0 + x**2, (-1.0, -0.15, 0.25, 1.0)),
    ]
    return cell2quasi(cols)


def _quasi1():
    return cell2quasi([_column(lambda x: 1.0 + 2.0 * x, (-1.0, 0.25, 1.0))])


def _expected_quasi3(x):
    return jnp.stack((1.0 + x, 2.0 - x, 1.0 + x**2), axis=-1)


def _expected_array3(x):
    return jnp.stack((2.0 + x, 1.0 + 2.0 * x, 3.0 - x), axis=-1)


def _assert_values(actual, expected):
    assert actual.shape == expected.shape
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(expected)))
    assert jnp.max(jnp.abs(actual - expected)) <= _TOL_FACTOR * _EPS * scale


def _op(name, a, b):
    if name == "plus":
        return a + b
    if name == "minus":
        return a - b
    if name == "times":
        return a * b
    raise AssertionError(name)


@pytest.mark.parametrize("name", ["plus", "minus", "times"])
@pytest.mark.parametrize("quasi_first", [True, False])
def test_three_matched_columns_quasi_array_chebfun(name, quasi_first):
    q = _quasi3()
    c = _array_chebfun3()
    result = _op(name, q, c) if quasi_first else _op(name, c, q)
    qv, cv = _expected_quasi3(_X), _expected_array3(_X)
    expected = {
        "plus": qv + cv,
        "minus": qv - cv if quasi_first else cv - qv,
        "times": qv * cv,
    }[name]
    _assert_values(result(_X), expected)
    assert result.n_cols == 3
    assert all(column.n_columns == 1 for column in result.cols)


@pytest.mark.parametrize("name", ["plus", "minus", "times"])
@pytest.mark.parametrize("quasi_first", [True, False])
def test_singleton_quasi_column_expands_against_array_chebfun(name, quasi_first):
    q = _quasi1()
    c = _array_chebfun3()
    result = _op(name, q, c) if quasi_first else _op(name, c, q)
    qv = 1.0 + 2.0 * _X
    cv = _expected_array3(_X)
    qv = jnp.broadcast_to(qv[..., None], cv.shape)
    expected = {
        "plus": qv + cv,
        "minus": qv - cv if quasi_first else cv - qv,
        "times": qv * cv,
    }[name]
    _assert_values(result(_X), expected)
    assert result.n_cols == 3
    assert all(column.n_columns == 1 for column in result.cols)


@pytest.mark.parametrize("name", ["plus", "minus", "times"])
@pytest.mark.parametrize("quasi_first", [True, False])
def test_scalar_chebfun_broadcasts_against_three_column_quasimatrix(name, quasi_first):
    q = _quasi3()
    scalar = _column(lambda x: 0.5 - x, (-1.0, 0.25, 1.0))
    result = _op(name, q, scalar) if quasi_first else _op(name, scalar, q)
    qv = _expected_quasi3(_X)
    sv = jnp.broadcast_to((0.5 - _X)[..., None], qv.shape)
    expected = {
        "plus": qv + sv,
        "minus": qv - sv if quasi_first else sv - qv,
        "times": qv * sv,
    }[name]
    _assert_values(result(_X), expected)
    assert result.n_cols == 3
    assert all(column.n_columns == 1 for column in result.cols)


def test_matched_columns_union_breaks_and_apply_stored_point_values():
    qcols = [
        _column(lambda x: 1.0 + x, (-1.0, -0.45, 0.25, 1.0)),
        _column(lambda x: 2.0 - x, (-1.0, -0.6, 0.25, 0.65, 1.0)),
        _column(lambda x: 1.0 + x**2, (-1.0, -0.15, 0.25, 1.0)),
    ]
    qcols = [col.define_point(0.25, val) for col, val in zip(qcols, (10.0, 20.0, 30.0))]
    q = cell2quasi(qcols)
    c = _array_chebfun3().define_point(0.25, [40.0, 50.0, 60.0])

    result = q + c
    # The smooth formulas at x=.25 are deliberately distinct from the stored
    # pointValues. The operation must combine the stored values columnwise.
    _assert_values(result(jnp.asarray([0.25])), jnp.asarray([[50.0, 70.0, 90.0]]))
    assert result.n_cols == 3
    assert all(column.n_columns == 1 for column in result.cols)


@pytest.mark.parametrize("name", ["plus", "minus", "times"])
@pytest.mark.parametrize("singleton_first", [True, False])
def test_quasimatrix_singleton_expansion(name, singleton_first):
    q, scalar = _quasi3(), _quasi1()
    result = _op(name, scalar, q) if singleton_first else _op(name, q, scalar)
    qv = _expected_quasi3(_X)
    sv = jnp.broadcast_to((1.0 + 2.0 * _X)[..., None], qv.shape)
    expected = {"plus": sv + qv,
                "minus": sv - qv if singleton_first else qv - sv,
                "times": sv * qv}[name]
    _assert_values(result(_X), expected)
    assert result.n_cols == 3
    assert all(column.n_columns == 1 for column in result.cols)


@pytest.mark.parametrize("name", ["plus", "minus", "times"])
def test_mixed_columns_reject_incompatible_widths(name):
    c = Chebfun.from_function(lambda x: jnp.stack((x, x**2), axis=-1),
                              domain=Domain((-1.0, 1.0)))
    with pytest.raises(ValueError, match="column counts"):
        _op(name, _quasi3(), c)


@pytest.mark.parametrize("name", ["plus", "minus", "times"])
def test_mixed_columns_reject_different_outer_domains(name):
    c = Chebfun.from_function(lambda x: jnp.stack((x, x**2, x**3), axis=-1),
                              domain=Domain((-2.0, 1.0)))
    with pytest.raises(ValueError):
        _op(name, _quasi3(), c)


@pytest.mark.parametrize("name", ["plus", "minus", "times"])
def test_mixed_columns_reject_transposed_operand(name):
    with pytest.raises(ValueError, match="transposed"):
        _op(name, _quasi3(), _array_chebfun3().transpose())


@pytest.mark.parametrize("name", ["plus", "minus", "times"])
@pytest.mark.parametrize("quasi_first", [True, False])
def test_scalar_chebfun_broadcast_preserves_stored_point_value(name, quasi_first):
    q = _quasi3()
    scalar = _column(lambda x: 0.5 - x, (-1.0, 0.25, 1.0)).define_point(0.25, 40.0)
    result = _op(name, q, scalar) if quasi_first else _op(name, scalar, q)
    x = jnp.asarray([0.25])
    qv = _expected_quasi3(x)
    sv = jnp.full(qv.shape, 40.0)
    expected = {"plus": qv + sv,
                "minus": qv - sv if quasi_first else sv - qv,
                "times": qv * sv}[name]
    _assert_values(result(x), expected)
    assert result.n_cols == 3
    assert all(column.n_columns == 1 for column in result.cols)
