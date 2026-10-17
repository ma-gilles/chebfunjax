"""Source factory branches, including the literal cell2quasi predicate.

Provenance: quasimatrix.m, @chebfun/cheb2quasi.m and
tests/chebfun/test_cell2quasi.m, Chebfun 7574c77.
Except for the named original predicate these are authored source controls.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.domain import Domain


def _op(x):
    return jnp.stack((x, x**2), axis=-1)


def test_cell2quasi_original_predicate():
    # Literal tests/chebfun/test_cell2quasi.m, predicate 1 (#1344).
    x35 = cj.chebfun(lambda x: x, domain=(3, 5))
    q = cj.quasimatrix([x35, 2])
    assert q[1].domain.breakpoints == (3, 5)


def test_existing_array_columns_preserve_coefficients_points_and_orientation():
    f = cj.chebfun(_op, domain=(-2, 3))
    for value in (f, f.T):
        q = cj.cheb2quasi(value)
        assert isinstance(q, Quasimatrix)
        assert q.n_cols == 2 and q.is_transposed == value.is_transposed
        assert q.shape == ((2, "inf") if value.is_transposed else ("inf", 2))
        assert q.cheb2quasi() is q
        for actual, expected in zip(q.cols, value.num2cell()):
            np.testing.assert_array_equal(actual.point_values, expected.point_values)
            for a, b in zip(actual.funs, expected.funs):
                np.testing.assert_array_equal(a.tech.coeffs, b.tech.coeffs)
        assert cj.quasimatrix(q, 'ignored invalid option') is q
    assert f.cheb2quasi().n_cols == 2


@pytest.mark.parametrize('operand', [None, [], jnp.empty((0,))])
def test_empty_constructor_precedes_option_validation(operand):
    result = cj.quasimatrix(operand, 'invalid missing option')
    assert isinstance(result, cj.Chebfun) and result.isempty()
    assert result.n_columns == 0


def test_scalar_and_existing_fast_paths_ignore_constructor_options():
    f = cj.chebfun('x', domain=(3, 5))
    q = cj.quasimatrix(f, 'invalid missing option', domain=(-1, 1))
    assert isinstance(q, cj.Chebfun) and q.domain == f.domain
    q = cj.quasimatrix([f, 2], 'invalid missing option', domain=(-1, 1))
    assert q.domain == f.domain
    np.testing.assert_array_equal(q[1](jnp.asarray([3., 4., 5.])), 2.)


@pytest.mark.parametrize('domain', [(-2, 3), Domain((-2, 3)), jnp.asarray([-2., 3.])])
def test_positional_domain_and_length(domain):
    q = cj.quasimatrix(_op, domain, 9)
    assert q.n_cols == 2 and q.domain.breakpoints == (-2, 3)
    assert [len(c) for c in q.cols] == [9, 9]
    x = jnp.asarray([-.7, .2, 1.5])
    np.testing.assert_allclose(q(x), _op(x), rtol=0, atol=100*np.finfo(float).eps)


def test_preferences_flags_and_python_keyword_forwarding():
    pref = ChebfunPref({'domain': (2, 4), 'techPrefs': {'fixedLength': 7}})
    q = cj.quasimatrix(_op, pref, 'chebkind', 1)
    assert q.domain.breakpoints == (2, 4)
    assert all(type(c.funs[0].tech).__name__ == 'Chebtech1' for c in q.cols)
    assert [len(c) for c in q.cols] == [7, 7]
    q = cj.quasimatrix(_op, pref, n=8, domain=(-1, 1))
    assert [len(c) for c in q.cols] == [8, 8]
    assert pref.domain == (2, 4) and pref.fixedLength == 7


def test_coefficients_and_explicit_breakpoint_splitting():
    q = cj.quasimatrix(jnp.asarray([[1., 2.], [3., 4.]]), 'coeffs')
    np.testing.assert_allclose(q(jnp.asarray([0., .5])), [[1., 2.], [2.5, 4.]],
                               rtol=0, atol=100*np.finfo(float).eps)
    q = cj.quasimatrix(lambda x: jnp.stack((x, jnp.abs(x)), axis=-1),
                      [-1, 0, 1], 'splitting', 'on')
    assert q.n_cols == 2
    np.testing.assert_allclose(q(jnp.asarray([-.5, .5])), [[-.5, .5], [.5, .5]],
                               rtol=0, atol=100*np.finfo(float).eps)


@pytest.mark.parametrize(('args', 'identifier'), [
    (('splitting', 1), 'badSplitting'),
    (('resampling', 'yes'), 'badResampling'),
    (('unknown_preference',), 'noPrefValue'),
    ((ChebfunPref(), ChebfunPref()), 'twoPrefs'),
    ((7, [-1, 1]), 'badInputNumeric'),
])
def test_constructor_argument_errors(args, identifier):
    with pytest.raises(ValueError, match=identifier):
        cj.quasimatrix(_op, *args)


def test_cell_errors_and_invalid_conversion():
    f = cj.chebfun('x')
    with pytest.raises(ValueError, match='domaion'):
        cj.quasimatrix([f, cj.chebfun('x', domain=(2, 3))])
    with pytest.raises(ValueError, match='notascalar'):
        cj.quasimatrix([cj.quasimatrix(_op), f])
    with pytest.raises(TypeError, match='Chebfun'):
        cj.cheb2quasi(1)


def test_known_missing_value_uses_source_indexing_error():
    with pytest.raises(IndexError, match='MATLAB:badsubscript'):
        cj.quasimatrix(_op, 'eps')
    with pytest.raises(ValueError, match='noPrefValue'):
        cj.quasimatrix(_op, 'coeffs')
    with pytest.raises(ValueError, match='coeffcell'):
        cj.quasimatrix([lambda x: x, lambda x: x*x], 'coeffs')
