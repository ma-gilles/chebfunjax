"""Native @chebfun/{roots,getRootsForBreaks}.m, Chebfun 7574c77 controls.

Exact piece coefficients and pointValues match native_reference.m operands.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece, _real_simple_roots
from chebfunjax.domain import Domain


def piecewise(coeffs, values=None):
    f = Chebfun(funs=[_Piece.from_coeffs(jnp.asarray(c), a, b)
                     for c, a, b in zip(coeffs, [-1., 0.], [0., 1.])],
                domain=Domain((-1., 0., 1.)))
    return f if values is None else f.set_point_values(jnp.asarray(values))


@pytest.mark.parametrize('values,expected', [([1., 0., 1.], [0.]),
                                           ([0., 1., 1.], [-1.]),
                                           ([1., 1., 0.], [1.])])
def test_isolated_point_values_and_nobreaks(values, expected):
    f = piecewise([[1.], [1.]], values)
    np.testing.assert_array_equal(f.roots(nojump=True, nozerofun=True), expected)
    wanted = [-1.] if values[0] == 0 else []
    np.testing.assert_array_equal(f.roots(nojump=True, nozerofun=True,
                                        nobreaks=True), wanted)
    np.testing.assert_array_equal(f.getRootsForBreaks(), wanted)


def test_continuous_breakpoint_root_survives_nobreaks():
    f = piecewise([[-.5, .5], [.5, .5]])
    np.testing.assert_allclose(f.getRootsForBreaks(), [0.], atol=2e-15, rtol=0)
    np.testing.assert_allclose(f.roots(nobreaks=True), [0.], atol=2e-15, rtol=0)


def test_jump_is_not_a_root_for_break_insertion_or_denominator():
    f = piecewise([[-1.], [1.]])
    np.testing.assert_array_equal(f.roots(), [0.])
    assert f.roots(nojump=True, nobreaks=True).size == 0
    assert f.getRootsForBreaks().size == 0
    assert _real_simple_roots(f).size == 0
    assert isinstance(_real_simple_roots(f), jax.Array)


def test_nozerofun_and_initial_point_value_grammar():
    f = piecewise([[0.], [1.]], [1., 1., 1.])
    np.testing.assert_array_equal(f.getRootsForBreaks(), [])
    # Native's initial pointValues check is unconditional, including nobreaks.
    g = f.set_point_values(jnp.asarray([0., 1., 1.]))
    np.testing.assert_array_equal(g.getRootsForBreaks(), [-1.])
    r = g.roots(nojump=True, nobreaks=True)
    assert np.any(np.asarray(r) == -.5)
    assert np.any(np.asarray(r) == -1.)


def test_array_columns_have_native_nan_padding():
    f = piecewise([[[1., 1.]], [[1., 1.]]],
                  [[1., 0.], [0., 1.], [1., 1.]])
    np.testing.assert_array_equal(f.roots(nojump=True), [[0., -1.]])
    roots = f.roots(nojump=True, nobreaks=True)
    assert roots.shape == (1, 2)
    assert np.isnan(roots[0, 0])
    assert float(roots[0, 1]) == -1.
    np.testing.assert_array_equal(f.getRootsForBreaks(), [-1.])


def test_complex_piece_does_not_create_jump_root():
    f = piecewise([[-1.+1j], [1.-1j]], [-1.+1j, 1.+1j, 1.-1j])
    assert f.roots().size == 0
    g = f.set_point_values(jnp.asarray([-1.+1j, 0., 1.-1j]))
    np.testing.assert_array_equal(g.roots(), [0.])
    assert g.getRootsForBreaks().size == 0


def test_get_roots_filter_is_simultaneous_and_jax(monkeypatch):
    f = Chebfun.from_coeffs(jnp.asarray([1.]))
    eps = jnp.finfo(jnp.float64).eps
    roots = jnp.asarray([[0., jnp.nan], [eps*.75, eps*1.5], [eps*3., jnp.nan]])
    def fake(self, **flags):
        assert flags == dict(nojump=True, nozerofun=True, nobreaks=True)
        return roots
    monkeypatch.setattr(Chebfun, 'roots', fake)
    filtered, raw = f.getRootsForBreaks(return_all=True)
    assert isinstance(filtered, jax.Array)
    np.testing.assert_array_equal(raw, roots)
    np.testing.assert_array_equal(filtered, [0., eps*3.])
    np.testing.assert_array_equal(f.getRootsForBreaks(tol=eps*4), [0.])
