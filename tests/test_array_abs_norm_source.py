"""Source array abs/norm/simplify controls, MATLAB Chebfun7574c77.

Native @chebfun/{abs,addBreaksAtRoots,norm,simplify}.m. Analytic linear
arrays establish union roots, own-column point zeros, and continuous extrema.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
@pytest.mark.parametrize('row', [False, True])
def test_array_abs_union_keeps_other_columns_and_values(tech, row):
    f = Chebfun(funs=[_Piece(tech=tech(coeffs=jnp.array([[.5, -.25, 0.], [1., 1., 0.]])), interval=(-1., 1.))], domain=Domain((-1., 1.)))
    f = f.T if row else f
    result = f.abs()
    assert result.is_transposed == row
    np.testing.assert_array_equal(result.domain.breakpoints, [-1., -.5, .25, 1.])
    np.testing.assert_allclose(result.point_values, [[.5, 1.25, 0.], [0., .75, 0.], [.75, 0., 0.], [1.5, .75, 0.]], rtol=0, atol=32*np.finfo(float).eps)
    assert result.point_values[1, 0] == 0 and result.point_values[2, 1] == 0


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
def test_array_abs_zeros_old_root_only_in_its_own_column(tech):
    pieces = []
    for a, b in [(-1., 0.), (0., 1.)]:
        center, half = (a+b)/2, (b-a)/2
        pieces.append(_Piece(tech=tech(coeffs=jnp.array([[center, center-.25], [half, half]])), interval=(a, b)))
    f = Chebfun(funs=pieces, domain=Domain((-1., 0., 1.))).set_point_values(jnp.array([[1., 2.], [7., 8.], [3., 4.]]))
    result = f.abs()
    old_zero = list(result.domain.breakpoints).index(0.)
    assert result.point_values[old_zero, 0] == 0
    assert result.point_values[old_zero, 1] == 8
    np.testing.assert_array_equal(result.point_values[[0, -1], :], [[1., 2.], [3., 4.]])


@pytest.mark.parametrize('p,expected', [(jnp.inf, 2.25), (-jnp.inf, .75)])
def test_native_array_norm_does_not_scalarize(monkeypatch, p, expected):
    f = Chebfun(funs=[_Piece(tech=Chebtech2(coeffs=jnp.array([[.5, -.25], [1., 1.]])), interval=(-1., 1.))], domain=Domain((-1., 1.)))
    def forbidden(*args, **kwargs):
        raise AssertionError('Native array abs/norm does not call mat2cell')
    monkeypatch.setattr(Chebfun, 'mat2cell', forbidden)
    assert abs(float(f.norm(p))-expected) < 64*np.finfo(float).eps


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
def test_simplify_uses_per_column_scales(monkeypatch, tech):
    f = Chebfun(funs=[_Piece(tech=tech(coeffs=jnp.array([[1., 100.]])), interval=(-1., 0.)), _Piece(tech=tech(coeffs=jnp.array([[10., 2.]])), interval=(0., 1.))], domain=Domain((-1., 0., 1.)))
    seen = []
    def observe(self, tol=None):
        seen.append(np.asarray(tol))
        return self
    monkeypatch.setattr(tech, 'simplify', observe)
    f.simplify()
    np.testing.assert_array_equal(seen, np.finfo(float).eps*np.array([[10., 1.], [1., 50.]]))


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
@pytest.mark.parametrize('operation', ['simplify', 'abs'])
def test_array_default_simplify_uses_session_preference(monkeypatch, tech, operation):
    # Native abs passes pref to simplify; simplify with omitted tol reads
    # current chebfunpref().techPrefs.chebfuneps. Restore even on assertion.
    saved = ChebfunPref()
    tolerance = 2.**-30
    f = Chebfun(funs=[_Piece(tech=tech(coeffs=jnp.array([[1., 2.]])),
                            interval=(-1., 1.))], domain=Domain((-1., 1.)))
    seen = []

    def observe(self, tol=None):
        seen.append(np.asarray(tol))
        return self

    monkeypatch.setattr(tech, 'simplify', observe)
    try:
        ChebfunPref.setDefaults('chebfuneps', tolerance)
        getattr(f, operation)()
        np.testing.assert_array_equal(seen, [[tolerance, tolerance]])
    finally:
        ChebfunPref.setDefaults(saved)
    assert ChebfunPref().techPrefs.chebfuneps == saved.techPrefs.chebfuneps
