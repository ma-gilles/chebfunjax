"""Independent bounded singular source representation/metadata contracts.

Provenance
----------
MATLAB source : tests/chebfun/test_merge.m, @fun/merge.m,
    @singfun/singfun.m, @unbndfun/unbndfun.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import _merge_bounded_fun_source, _merge_limit_row, _Piece
from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech2


def test_full_singular_limits_not_smooth_factor_limits():
    f = _Piece(Singfun(Chebtech2.from_coeffs(jnp.array([2.])), (-1., 0.)), (-2., 0.))
    assert jnp.isposinf(_merge_limit_row(f, False)[0])
    assert float(_merge_limit_row(f, True)[0]) == 1.


def test_trial_keeps_outer_exponents_and_source_preferences(monkeypatch):
    left = _Piece(Singfun(Chebtech2.from_coeffs(jnp.array([2.])), (-1., 0.)), (-2., 0.))
    right = _Piece(Singfun(Chebtech2.from_coeffs(jnp.array([3.])), (0., -1.)), (0., 7.))
    observed = {}
    def fit(cls, op, **kwargs):
        observed.update(kwargs)
        t = jnp.array([-.75, .5])
        x = 7*(t+1)/2 - 2*(1-t)/2
        full = jnp.where(x <= 0, 2/(x+2), 3/(2-2*x/7))
        np.testing.assert_allclose(op(t), full / ((1+t)**-1*(1-t)**-1), rtol=0, atol=4e-15)
        return cls.from_coeffs(jnp.array([1.]))
    monkeypatch.setattr(Chebtech2, 'from_function', classmethod(fit))
    out = _merge_bounded_fun_source(left, right, maxpow2=5, max_length=19,
        tol=jnp.finfo(jnp.float64).eps, splitting=False, vscale=4., hscale=7.,
        sample_test=False, min_samples=9, refinement_function='resampling',
        turbo=False, check='classic')
    assert out.tech.exponents == (-1., -1.)
    assert observed['max_length'] == 19 and observed['min_samples'] == 9
    assert observed['hscale'] == 7/9 and observed['vscale'] == 4.
    assert float(observed['tol']) == 1e-14 and observed['extrapolate']
    assert observed['sample_test'] is False
    assert observed['refinement_function'] == 'resampling'
    assert observed['check'] == 'classic'


def test_root_breaks_store_source_zero_pointvalues():
    f = cj.chebfun(lambda x: (x-.25)/(x+2), domain=(-2., 1.), exps=[-1., 0.])
    g = f.addBreaksAtRoots()
    assert len(g.funs) > 1
    pv = jnp.asarray(g.point_values)
    np.testing.assert_array_equal(pv[1:-1], jnp.zeros_like(pv[1:-1]))


def test_singular_evaluation_preserves_complex_contour():
    f = Singfun(Chebtech2.from_coeffs(jnp.array([2., 1.])), (-1., -1.))
    z = jnp.array([.2+.3j, -.4+.1j])
    np.testing.assert_allclose(f(z), (2+z)/(1+z)/(1-z), rtol=0, atol=4e-15)
