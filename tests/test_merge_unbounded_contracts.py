"""Independent source merge map ownership and construction preference seams.

Provenance
----------
MATLAB source : tests/chebfun/test_merge.m, @fun/merge.m,
    @singfun/singfun.m, @unbndfun/unbndfun.m
Chebfun commit: 7574c77
"""
import importlib

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import _merge_fun_source, _Piece
from chebfunjax.domain import Domain
from chebfunjax.fun.singfun import Singfun
from chebfunjax.fun.unbndfun import Unbndfun, _inverse_left, _inverse_right
from chebfunjax.tech.chebtech import Chebtech2


def options():
    return dict(maxpow2=6, max_length=73, tol=jnp.finfo(jnp.float64).eps,
                splitting=False, vscale=8., hscale=1., sample_test=False,
                min_samples=17, refinement_function='resampling', turbo=False,
                check='classic')


def pair():
    left = _Piece(Chebtech2.from_coeffs(jnp.array([5., 5.])), (0., 10.))
    right = Unbndfun.from_chebtech(Chebtech2.from_coeffs(jnp.array([3.])), Domain((10., jnp.inf)))
    return left, right


def test_union_mapping_endpoint_representation_and_raw_preferences(monkeypatch):
    left, right = pair()
    observed = {}
    def fit(cls, op, **kwargs):
        observed.update(kwargs)
        # Union map [0,Inf] sends -.5 ->5, 0 ->15, 1 ->Inf.
        # Existing right representation owns its finite value3 at Inf.
        np.testing.assert_allclose(op(jnp.array([-.5, 0., 1.])), [5., 3., 3.], rtol=0, atol=4e-15)
        return cls.from_coeffs(jnp.array([1.]), ishappy=False)
    monkeypatch.setattr(Chebtech2, 'from_function', classmethod(fit))
    out = _merge_fun_source(left, right, **options())
    assert out.interval == (0., float('inf'))
    assert out.mapping_type == 'right_inf' and not out.ishappy
    assert observed['hscale'] == 1. and observed['max_length'] == 73
    assert observed['vscale'] == 8. and observed['min_samples'] == 17
    assert observed['sample_test'] is False
    assert observed['refinement_function'] == 'resampling'
    assert observed['check'] == 'classic' and observed['extrapolate'] is False


@pytest.mark.parametrize('endpoint', [jnp.nan, jnp.inf])
def test_inf_only_detection_and_no_unhappy_retry(monkeypatch, endpoint):
    mod = importlib.import_module('chebfunjax.chebfun1d.chebfun')
    sf = importlib.import_module('chebfunjax.fun.singfun')
    calls = []
    def mapped_values(x, left, right):
        return jnp.where(jnp.isinf(x), endpoint, 1.)
    def finder(op):
        calls.append('detect')
        return (0., 0.)
    def fit(cls, op, **kwargs):
        assert bool(jnp.isnan(op(jnp.array([1.])))[0]) == bool(jnp.isnan(endpoint))
        return cls.from_coeffs(jnp.array([1.]), ishappy=False)
    monkeypatch.setattr(mod, '_merge_pair_values', mapped_values)
    monkeypatch.setattr(sf, '_find_sing_exponents', finder)
    monkeypatch.setattr(Chebtech2, 'from_function', classmethod(fit))
    out = _merge_fun_source(*pair(), **options())
    assert len(calls) == (1 if bool(jnp.isinf(endpoint)) else 0)
    assert not out.ishappy


def test_stored_exponent_infinite_end_sign_transition(monkeypatch):
    left, right = pair()
    right = right.with_tech(Singfun(right.tech, (0., -2.)))
    observed = {}
    def fit(cls, op, **kwargs):
        observed.update(kwargs)
        return cls.from_coeffs(jnp.array([1.]))
    monkeypatch.setattr(Chebtech2, 'from_function', classmethod(fit))
    out = _merge_fun_source(left, right, **options())
    assert out.tech.exponents == (0., 2.)
    assert float(observed['tol']) == 1e-14
    assert observed['hscale'] == 1.


def test_inverse_maps_preserve_written_source_order():
    a = jnp.asarray(1e16)
    x = a + 2.
    np.testing.assert_array_equal(_inverse_right(x, a), (-15.+x-a)/(15.+x-a))
    np.testing.assert_array_equal(_inverse_left(x, a), (15.+x-a)/(15.-x+a))
