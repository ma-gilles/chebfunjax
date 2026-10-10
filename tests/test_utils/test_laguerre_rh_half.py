"""Independent public half-alpha Laguerre RH and Hermite LAG controls.

Provenance: MATLAB lagpts.m/hermpts.m at Chebfun commit
7574c77680d7e82b79626300bf255498271a72df. Newbounds are diagnostic, inherited
from the alpha0 RH independent envelope (8e-10 nodes,4e-9 weights>1e-25,
2e-9 moments); original MATLAB test assertions are unchanged. Hermite node
5e-10 and weight4e-9 bounds are predeclared independent large-LAG controls.
Exact half-order Bessel roots seed Newton instead of Piessens rounded seeds.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from scipy.linalg import eigh_tridiagonal

from chebfunjax.utils import quadrature
from chebfunjax.utils.quadrature import hermpts, lagpts

EPS = np.finfo(float).eps


@pytest.fixture(scope='module', params=[-0.5, 0.5])
def half_rule(request):
    alpha = request.param
    return alpha, tuple(np.asarray(a) for a in lagpts(3000, alpha, bary=True, method='RH'))


def test_half_rh_against_independent_tridiagonal(half_rule):
    alpha, (x, w, _) = half_rule
    k = np.arange(1, 3001, dtype=float)
    nodes, vectors = eigh_tridiagonal(2*k-1+alpha, np.sqrt(k[:-1]*(k[:-1]+alpha)),
                                     lapack_driver='stemr')
    expected = math.gamma(alpha+1)*vectors[0]**2
    assert np.max(np.abs(x-nodes)) <= 8e-10
    mask = expected > 1e-25
    assert mask.sum() > 200
    npt.assert_allclose(w[mask], expected[mask], rtol=4e-9, atol=0)


def test_half_rh_structure_barycentric_and_moments(half_rule):
    alpha, (x, w, v) = half_rule
    assert x.shape == w.shape == v.shape == (3000,)
    assert np.isfinite(x).all() and np.isfinite(w).all() and np.isfinite(v).all()
    assert np.all(x > 0) and np.all(np.diff(x) > 0) and np.all(w >= 0)
    positive = np.flatnonzero(w > 0)
    npt.assert_array_equal(positive, np.arange(len(positive)))
    expected_v = (-1.)**np.arange(3000)*np.sqrt(w*x)
    expected_v /= np.max(np.abs(expected_v))
    npt.assert_allclose(v, expected_v, rtol=32*EPS, atol=32*EPS)
    expected = [math.gamma(alpha+k+1) for k in range(5)]
    npt.assert_allclose([w @ x**k for k in range(5)], expected, rtol=2e-9, atol=0)


def test_large_half_default_dispatch_matches_explicit_rh(half_rule):
    alpha, explicit = half_rule
    for got, want in zip(lagpts(3000, alpha, bary=True), explicit, strict=True):
        npt.assert_array_equal(got, want)


def test_half_public_jit_matches_eager(half_rule):
    alpha, eager = half_rule
    compiled = jax.jit(lambda: lagpts(3000, alpha, bary=True, method='RH'))()
    for got, want in zip(compiled, eager, strict=True):
        npt.assert_allclose(got, want, rtol=32*EPS, atol=32*EPS)


@pytest.mark.parametrize('interval', [(1., np.inf), (-np.inf, -1.)])
def test_half_rh_source_interval_mapping(half_rule, interval):
    alpha, (x, w, v) = half_rule
    actual = lagpts(3000, alpha, interval, bary=True, method='RH')
    mapped_x = x+interval[0] if np.isinf(interval[1]) else -x+interval[1]
    factor = np.exp(-interval[0]) if np.isinf(interval[1]) else np.exp(interval[1])
    for got, want in zip(actual, (mapped_x, w*factor, v), strict=True):
        npt.assert_allclose(got, want, rtol=32*EPS, atol=32*EPS)


@pytest.mark.parametrize('alpha', [-0.5, 0.5])
def test_half_rh_n10000_structure_and_moments(alpha):
    x, w = (np.asarray(a) for a in lagpts(10000, alpha))
    assert np.isfinite(x).all() and np.isfinite(w).all()
    assert np.all(x > 0) and np.all(np.diff(x) > 0) and np.all(w >= 0)
    positive = np.flatnonzero(w > 0)
    npt.assert_array_equal(positive, np.arange(len(positive)))
    npt.assert_allclose([w @ x**k for k in range(5)],
                        [math.gamma(alpha+k+1) for k in range(5)], rtol=2e-9, atol=0)


@pytest.mark.parametrize('n', [6000, 6001])
def test_large_hermite_lag_against_independent_hermite_tridiagonal(n):
    x, w, v = (np.asarray(a) for a in hermpts(n, 'LAG', bary=True))
    nodes, vectors = eigh_tridiagonal(np.zeros(n), np.sqrt(np.arange(1, n)/2),
                                     lapack_driver='stemr')
    expected = np.sqrt(np.pi)*vectors[0]**2
    assert np.max(np.abs(x-nodes)) <= 5e-10
    mask = expected > 1e-25
    assert mask.sum() > 400
    npt.assert_allclose(w[mask], expected[mask], rtol=4e-9, atol=0)
    assert np.isfinite(v).all() and np.max(np.abs(v)) == 1
    npt.assert_allclose([w @ x**k for k in (0, 2, 4)],
                        [np.sqrt(np.pi), np.sqrt(np.pi)/2, 3*np.sqrt(np.pi)/4],
                        rtol=2e-9, atol=0)


@pytest.mark.parametrize('alpha', [-0.5, 0.5, 1.5])
def test_large_default_method_selection_without_dense_allocations(monkeypatch, alpha):
    methods = []
    def probe(n, alpha, interval, method):
        methods.append(method)
        return jnp.ones(n), jnp.ones(n)
    monkeypatch.setattr(quadrature, '_lagpts_core', probe)
    lagpts(3000, alpha)
    # Native lagpts.m selects RH for every alpha when n >= 3000.
    assert methods == ['rh']


def test_dynamic_alpha_default_retains_documented_gw_adapter(monkeypatch):
    methods = []
    def probe(n, alpha, interval, method):
        methods.append(method)
        return jnp.ones(n)*alpha, jnp.ones(n)
    monkeypatch.setattr(quadrature, '_lagpts_core', probe)
    jax.jit(lambda a: lagpts(3000, a))(jnp.float64(0.5))
    assert methods == ['gw']
