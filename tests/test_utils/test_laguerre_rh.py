"""Independent controls for the source alpha=0 Laguerre RH integration.

Provenance: lagpts.m newton/polyAsyRH/asyBulk/asyBessel/asyAiry, Chebfun7574c77.
Large-n values use an independent SciPy tridiagonal Golub--Welsch oracle;
these bounds are not new MATLAB assertions. Original source tests stay intact.
"""
import math

import jax
import numpy as np
import numpy.testing as npt
import pytest
from scipy.linalg import eigh_tridiagonal

from chebfunjax.utils.quadrature import lagpts

EPS = np.finfo(float).eps


@pytest.fixture(scope='module')
def rule3000():
    return tuple(map(np.asarray, lagpts(3000, method='RH', bary=True)))


def test_rh_against_independent_tridiagonal_rule(rule3000):
    x, w, _ = rule3000
    k = np.arange(1, 3001, dtype=float)
    expected_x, vectors = eigh_tridiagonal(2*k-1, k[:-1], lapack_driver='stemr')
    expected_w = vectors[0]**2
    assert np.max(np.abs(x-expected_x)) <= 8e-10
    # MRRR eigenvector components for tiny weights are not a relative oracle.
    mask = expected_w > 1e-25
    assert mask.sum() > 200
    assert np.max(np.abs(w[mask]/expected_w[mask]-1)) <= 4e-9


def test_rh_structure_barycentric_weights_and_normalized_moments(rule3000):
    x, w, v = rule3000
    assert x.shape == w.shape == v.shape == (3000,)
    assert np.isfinite(x).all() and np.isfinite(w).all() and np.isfinite(v).all()
    assert np.all(x > 0) and np.all(np.diff(x) > 0) and np.all(w >= 0)
    positive = np.flatnonzero(w > 0)
    npt.assert_array_equal(positive, np.arange(len(positive)))
    expected_v = (-1.)**np.arange(3000) * np.sqrt(w*x)
    expected_v /= np.max(np.abs(expected_v))
    npt.assert_allclose(v, expected_v, rtol=32*EPS, atol=32*EPS)
    expected_moments = np.array([math.factorial(p) for p in range(5)])
    actual = np.array([w @ x**p for p in range(5)])
    npt.assert_allclose(actual, expected_moments, rtol=100_000*EPS, atol=0)


def test_default_alpha_zero_n3000_selects_source_rh(rule3000):
    for got, expected in zip(lagpts(3000, bary=True), rule3000, strict=True):
        npt.assert_array_equal(np.asarray(got), expected)


def test_public_rh_under_jit_matches_eager(rule3000):
    compiled = jax.jit(lambda: lagpts(3000, method='RH', bary=True))
    for got, expected in zip(compiled(), rule3000, strict=True):
        npt.assert_allclose(np.asarray(got), expected, rtol=32*EPS, atol=32*EPS)


@pytest.mark.parametrize('interval', [(1., math.inf), (-math.inf, -1.)])
def test_rh_interval_mapping_preserves_source_bary_order(rule3000, interval):
    x, w, v = rule3000
    mapped_x, mapped_w, mapped_v = map(np.asarray, lagpts(
        3000, interval=interval, method='RH', bary=True))
    expected_x = x+1 if math.isinf(interval[1]) else -x-1
    npt.assert_allclose(mapped_x, expected_x, rtol=32*EPS, atol=32*EPS)
    npt.assert_allclose(mapped_w, w*np.exp(-1.), rtol=32*EPS, atol=32*EPS)
    npt.assert_array_equal(mapped_v, v)


def test_explicit_generalized_rh_stays_explicitly_unported():
    with pytest.raises(NotImplementedError, match='RH variant is not yet supported'):
        lagpts(3000, .5, method='RH')


def test_n10000_rh_has_finite_ordered_nodes_and_gamma_moments():
    x, w = map(np.asarray, lagpts(10000))
    assert x.shape == w.shape == (10000,)
    assert np.isfinite(x).all() and np.isfinite(w).all()
    assert np.all(x > 0) and np.all(np.diff(x) > 0) and np.all(w >= 0)
    positive = np.flatnonzero(w > 0)
    npt.assert_array_equal(positive, np.arange(len(positive)))
    # Same independent moment envelope used by the pre-integration RH pilot.
    npt.assert_allclose([w @ x**p for p in range(5)],
                       [math.factorial(p) for p in range(5)], rtol=2e-9, atol=0)
