# uses-numpy: independent reference moments, selected eigenvalues and special functions.
"""General-alpha RH diagnostics with unchanged inherited RH envelopes.

Provenance
----------
MATLAB source: lagpts.m and besselroots.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
RH envelopes8e-10 nodes/4e-9 weights/2e-9 moments are those already used for
half/zero alpha. Source-expression fixture is interpreted, not fresh MATLAB.
Bessel adapter envelope is separately predeclared5e-12 relative+1e-13 absolute.
"""
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import eigh_tridiagonal
from scipy.special import eval_genlaguerre, gammaln, jv

from chebfunjax.utils.bessel_general import _bessel_j_general, _bessel_j_general_complex
from chebfunjax.utils.bessel_roots_general import _bessel_roots_general
from chebfunjax.utils.laguerre_rh_expansions import (
    _asyairy_general,
    _asybessel_general,
    _asybulk_general,
)
from chebfunjax.utils.quadrature import lagpts
from chebfunjax.utils.specfun import besselroots


@pytest.mark.parametrize('order', [-1.9, -.7, .3, 1., 2., 5., 6., 10., 11.])
def test_bessel_real_adapter(order):
    x = np.r_[np.geomspace(.02, 3.99, 20), np.linspace(4., max(32., 2*order**2), 40),
              np.geomspace(max(32.1, 2*order**2+.1), 3000., 40)]
    got = jax.jit(jax.vmap(lambda z: _bessel_j_general(order, z)))(jnp.asarray(x))
    np.testing.assert_allclose(got, jv(order, x), rtol=5e-12, atol=1e-13)


@pytest.mark.parametrize('order', [-1.7, -.7, .3, 1., 1.3, 3.])
def test_bessel_negative_trial_continuation(order):
    z = 1j*np.asarray([.02, .2, 2., 12., 40.])
    got = jax.jit(jax.vmap(lambda a: _bessel_j_general_complex(order, a)))(jnp.asarray(z))
    np.testing.assert_allclose(got, jv(order, z), rtol=5e-12, atol=1e-13)


@pytest.mark.parametrize('order', [-.99, -.7, .3, 1., 2., 5., 10.])
def test_piessens_mcmahon_source_seeds(order):
    got = jax.jit(lambda: _bessel_roots_general(order, 18))()
    np.testing.assert_allclose(got, besselroots(order, 18), rtol=4e-15, atol=4e-15)


@pytest.mark.parametrize('branch', ['asyBulk', 'asyBessel', 'asyAiry'])
def test_literal_source_general_correction_tables(branch):
    fixture = json.loads((Path(__file__).parent/'fixtures/laguerre_general_source_expressions.json').read_text())
    functions = {'asyBulk': _asybulk_general, 'asyAiry': _asyairy_general,
                 'asyBessel': lambda n,y,a,T: _asybessel_general(n,y,a,T,_bessel_j_general)}
    for row in fixture['cases']:
        if row['branch'] != branch:
            continue
        got = functions[branch](row['n'], jnp.asarray(row['y']), row['alpha'], row['T'])
        np.testing.assert_allclose(got, row['expected'], rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize('alpha', [-.99, -.7, .3, 1., 2., 5., 5.5, 10.])
def test_general_rule_moments_selected_nodes_and_weights(alpha):
    x, w, v = map(np.asarray, lagpts(3000, alpha, bary=True))
    assert np.isfinite(x).all() and np.isfinite(w).all() and np.isfinite(v).all()
    assert np.all(x > 0) and np.all(np.diff(x) > 0) and np.all(w >= 0)
    np.testing.assert_allclose([w@x**k for k in range(5)],
                              [math.gamma(alpha+k+1) for k in range(5)], rtol=2e-9, atol=0)
    k = np.arange(1, 3001, dtype=float)
    expected = eigh_tridiagonal(2*k-1+alpha, np.sqrt(k[:-1]*(k[:-1]+alpha)),
        eigvals_only=True, select='i', select_range=(0,15), lapack_driver='stebz')
    np.testing.assert_allclose(x[:16], expected, rtol=0, atol=8e-10)
    derivative = -eval_genlaguerre(2999, alpha+1, x[:16])
    expected_weights = np.exp(gammaln(3001+alpha)-gammaln(3001))/(x[:16]*derivative**2)
    np.testing.assert_allclose(w[:16], expected_weights, rtol=4e-9, atol=0)
    for got, want in zip(lagpts(3000, alpha, bary=True, method='RH'), (x,w,v), strict=True):
        np.testing.assert_array_equal(got, want)
