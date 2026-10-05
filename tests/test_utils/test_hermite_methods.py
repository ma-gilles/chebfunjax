"""Independent method and dispatch checks for Gauss--Hermite quadrature.

Provenance
----------
MATLAB source : hermpts.m
Chebfun commit: 7574c77
"""

import os
import subprocess
import sys

import numpy as np
import numpy.testing as npt
import pytest
from scipy.special import roots_hermite

from chebfunjax.utils.quadrature import hermpts

EPS = np.finfo(float).eps


@pytest.mark.parametrize('n', [251, 1000, 1001])
def test_asy_against_independent_hermite_roots(n):
    x, w, v = map(np.asarray, hermpts(n, 'ASY', bary=True))
    expected_x, expected_w = roots_hermite(n)
    npt.assert_allclose(x, expected_x, atol=128 * EPS * np.sqrt(n), rtol=0)
    # The four-term source ASY formula has a truncation error near n=251;
    # this uses the source moment tolerance there, rather than REC accuracy.
    npt.assert_allclose(w, expected_w, atol=3000 * EPS, rtol=0)
    assert abs(w @ x**2 - np.sqrt(np.pi) / 2) < 3000 * EPS
    assert abs(np.max(np.abs(v)) - 1) <= 2 * EPS
    assert np.all(np.diff(x) > 0)
    assert np.all(w >= 0)
    for got, default in zip((x, w, v), hermpts(n, bary=True), strict=True):
        npt.assert_array_equal(got, default)


@pytest.mark.parametrize('n', [2, 3, 20, 21, 42, 128, 129, 130, 131, 251])
def test_lag_relation_against_independent_hermite_roots(n):
    x, w, v = map(np.asarray, hermpts(n, 'LAG', bary=True))
    expected_x, expected_w = roots_hermite(n)
    npt.assert_allclose(x, expected_x, atol=256 * EPS * np.sqrt(n), rtol=0)
    npt.assert_allclose(w, expected_w, atol=256 * EPS, rtol=0)
    npt.assert_array_equal(x, -x[::-1])
    npt.assert_array_equal(w, w[::-1])
    assert abs(w.sum() - np.sqrt(np.pi)) < 10 * EPS
    assert abs(w @ x**2 - np.sqrt(np.pi) / 2) < 256 * EPS
    # MATLAB's mathematical normalization target is 1; CPU JIT reciprocal
    # lowering gave 0.9999999999999999 at n21/128 in the first qualification.
    assert abs(np.max(abs(v)) - 1) <= 2 * EPS
    npt.assert_array_equal(np.sign(v), (-1.0)**np.arange(n))


@pytest.mark.parametrize('method', ['GW', 'REC', 'GLR', 'ASY', 'LAG'])
@pytest.mark.parametrize('n', [0, 1])
def test_all_methods_share_empty_and_singleton_source_dispatch(method, n):
    x, w, v = map(np.asarray, hermpts(n, method, bary=True))
    assert x.shape == w.shape == v.shape == (n,)
    if n:
        npt.assert_array_equal(x, [0.0])
        npt.assert_array_equal(v, [1.0])
        npt.assert_array_equal(w, [np.sqrt(np.pi)])


def test_source_three_character_type_prefixes():
    for prefix, full in [('phy', 'phys'), ('pro', 'prob')]:
        for got, expected in zip(hermpts(42, prefix), hermpts(42, full), strict=True):
            npt.assert_array_equal(got, expected)


def test_lazy_method_imports_do_not_capture_jit_tracers():
    # Fresh interpreters reproduce the integration failure where an import
    # inside a jitted method used to create a global JAX-array tracer.
    for method, orders in [('ASY', (251, 1000)), ('REC', (21, 42)), ('LAG', (20, 42))]:
        program = f'''
import jax
import numpy as np
from chebfunjax.utils.quadrature import hermpts
with jax.checking_leaks():
    for n in {orders!r}:
        rule = jax.jit(lambda: hermpts(n, {method!r}, bary=True))()
        jax.block_until_ready(rule)
        assert all(np.isfinite(np.asarray(a)).all() for a in rule)
'''
        completed = subprocess.run([sys.executable, '-c', program],
                                    env=dict(os.environ, JAX_PLATFORMS='cpu'),
                                    capture_output=True, text=True, timeout=120)
        assert completed.returncode == 0, completed.stdout + completed.stderr
