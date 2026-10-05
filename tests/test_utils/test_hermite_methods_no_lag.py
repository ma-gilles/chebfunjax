"""Focused REC/GLR/ASY/LAG dispatch checks for the Hermite/Laguerre overlay.

Provenance: MATLAB ``hermpts.m``, Chebfun commit
7574c77680d7e82b79626300bf255498271a72df. Independent references use
SciPy/NumPy quadrature routines; source tolerances are retained.
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
    npt.assert_allclose(w, expected_w, atol=3000 * EPS, rtol=0)
    assert abs(w @ x**2 - np.sqrt(np.pi) / 2) < 3000 * EPS
    assert abs(np.max(np.abs(v)) - 1) <= 2 * EPS
    assert np.all(np.diff(x) > 0) and np.all(w >= 0)
    for got, default in zip((x, w, v), hermpts(n, bary=True), strict=True):
        npt.assert_array_equal(got, default)


@pytest.mark.parametrize('method', ['GW', 'REC', 'GLR', 'ASY', 'LAG'])
@pytest.mark.parametrize('n', [0, 1])
def test_supported_methods_share_empty_and_singleton_source_dispatch(method, n):
    x, w, v = map(np.asarray, hermpts(n, method, bary=True))
    assert x.shape == w.shape == v.shape == (n,)
    if n:
        npt.assert_array_equal(x, [0.0])
        npt.assert_array_equal(v, [1.0])
        npt.assert_array_equal(w, [np.sqrt(np.pi)])


def test_n_zero_returns_before_option_validation():
    x, w, v = hermpts(0, 'bad-kind', 'bad-method', bary=True)
    assert np.asarray(x).shape == np.asarray(w).shape == np.asarray(v).shape == (0,)


def test_source_three_character_type_prefixes():
    for prefix, full in [('phy', 'phys'), ('pro', 'prob')]:
        for got, expected in zip(hermpts(42, prefix), hermpts(42, full), strict=True):
            npt.assert_array_equal(got, expected)


def test_lazy_method_imports_do_not_capture_jit_tracers():
    for method, orders in [('ASY', (251, 1000)), ('REC', (21, 42))]:
        body = "import jax, numpy as np\nfrom chebfunjax.utils.quadrature import hermpts\nwith jax.checking_leaks():\n"
        body += ''.join("    rule=jax.jit(lambda: hermpts(%d, %r, bary=True))()\n    jax.block_until_ready(rule)\n    assert all(np.isfinite(np.asarray(a)).all() for a in rule)\n" % (n, method) for n in orders)
        completed = subprocess.run([sys.executable, '-c', body],
                                   env=dict(os.environ, JAX_PLATFORMS='cpu'),
                                   capture_output=True, text=True, timeout=120)
        assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(('args', 'expected'), [
    (('REC', 'prob'), ('REC', 'prob')),
    (('prob', 'REC'), ('REC', 'prob')),
    (('REC', 'GLR', 'prob'), ('GLR', 'prob')),
    (('prob', 'phys', 'REC'), ('REC', 'phys')),
])
def test_source_options_any_order_and_last_flag_wins(args, expected):
    # Use independent explicit calls as parser controls; all orders should
    # resolve to the same method and type while retaining the original values.
    got = hermpts(42, *args)
    want = hermpts(42, expected[0], expected[1])
    for a, b in zip(got, want, strict=True):
        npt.assert_array_equal(a, b)


def test_nondefault_method_keyword_is_final_method_override():
    got = hermpts(42, 'REC', 'prob', method='GLR')
    expected = hermpts(42, 'prob', 'GLR')
    for a, b in zip(got, expected, strict=True):
        npt.assert_array_equal(a, b)


@pytest.mark.parametrize(('option', 'kind'), [
    ('GW', 'phys'), ('REC', 'phys'), ('GLR', 'prob'),
    ('ASY', 'prob'), ('LAG', 'phys'),
])
def test_all_valid_methods_use_trivial_singleton_before_algorithm(option, kind):
    x, w, v = map(np.asarray, hermpts(1, option, kind, bary=True))
    npt.assert_array_equal(x, [0.0 if kind == 'phys' else 0.0])
    npt.assert_array_equal(w, [np.sqrt(np.pi) * (np.sqrt(2.0) if kind == 'prob' else 1.0)])
    npt.assert_array_equal(v, [1.0])
