"""Independent generalized Laguerre recurrence and mapping checks.

Provenance
----------
MATLAB source : lagpts.m (lag_rec and public wrapper)
Chebfun commit: 7574c77
"""

import jax
import numpy as np
import numpy.testing as npt
import pytest
from scipy.special import gamma, roots_genlaguerre

from chebfunjax.utils.quadrature import lagpts

EPS = np.finfo(float).eps


@pytest.mark.parametrize('alpha', [-0.5, 0.0, 0.5, 2.5])
@pytest.mark.parametrize('n', [1, 2, 17, 42, 251, 299])
def test_recurrence_against_independent_generalized_laguerre_rule(n, alpha):
    x, w, v = map(np.asarray, lagpts(n, alpha, method='REC', bary=True))
    expected_x, expected_w = roots_genlaguerre(n, alpha)
    npt.assert_allclose(x, expected_x, atol=1000 * EPS, rtol=100 * EPS)
    npt.assert_allclose(w, expected_w, atol=1000 * EPS * gamma(alpha + 1), rtol=0)
    assert np.isfinite(x).all() and np.isfinite(w).all() and np.isfinite(v).all()
    assert np.all(x > 0) and np.all(np.diff(x) > 0) and np.all(w >= 0)
    # An n-node Gaussian rule is exact only through degree 2*n-1.
    for degree in range(min(4, 2 * n)):
        exact = gamma(alpha + degree + 1)
        # test_lagpts.m uses 100*TOL at n251. Apply that source large-n
        # bound here too: at n299/alpha=-.5 the exact first moment is
        # 0.8862269254527579; binary64 scalar source emulation gives
        # 0.8862269254525114 and JAX 0.8862269254525070. Both reject the
        # initial small-n bound, so that new bound did not qualify the
        # source algorithm itself. The original MATLAB assertions retain
        # their exact, unmodified tolerances in the separate port test.
        tolerance = (100000 if n >= 200 else 1000) * EPS
        assert abs(w @ x**degree - exact) <= tolerance * exact
    # Underflowed weights and their barycentric weights may be zero.
    nonzero = v != 0
    npt.assert_array_equal(np.sign(v[nonzero]), ((-1.0)**np.arange(n))[nonzero])


def test_dynamic_alpha_and_static_interval_under_jit():
    for interval in (None, (1.0, np.inf), (-np.inf, -1.0)):
        function = jax.jit(lambda alpha: lagpts(42, alpha, interval, bary=True))
        for alpha in (-0.5, 0.5):
            for actual, expected in zip(function(alpha), lagpts(42, alpha, interval, bary=True), strict=True):
                npt.assert_allclose(actual, expected, rtol=64 * EPS, atol=64 * EPS)


def test_barycentric_weights_are_computed_before_interval_weight_underflow():
    x, w, v = lagpts(17, bary=True)
    shifted_x, shifted_w, shifted_v = lagpts(17, interval=(1000.0, np.inf), bary=True)
    npt.assert_array_equal(shifted_x, x + 1000)
    npt.assert_array_equal(shifted_w, np.zeros(17))
    npt.assert_array_equal(shifted_v, v)


@pytest.mark.parametrize('alpha', [-1.001, 0.5j])
def test_invalid_generalized_parameter(alpha):
    with pytest.raises(ValueError, match='alpha'):
        lagpts(5, alpha)


@pytest.mark.parametrize('interval', [(0.0, 1.0), (-np.inf, np.inf)])
def test_invalid_semi_infinite_interval(interval):
    with pytest.raises(ValueError, match='semi-infinite'):
        lagpts(5, interval=interval)


def test_piecewise_interval_warns_and_uses_endpoints_like_matlab():
    with pytest.warns(UserWarning, match="piecewise intervals"):
        actual = lagpts(17, interval=(1.0, 2.0, np.inf), bary=True)
    expected = lagpts(17, interval=(1.0, np.inf), bary=True)
    for got, want in zip(actual, expected, strict=True):
        npt.assert_array_equal(got, want)
