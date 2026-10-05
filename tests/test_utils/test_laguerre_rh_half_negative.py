"""Independent negative-trial RH Newton-step checks .

This is a selected diagnostic of the two half-integer alpha branches at
n=3000. It is not an exhaustive validation or a MATLAB assertion. The
reference uses the generalized Laguerre three-term recurrence in mpmath at
80 decimal digits, independently of the RH asymptotics.

Source: pinned Chebfun ``lagpts.m`` at commit
7574c77680d7e82b79626300bf255498271a72df. At lines 473–486, MATLAB updates
``step = pe / (polyAsyRH(n-1,x,alpha+1,T)*factorx - pe/2)`` and then
``x = x - step``. Lines 407–453 define the factors, and lines 515–530 select
Bessel/bulk/Airy expansions. For the mathematical oracle, use
``d/dx L_n^alpha(x) = -L_(n-1)^(alpha+1)(x)`` and the source weighted polynomial
``exp(-x/2) L_n^alpha(x)``; its logarithmic derivative is
``-L_(n-1)^(alpha+1) - L_n^alpha/2``. Thus the corresponding Newton step is
``L_n^alpha / (-L_(n-1)^(alpha+1) - L_n^alpha/2)``.

The three negative x values deliberately exercise negative Newton trials in
the near-zero/Bessel path. The predeclared 2e-9 relative / 8e-10 absolute
bound is a diagnostic envelope for the asymptotic polynomial and conversion
factors; it is not copied from a MATLAB test. These checks use
``chebfunjax.utils.laguerre_rh_half._poly_asy_rh_half`` and
``_rh_factors_half`` interfaces. The selected private candidate pilot was checked before public integration.
"""

import math

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

from chebfunjax.utils.laguerre_rh_half import (
    _poly_asy_rh_half,
    _rh_factors_half,
)

N = 3000
NEGATIVE_TRIALS = (-1.0e-4, -1.0e-2, -1.0e-1)
ALPHAS = (-0.5, 0.5)
RTOL = 2.0e-9
ATOL = 8.0e-10


def _laguerre_mp(n: int, alpha: float, x: mp.mpf) -> mp.mpf:
    """Generalized Laguerre recurrence, independent of the RH formulas."""
    if n == 0:
        return mp.mpf(1)
    previous = mp.mpf(1)
    current = 1 + mp.mpf(alpha) - x
    for degree in range(1, n):
        following = (
            (2 * degree + 1 + mp.mpf(alpha) - x) * current
            - (degree + mp.mpf(alpha)) * previous
        ) / (degree + 1)
        previous, current = current, following
    return current


def _reference_step(n: int, alpha: float, x: float) -> float:
    with mp.workdps(80):
        xx = mp.mpf(x)
        value = _laguerre_mp(n, alpha, xx)
        derivative_factor = _laguerre_mp(n - 1, alpha + 1.0, xx)
        step = value / (-derivative_factor - value / 2)
        return float(step)


def _candidate_step(y, alpha):
    # Match source T=ceil(34/log(n)); alpha is closed over and therefore static.
    terms = math.ceil(34.0 / math.log(N))
    factorx, _factorw = _rh_factors_half(N, alpha)
    value = _poly_asy_rh_half(N, y, alpha, terms)
    derivative_polynomial = _poly_asy_rh_half(N - 1, y, alpha + 1.0, terms)
    return value / (factorx * derivative_polynomial - value / 2.0)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_negative_trial_newton_step_matches_independent_laguerre(alpha):
    points = jnp.asarray(NEGATIVE_TRIALS, dtype=jnp.float64)
    eager = jax.vmap(lambda y: _candidate_step(y, alpha))(points)
    compiled = jax.jit(jax.vmap(lambda y: _candidate_step(y, alpha)))(points)
    actual = np.asarray(eager)
    compiled_values = np.asarray(compiled)
    expected = np.asarray(
        [_reference_step(N, alpha, y) for y in NEGATIVE_TRIALS],
        dtype=np.float64,
    )

    assert np.isfinite(actual).all()
    assert np.isfinite(compiled_values).all()
    np.testing.assert_allclose(compiled_values, actual, rtol=64 * np.finfo(float).eps,
                               atol=64 * np.finfo(float).eps)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
