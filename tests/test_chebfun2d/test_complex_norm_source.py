"""Independent complex L2 counterexamples for represented separable functions.

Native tests/chebfun2/test_norm.m uses1000*chebfun2eps (Chebfun7574c77).
The ix target is native; the rank2 representation/orientation controls are
additional analytic regressions, not a port of the native SVD norm algorithm.
"""
import math
from fractions import Fraction

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.separable_approx import SeparableApprox
from chebfunjax.tech.chebtech import Chebtech2


@pytest.mark.parametrize('api', ['chebfun2', 'separable'])
@pytest.mark.parametrize('fixture', ['one_plus_ix', 'complex_factor_cancellation'])
def test_represented_complex_l2_analytic(api, fixture):
    one = Chebtech2.from_coeffs(jnp.asarray([1.]))
    if fixture == 'one_plus_ix':
        row = Chebtech2.from_coeffs(jnp.asarray([0., 1.]))
        # f=1+i*x: integral_y integral_x(1+x²) =16/3.
        exact_squared = Fraction(16, 3)
    else:
        row = Chebtech2.from_coeffs(jnp.asarray([1j, 1.]))
        # f=1+i*(i+x)=i*x: integral_y integral_x x² =4/3.
        # Conjugating the SECOND weight instead would yield52/3.
        exact_squared = Fraction(4, 3)
    approx = SeparableApprox(cols=[one, one], rows=[one, row],
                             pivots=jnp.asarray([1., 1j]),
                             domain=(-1., 1., -1., 1.))
    actual = Chebfun2(approx=approx).norm() if api == 'chebfun2' else approx.norm('fro')
    # Original native norm test absolute tolerance, no residual-derived bound.
    tolerance = 1000*jnp.finfo(jnp.float64).eps
    assert abs(float(actual)-math.sqrt(float(exact_squared))) < tolerance
