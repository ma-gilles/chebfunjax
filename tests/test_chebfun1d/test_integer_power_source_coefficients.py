"""Exact dyadic coefficients guard bounded square/cube degree and tech kind.

Provenance
----------
MATLAB source : @chebfun/power.m (columnPower), @chebtech/compose.m
Chebfun commit: 7574c77
Independent oracle: T32^2=(1+T64)/2 and T32^3=(3*T32+T96)/4.
The positive input 2+T32/8 stays away from zero and the power branch cut.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize('Tech', [Chebtech1, Chebtech2])
@pytest.mark.parametrize('exponent', [2, 3])
def test_positive_high_degree_integer_power_exact_coefficients(Tech, exponent):
    coefficients = jnp.zeros(33).at[0].set(2.).at[32].set(.125)
    f = Chebfun(funs=[_Piece(tech=Tech.from_coeffs(coefficients),
                            interval=(-1., 1.))], domain=Domain((-1., 1.)))
    result = f ** exponent
    if exponent == 2:
        expected = np.zeros(65)
        expected[[0, 32, 64]] = [4.0078125, .5, .0078125]
    else:
        expected = np.zeros(97)
        expected[[0, 32, 64, 96]] = [8.046875, 1.50146484375, .046875, .00048828125]
    tech = result.funs[0].tech
    assert type(tech) is Tech and tech.ishappy and tech.n == expected.size
    tolerance = 20*np.finfo(float).eps*np.max(np.abs(expected))
    assert np.max(np.abs(np.asarray(tech.coeffs)-expected)) < tolerance
