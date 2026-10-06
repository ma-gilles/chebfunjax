"""Public Unbndfun complex-coordinate regression, independent rational fields.

Provenance
----------
MATLAB source : @unbndfun/feval.m, @mapping/mapping.m
Chebfun commit: 7574c77
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.unbndfun import Unbndfun
from chebfunjax.tech.chebtech import Chebtech2


@pytest.mark.parametrize('side', [1, -1])
def test_public_unbounded_complex_queries_eager_and_jit(side):
    # On[0,Inf], (1-t)/30 is1/(x+15). On[-Inf,0],
    # (1+t)/30 is1/(15-x). These independent analytic fields avoid fitting
    # error and do not import source-produced coefficients or expected output.
    dom = Domain((0., float('inf')) if side == 1 else (-float('inf'), 0.))
    f = Unbndfun.from_chebtech(
        Chebtech2.from_coeffs(jnp.array([1/30, -side/30])), dom)
    z = jnp.array([side*3+2j, side*12-.75j, side*30+3j])
    expected = 1/(15+side*z)
    assert bool(jnp.all(jnp.abs(jnp.imag(expected)) > 1e-5))
    # |expected|<1; 64eps absolute budgets complex rational arithmetic.
    bound = 64*np.finfo(float).eps
    for actual in [f(z), jax.jit(lambda x: f(x))(z)]:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=bound)
    # The other changed public cast is forward_map, not inverse_map.
    t = jnp.array([-.5+.125j, .25-.25j])
    expected_map = side*15*(1+side*t)/(1-side*t)
    for actual in [f.forward_map(t), jax.jit(lambda x: f.forward_map(x))(t)]:
        np.testing.assert_allclose(actual, expected_map, rtol=0,
                                   atol=64*60*np.finfo(float).eps)
