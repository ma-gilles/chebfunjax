# uses-numpy: Independent reference fixtures and numeric assertions in tests.
"""Unrun zero representation controls for the proposed axis mean.

Pinned source: Chebfun 7574c77 @spherefun/constructor.m zero-matrix
PhaseOne branch and periodic-factor construction; @spherefun/sum.m
Chebfun versus numeric fallback; @separableApprox/cdr.m zero reciprocal.
The explicit empty-factor adapter case is not a MATLAB zero-constructor
oracle. It tests the Python representation's numeric-fallback contract.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize("from_values", [False, True])
def test_public_zero_mean_representation(from_values):
    if from_values:
        zero = Spherefun.from_values(jnp.zeros((5, 8)))
    else:
        zero = Spherefun.from_function(lambda lam, theta: jnp.zeros_like(lam))
    assert not zero.isempty()
    first, second = zero.mean(1), zero.mean(2)
    assert first.is_transposed and not second.is_transposed
    assert isinstance(first.funs[0].tech, Trigtech)
    assert isinstance(second.funs[0].tech, Chebtech2)
    np.testing.assert_array_equal(first(jnp.array([-jnp.pi, 0.0, jnp.pi])), 0.0)
    np.testing.assert_array_equal(second(jnp.array([0.0, jnp.pi / 2, jnp.pi])), 0.0)


def test_nonempty_empty_factor_numeric_fallback():
    zero = Spherefun(cols=[], rows=[], pivots=jnp.empty(0), idx_plus=(), idx_minus=())
    assert not zero.isempty()
    first, second = zero.mean(1), zero.mean(2)
    assert first.is_transposed and not second.is_transposed
    assert isinstance(first.funs[0].tech, Chebtech2)
    assert isinstance(second.funs[0].tech, Chebtech2)
    np.testing.assert_array_equal(first(jnp.array([-jnp.pi, 0.0, jnp.pi])), 0.0)
    np.testing.assert_array_equal(second(jnp.array([0.0, jnp.pi / 2, jnp.pi])), 0.0)
