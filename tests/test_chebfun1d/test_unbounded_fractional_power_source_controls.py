"""Focused mapping-preservation control for unbounded fractional powers.

The simple positive analytic function isolates retention of an Unbndfun map
when its mapped onefun is a Singfun. It complements, but does not replace,
MATLAB test_power.m pass 35.

Provenance
----------
MATLAB sources : @chebfun/power.m, @classicfun/power.m,
                 @unbndfun/unbndfun.m
Chebfun commit: 7574c77
Original: Copyright 2017 by The University of Oxford and the Chebfun Developers.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.fun.unbndfun import Unbndfun

EPS = float(np.finfo(np.float64).eps)


@pytest.mark.parametrize("transposed", [False, True])
def test_fractional_power_keeps_unbounded_domain_mapping(transposed):
    f = cj.chebfun(
        lambda x: x**2 + 2.0,
        domain=(-jnp.inf, jnp.inf),
        exps=(2.0, 2.0),
    )
    f = f.set_point_values(jnp.asarray([4.0, 9.0]))
    if transposed:
        f = f.T
    result = f**1.5
    assert result.is_transposed == transposed
    np.testing.assert_allclose(np.asarray(result.point_values), [8.0, 27.0],
                               rtol=100 * EPS, atol=0.0)
    np.testing.assert_allclose(np.asarray(result(jnp.asarray([-jnp.inf, jnp.inf]))),
                               [8.0, 27.0], rtol=100 * EPS, atol=0.0)

    assert isinstance(f.funs[0], Unbndfun)
    assert isinstance(result.funs[0], Unbndfun)
    assert result.funs[0].mapping_type == f.funs[0].mapping_type
    assert tuple(result.funs[0].domain.breakpoints) == tuple(
        f.funs[0].domain.breakpoints)

    points = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
    exact = (points**2 + 2.0) ** 1.5
    # Independent analytic control at finite physical points. The original
    # pass35 retains its separate 1e7*eps*vscale bound unchanged.
    np.testing.assert_allclose(np.asarray(result(points)), exact,
                               rtol=100 * EPS, atol=0.0)
