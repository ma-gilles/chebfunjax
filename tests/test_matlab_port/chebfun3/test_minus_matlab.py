"""All three original assertions from Chebfun tests/chebfun3/test_minus.m.

Native commit 7574c77. Continuous norms, thresholds, inputs and construction
order are preserved. The cheb.xyz shorthand is expanded into its three
coordinate constructors. Qualification uses fresh serial processes per clause.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref

TOL = 1e3 * ChebfunPref().cheb3Prefs.chebfun3eps


def ff(x, y, z):
    return jnp.cos(x*y*z)


def gg(x, y, z):
    return x+y+z+x*y*z


@pytest.mark.parametrize("statement", [1, 2])
def test_native_minus_domain_norm(statement):
    if statement == 1:
        f = chebfun3(ff)
        g = chebfun3(gg)
        exact = chebfun3(lambda x, y, z: ff(x, y, z)-gg(x, y, z))
    else:
        domain = (-1., math.pi, 0., 2*math.pi, -math.pi, math.pi)
        f = chebfun3(ff, domain)
        g = chebfun3(gg, domain)
        exact = chebfun3(lambda x, y, z: ff(x, y, z)-gg(x, y, z), domain)
    assert ((f-g)-exact).norm() < TOL


def test_native_minus_near_cancellation():
    x = chebfun3(lambda x, y, z: x)
    chebfun3(lambda x, y, z: y)
    chebfun3(lambda x, y, z: z)
    f = x
    g = 0.9999999*x
    assert (f-g).norm() > 0
