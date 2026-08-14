"""Port of MATLAB Chebfun tests/chebfun2/test_isPeriodicTech.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_isPeriodicTech.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2


class TestChebfun2IsPeriodicTech:
    def test_cheb_not_periodic(self):
        # pass(1)
        f = Chebfun2.from_function(lambda x, y: jnp.cos(jnp.pi * x) + 0 * y)
        assert not f.isPeriodicTech()

    def test_trig_periodic(self):
        # pass(2)
        f = Chebfun2.from_function(
            lambda x, y: jnp.cos(jnp.pi * x) + 0 * y, trig=True)
        assert f.isPeriodicTech()

    def test_trig_periodic_pi_domain(self):
        # pass(3)
        f = Chebfun2.from_function(
            lambda x, y: jnp.cos(x) * jnp.sin(y) + jnp.cos(3 * x)
            + jnp.cos(5 * x),
            domain=(-np.pi, np.pi, -np.pi, np.pi), trig=True)
        assert f.isPeriodicTech()
        xv, yv = 0.4, -1.1
        want = (np.cos(xv) * np.sin(yv) + np.cos(3 * xv)
                + np.cos(5 * xv))
        assert abs(float(f(jnp.asarray(xv), jnp.asarray(yv))) - want) \
            < 1e-11
