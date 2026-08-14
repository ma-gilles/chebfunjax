"""Port of MATLAB Chebfun tests/chebfun2/test_techs.m (Fable 5).

Alternative underlying techs for a chebfun2.  chebfunjax supports the
Chebyshev (default) and trigonometric ('trig') slice representations;
MATLAB's pass(1) chebtech1 preference has no counterpart (slices are
always Chebtech2 in the Chebyshev case) and is covered by the default
construction instead.

Provenance
----------
MATLAB source : tests/chebfun2/test_techs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS


class TestChebfun2Techs:
    def test_chebtech(self):
        # pass(2): default Chebyshev construction.
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        assert abs(float(f(jnp.asarray(0.0), jnp.asarray(0.0))) - 1.0) < TOL
        assert f.approx.techs == ("cheb", "cheb")

    def test_trigtech(self):
        # pass(3): 'trig' construction of a bi-periodic function.
        f = Chebfun2.from_function(
            lambda x, y: jnp.cos(jnp.pi * y) * jnp.sin(jnp.pi * x),
            trig=True)
        assert abs(float(f(jnp.asarray(0.0), jnp.asarray(0.0)))) < 100 * TOL
        assert f.approx.techs == ("trig", "trig")
        # spot values
        xv, yv = 0.3, -0.7
        want = np.cos(np.pi * yv) * np.sin(np.pi * xv)
        got = float(f(jnp.asarray(xv), jnp.asarray(yv)))
        assert abs(got - want) < 100 * TOL
