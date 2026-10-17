"""Port of MATLAB Chebfun tests/chebfun2/test_techs.m (Fable 5).

Literal three preference predicates, with Python keyword preference syntax.

Provenance
----------
MATLAB source : tests/chebfun2/test_techs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS


class TestChebfun2Techs:
    def test_chebtech1(self):
        pref = ChebfunPref()
        pref.tech = Chebtech1
        f = chebfun2(lambda x, y: jnp.cos(x * y), (-1, 1, -1, 1), pref=pref)
        assert abs(float(f(0., 0.)) - 1.) < TOL
        assert all(isinstance(t, Chebtech1) for t in f.approx.cols + f.approx.rows)

    def test_chebtech(self):
        # pass(2): default Chebyshev construction.
        pref = ChebfunPref()
        pref.tech = Chebtech2
        f = chebfun2(lambda x, y: jnp.cos(x * y), (-1, 1, -1, 1), pref=pref)
        assert abs(float(f(jnp.asarray(0.0), jnp.asarray(0.0))) - 1.0) < TOL
        assert f.approx.techs == ("cheb", "cheb")

    def test_trigtech(self):
        # pass(3): 'trig' construction of a bi-periodic function.
        pref = ChebfunPref()
        pref.tech = Trigtech
        f = chebfun2(
            lambda x, y: jnp.cos(jnp.pi * y) * jnp.sin(jnp.pi * x),
            (-1, 1, -1, 1), pref=pref)
        assert abs(float(f(jnp.asarray(0.0), jnp.asarray(0.0)))) < 100 * TOL
        assert f.approx.techs == ("trig", "trig")
        # spot values
        xv, yv = 0.3, -0.7
        want = np.cos(np.pi * yv) * np.sin(np.pi * xv)
        got = float(f(jnp.asarray(xv), jnp.asarray(yv)))
        assert abs(got - want) < 100 * TOL
