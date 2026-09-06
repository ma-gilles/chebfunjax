"""Port of MATLAB Chebfun tests/chebfun3/test_techs.m (Fable 5).

``pref.tech = @chebtech1`` etc. is the ``tech=`` keyword of ``chebfun3``.

Provenance
----------
MATLAB source : tests/chebfun3/test_techs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech

jax.config.update("jax_enable_x64", True)

pi = np.pi


class TestChebfun3Techs:
    def test_all_matlab_assertions(self):
        tol = 100 * ChebfunPref().cheb3Prefs.chebfun3eps
        ff = lambda x, y, z: jnp.cos(x * y * z)  # noqa: E731
        dom = (-1, 1, -1, 1, -1, 1)
        f = chebfun3(ff, dom, tech="chebtech1")
        assert isinstance(f.cols[0], Chebtech1)                              # pass(1)
        assert abs(float(f(0, 0, 0)) - float(ff(0.0, 0.0, 0.0))) < tol        # pass(2)
        f = chebfun3(ff, dom, tech="chebtech2")
        assert isinstance(f.cols[0], Chebtech2)                              # pass(3)
        assert abs(float(f(0, 0, 0)) - float(ff(0.0, 0.0, 0.0))) < tol        # pass(4)
        ff = lambda x, y, z: jnp.cos(pi * z) * jnp.sin(pi * y) * jnp.sin(pi * x)  # noqa: E731
        f = chebfun3(ff, dom, tech="trigtech")
        assert isinstance(f.cols[0], Trigtech)                               # pass(5)
        assert abs(float(f(0, 0, 0)) - float(ff(0.0, 0.0, 0.0))) < 100 * tol  # pass(6)
