"""Port of MATLAB Chebfun tests/chebfun/test_changeTech.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_changeTech.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.tech.trigtech import Trigtech

jax.config.update("jax_enable_x64", True)

TOL = 1e-14


class TestChebfunChangeTech:
    def test_all_matlab_assertions(self):
        f = chebfun(jnp.cos, domain=(0.0, 2 * np.pi))
        g = f.change_tech("trigtech")
        assert isinstance(g.funs[0].tech, Trigtech)                 # pass(1)
        assert float((f - g).norm(jnp.inf)) < TOL                   # pass(2)

        f = chebfun(jnp.cos, domain=(0.0, 2 * np.pi), trig=True)
        g = f.change_tech(Chebtech2)                                # pref.tech
        assert isinstance(g.funs[0].tech, Chebtech2)                # pass(3)
        assert float((f - g).norm(jnp.inf)) < TOL                   # pass(4)
