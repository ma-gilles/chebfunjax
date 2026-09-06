"""Port of MATLAB Chebfun tests/chebfun/test_truncate.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_truncate.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.tech.trigtech import Trigtech

jax.config.update("jax_enable_x64", True)

EPS = np.finfo(float).eps


class TestChebfunTruncate:
    def test_all_matlab_assertions(self):
        x = chebfun("x")
        f = x.sign()
        g = f.truncate(10)
        assert len(g) == 10                                         # pass(1)
        c = g.chebcoeffs()
        assert abs(float(c[-1]) - 4 / (9 * np.pi)) < 10 * EPS       # pass(2)

        G = chebfun(jnp.sign, trunc=10, splitting=True)
        assert len(g) == 10                                         # pass(3)
        assert float((g - G).norm(jnp.inf)) < 10 * EPS              # pass(4)
        c = g.chebcoeffs()
        assert abs(float(c[-1]) - 4 / (9 * np.pi)) < 10 * EPS       # pass(5)

        f = chebfun(lambda t: jnp.exp(jnp.sin(np.pi * t)), trig=True)
        g = f.truncate(10)
        assert len(g) == 10                                         # pass(6)
        assert isinstance(g.funs[0].tech, Trigtech)                 # pass(7)

        f = chebfun("exp(sin(t))", domain=(0.0, 2 * np.pi))
        p = chebfun(f, trunc=31, trig=True)
        assert list(p.domain.breakpoints) == [0.0, 2 * np.pi]       # pass(8)
        assert float((f - p).norm(2)) < 100 * EPS                   # pass(9)
