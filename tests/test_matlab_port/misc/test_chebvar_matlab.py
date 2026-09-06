"""Port of MATLAB Chebfun tests/misc/test_chebvar.m (Fable 5).

MATLAB's ``chebvar x y [a b]`` workspace command is ``chebvar("x", "y",
domain=(a, b))``, returning the identity chebfun(s).

Provenance
----------
MATLAB source : tests/misc/test_chebvar.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun, chebvar

jax.config.update("jax_enable_x64", True)

EPS = np.finfo(float).eps


class TestMiscChebvar:
    def test_all_matlab_assertions(self):
        x = chebvar("x")
        f = x.sin()
        g = chebfun(jnp.sin)
        assert float((f - g).norm(jnp.inf)) < EPS                   # pass(1)

        x = chebvar("x", domain=(0.0, 10.0))
        f = (x ** 2).sin() + x.sin() ** 2
        g = chebfun(lambda t: jnp.sin(t ** 2) + jnp.sin(t) ** 2, domain=(0.0, 10.0))
        assert float((f - g).norm(jnp.inf)) < 1e4 * EPS             # pass(2)

        x, y, z = chebvar("x", "y", "z", domain=(0.0, 1.0))
        f = x.sin()
        g = y.cos()
        h = chebfun(lambda t: jnp.sin(t) + jnp.cos(t), domain=(0.0, 1.0))
        assert float(((f + g) - h).norm(jnp.inf)) < 10 * EPS        # pass(3)
