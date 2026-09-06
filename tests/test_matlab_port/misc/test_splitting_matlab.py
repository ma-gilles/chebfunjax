"""Port of MATLAB Chebfun tests/misc/test_splitting.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_splitting.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebpref import ChebfunPref, splitting

jax.config.update("jax_enable_x64", True)


class TestMiscSplitting:
    def test_all_matlab_assertions(self):
        saved = ChebfunPref()
        ChebfunPref.setDefaults("factory")
        try:
            assert splitting() == "off"                             # pass(1)
            assert splitting("on") == "off"                         # pass(2)
            assert splitting() == "on"                              # pass(3)
            F = lambda x: jnp.cos(np.pi * x) * jnp.sign(x)  # noqa: E731
            f = chebfun(lambda x: F(x), domain=(-1.0, 1.0))
            x = jnp.asarray(np.linspace(-1, 1, 20))
            err = float(np.max(np.abs(np.asarray(f(x)) - np.asarray(F(x)))))
            assert len(f.domain.breakpoints) > 2 and err < 1e-5    # pass(4)
            assert splitting("off") == "on"                         # pass(5)
            assert splitting() == "off"                             # pass(6)
        finally:
            ChebfunPref.setDefaults(saved)
