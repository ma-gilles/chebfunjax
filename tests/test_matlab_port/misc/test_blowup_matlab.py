"""Port of MATLAB Chebfun tests/misc/test_blowup.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_blowup.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebpref import ChebfunPref, blowup

jax.config.update("jax_enable_x64", True)

EPS = np.finfo(float).eps


class TestMiscBlowup:
    def test_all_matlab_assertions(self):
        saved = ChebfunPref()
        ChebfunPref.setDefaults("factory")
        try:
            assert blowup() == 0                                    # pass(1)
            assert blowup("on") == 0                                # pass(2)
            assert blowup() == 1                                    # pass(3)
            F = lambda x: 1.0 / (x + 1)  # noqa: E731
            f = chebfun(lambda x: F(x), domain=(-1.0, 1.0))
            x = jnp.asarray(0.9 * np.linspace(-1, 1, 20))
            err = float(np.max(np.abs(np.asarray(f(x)) - np.asarray(F(x)))))
            assert err < 1e2 * EPS * f.vscale                       # pass(4)
            assert ChebfunPref().blowupPrefs.defaultSingType == "pole"  # pass(5)
            assert blowup(2) == 1                                   # pass(6)
            assert blowup() == 2                                    # pass(7)
            F = lambda x: 1.0 / jnp.sqrt(x + 1)  # noqa: E731
            f = chebfun(lambda x: F(x), domain=(-1.0, 1.0))
            err = float(np.max(np.abs(np.asarray(f(x)) - np.asarray(F(x)))))
            assert err < 1e2 * EPS * f.vscale                       # pass(8)
            assert blowup("off") == 2                               # pass(9)
            assert blowup() == 0                                    # pass(10)
        finally:
            ChebfunPref.setDefaults(saved)
