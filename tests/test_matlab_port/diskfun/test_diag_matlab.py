"""Port of MATLAB Chebfun tests/diskfun/test_diag.m (Fable 5).

MATLAB's ``pass(1+j) = norm(g - d)`` records the error itself; the port
checks it against the file's tolerance.

Provenance
----------
MATLAB source : tests/diskfun/test_diag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.diskfun.diskfun import Diskfun

from ._cart import disk_polar

jax.config.update("jax_enable_x64", True)


class TestDiskfunDiag:
    def test_all_matlab_assertions(self):
        tol = 10 * ChebfunPref().cheb2Prefs.chebfun2eps
        f = Diskfun.coeffs2diskfun(jnp.zeros((1, 1)))
        assert f.diag().iszero()                                             # pass(1)
        ff = lambda th, r: jnp.exp(-r ** 2 * jnp.cos(th) * jnp.cos(th))  # noqa: E731
        f = disk_polar(ff)
        rng = np.random.RandomState(10062001)
        alp = np.pi * (1 - 2 * rng.rand(10))
        for a in alp:
            g = chebfun(lambda r, _a=a: ff(_a, r), domain=(-1.0, 1.0))
            d = f.diag(a)
            assert float((g - d).norm(2)) < 100 * tol                        # pass(2..11)
        with pytest.raises(ValueError, match="angleOutOfRange"):
            f.diag(10 * np.pi)                                               # pass(12)
