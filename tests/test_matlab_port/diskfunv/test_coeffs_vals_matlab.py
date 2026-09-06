"""Port of MATLAB Chebfun tests/diskfunv/test_coeffs_vals.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfunv/test_coeffs_vals.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.diskfun.diskfun import Diskfun
from chebfunjax.diskfun.diskfunv import Diskfunv

from ..diskfun._cart import disk_xy

jax.config.update("jax_enable_x64", True)


def _ninf(A):
    return float(np.max(np.abs(np.asarray(A))))


class TestDiskfunvCoeffsVals:
    def test_all_matlab_assertions(self):
        tol = 1e3 * ChebfunPref().techPrefs.chebfuneps
        u = disk_xy(lambda x, y: jnp.exp(-jnp.cos(np.pi * (x + y))))
        v = disk_xy(lambda x, y: x * jnp.sin(x * y))
        f = Diskfunv(u, v)
        x, y = f.coeffs2()
        assert float((Diskfun.coeffs2diskfun(x) - u).norm("inf")) < tol      # pass(1)
        assert float((Diskfun.coeffs2diskfun(y) - v).norm("inf")) < tol      # pass(2)
        x, y = f.coeffs2(50, 60)
        assert _ninf(np.asarray(x) - np.asarray(u.coeffs2(50, 60))) < tol    # pass(3)
        assert _ninf(np.asarray(y) - np.asarray(v.coeffs2(50, 60))) < tol    # pass(4)
        f2 = Diskfunv.coeffs2diskfunv(u.coeffs2(), v.coeffs2())
        assert float((f - f2).norm()) < tol                                  # pass(5)
        x, y = f2.coeffs2()
        uu, vv = Diskfunv.coeffs2vals(x, y)
        assert _ninf(np.asarray(Diskfun.coeffs2vals(x)) - np.asarray(uu)) < tol  # pass(6)
        assert _ninf(np.asarray(Diskfun.coeffs2vals(y)) - np.asarray(vv)) < tol  # pass(7)
        a, b = Diskfunv.vals2coeffs(uu, vv)
        assert _ninf(np.asarray(x) - np.asarray(a)) < tol                    # pass(8)
        assert _ninf(np.asarray(y) - np.asarray(b)) < tol                    # pass(9)
