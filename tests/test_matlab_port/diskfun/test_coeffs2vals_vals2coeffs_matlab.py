"""Port of MATLAB Chebfun tests/diskfun/test_coeffs2vals_vals2coeffs.m
(Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_coeffs2vals_vals2coeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.randfuns import randnfundisk
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.diskfun.diskfun import Diskfun
from chebfunjax.tech.trigtech import _trig_eval_np, trig_vals2coeffs
from chebfunjax.utils.quadrature import chebpts
from chebfunjax.utils.transforms import vals2coeffs

from ._cart import disk_xy

jax.config.update("jax_enable_x64", True)


def _trigpts(n):
    return -np.pi + 2 * np.pi * np.arange(n) / n


def _cvals(cols, r):
    return np.column_stack([np.asarray(c(jnp.asarray(r))).ravel() for c in cols])


def _rvals(rows, t):
    return np.column_stack([np.real(np.asarray(_trig_eval_np(
        np.asarray(rw.coeffs)[:, None], np.asarray(t) / np.pi,
        is_real=rw.is_real))).ravel() for rw in rows])


def _ninf(A):
    return float(np.max(np.abs(np.asarray(A))))


class TestDiskfunCoeffs2valsVals2coeffs:
    def test_all_matlab_assertions(self):
        tol = 1e2 * ChebfunPref().techPrefs.chebfuneps
        f = disk_xy(lambda x, y: jnp.sin(4 * ((3 * x - .4) ** 2 + y)))
        exact = f.coeffs2()
        cfs = Diskfun.vals2coeffs(Diskfun.coeffs2vals(exact))
        assert _ninf(np.asarray(exact) - np.asarray(cfs)) < tol              # pass(1)

        f = disk_xy(lambda x, y: jnp.exp(-2 * x ** 3 - (y - .4) ** 3))
        c, d, r = f.cdr()
        rr = np.asarray(chebpts(51))
        t = _trigpts(106)
        F = _cvals(c, rr) @ np.asarray(d) @ _rvals(r, t).T                   # g(tt, rr)
        cfs = Diskfun.vals2coeffs(F)
        assert _ninf(np.asarray(cfs) - np.asarray(f.coeffs2(106, 51))) < tol  # pass(2)
        vals = np.asarray(Diskfun.coeffs2vals(f.coeffs2()))
        m, n = vals.shape
        t = _trigpts(n)
        rr = np.asarray(chebpts(m))
        G = _cvals(c, rr) @ np.asarray(d) @ _rvals(r, t).T
        assert _ninf(G - vals) < 1e2 * tol                                   # pass(3)

        np.random.seed(0)
        f = randnfundisk(.3)
        c, d, r = f.cdr()
        vals = np.asarray(Diskfun.coeffs2vals(f.coeffs2()))
        m, n = vals.shape
        # [u, s, v] = diskfun.coeffs2vals(c, d, r): the factor values
        U = _cvals(c, np.asarray(chebpts(m)))
        V = _rvals(r, _trigpts(n))
        assert _ninf(vals - U @ np.asarray(d) @ V.T) < 10 * tol              # pass(4)

        tt = _trigpts(100)
        rr = np.asarray(chebpts(51))
        cv = _cvals(c, rr)
        rv = _rvals(r, tt)
        u = np.column_stack([np.asarray(vals2coeffs(jnp.asarray(cv[:, j]))).ravel()
                             for j in range(cv.shape[1])])
        v = np.column_stack([np.asarray(trig_vals2coeffs(jnp.asarray(rv[:, j]))).ravel()
                             for j in range(rv.shape[1])])
        F = np.asarray(f.coeffs2(100, 51))
        assert _ninf(F - u @ np.asarray(d) @ v.T) < 3 * tol                  # pass(5)
