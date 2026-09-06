"""Port of MATLAB Chebfun tests/spherefun/test_coeffs2vals_vals2coeffs.m
(Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_coeffs2vals_vals2coeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.randfuns import randnfunsphere
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun

from ._cart import sph_xyz

jax.config.update("jax_enable_x64", True)


def _trigpts(n):
    return -np.pi + 2 * np.pi * np.arange(n) / n


def _vals(techs, x):
    from chebfunjax.tech.trigtech import _trig_eval_np
    return np.column_stack([np.real(np.asarray(_trig_eval_np(
        np.asarray(t.coeffs)[:, None], np.asarray(x) / np.pi,
        is_real=t.is_real))).ravel() for t in techs])


def _ninf(A):
    return float(np.max(np.abs(np.asarray(A))))


class TestSpherefunCoeffs2valsVals2coeffs:
    def test_all_matlab_assertions(self):
        tol = 1e2 * ChebfunPref().techPrefs.chebfuneps
        f = sph_xyz(lambda x, y, z: jnp.exp(-((x - .4) ** 2 + (y - .9) ** 2 + (z - .1) ** 2)))
        exact = f.coeffs2()
        cfs = Spherefun.vals2coeffs(Spherefun.coeffs2vals(exact))
        assert _ninf(np.asarray(exact) - np.asarray(cfs)) < tol              # pass(1)

        f = sph_xyz(lambda x, y, z: jnp.sin(np.pi * x * y) + jnp.sin(np.pi * x * z))
        c, d, r = f.cdr()
        lam = _trigpts(50)
        th = _trigpts(60)
        # g = c*d*r.' evaluated on [tt, ll] = meshgrid(th, lam): the
        # columns are sampled on the 50-point grid, the rows on the
        # 60-point grid (a 50 x 60 matrix).
        F = _vals(c, lam) @ np.asarray(d) @ _vals(r, th).T                   # g(tt, ll)
        cfs = Spherefun.vals2coeffs(F)
        assert _ninf(np.asarray(cfs) - np.asarray(f.coeffs2(60, 50))) < tol  # pass(2)
        vals = np.asarray(Spherefun.coeffs2vals(f.coeffs2()))
        m, n = vals.shape
        th = _trigpts(n)
        lam = _trigpts(m)
        G = _vals(c, lam) @ np.asarray(d) @ _vals(r, th).T
        assert _ninf(G - vals) < 1e2 * tol                                   # pass(3)

        np.random.seed(0)
        f = randnfunsphere(.3)
        c, d, r = f.cdr()
        vals = np.asarray(Spherefun.coeffs2vals(f.coeffs2()))
        # [u, s, v] = spherefun.coeffs2vals(c, d, r) -> u*s*v.' is the value
        # matrix of the CDR factors on the same grid.
        m, n = vals.shape
        U = _vals(c, _trigpts(m))
        V = _vals(r, _trigpts(n))
        assert _ninf(vals - U @ np.asarray(d) @ V.T) < 100 * tol             # pass(4)

        tt = _trigpts(101)
        ll = _trigpts(101)
        cv = _vals(c, ll)
        rv = _vals(r, tt)
        F = np.asarray(f.coeffs2(101, 101))
        # [u, s, v] = spherefun.vals2coeffs(c, d, r): coefficients of the
        # sampled factors, u*s*v.' == coeffs2(f, 101, 101)
        from chebfunjax.tech.trigtech import trig_vals2coeffs
        u = np.column_stack([np.asarray(trig_vals2coeffs(jnp.asarray(cv[:, j]))).ravel()
                             for j in range(cv.shape[1])])
        v = np.column_stack([np.asarray(trig_vals2coeffs(jnp.asarray(rv[:, j]))).ravel()
                             for j in range(rv.shape[1])])
        assert _ninf(F - u @ np.asarray(d) @ v.T) < 3 * tol                  # pass(5)
