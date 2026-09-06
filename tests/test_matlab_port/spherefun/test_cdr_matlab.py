"""Port of MATLAB Chebfun tests/spherefun/test_cdr.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_cdr.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebpref import ChebfunPref

from ._cart import sph_lt, sph_xyz

jax.config.update("jax_enable_x64", True)


def _vals(techs, x):
    from chebfunjax.tech.trigtech import _trig_eval_np
    return np.column_stack([np.real(np.asarray(_trig_eval_np(
        np.asarray(t.coeffs)[:, None], np.asarray(x) / np.pi,
        is_real=t.is_real))).ravel() for t in techs])


class TestSpherefunCdr:
    def test_all_matlab_assertions(self):
        tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
        f = sph_lt(lambda lam, th: jnp.sin(th) * jnp.sin(lam))
        C, D, R = f.cdr()
        x = np.linspace(-np.pi, np.pi, 200)
        assert np.linalg.norm(_vals(C, x) - _vals(R, x)) < tol              # pass(1)

        f = sph_xyz(lambda x, y, z: jnp.exp(-((x - .4) ** 2 + (y - .9) ** 2 + (z - .1) ** 2)))
        C, D, R = f.cdr()
        g = chebfun2(lambda lam, th: f(lam, th), domain=(-np.pi, np.pi, -np.pi, np.pi))
        lam = np.linspace(-np.pi, np.pi, 41)
        th = np.linspace(-np.pi, np.pi, 37)
        CDR = _vals(C, th) @ np.asarray(D) @ _vals(R, lam).T          # C * D * R'
        G = np.asarray(g.fevalm(jnp.asarray(lam), jnp.asarray(th)))
        assert np.max(np.abs(G - CDR)) < tol                                 # pass(2)
