"""Port of MATLAB Chebfun tests/misc/test_chebpolyval.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_chebpolyval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun, chebpoly
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.utils.polynomials import chebpolyval

jax.config.update("jax_enable_x64", True)


class TestMiscChebpolyval:
    def test_all_matlab_assertions(self):
        rng = np.random.RandomState(42)
        tol = 10 * ChebfunPref().chebfuneps
        n = 10
        c = rng.rand(10, 2)
        x = rng.rand(3, 3)
        fx1 = np.asarray(chebpolyval(c, x))
        T = chebpoly(np.arange(10))
        cf = c[::-1, :]                                   # T*flipud(c)
        Tc = [sum((T.extract_columns(k) * float(cf[k, j]) for k in range(10)),
                  0.0 * T.extract_columns(0)) for j in range(2)]
        fx2 = np.stack([np.asarray(Tc[j](jnp.asarray(x))) for j in range(2)], axis=-1)
        assert np.max(np.abs(fx1 - fx2)) < n * tol                  # pass(1)

        x = chebfun("x")
        f = chebpolyval(c, x)
        err = max(float((f[j] - Tc[j]).norm(jnp.inf)) for j in range(2))
        assert err < n * tol                                        # pass(2)
