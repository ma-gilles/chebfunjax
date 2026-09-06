"""Port of MATLAB Chebfun tests/chebfun/test_deltaOps.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_deltaOps.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)

EPS = np.finfo(float).eps


class TestChebfunDeltaOps:
    def test_all_matlab_assertions(self):
        tol = ChebfunPref().deltaPrefs.deltaTol
        rng = np.random.RandomState(0)

        n = 6
        x = chebfun("x", domain=(0.0, n))
        f = 0.5 * x.sin()
        for j in range(1, n):
            f = f + float(rng.randn()) * cj.dirac(x - j)
        assert float((f.cumsum().diff() - f).norm(2)) < tol         # pass(1)

        x = chebfun("x")
        f = cj.dirac(x - .5) + cj.dirac(x) + cj.dirac(x + .5) + cj.heaviside(x)
        assert float((f.cumsum().diff() - f).norm(2)) < tol         # pass(2)

        x = chebfun("x")
        f = x.sign() + (x - .5).sign()
        f2 = float(f(jnp.asarray(-1.0))) + f.diff().cumsum()
        assert float((f - f2).norm(2)) < tol                        # pass(3)

        x = chebfun("x", domain=(0.0, 5.0))
        f = 0.5 * x.sin()
        A = rng.randn(4)
        for j in range(1, 5):
            f = f + float(A[j - 1]) * cj.dirac(x - j)
        F = (.5 * x.sin()).cumsum()
        for j in range(1, 5):
            F = F + float(A[j - 1]) * cj.heaviside(x - j)
        assert float((F.diff() - (f - float(f(jnp.asarray(0.0))))).norm(2)) < tol  # pass(4)

        x = chebfun("x")
        assert float(cj.innerProduct(cj.dirac(x), x)) < EPS         # pass(5)

        x = chebfun("x")
        A = [cj.dirac(x - 1), cj.dirac(x - 0.5), cj.dirac(x - 0.0)]
        assert float((A[1] - cj.dirac(x - .5)).norm(2)) < EPS and len(A) == 3  # pass(6)
