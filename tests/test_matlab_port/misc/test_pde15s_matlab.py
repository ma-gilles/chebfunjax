"""Port of MATLAB Chebfun tests/misc/test_pde15s.m (Fable 5).

MATLAB solves the same PDE with the ``chebtech1`` and ``chebtech2``
techs and checks the solutions agree; here the initial conditions are
built on first- and second-kind points (``chebkind``) and the two runs
are compared.  MATLAB's ``bc.left = @(u) diff(u); bc.right = 0`` is
``lbc=lambda u: u.diff(), rbc=0``.

Provenance
----------
MATLAB source : tests/misc/test_pde15s.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun1d.pde15s import pde15s
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)


class TestMiscPde15s:
    def test_all_matlab_assertions(self):
        tol = 1e5 * ChebfunPref().chebfuneps
        t = np.arange(0, 1.0001, 0.1)
        pde = lambda tt, x, u: .1 * u.diff(2) + u.diff()  # noqa: E731

        f = chebfun(lambda x: jnp.sin(np.pi * x), domain=(-2.5, 3.0), chebkind=1)
        uu = pde15s(pde, t, f, lbc=lambda u: u.diff(), rbc=0.0, rtol=1e-10, atol=1e-12)

        f = chebfun(lambda x: jnp.sin(np.pi * x), domain=(-2.5, 3.0))
        vv = pde15s(pde, t, f, lbc=lambda u: u.diff(), rbc=0.0, rtol=1e-10, atol=1e-12)

        err = max(float((u - v).norm(2)) for u, v in zip(uu, vv))
        assert err < tol                                            # pass(1)
