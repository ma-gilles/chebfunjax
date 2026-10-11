"""Native pde15s technology-session and quasimatrix Frobenius predicate.

MATLAB source: tests/misc/test_pde15s.m; @chebfun/pdeSolve.m.
Chebfun commit: 7574c77.
The list of output Chebfuns and lbc/rbc keywords are Python API adapters.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun1d.pde15s import pde15s
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


class TestMiscPde15s:
    def test_all_matlab_assertions(self):
        tol = 1e5 * ChebfunPref().chebfuneps
        t = np.arange(0, 1.0001, 0.1)
        pde = lambda tt, x, u: 0.1 * u.diff(2) + u.diff()
        saved = ChebfunPref._defaults
        try:
            ChebfunPref.setDefaults("tech", Chebtech1)
            f = chebfun(lambda x: jnp.sin(np.pi * x), domain=(-2.5, 3.0))
            assert isinstance(f.funs[0].tech, Chebtech1)
            # Native pdeset('plot','off') leaves ODE/spatial tolerances at1e-6.
            # The Python returning-output API does not plot by default.
            uu = pde15s(pde, t, f, lbc=lambda u: u.diff(), rbc=0.0)

            ChebfunPref.setDefaults("tech", Chebtech2)
            f = chebfun(lambda x: jnp.sin(np.pi * x), domain=(-2.5, 3.0))
            assert isinstance(f.funs[0].tech, Chebtech2)
            vv = pde15s(pde, t, f, lbc=lambda u: u.diff(), rbc=0.0)

            # Native uu/vv each have11 columns. Check list shape before zip.
            assert len(uu) == len(vv) == len(t) == 11
            column_norms = np.asarray([float((u - v).norm(2)) for u, v in zip(uu, vv, strict=True)])
            # @chebfun/norm.m: default array/quasimatrix norm is Frobenius.
            err = float(np.linalg.norm(column_norms, 2))
            print(
                {
                    "frobenius_error": err,
                    "max_column_error": float(column_norms.max()),
                    "source_bound": tol,
                    "column_L2_errors": column_norms.tolist(),
                    "first_run_techs": [type(u.funs[0].tech).__name__ for u in uu],
                },
                flush=True,
            )
            assert err < tol
        finally:
            ChebfunPref._defaults = saved
