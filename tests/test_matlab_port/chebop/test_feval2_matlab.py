"""Port of MATLAB Chebfun tests/chebop/test_feval2.m (Fable 5).

FIXED: chebop application to eigenfunctions (MATLAB A*ev) works via
Chebop.__call__; each eigenfunction column wraps as a chebfun and
A(v) = lambda v holds.

Provenance
----------
MATLAB source : tests/chebop/test_feval2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.chebop import Chebop, _chebfun_from_values

XS = jnp.asarray(np.linspace(-0.9, 0.9, 15))


class TestChebopFeval2:
    def test_apply_to_eigenfunctions(self):
        A = Chebop(lambda x, u: u.diff(2) + x * u, (-1.0, 1.0))
        A.lbc = 0.0
        A.rbc = 0.0
        out1, out2 = A.eigs(k=3, return_eigenfunctions=True)
        # scalar eigs returns (values, functions); systems return the
        # reverse -- detect by which element holds chebfuns
        if hasattr(out1, "__len__") and len(out1) and \
                hasattr(out1[0], "funs"):
            evs, lam = out1, out2
        else:
            lam, evs = out1, out2
        lams = np.real(np.asarray(jnp.asarray(lam))).ravel()
        for i in range(3):
            ev_i = evs[i]
            v = ev_i if hasattr(ev_i, "funs") else \
                _chebfun_from_values(jnp.real(jnp.asarray(ev_i)),
                                     (-1.0, 1.0))
            g = A(v)  # pass(1)-(3): application does not crash
            resid = float(jnp.max(jnp.abs(
                g(XS) - float(lams[i]) * v(XS))))
            scale = float(jnp.max(jnp.abs(v(XS)))) * abs(lams[i])
            assert resid < 1e-6 * max(scale, 1.0)
