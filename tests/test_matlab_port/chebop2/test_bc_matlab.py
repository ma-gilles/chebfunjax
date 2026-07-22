"""Port of MATLAB Chebfun tests/chebop2/test_bc.m (Fable 5).

Checks that the ``N.bc`` shorthand (set all four edges at once) produces the
same solution as setting ``lbc``/``rbc``/``ubc``/``dbc`` individually.  This
is a same-solver identity, so the two solves are bit-for-bit identical.

Provenance
----------
MATLAB source : tests/chebop2/test_bc.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.chebop2 import Chebop2, laplacian


class TestChebop2Bc:
    def test_bc_shorthand_matches_individual_edges(self):
        # tol = 1e-14 (MATLAB)
        tol = 1e-14

        # Individual edges: lbc = rbc = ubc = dbc = 0.
        N = Chebop2(lambda u: laplacian(u))
        N.lbc = 0.0
        N.rbc = 0.0
        N.ubc = 0.0
        N.dbc = 0.0
        exact = N.solve(1.0, n=17)

        # Shorthand: N.bc = 0.
        M = Chebop2(lambda u: laplacian(u))
        M.bc = 0.0
        u = M.solve(1.0, n=17)

        xx = np.linspace(-1.0, 1.0, 40)
        X, Y = np.meshgrid(xx, xx)
        a = np.asarray(u(jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())))
        b = np.asarray(exact(jnp.asarray(X.ravel()), jnp.asarray(Y.ravel())))
        assert np.max(np.abs(a - b)) < tol
