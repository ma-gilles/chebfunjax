"""Port of MATLAB Chebfun tests/chebfun3/test_plus.m (Fable 5).

MATLAB checks norm(f+g - FplusG) < tol; the port checks the max error
on an interior lattice at the same tolerance (the uncompressed sum is
exact pointwise, so this is the same assertion).

Provenance
----------
MATLAB source : tests/chebfun3/test_plus.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

from ._helpers import EPS, maxdiff

TOL = 1e5 * EPS
DOMS = [(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        (-2.0, 2.0, -2.0, 2.0, -2.0, 2.0),
        (-1.0, float(np.pi), 0.0, float(2 * np.pi),
         -float(np.pi), float(np.pi))]


def ff(x, y, z):
    return jnp.cos(x * y * z)


def gg(x, y, z):
    return x + y + z + x * y * z


class TestChebfun3Plus:
    @pytest.mark.parametrize("dom", DOMS)
    def test_plus_matches_direct_construction(self, dom):
        f = Chebfun3.from_function(ff, domain=dom)
        g = Chebfun3.from_function(gg, domain=dom)
        tolk = float(np.max(np.abs(dom))) * TOL
        assert maxdiff(f + g, lambda x, y, z: ff(x, y, z) + gg(x, y, z),
                       dom) < tolk

    def test_rank_not_inflated(self):
        # MATLAB pass(4): rank(f+f) == rank(f) after compression.
        # Chebfun3.plus rebuilds the sum through the constructor (MATLAB's
        # active plus path), so the Tucker rank stays minimal.
        f = Chebfun3.from_function(lambda x, y, z: x)
        g = f + f
        assert maxdiff(g, lambda x, y, z: 2 * x) < TOL
        assert g.rank == f.rank == (1, 1, 1)
