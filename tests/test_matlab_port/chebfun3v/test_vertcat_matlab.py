"""Port of MATLAB Chebfun tests/chebfun3v/test_vertcat.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_vertcat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = 2.220446049250313e-16
TOL = 1e3 * EPS

DOMS = [(-1, 1, -1, 1, -1, 1), (-3, 2, -1, 2, -1, 1)]


class TestChebfun3vVertcat:
    @pytest.mark.parametrize("dom", DOMS)
    def test_vertcat(self, dom):
        f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x * y * z),
                                   domain=dom)
        F = Chebfun3v([f, f])
        G = Chebfun3v([f, f, f])
        # [F; f] appends a scalar component; [f; F] prepends one.
        H = Chebfun3v(list(F.components) + [f])
        K = Chebfun3v([f] + list(F.components))

        Fc = F.components
        Gc = G.components
        assert float((Fc[0] - f).norm()) < TOL
        assert float((Fc[1] - f).norm()) < TOL
        assert float((Gc[0] - f).norm()) < TOL
        assert float((Gc[1] - f).norm()) < TOL
        assert float((Gc[2] - f).norm()) < TOL
        assert float((G - H).norm()) < TOL
        assert float((G - K).norm()) < TOL
