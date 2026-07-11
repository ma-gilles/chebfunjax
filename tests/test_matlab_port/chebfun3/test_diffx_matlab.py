"""Port of MATLAB Chebfun tests/chebfun3/test_diffx.m (Fable 5).

MATLAB diffx(f, k) maps to chebfunjax f.diff(dim=1, k).

Provenance
----------
MATLAB source : tests/chebfun3/test_diffx.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

from ._helpers import EPS, maxdiff

TOL = 1e4 * EPS
DOMS = [(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0),
        (-2.0, 0.0, 0.0, 2.0, -1.0, 1.0)]


FF = [lambda x, y, z: x, lambda x, y, z: jnp.cos(x) * jnp.exp(y) * jnp.sin(z), lambda x, y, z: x ** 2 + x * y ** 2 * z ** 3]
FD1 = [lambda x, y, z: 1 + 0 * x, lambda x, y, z: -jnp.sin(x) * jnp.exp(y) * jnp.sin(z), lambda x, y, z: 2 * x + y ** 2 * z ** 3]
FD2 = [lambda x, y, z: 0 * x, lambda x, y, z: -jnp.cos(x) * jnp.exp(y) * jnp.sin(z), lambda x, y, z: 2.0 + 0 * x]


class TestChebfun3DiffX:
    @pytest.mark.parametrize("jj", range(3))
    @pytest.mark.parametrize("dom", DOMS)
    def test_first_and_second_derivative(self, jj, dom):
        f = Chebfun3.from_function(FF[jj], domain=dom)
        assert maxdiff(f.diff(dim=1, k=1), FD1[jj], dom) < TOL
        assert maxdiff(f.diff(dim=1, k=2), FD2[jj], dom) < 10 * TOL
