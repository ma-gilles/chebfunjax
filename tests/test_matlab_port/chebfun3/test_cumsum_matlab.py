"""Port of MATLAB Chebfun tests/chebfun3/test_cumsum.m (Fable 5).

FIXED (Fable 5): Chebfun3.cumsum / cumsum3 added in the audit.

Provenance
----------
MATLAB source : tests/chebfun3/test_cumsum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import Chebfun, Domain
from chebfunjax.chebfun3d.chebfun3 import Chebfun3

from ._helpers import EPS, maxdiff

TOL = 100 * EPS


class TestChebfun3Cumsum:
    def test_cube_domain(self):
        cf = Chebfun3.from_function
        x = cf(lambda x, y, z: x)
        y = cf(lambda x, y, z: y)
        z = cf(lambda x, y, z: z)

        assert maxdiff(x.cumsum(), lambda x, y, z: 0.5 * (x ** 2 - 1)) < TOL
        assert maxdiff(x.cumsum(1), lambda x, y, z: 0.5 * (x ** 2 - 1)) < TOL
        assert maxdiff(x.cumsum(2), lambda x, y, z: x * (y + 1)) < TOL
        assert maxdiff(x.cumsum(3), lambda x, y, z: x * (z + 1)) < TOL

        assert maxdiff(y.cumsum(), lambda x, y, z: y * (x + 1)) < TOL
        assert maxdiff(y.cumsum(1), lambda x, y, z: y * (x + 1)) < TOL
        assert maxdiff(y.cumsum(2), lambda x, y, z: 0.5 * (y ** 2 - 1)) < TOL
        assert maxdiff(y.cumsum(3), lambda x, y, z: y * (z + 1)) < TOL

        assert maxdiff(z.cumsum(), lambda x, y, z: z * (x + 1)) < TOL
        assert maxdiff(z.cumsum(1), lambda x, y, z: z * (x + 1)) < TOL
        assert maxdiff(z.cumsum(2), lambda x, y, z: z * (y + 1)) < TOL
        assert maxdiff(z.cumsum(3), lambda x, y, z: 0.5 * (z ** 2 - 1)) < TOL

    def test_box_domain(self):
        dom = (-1.1, 2.0, -0.2, 3.0, 5.0, 6.0)
        cf = Chebfun3.from_function
        x = cf(lambda x, y, z: x, domain=dom)
        y = cf(lambda x, y, z: y, domain=dom)
        z = cf(lambda x, y, z: z, domain=dom)

        def md(f, ref):
            return maxdiff(f, ref, dom=dom)

        assert md(x.cumsum(), lambda x, y, z: 0.5 * (x ** 2 - 1.1 ** 2)) < TOL
        assert md(x.cumsum(1), lambda x, y, z: 0.5 * (x ** 2 - 1.1 ** 2)) < TOL
        assert md(x.cumsum(2), lambda x, y, z: x * (y + 0.2)) < TOL
        assert md(x.cumsum(3), lambda x, y, z: x * (z - 5)) < TOL

        assert md(y.cumsum(), lambda x, y, z: y * (x + 1.1)) < TOL
        assert md(y.cumsum(1), lambda x, y, z: y * (x + 1.1)) < TOL
        assert md(y.cumsum(2), lambda x, y, z: 0.5 * (y ** 2 - 0.2 ** 2)) < TOL
        assert md(y.cumsum(3), lambda x, y, z: y * (z - 5)) < TOL

        assert md(z.cumsum(), lambda x, y, z: z * (x + 1.1)) < TOL
        assert md(z.cumsum(1), lambda x, y, z: z * (x + 1.1)) < TOL
        assert md(z.cumsum(2), lambda x, y, z: z * (y + 0.2)) < TOL
        assert md(z.cumsum(3), lambda x, y, z: 0.5 * (z ** 2 - 25)) < TOL

    def test_triple_cumsum_is_cumsum3(self):
        dom = (-1.1, 2.0, -0.2, 3.0, 5.0, 6.0)
        f = Chebfun3.from_function(
            lambda x, y, z: jnp.sin((x - 0.1) * (y + 0.1) * (z + 0.1)),
            domain=dom)
        c3 = f.cumsum3()
        assert maxdiff(f.cumsum().cumsum(2).cumsum(3),
                       lambda x, y, z: c3(x, y, z), dom=dom) < TOL
        assert maxdiff(f.cumsum(1).cumsum(2).cumsum(3),
                       lambda x, y, z: c3(x, y, z), dom=dom) < TOL

    def test_matches_1d_cumsum_slice(self):
        # cumsum along x of exp(x) equals the 1D cumsum of exp on a slice.
        f = Chebfun.from_function(lambda t: jnp.exp(t), Domain((-1.0, 1.0)))
        f3 = Chebfun3.from_function(lambda x, y, z: jnp.exp(x))
        g = f.cumsum()
        g3 = f3.cumsum()
        s = np.linspace(-1, 1, 11)
        S = jnp.asarray(s)
        g3x = g3(S, jnp.zeros_like(S), jnp.full_like(S, 0.3))
        assert float(jnp.max(jnp.abs(g3x - g(S)))) < TOL
