"""Port of MATLAB Chebfun tests/unbndfun/test_restrict.m (Fable 5).

FIXED: Unbndfun.restrict added in the Fable 5 audit -- a breakpoint
partition of an unbounded domain yields Bndfun pieces for finite
subintervals and Unbndfun pieces for semi-infinite ends.

Provenance
----------
MATLAB source : tests/unbndfun/test_restrict.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.unbndfun import Unbndfun

TOL = 1e-12


class TestUnbndfunRestrict:
    def test_partition_of_real_line(self):
        def op(x):
            return x ** 2 * jnp.exp(-(x ** 2))

        f = Unbndfun.from_function(op, Domain((-np.inf, np.inf)))
        parts = f.restrict((-np.inf, -2.0, 7.0, np.inf))
        assert isinstance(parts[0], Unbndfun)
        assert isinstance(parts[1], Bndfun)
        assert isinstance(parts[2], Unbndfun)
        rng = np.random.default_rng(6178)
        grids = [
            jnp.asarray(-100 + 98 * rng.random(100)),
            jnp.asarray(-2 + 9 * rng.random(100)),
            jnp.asarray(7 + 93 * rng.random(100)),
        ]
        for g, xs in zip(parts, grids):
            assert float(jnp.max(jnp.abs(g(xs) - op(xs)))) < TOL
