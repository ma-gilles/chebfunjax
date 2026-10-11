"""Port of MATLAB Chebfun tests/deltafun/test_chebcoeffs.m (Opus 4.8).

Literal original predicate calls both public coefficient methods.
The Python constructor uses labeled location/magnitude positions matching
native data.deltaLoc/data.deltaMag; scalar coefficient vectors stay1D.

Provenance
----------
MATLAB source : tests/deltafun/test_chebcoeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun


class TestDeltafunChebcoeffs:
    def test_chebcoeffs_delegates_to_funpart(self):
        # pass(1): all( chebcoeffs(f) == chebcoeffs(df) )
        a, b = -4.0, 4.0
        f = Bndfun.from_function(jnp.sin, Domain((a, b)))
        mag = 0.9 * (a + (b - a) * np.random.rand(3, 3))
        loc = 0.9 * (a + (b - a) * np.random.rand(3))
        df = Deltafun(f, jnp.asarray(loc), jnp.asarray(mag))
        npt.assert_array_equal(np.asarray(df.chebcoeffs()), np.asarray(f.chebcoeffs()))
