"""Port of MATLAB Chebfun tests/deltafun/test_chebcoeffs.m (Opus 4.8).

MATLAB ``chebcoeffs(df)`` delegates to the Chebyshev coefficients of the
Deltafun's smooth part.  chebfunjax has no ``chebcoeffs`` method on Deltafun,
but exposes the same coefficients via ``df.funPart.coeffs``; the constructor
must leave those coefficients untouched.

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
        npt.assert_array_equal(np.array(df.funPart.coeffs), np.array(f.coeffs))
