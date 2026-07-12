"""Port of MATLAB Chebfun tests/chebfun2/test_trace.m (Fable 5).

FIXED: Chebfun2.trace added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun2/test_trace.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2


class TestChebfun2Trace:
    def test_trace_is_diag_integral(self):
        f = Chebfun2.from_function(lambda x, y: jnp.exp(x * y))
        # int exp(t^2) dt on [-1,1] via scipy
        from scipy.integrate import quad
        ref = quad(lambda t: np.exp(t * t), -1, 1, epsabs=1e-14)[0]
        assert abs(float(f.trace()) - ref) < 1e-11
