"""Independent quadrature control for a diagonal integral.

These are supplemental Python checks, not translated native test assertions.
The formerly cited native test file does not exist at Chebfun7574c77.
The corresponding library operation is documented in @chebfun2/trace.m;
the analytic formulas and diagnostic bounds below are supplemental oracles.
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
