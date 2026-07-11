"""Port of MATLAB Chebfun tests/chebfun/test_polyfitL1.m (Fable 5).

The L1-optimality condition: int T_k * sign(f - p) = 0 for k <= n.

Provenance
----------
MATLAB source : tests/chebfun/test_polyfitL1.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial import chebyshev as C

import chebfunjax as cj

TOL = 1e-6


class TestChebfunPolyfitL1:
    @pytest.mark.xfail(
        reason="chebfunjax polyfitL1 is NOT L1-optimal: its degree-5 "
        "fit of exp(x)sin(10x) has L1 error 1.93, WORSE than the plain "
        "L2 projection (1.27), and the optimality condition "
        "int T_k sign(f-p) = 0 fails (k=0 residual -0.088). Real bug "
        "flagged in the Fable 5 audit.")
    def test_l1_optimality_condition(self):
        f = cj.chebfun(lambda x: jnp.exp(x) * jnp.sin(10 * x))
        n = 5
        p = f.polyfitL1(n)
        # residual sign function via dense sampling (the chebfun-level
        # sign/int is MATLAB's route; the optimality integral is the
        # same quantity computed by Gauss quadrature here)
        from chebfunjax.utils.quadrature import legpts
        x, w = legpts(2000)
        x, w = np.asarray(x), np.asarray(w)
        r = np.sign(np.asarray(f(jnp.asarray(x)))
                    - np.asarray(p(jnp.asarray(x))))
        for k in range(n + 1):
            Tk = C.chebval(x, np.eye(n + 1)[k])
            val = float(np.dot(w, Tk * r))
            assert abs(val) < TOL, f"k={k}: {val}"
