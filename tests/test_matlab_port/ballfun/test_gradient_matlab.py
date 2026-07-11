"""Port of MATLAB Chebfun tests/ballfun/test_gradient.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_gradient.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import X0, val


class TestBallfunGradient:
    def test_grad_of_exp_x(self):
        f = Ballfun.from_function(lambda x, y, z: jnp.exp(x))
        gx, gy, gz = f.grad()
        assert abs(val(gx) - np.exp(X0)) < 1e-7
        assert abs(val(gy)) < 1e-7
        assert abs(val(gz)) < 1e-7
