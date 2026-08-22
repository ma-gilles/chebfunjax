"""Port of MATLAB Chebfun tests/ballfun/test_coeffs2vals.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_coeffs2vals.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.utils.quadrature import trigpts

jax.config.update("jax_enable_x64", True)

TOL = 1e2 * 2.220446049250313e-16


class TestBallfunCoeffs2Vals:
    def test_all_matlab_assertions(self):
        # Round trip with f = sin(x)*y*z
        f = Ballfun.from_function(lambda x, y, z: jnp.sin(x) * y * z)
        exact = np.array(f.coeffs)
        cfs = np.array(Ballfun.vals2coeffs(Ballfun.coeffs2vals(f.coeffs)))
        assert np.max(np.abs(exact - cfs)) < TOL  # pass(1)

        # Example 2
        lam = np.pi * np.array(trigpts(3)[0])
        vals = np.exp(1j * lam)[None, :]
        cfs = np.array([0.0, 0.0, 1.0])[None, :]
        out = np.array(Ballfun.coeffs2vals(cfs))
        assert np.linalg.norm((out - vals).ravel()) < TOL  # pass(2)
