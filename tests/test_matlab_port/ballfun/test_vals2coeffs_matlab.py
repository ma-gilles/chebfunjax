"""Port of MATLAB Chebfun tests/ballfun/test_vals2coeffs.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_vals2coeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.utils.quadrature import trigpts

jax.config.update("jax_enable_x64", True)

TOL = 1e4 * 2.220446049250313e-16


class TestBallfunVals2Coeffs:
    def test_all_matlab_assertions(self):
        # Example 1
        X = np.ones((21, 22, 23))
        V = np.array(Ballfun.vals2coeffs(X))
        V2 = np.zeros((21, 22, 23), dtype=complex)
        V2[0, 11, 11] = 1.0
        assert np.linalg.norm((V - V2).ravel()) < TOL  # pass(1)

        # Example 2
        S = (100, 150, 200)
        n, p = S[1], S[2]
        X = np.ones(S)
        V = np.array(Ballfun.vals2coeffs(X))
        V2 = np.zeros(S, dtype=complex)
        V2[0, n // 2, p // 2] = 1.0
        assert np.linalg.norm((V - V2).ravel()) < TOL  # pass(2)

        # Example 3
        lam = np.pi * np.array(trigpts(3)[0])
        vals = np.exp(1j * lam)[None, :]  # row vector, as in MATLAB
        cfs = np.array([0.0, 0.0, 1.0])[None, :]
        out = np.array(Ballfun.vals2coeffs(vals))
        assert np.linalg.norm((out - cfs).ravel()) < TOL  # pass(3)
