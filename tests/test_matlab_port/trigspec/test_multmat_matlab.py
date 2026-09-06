"""Port of MATLAB Chebfun tests/trigspec/test_multmat.m (Fable 5).

Provenance
----------
MATLAB source : tests/trigspec/test_multmat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import toeplitz

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.operators.trigspec import multmat

jax.config.update("jax_enable_x64", True)


def _exact(n):
    C = np.zeros(n, dtype=complex)
    R = np.zeros(n, dtype=complex)
    C[1] = -1j / 2
    R[1] = 1j / 2
    return toeplitz(C, R)


class TestTrigspecMultmat:
    def test_all_matlab_assertions(self):
        tol = 10 * ChebfunPref().techPrefs.chebfuneps
        for m, n in ((11, 31), (11, 32), (12, 15), (12, 16)):
            f = chebfun(lambda x: jnp.sin(np.pi * x), n=m, trig=True)
            M = np.asarray(multmat(n, f))
            assert np.linalg.norm(M - _exact(n), 2) < tol                    # pass(1)-(4)
