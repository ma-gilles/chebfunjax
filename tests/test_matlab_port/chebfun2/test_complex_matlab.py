"""Port of MATLAB Chebfun tests/chebfun2/test_complex.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_complex.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

TOL = 1000 * float(np.finfo(np.float64).eps)
_T = np.linspace(-1.0, 1.0, 61)
_A, _B = np.meshgrid(_T, _T)
_JA, _JB = jnp.asarray(_A), jnp.asarray(_B)


class TestChebfun2Complex:
    def test_complex_one_arg(self):
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        d = f - Chebfun2.complex(f)
        assert float(np.max(np.abs(np.asarray(d(_JA, _JB))))) < TOL

    def test_complex_two_args(self):
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        d = (f + 1j * f) - Chebfun2.complex(f, f)
        assert float(np.max(np.abs(np.asarray(d(_JA, _JB))))) < TOL
