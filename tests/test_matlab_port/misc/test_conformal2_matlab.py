"""Port of MATLAB Chebfun tests/misc/test_conformal2.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_conformal2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.utils.conformal2 import conformal2

jax.config.update("jax_enable_x64", True)


class TestMiscConformal2:
    def test_all_matlab_assertions(self):
        z = chebfun(lambda t: jnp.exp(1j * np.pi * t), trig=True)
        C1 = z * abs(1 + .1 * z ** 4)
        C2 = .5 * z * abs(1 + .2 * z ** 3)
        f, finv, rho, *_ = conformal2(C1, C2)
        w = 1 + 0.1j
        assert abs(rho - .539197) < 1e-3                             # pass(1)
        assert abs(complex(np.asarray(finv(np.asarray(f(w))))) - w) < 1e-3
