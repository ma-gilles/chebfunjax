"""Port of MATLAB Chebfun tests/misc/test_randnfunsphere.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_randnfunsphere.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import numpy as np

from chebfunjax.chebfun1d.randfuns import randnfunsphere

jax.config.update("jax_enable_x64", True)


class TestMiscRandnfunsphere:
    def test_all_matlab_assertions(self):
        np.random.seed(0)
        f = randnfunsphere(.2)
        assert abs(float((f ** 2).mean2()) - 1) < .1                # pass(1)
        assert abs(float(f.mean2())) < .1                           # pass(2)
        f = randnfunsphere(1e6)
        assert float(f.diff().norm("fro")) < 1e-4                   # pass(3)
        f = randnfunsphere(3.1)
        assert f.rank == 4                                        # pass(4)
        f = randnfunsphere(3.1, "mono")
        assert f.rank == 3                                        # pass(5)
