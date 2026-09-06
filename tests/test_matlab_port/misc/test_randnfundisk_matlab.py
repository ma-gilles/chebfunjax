"""Port of MATLAB Chebfun tests/misc/test_randnfundisk.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_randnfundisk.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import numpy as np

from chebfunjax.chebfun1d.randfuns import randnfundisk

jax.config.update("jax_enable_x64", True)


class TestMiscRandnfundisk:
    def test_all_matlab_assertions(self):
        np.random.seed(0)
        f = randnfundisk(.2)
        assert abs(float((f ** 2).mean2()) - 1) < .1                # pass(1)
        assert abs(float(f.mean2())) < .1                           # pass(2)
        f = randnfundisk(40)
        d = float(f.max2()) - float(f.min2())
        assert 0.001 < d < 0.5                                      # pass(3)
