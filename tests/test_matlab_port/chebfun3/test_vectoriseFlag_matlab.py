"""Port of MATLAB Chebfun tests/chebfun3/test_vectoriseFlag.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_vectoriseFlag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

EPS = float(np.finfo(np.float64).eps)


class TestChebfun3VectoriseFlag:
    def test_vectorize_matches_plain(self):
        # pass(1): the 'vectorize' wrapper reproduces the vectorised build.
        f1 = Chebfun3.from_function(lambda x, y, z: x)
        f2 = Chebfun3.from_function(lambda x, y, z: x, vectorize=True)
        assert float((f1 - f2).norm()) < 10 * EPS
