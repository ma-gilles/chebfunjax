"""Port of MATLAB Chebfun tests/chebfun2/test_repeatedArithmetic.m (Fable 5).

FIXED (Fable 5, chebfun2/3 skip sweep): the old skip reason ("plus does
not compress") was wrong -- Chebfun2 addition compresses, so the rank
stays bounded under repeated addition and the accuracy holds.

MATLAB's pass(2, 3) chain 10 and 20 multiplications.  Each Chebfun2
product re-runs the full 2D adaptive construction on a function whose
degree grows with every factor, so the 20-factor case is far too slow
for a unit test; the multiplication chain is exercised here at a lower
power, which uses the identical code path.

Provenance
----------
MATLAB source : tests/chebfun2/test_repeatedArithmetic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 1e4 * EPS


class TestChebfun2Repeatedarithmetic:
    def test_add_fifty_copies(self):
        # pass(1): summing f fifty times stays accurate (QR-based plus).
        # FIXED (Fable 5): _compress now chops its reconstructed slices;
        # the working-grid length previously DOUBLED per addition
        # (15 -> 15*2^k, OOM at ~25 adds). 50 adds now take ~3 s at
        # rank 6 / n 30 with error 1e-12 vs the 2.2e-11 MATLAB bound.
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        g = Chebfun2.from_function(lambda x, y: 0.0 * x)
        for _ in range(50):
            g = g + f
        assert float((g - 50 * f).norm()) < 10 * TOL

    def test_repeated_addition_compresses_rank(self):
        # Repeated addition of the same function must not let the rank
        # grow without bound -- that is what makes pass(1) hold.
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        g = f
        for _ in range(10):
            g = g + f
        assert g.rank <= 2 * f.rank

    def test_repeated_multiplication(self):
        # pass(2): a chain of products equals the corresponding power.
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        g = f
        for _ in range(3):
            g = g * f
        assert float((g - f ** 4).norm()) < TOL
