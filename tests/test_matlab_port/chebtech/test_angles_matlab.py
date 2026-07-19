"""Port of MATLAB Chebfun tests/chebtech/test_angles.m (Opus 4.8).

MATLAB ``chebtech{1,2}.angles(n)`` returns ``acos(chebpts(n))`` (the angles of
the Chebyshev points).  chebfunjax now implements ``Chebtech{1,2}.angles`` as an
exact port of ``@chebtech{1,2}/angles.m``; the check ``cos(angles(n)) ==
chebpts(n)`` reproduces the MATLAB test.

Provenance
----------
MATLAB source : tests/chebtech/test_angles.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.utils.quadrature import chebpts

_TOL = 1e-15


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestChebtechAngles:
    def test_chebtech1(self):
        # pass(1): cos(chebtech1.angles(10)) == chebtech1.chebpts(10).
        x = chebpts(10, 1)
        t = Chebtech1.angles(10)
        assert _ninf(jnp.cos(t) - x) < _TOL

    def test_chebtech2(self):
        # pass(2): cos(chebtech2.angles(10)) == chebtech2.chebpts(10).
        x = chebpts(10, 2)
        t = Chebtech2.angles(10)
        assert _ninf(jnp.cos(t) - x) < _TOL
