"""Port of MATLAB Chebfun tests/chebfun2v/test_roots_syntax.m (Fable 5).

FIXED (Fable 5): with both the marching-squares and Bezout-resultant
backends available, the full method-selection syntax is now cross-checked.
James Whidborne's problem (which once broke the rootfinding syntax in
Chebfun) is solved through every spelling and all must agree:

    roots(f, g, 'ms'), roots(f, g), roots(f, g, 'resultant'),
    roots([f; g], 'ms'), roots([f; g]), roots([f; g], 'resultant').

Provenance
----------
MATLAB source : tests/chebfun2v/test_roots_syntax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d import chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

from ._helpers import TOL, match_points

_Q = 1.4
_V1 = (0.0, 2 * np.pi, 0.01, 25.0)   # [phidomain, pdomain]


def _AB(phi, p):
    A = ((_Q * np.cos(_Q * np.pi) * jnp.cos(phi)
          + (2 * _Q ** 2 - 1) * np.sin(_Q * np.pi) * jnp.sin(phi))
         / (_Q ** 2 - 1)
         + np.pi * np.sin(_Q * np.pi) / p + np.cos(_Q * np.pi) / (_Q * p))
    B = ((-_Q * np.sin(_Q * np.pi) * jnp.cos(phi)
          + (2 * _Q ** 2 - 1) * np.cos(_Q * np.pi) * jnp.sin(phi))
         / (_Q ** 2 - 1)
         + np.pi * np.cos(_Q * np.pi) / p - np.sin(_Q * np.pi) / (_Q * p))
    return A, B


def _eq43(phi, p):
    A, B = _AB(phi, p)
    return (1 - _Q * A * p * np.cos(2 * np.pi * _Q)
            + _Q * B * p * np.sin(2 * np.pi * _Q)
            - _Q ** 2 * p * jnp.cos(phi) / (_Q ** 2 - 1))


def _eq44(phi, p):
    A, B = _AB(phi, p)
    return (_Q ** 2 * A * np.sin(2 * np.pi * _Q)
            + _Q ** 2 * B * np.cos(2 * np.pi * _Q)
            + _Q ** 2 * jnp.sin(phi) / (_Q ** 2 - 1))


class TestChebfun2vRootsSyntax:
    def test_all_matlab_assertions(self):
        eq43 = chebfun2(_eq43, _V1)
        eq44 = chebfun2(_eq44, _V1)

        r1 = eq43.roots(eq44, method="ms")            # marching squares
        r2 = eq43.roots(eq44)                          # auto (defaults to ms)
        r3 = eq43.roots(eq44, method="resultant")      # resultant

        F = Chebfun2v([eq43.approx, eq44.approx])
        r4 = F.roots(method="ms")                      # marching squares
        r5 = F.roots()                                 # auto
        r6 = F.roots(method="resultant")               # resultant

        assert match_points(r1, r2, TOL)
        assert match_points(r1, r3, TOL)
        assert match_points(r1, r4, TOL)
        assert match_points(r1, r5, TOL)
        assert match_points(r1, r6, TOL)
