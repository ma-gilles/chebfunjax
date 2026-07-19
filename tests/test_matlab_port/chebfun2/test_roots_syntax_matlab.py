"""Port of MATLAB Chebfun tests/chebfun2/test_roots_syntax.m (Fable 5).

FIXED (Fable 5): ``roots(f, g)`` now dispatches to both the marching-squares
and the Bezout-resultant common-zero finders, so every spelling of the call
is exercised and cross-checked:

    roots(f, g, 'resultant'), roots(f, g, 'ms'), roots(f, g,
    'marchingsquares'), roots(f, g), roots([f; g], 'resultant'),
    roots([f; g], 'marchingsquares'), roots([f; g]).

Same-method spellings must be byte-identical; the marching-squares and
resultant results must agree to ``10*eps`` (the MATLAB tolerance).

Provenance
----------
MATLAB source : tests/chebfun2/test_roots_syntax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d import chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

_TOL = 10 * float(np.finfo(np.float64).eps)


def _sorted_cols(r):
    r = np.asarray(r)
    return np.sort(r[:, 0]), np.sort(r[:, 1])


class TestChebfun2Rootssyntax:
    def test_all_matlab_assertions(self):
        f = chebfun2(lambda x, y: jnp.cos(7 * x ** 2 * y + y))
        g = chebfun2(lambda x, y: jnp.cos(7 * x * y))
        F = Chebfun2v([f.approx, g.approx])

        r1 = np.asarray(f.roots(g, "resultant"))
        r2 = np.asarray(f.roots(g, "ms"))
        r3 = np.asarray(f.roots(g, "marchingsquares"))
        r4 = np.asarray(f.roots(g))
        r5 = np.asarray(F.roots("resultant"))
        r6 = np.asarray(F.roots("marchingsquares"))
        r7 = np.asarray(F.roots())

        # Same algorithm through different syntaxes -> identical arrays.
        assert np.array_equal(r1, r5)
        assert np.array_equal(r2, r3)
        assert np.array_equal(r2, r6)
        assert np.array_equal(r4, r7)

        # Marching-squares vs resultant agree to the MATLAB tolerance.
        x4, y4 = _sorted_cols(r4)
        x6, y6 = _sorted_cols(r6)
        x5, y5 = _sorted_cols(r5)
        assert np.linalg.norm(x4 - x6) < _TOL and np.linalg.norm(y4 - y6) < _TOL
        assert np.linalg.norm(x4 - x5) < _TOL and np.linalg.norm(y4 - y5) < _TOL
