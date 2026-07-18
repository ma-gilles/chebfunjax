"""Port of MATLAB Chebfun tests/chebfun2v/test_twocomponents.m (Fable 5).

FIXED (Fable 5 audit): two-component Chebfun2v construction (from Chebfun2
objects and from handles, with explicit domains), scalar arithmetic, and
vector-valued point evaluation work.

The remaining MATLAB assertions target features chebfunjax's Chebfun2v does
not have: adding/multiplying by a length-2 row/column vector (``G+[1 2]``,
``G.*[1 2]'``), the ``G'*[1 2]'`` inner product, and the vector-calculus
identities that need ``grad`` on a scalar Chebfun2 (absent).  Those are not
ported.

Provenance
----------
MATLAB source : tests/chebfun2v/test_twocomponents.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

EPS = float(np.finfo(np.float64).eps)
TOL = 1e5 * EPS


def _ev(H, x, y):
    return np.asarray(H(jnp.asarray(x), jnp.asarray(y)))


class TestChebfun2vTwocomponents:
    def test_construction_equivalence(self):
        # pass(1,2): building from Chebfun2 objects and from handles agree,
        # on the default domain and on an explicit domain.
        f = Chebfun2.from_function(lambda x, y: x)
        assert float((Chebfun2v.from_functions(f, f)
                      - Chebfun2v.from_functions(lambda x, y: x, lambda x, y: x)).norm()) < TOL
        d = (-1.0, 1.0, -1.0, 1.0)
        fd = Chebfun2.from_function(lambda x, y: x, domain=d)
        assert float((Chebfun2v.from_functions(fd, fd, domain=d)
                      - Chebfun2v.from_functions(lambda x, y: x, lambda x, y: x,
                                                 domain=d)).norm()) < TOL

    def test_scalar_arithmetic_and_eval(self):
        # pass(3,4,6): F1 + G and scalar +/-, with vector-valued evaluation.
        f = Chebfun2.from_function(lambda x, y: x)
        F1 = Chebfun2v.from_functions(f, f)
        G = Chebfun2v.from_functions(f, 2 * f)
        # pass(3): (F1 + G)(1, 1) == [2, 3].
        assert float(np.max(np.abs(_ev(F1 + G, 1.0, 1.0) - np.array([2.0, 3.0])))) < TOL
        p = np.pi / 6
        # pass(4): (G + 1)(pi/6, 1) == pi/6 * [1, 2] + 1.
        assert float(np.max(np.abs(_ev(G + 1, p, 1.0) - (p * np.array([1.0, 2.0]) + 1)))) < TOL
        # pass(6): (1 + G)(pi/6, 1) == pi/6 * [1, 2] + 1.
        assert float(np.max(np.abs(_ev(1 + G, p, 1.0) - (p * np.array([1.0, 2.0]) + 1)))) < TOL
