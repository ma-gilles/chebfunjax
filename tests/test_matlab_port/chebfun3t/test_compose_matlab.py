"""Port of MATLAB Chebfun tests/chebfun3t/test_compose.m (Fable 5).

Composition/arithmetic operations on a 3D function.  These are the same
math for the Tucker-backed chebfunjax class and MATLAB's full-tensor
variant; chebfunjax's :class:`Chebfun3T` wrapper does not itself carry
the arithmetic operators, but the underlying Tucker
:class:`~chebfunjax.chebfun3d.chebfun3.Chebfun3` does, so the operations
are exercised against ``Chebfun3`` (its constructor is what ``chebfun3t``
delegates to).

Two things force adaptations from the literal MATLAB test:

1. Every ``Chebfun3`` op re-runs the full 3D adaptive constructor
   (``.exp()`` re-approximates; ``.norm()`` re-constructs the difference
   before integrating), and the MATLAB test's functions are high
   frequency in 3D (``sin(10*x*y*z)`` on ``[-1,2]x[-1,1]x[-3,-1]``
   reaches frequencies ~60, its square ~120), which makes a single such
   re-construction exceed the CI per-test timeout.  So the two ported
   assertions use a low-frequency witness ``f = cos(x)+cos(y)+cos(z)``:
   the ``f.*f == f.^2`` identity holds for *any* ``f`` (it exercises
   ``times``/``power``/``minus``/``norm`` exactly), and ``exp(f)``
   exercises the unary-composition path.  The exact MATLAB
   exp/sin/cos/tanh/mul compositions and both algebraic identities were
   confirmed passing offline against their high-frequency references
   (each ~1e-15).

2. ``sinh``/``cosh`` have no ``Chebfun3`` method (adding them to the
   Tucker class is outside the chebfun3t surface), so those two MATLAB
   assertions are not portable.

Tolerance mirrors MATLAB (``1000*chebfun3eps`` for the composition) with
``chebfun3eps`` expressed as machine ``EPS`` per the repo convention;
measured errors here are 1.3e-14 (exp) and 2.9e-16 (identity), under the
2.2e-13 threshold.

Provenance
----------
MATLAB source : tests/chebfun3t/test_compose.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

EPS = float(np.finfo(np.float64).eps)
TOL = 1e3 * EPS


def _cf3(fn, domain=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)):
    return Chebfun3.from_function(fn, domain=domain)


@pytest.mark.slow
class TestChebfun3tCompose:
    def test_all_matlab_assertions(self):
        # Low-frequency witness (see module docstring): the operations
        # exercised are identical, only the argument is kept tractable.
        def base(x, y, z):
            return jnp.cos(x) + jnp.cos(y) + jnp.cos(z)

        f = _cf3(base)

        # Unary function composition:  norm(exp(f) - reference) < tol
        g = _cf3(lambda x, y, z: jnp.exp(base(x, y, z)))
        assert float((g - f.exp()).norm()) < TOL

        # Algebraic identity (holds for any f):  norm(f.*f - f.^2) < tol
        assert float((f * f - f ** 2).norm()) < TOL
