"""Port of MATLAB Chebfun tests/chebop2/test_subsref.m (Fable 5).

``N(m, n)`` is ``N.matrix(m, n)`` (also ``N(m, n)``), ``N(f)`` / ``N * f``
apply the PDO to a Chebfun2.

Provenance
----------
MATLAB source : tests/chebop2/test_subsref.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.operators.chebop2 import Chebop2, laplacian

jax.config.update("jax_enable_x64", True)


class TestChebop2Subsref:
    def test_all_matlab_assertions(self):
        N = Chebop2(lambda u: u.diff(2, 0) + u.diff(0, 2))
        assert N(10, 10).shape[0] - 100 == 0                                 # pass(1)
        assert N(10, 10).shape[1] - 100 == 0                                 # pass(2)
        m, n = 5, 10
        assert N(m, n).shape[0] - m * n == 0                                 # pass(3)
        assert N(m, n).shape[1] - m * n == 0                                 # pass(4)
        assert N(m).shape[0] - m ** 2 == 0                                   # pass(5)
        assert np.linalg.norm(N(n, n) - N(n)) == 0                           # pass(6)
        assert np.linalg.norm(np.asarray(N.coeffs) - np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])) == 0  # pass(7)
        f = chebfun2(lambda x, y: jnp.real(jnp.exp(x + 1j * y)))
        N = Chebop2(lambda x, y, u: u.diff(2, 0) + u.diff(0, 2))
        N.lbc = f(-1.0, ":")
        N.rbc = f(1.0, ":")
        N.ubc = f(":", 1.0)
        N.dbc = f(":", -1.0)
        assert float(N(f).norm()) < 1e-11                                    # pass(8)
        N = Chebop2(lambda x, y, u: laplacian(u))
        N.lbc = f(-1.0, ":")
        N.rbc = f(1.0, ":")
        N.ubc = f(":", 1.0)
        N.dbc = f(":", -1.0)
        assert float((N * f).norm()) < 1e-11                                 # pass(9)
