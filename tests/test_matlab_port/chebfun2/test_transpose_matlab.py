"""Port of MATLAB Chebfun tests/chebfun2/test_transpose.m (Fable 5).

FIXED (Fable 5, chebfun2/3 skip sweep): ``Chebfun2.transpose()`` /
``ctranspose()`` now exist, swapping the low-rank column and row slices
and the two halves of the domain.

Provenance
----------
MATLAB source : tests/chebfun2/test_transpose.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 1000 * EPS


class TestChebfun2Transpose:
    def test_symmetric_function(self):
        # pass(1, 2): cos(x*y) is symmetric, so f == f' == f.'.
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        assert float((f - f.ctranspose()).norm()) < TOL
        assert float((f - f.transpose()).norm()) < TOL
        assert float((f - f.T).norm()) < TOL
        assert float((f - f.H).norm()) < TOL

    def test_real_unsymmetric(self):
        # pass(3, 4): (cos(x*y) + x)' on [-3 4 -1 0] is cos(x*y) + y on
        # [-1 0 -3 4].
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y) + x,
                                   domain=(-3.0, 4.0, -1.0, 0.0))
        g = Chebfun2.from_function(lambda x, y: jnp.cos(x * y) + y,
                                   domain=(-1.0, 0.0, -3.0, 4.0))
        assert float((f.ctranspose() - g).norm()) < TOL
        assert float((f.transpose() - g).norm()) < TOL

    def test_complex_conjugation(self):
        # pass(5, 6): f = cos(x*y) + 1i*x.  The conjugate transpose
        # negates the imaginary part; the plain transpose does not.
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y) + 1j * x,
                                   domain=(-3.0, 4.0, -1.0, 0.0))
        g1 = Chebfun2.from_function(lambda x, y: jnp.cos(x * y) - 1j * y,
                                    domain=(-1.0, 0.0, -3.0, 4.0))
        g2 = Chebfun2.from_function(lambda x, y: jnp.cos(x * y) + 1j * y,
                                    domain=(-1.0, 0.0, -3.0, 4.0))
        assert float((f.ctranspose() - g1).norm()) < TOL
        assert float((f.transpose() - g2).norm()) < TOL

    def test_unsymmetric_domain_swaps(self):
        # pass(7): the transposed domain is the input domain with its two
        # halves exchanged.
        f = Chebfun2.from_function(lambda x, y: x,
                                   domain=(0.0, 1.0, 2.0, 3.0))
        d = f.domain
        assert f.transpose().domain == (d[2], d[3], d[0], d[1])
