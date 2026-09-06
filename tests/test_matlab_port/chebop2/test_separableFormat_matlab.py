"""Port of MATLAB Chebfun tests/chebop2/test_separableFormat.m (Fable 5).

``chebop2.separableFormat(N)`` is ``Chebop2.separable_format(N)``; the
coefficient cell ``N.coeffs`` (``(yorder+1) x (xorder+1)`` scalars /
chebfun2s) is ``N.coeffs_cell()``.

Provenance
----------
MATLAB source : tests/chebop2/test_separableFormat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.operators.chebop2 import Chebop2

jax.config.update("jax_enable_x64", True)

pi = np.pi
s3 = np.sqrt(3)


def _check_coeffs(A, cellU, S, cellV, dom):
    """MATLAB checkCoeffs: rebuild every coefficient from the separable
    format and compare (norm of the chebfun2 difference)."""
    err = np.zeros((len(cellU), len(cellV)))
    for jj in range(len(cellU)):
        for kk in range(len(cellV)):
            cols = [cellU[jj][r] for r in range(S.shape[0])]
            rows = [cellV[kk][r] for r in range(S.shape[0])]
            a = Chebfun2.from_cdr(cols, np.diag(S), rows, dom)
            target = A[jj][kk]
            diff = a - target if isinstance(target, Chebfun2) else a - float(np.real(target)) - (1j * float(np.imag(target)) if np.iscomplex(target) else 0)
            err[jj, kk] = float(diff.norm())
    return np.linalg.norm(err, 2) < 1e-8


def _run(op, dom=(-1, 1, -1, 1)):
    N = Chebop2(op, domain=dom)
    U, S, V = Chebop2.separable_format(N)
    return _check_coeffs(N.coeffs_cell(), U, S, V, tuple(float(v) for v in dom))


class TestChebop2SeparableFormat:
    def test_all_matlab_assertions(self):
        assert _run(lambda x, y, u: u.diff(2, 0) + u.diff(0, 2))                          # pass(1)
        assert _run(lambda x, y, u: u.diff(2, 0) + u.diff(0, 2) + 100 * u)                # pass(2)
        assert _run(lambda x, y, u: u.diff(2, 0) + u.diff(0, 2) + (10 * x + y) * u)       # pass(3)
        assert _run(lambda x, y, u: u.diff(2, 0) + u.diff(0, 2) + np.cos(100 * x) * u)   # pass(4)
        assert _run(lambda x, y, u: u.diff(2, 0) + u.diff(0, 2) + (10 + y) * u)           # pass(5)
        d = (-1, pi / 3, -s3 / 2, 1.1)
        assert _run(lambda x, y, u: u.diff(2, 0) + u.diff(0, 2) + (10 + y) * u, d)        # pass(6)
        assert _run(lambda x, y, u: (1 + x ** 2) * u.diff(2, 0) + u.diff(0, 2) + u, d)    # pass(7)
        assert _run(lambda x, y, u: u.diff(1, 0) - u.diff(0, 2), (-1, 1, 0, 10))          # pass(8)
        assert _run(lambda x, y, u: u.diff(0, 1) - x ** 2 * u.diff(0, 2) + y * u, (-1, 1, 0, 10))  # pass(9)
        assert _run(lambda x, y, u: 1j * u.diff(1, 0))                                    # pass(10)
        assert _run(lambda x, y, u: 1j * u.diff(1, 0) + (1 + 1j) * u.diff(0, 1))          # pass(11)
        hb = 0.0256
        assert _run(lambda x, y, u: 1j * hb * u.diff(1, 0) + hb ** 2 * u.diff(0, 2) - x ** 2 * u, (-1, 1, 0, 10))  # pass(12)
