"""Chebop.null -- nullspace bases of linear operators.

Pins the behaviors of the ode-eig/NullSpace chebfun.org example:
orthonormality, vanishing residual, the {1, x} span for u'', and the
dimension drop when a boundary condition is attached.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from chebfunjax.chebfun1d.chebfun import chebfun, subspace  # noqa: E402
from chebfunjax.operators.chebop import Chebop  # noqa: E402


class TestNull:
    def test_second_derivative_spans_one_x(self):
        L = Chebop(lambda u: u.diff(2))
        V = L.null()
        assert len(V) == 2
        G = np.array([[float((a * b).sum()) for b in V] for a in V])
        assert np.max(np.abs(G - np.eye(2))) < 1e-12
        assert max(float(L(v).norm(2)) for v in V) < 1e-8
        one = chebfun(lambda t: 0 * t + 1.0)
        x = chebfun(lambda t: t)
        assert float(subspace([one, x], V)) < 1e-10

    def test_incomplete_bcs_reduce_dimension(self):
        dom = (-np.pi, np.pi)
        L = Chebop(lambda x, u: (u.diff(2) + 0.1 * x * (1 - x**2) * u.diff()
                                 + x.sin() * u), domain=dom)
        assert len(L.null()) == 2
        L.lbc = 0.0
        V = L.null()
        assert len(V) == 1
        assert abs(float(V[0](-np.pi))) < 1e-10
        assert float(L(V[0]).norm(2)) < 1e-8

    def test_integral_side_condition(self):
        # 3rd-order operator with int(u) = u(0): dim = 3 - 1 = 2.
        L = Chebop(lambda x, u: 0.1 * u.diff(3) + x.sin() * u.diff(2) + u,
                   domain=(-1, 1))
        L.bc = lambda x, u: u.sum() - u(0.0)
        V = L.null()
        assert len(V) == 2
        for v in V:
            assert abs(float(v.sum()) - float(v(0.0))) < 1e-9
