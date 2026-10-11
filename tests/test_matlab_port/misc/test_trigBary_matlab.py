"""Native tests/misc/test_trigBary.m predicates1–9(Chebfun7574c77).

MATLAB seedRNG(3453) maps to one numpy.default_rng(3453) stream;
random realization differs but source sizes/distributions/order are retained.
"""

import numpy as np

import chebfunjax as cj
from chebfunjax.utils.quadrature import trigpts
from chebfunjax.utils.trigutils import trigBaryWeights


def _p1(x):
    return (
        np.cos(4 * np.pi * x)
        - 2 * np.sin(3 * np.pi * x)
        + 3 * np.sin(2 * np.pi * x)
        - 2 * np.cos(np.pi * x)
        + 1
    )


class TestTrigBary:
    def test_all_matlab_assertions(self):
        rng = np.random.default_rng(3453)
        xr = 2 * rng.random(1000) - 1
        tol = 1e-12
        xk = np.asarray(trigpts(10, (-1, 1))[0])
        y = cj.trigBary(xr, _p1(xk), xk, (-1, 1))
        assert np.max(np.abs(_p1(xr) - y)) < tol  # pass(1)
        xk = np.asarray(trigpts(1001, (-1, 1))[0])
        y = cj.trigBary(xr, _p1(xk), xk, (-1, 1))
        assert np.max(np.abs(_p1(xr) - y)) < tol  # pass(2)
        xk = np.asarray(trigpts(10, (-1, 1))[0])
        y = cj.trigBary(xk, _p1(xk), xk, (-1, 1))
        assert np.max(np.abs(_p1(xk) - y)) < tol  # pass(3)
        xk = -np.cos((2 * np.arange(1, 9) - 1) * np.pi / 16)

        def p2(x):
            return 2 * np.sin(3 * np.pi * x) + 3 * np.sin(2 * np.pi * x) - 2 * np.cos(np.pi * x) + 1

        y = cj.trigBary(xr, p2(xk), xk, (-1, 1))
        assert np.max(np.abs(p2(xr) - y)) < tol  # pass(4)

        def q(x):
            return np.stack(
                [0.45 + np.sin(np.pi * x), 0.32 + np.sin(np.pi * x) + np.cos(2 * np.pi * x)],
                axis=-1,
            )

        tol = 1e-10
        xk = -2 + 4 * rng.random(8)
        xr = -2 + 4 * rng.random(100)
        difference = q(xr) - cj.trigBary(xr, q(xk), xk, (-2, 2))
        assert np.max(np.abs(difference)) < tol  # pass(5)
        difference = q(xk) - cj.trigBary(xk, q(xk), xk, (-2, 2))
        assert np.max(np.abs(difference)) < tol  # pass(6)
        n = 2500
        w = trigBaryWeights(np.sort(rng.random(n)))
        assert len(w) == n  # pass(7)
        eps = np.finfo(float).eps
        xk = np.array([-1, -1 / 3 - eps, 1 / 3 - eps, 1])
        fvals = np.array([1, 1, 1, 1 + eps])
        y = cj.trigBary(2, fvals, xk, (-np.pi, np.pi))
        assert np.all(np.isfinite(y)) and np.all(np.abs(y - 1) < 10 * eps)  # pass(8)
        w = trigBaryWeights(np.linspace(-1, 1, 4))
        w_ex = np.array([0.38883657630910357, -1, 1, -0.38883657630910357])
        tol = 10 * eps
        assert np.max(np.abs(np.abs(w) - np.abs(w_ex))) < tol  # pass(9)
