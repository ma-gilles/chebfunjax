"""Port of MATLAB Chebfun tests/treeVar/test_univariate.m (Fable 5).

Converts ``@(u) diff(u,2) + diff(u) + alp*method(u)`` to first-order
form for every univariate treeVar method and compares the converted
right-hand side against the manually constructed one.

Provenance
----------
MATLAB source : tests/treeVar/test_univariate.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
import pytest

from chebfunjax.operators.treevar import (
    _UNARY_FNS,
    _UNIVARIATE,
    to_first_order,
)

DOM = (0.0, 2.0)
RHS = 3.2
ALP = 2.0
T_ARG = 1.0
U_ARG = [0.4, 0.2]

METHODS = [m for m in _UNIVARIATE if m not in ("uminus", "uplus")]


def _op(name):
    return lambda u: u.diff(2) + u.diff() + ALP * getattr(u, name)()


class TestTreevarUnivariate:
    @pytest.mark.parametrize("name", METHODS)
    def test_method(self, name):
        anon_fun, *_ = to_first_order(_op(name), RHS, DOM)
        fn = _UNARY_FNS[name]
        got = np.ravel(anon_fun(T_ARG, U_ARG))
        want = np.asarray(
            [U_ARG[1], RHS - U_ARG[1] - ALP * fn(U_ARG[0])])
        assert np.linalg.norm(got - want) < 10 * np.finfo(float).eps

    def test_uminus_uplus(self):
        def mk(is_minus):
            return lambda u: (u.diff(2) + u.diff()
                              + ALP * ((-u) if is_minus else (+u)))

        for name, sign in (("uminus", -1.0), ("uplus", 1.0)):
            anon_fun, *_ = to_first_order(mk(name == "uminus"),
                                          RHS, DOM)
            got = np.ravel(anon_fun(T_ARG, U_ARG))
            want = np.asarray(
                [U_ARG[1], RHS - U_ARG[1] - ALP * sign * U_ARG[0]])
            assert np.linalg.norm(got - want) < 1e-14
