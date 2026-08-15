"""Port of MATLAB Chebfun tests/treeVar/test_bivariate.m (Fable 5).

Provenance
----------
MATLAB source : tests/treeVar/test_bivariate.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import operator

import numpy as np
import pytest

from chebfunjax.operators.treevar import to_first_order

DOM = (0.0, 2.0)
RHS = 3.2
ALP = 2.0
T_ARG = 1.0
U_ARG = [0.4, 0.2]
S_ARG = 2.0

BIVARIATE = [operator.sub, operator.add, operator.pow,
             operator.truediv, operator.mul]


def _op_tv(method):
    return lambda u: (u.diff(2) + u.diff()
                      + ALP * method(u, u.diff()))


def _op_sc(method):
    return lambda u: (u.diff(2) + u.diff()
                      + ALP * method(u, S_ARG))


class TestTreevarBivariate:
    @pytest.mark.parametrize("method", BIVARIATE)
    def test_treevar_treevar(self, method):
        # diff(u,2) + diff(u) + alp*method(u, diff(u))
        anon_fun, *_ = to_first_order(_op_tv(method), RHS, DOM)
        got = np.ravel(anon_fun(T_ARG, U_ARG))
        want = np.asarray([U_ARG[1], RHS - U_ARG[1]
                           - ALP * method(U_ARG[0], U_ARG[1])])
        assert np.linalg.norm(got - want) < 10 * np.finfo(float).eps

    @pytest.mark.parametrize("method", [operator.truediv, operator.mul])
    def test_treevar_scalar(self, method):
        # MATLAB mrdivide/mtimes with a scalar right argument.
        anon_fun, *_ = to_first_order(_op_sc(method), RHS, DOM)
        got = np.ravel(anon_fun(T_ARG, U_ARG))
        want = np.asarray([U_ARG[1], RHS - U_ARG[1]
                           - ALP * method(U_ARG[0], S_ARG)])
        assert np.linalg.norm(got - want) < 10 * np.finfo(float).eps
