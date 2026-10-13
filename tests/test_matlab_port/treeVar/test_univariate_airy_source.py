"""Captured native ``airy`` clause from TreeVar's unary source test.

MATLAB source: tests/treeVar/test_univariate.m, ``@airy`` entry.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
Native R2025b capture: treevar_unary_native_capture_20261014/capture.json.
The original 10*eps norm predicate and inputs are retained here separately
so the Airy clause can be qualified without rerunning unrelated unary cases.
"""

import numpy as np

from chebfunjax.operators.treevar import to_first_order

DOM = (0.0, 2.0)
RHS = 3.2
ALP = 2.0
T_ARG = 1.0
U_ARG = [0.4, 0.2]
NATIVE_EXPECTED = np.asarray([0.2, 2.4905152914086477])


def test_captured_native_airy_unary_clause():
    def op(u):
        return u.diff(2) + u.diff() + ALP * u.airy()

    converted, *_ = to_first_order(op, RHS, DOM)
    got = np.ravel(converted(T_ARG, U_ARG))
    assert np.linalg.norm(got - NATIVE_EXPECTED) < 10 * np.finfo(float).eps
