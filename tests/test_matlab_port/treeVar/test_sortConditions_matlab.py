"""Port of MATLAB Chebfun tests/treeVar/test_sortConditions.m (Fable 5).

All 12 sorting passes and all 13 error cases.

Provenance
----------
MATLAB source : tests/treeVar/test_sortConditions.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.operators.treevar import TreeVarError, sort_conditions

DOM = (0.4, 2.0)


class TestTreevarSortConditions:
    @pytest.mark.parametrize("fun,orders,correct", [
        (lambda u: u, 1, [1]),
        (lambda u: [u, u.diff()], 2, [1, 2]),
        (lambda u: [u.diff(), u], 2, [2, 1]),
        (lambda u: [u.diff(0), u.diff(1), u.diff(2), u.diff(3),
                    u.diff(4), u.diff(5)], 6, [1, 2, 3, 4, 5, 6]),
        (lambda u: [u.diff(2), u.diff(4), u.diff(5), u.diff(0),
                    u.diff(1), u.diff(3)], 6, [4, 5, 1, 6, 2, 3]),
        (lambda u, v: [u, v], [1, 1], [1, 2]),
        (lambda u, v: [v, u], [1, 1], [2, 1]),
        (lambda u, v: [u, u.diff(), v, v.diff()], [2, 2],
         [1, 2, 3, 4]),
        (lambda u, v: [u, v.diff(), u.diff(), v], [2, 2],
         [1, 3, 4, 2]),
        (lambda u, v, w, y: [
            u.diff(3), w.diff(3), w, u.diff(2),
            v.diff(3), v.diff(2), w.diff(), y,
            u.diff(), y.diff(2), v.diff(), v,
            y.diff(1), y.diff(3), u, w.diff(2)], [4, 4, 4, 4],
         [15, 9, 4, 1, 12, 11, 6, 5, 3, 7, 16, 2, 8, 13, 10, 14]),
        (lambda u: [u[0], u[1].diff(), u[0].diff(), u[1]], [2, 2],
         [1, 3, 4, 2]),
        (lambda u: [
            u[0].diff(3), u[2].diff(3), u[2], u[0].diff(2),
            u[1].diff(3), u[1].diff(2), u[2].diff(), u[3],
            u[0].diff(), u[3].diff(2), u[1].diff(), u[1],
            u[3].diff(1), u[3].diff(3), u[0], u[2].diff(2)],
         [4, 4, 4, 4],
         [15, 9, 4, 1, 12, 11, 6, 5, 3, 7, 16, 2, 8, 13, 10, 14]),
    ])
    def test_sorting(self, fun, orders, correct):
        idx = sort_conditions(fun, DOM, orders)
        assert list(idx) == correct

    @pytest.mark.parametrize("fun,orders,ident", [
        (lambda x, u: 5 * u - 1, 1, "unsupportedCondition"),
        (lambda x, u: 5 * u.diff() - 1, 1, "unsupportedCondition"),
        (lambda x, u: u + u.diff(), 2, "unsupportedCondition"),
        (lambda x, u, v: u + u.diff(), 2, "unsupportedCondition"),
        (lambda x, u, v: u + v.diff(), [1, 2], "nonSeparated"),
        (lambda u: [u - 1, u - 2], 2,
         "multipleConditionsSameVariable"),
        (lambda y: [y - 1, y.diff(2)], 2, "tooHighOrderCondition"),
        (lambda y: [y - 1, y.diff(2)], 3, "missingConditions"),
        (lambda y: [y - 1, y.diff()], 3, "missingConditions"),
        (lambda u, v: [u - 1, v - 2, u.diff() - 2, v - 3], [2, 1],
         "multipleConditionsSameVariable"),
        (lambda u, v: [u - 1, v.diff(2), u.diff(), v - 3], [2, 2],
         "tooHighOrderCondition"),
        (lambda u, v: [u - 1, v.diff(2), v.diff(), u.diff(2)],
         [3, 3], "missingConditions"),
        (lambda u, v: [u - 1, u.diff(), v.diff(), v - 3], [3, 3],
         "missingConditions"),
    ])
    def test_errors(self, fun, orders, ident):
        with pytest.raises(TreeVarError) as exc:
            sort_conditions(fun, DOM, orders)
        assert ident in exc.value.identifier
