"""Port of MATLAB Chebfun tests/treeVar/test_toFirstOrder.m (Fable 5).

All 19 conversion problems plus the two error cases.  ``anonFun`` in
MATLAB is a column vector function; here the converted function returns
an (n, 1) array and comparisons are made on the raveled values.

Provenance
----------
MATLAB source : tests/treeVar/test_toFirstOrder.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.operators.treevar import TreeVarError, to_first_order

jax.config.update("jax_enable_x64", True)

DOM = (-1.0, 4.0)
TOL = 5e-14


def _x():
    return cj.chebfun(lambda t: t, domain=DOM)


def _cnorm(c, target):
    """norm(coeffs{k} - target) for scalar or chebfun coefficients."""
    if isinstance(c, (int, float)):
        if isinstance(target, (int, float)):
            return abs(c - target)
        return float((target - c).norm())
    if isinstance(target, (int, float)):
        return float((c - target).norm())
    return float((c - target).norm())


class TestTreevarToFirstOrder:
    def test_simple(self):
        f, idx, dom_out, coeffs, orders = to_first_order(
            lambda u: 5 * (u.diff(2) + 3 * u), 0, DOM)
        assert np.linalg.norm(np.ravel(f(1, [2, 1])) - [1, -6]) < TOL
        assert list(idx) == [1]
        assert dom_out == DOM
        assert _cnorm(coeffs[0], 5.0) < TOL
        assert list(orders) == [2]

    def test_scalar_variable_in_op(self):
        alpha = 4.0
        f, idx, dom_out, coeffs, orders = to_first_order(
            lambda u: 3.5 * (u.diff(2) + alpha * u), 5, DOM)
        assert np.linalg.norm(np.ravel(f(1, [2, 1]))
                              - [1, 5 / 3.5 - 4 * 2]) < TOL
        assert _cnorm(coeffs[0], 3.5) < TOL

    def test_chebfun_coefficient(self):
        x = _x()
        f, idx, dom_out, coeffs, orders = to_first_order(
            lambda u: 5 * ((x + 1) * u.diff(2) + u), -5, DOM)
        assert np.linalg.norm(np.ravel(f(-0.5, [2, 1]))
                              - [1, -6]) < TOL
        assert _cnorm(coeffs[0], 5 * (x + 1)) < 1e-10

    def test_chebfun_in_lower_order_term(self):
        x = _x()
        f, idx, dom_out, coeffs, orders = to_first_order(
            lambda u: u.diff(2) + 2 * x.sin() * u, x.cos(), DOM)
        want = [1, np.cos(0.5) - 2 * np.sin(0.5) * 2]
        assert np.linalg.norm(np.ravel(f(0.5, [2, 1])) - want) < TOL
        assert _cnorm(coeffs[0], 1.0) < 1e-10

    def test_chebfun_at_start(self):
        x = _x()
        f, idx, dom_out, coeffs, orders = to_first_order(
            lambda u: 3 * (x + 2) * ((x + 1) * u.diff(2) + u), 0, DOM)
        assert np.linalg.norm(np.ravel(f(-0.5, [2, 2]))
                              - [2, -4]) < TOL
        assert _cnorm(coeffs[0], 3 * (x + 2) * (x + 1)) < 1e-9

    def test_breakpoints_from_abs(self):
        x = _x()
        f, idx, dom_out, coeffs, orders = to_first_order(
            lambda u: x.cos() * u.diff(2)
            + 2 * (np.pi * x).sin().abs() * u,
            (2 * x).cos(), DOM)
        t, u = 0.5, [2, 1]
        want = [u[1], (np.cos(2 * t)
                       - 2 * abs(np.sin(np.pi * t)) * u[0])
                / np.cos(t)]
        assert np.linalg.norm(np.ravel(f(t, u)) - want) < TOL
        assert np.linalg.norm(np.asarray(dom_out)
                              - np.arange(-1.0, 5.0)) < TOL
        assert _cnorm(coeffs[0], x.cos()) < 1e-10
        assert list(orders) == [2]

    def test_breakpoints_higher_order(self):
        x = _x()
        f, idx, dom_out, coeffs, orders = to_first_order(
            lambda u: x.cos() * u.diff(4)
            + 2 * (np.pi * x).sin().abs() * u.diff(2),
            (2 * x).cos(), DOM)
        t, u = 0.5, [2, 1, 3, 4]
        want = [u[1], u[2], u[3],
                (np.cos(2 * t)
                 - 2 * abs(np.sin(np.pi * t)) * u[2]) / np.cos(t)]
        assert np.linalg.norm(np.ravel(f(t, u)) - want) < TOL
        assert list(orders) == [4]

    def test_coupled_first_order(self):
        x = _x()
        f, idx, dom_out, coeffs, orders = to_first_order(
            lambda t, u, v: [5 * (u.diff() + 3 * v),
                             t.cos() * v.diff() + t.sin() * u],
            [x.tanh(), 2], DOM)
        pt = [2.4, 2.3]
        want = [np.tanh(1) / 5 - 3 * pt[1],
                (2 - np.sin(1) * pt[0]) / np.cos(1)]
        assert np.linalg.norm(np.ravel(f(1, pt)) - want) < TOL
        assert list(idx) == [1, 2]
        assert _cnorm(coeffs[0], 5.0) + _cnorm(coeffs[1],
                                               x.cos()) < 1e-10
        assert list(orders) == [1, 1]

    def test_coupled_second_order(self):
        x = _x()
        f, idx, dom_out, coeffs, orders = to_first_order(
            lambda t, u, v: [5 * (u.diff(2) + 3 * v),
                             t.cos() * v.diff(2) + t.sin() * u],
            [x.tanh(), 2], DOM)
        pt = [2, 1, 2.4, 2.3]
        want = [pt[1], np.tanh(1) / 5 - 3 * pt[2], pt[3],
                (2 - np.sin(1) * pt[0]) / np.cos(1)]
        assert np.linalg.norm(np.ravel(f(1, pt)) - want) < TOL
        assert list(idx) == [1, 3]
        assert list(orders) == [2, 2]

    def test_coupled_second_order_breakpoints(self):
        x = _x()
        f, idx, dom_out, coeffs, orders = to_first_order(
            lambda t, u, v: [
                5 * (u.diff(2) + 3 * v),
                t.cos() * v.diff(2)
                + (np.pi * t).sin().abs() * u],
            [x.tanh(), 2], DOM)
        pt = [2, 1, 2.4, 2.3]
        want = [pt[1], np.tanh(1) / 5 - 3 * pt[2], pt[3],
                (2 - abs(np.sin(np.pi * 1.0)) * pt[0]) / np.cos(1)]
        assert np.linalg.norm(np.ravel(f(1, pt)) - want) < TOL
        assert np.linalg.norm(np.asarray(dom_out)
                              - np.arange(-1.0, 5.0)) < TOL

    def test_chebfun_inside_not_rhs(self):
        x = _x()
        f, idx, dom_out, coeffs, orders = to_first_order(
            lambda u: 5 * (u.diff(2) + (x + 5).coth()), -5, DOM)
        pt = [2.4, 2.3]
        want = [pt[1], -1 - 1 / np.tanh(1 + 5)]
        assert np.linalg.norm(np.ravel(f(1, pt)) - want) < TOL

    def test_four_variables_mixed(self):
        x = _x()
        f, idx, dom_out, coeffs, orders = to_first_order(
            lambda t, u, v, w, y: [
                y.diff(3) + w + u.diff(),
                7 * w.diff() + v.diff(),
                5 * (u.diff(2) + 3 * v + (t + 5).coth()),
                t.cos() * v.diff(2) + u.diff() - y.diff(2)],
            [x.exp(), x ** 2, x.tanh(), 2], DOM)
        pt = np.array([2, 1, 2.4, 2.3, 1.2, 3.2, 5.1, 4.6])
        t = 1.0
        want = [pt[1],
                np.tanh(t) / 5 - 3 * pt[2] - 1 / np.tanh(t + 5),
                pt[3], (2 - pt[1] + pt[7]) / np.cos(t),
                (t ** 2 - pt[3]) / 7,
                pt[6], pt[7], np.exp(t) - pt[4] - pt[1]]
        assert np.linalg.norm(np.ravel(f(t, pt)) - want) < 1e-12
        assert list(idx) == [1, 3, 5, 6]
        assert list(orders) == [2, 2, 1, 3]

    def test_multiple_chebfuns_scalars_v1(self):
        x = _x()
        f, idx, dom_out, coeffs, orders = to_first_order(
            lambda t, u: u.diff() + t + 3 * t, x.tanh(), DOM)
        want = [np.tanh(1) - 4 * 1.0]
        assert np.linalg.norm(np.ravel(f(1, [2.4, 2.3])[:1])
                              - want) < TOL
        assert list(orders) == [1]

    def test_multiple_chebfuns_scalars_v2(self):
        x = _x()
        f, idx, dom_out, coeffs, orders = to_first_order(
            lambda t, u: t.cos() * (2 - t.sin() + u.diff())
            + t + 3 * t, x.tanh(), DOM)
        want = [(np.tanh(1) - 4) / np.cos(1) + np.sin(1) - 2]
        assert np.linalg.norm(np.ravel(f(1, [2.4, 2.3])[:1])
                              - want) < TOL
        assert _cnorm(coeffs[0], x.cos()) < 1e-10

    def test_multiple_chebfuns_scalars_v3(self):
        x = _x()
        f, idx, dom_out, coeffs, orders = to_first_order(
            lambda t, u: 2 - t.sin() + u.diff() + t + 3 * t,
            x.tanh(), DOM)
        want = [np.tanh(1) - 4 + np.sin(1) - 2]
        assert np.linalg.norm(np.ravel(f(1, [2.4, 2.3])[:1])
                              - want) < TOL

    def test_coupled_multiple_chebfuns_scalars(self):
        x = _x()
        f, idx, dom_out, coeffs, orders = to_first_order(
            lambda t, u, v: [
                4 * (2 + u.diff()) + t + 3 * t + 3 * v,
                t.cos() * v.diff() + t.sin() * u + t + t],
            [x.tanh(), 2], DOM)
        pt = [2.4, 2.3]
        want = [(np.tanh(1) - 4 - 3 * pt[1]) / 4 - 2,
                (2 - 2 - np.sin(1) * pt[0]) / np.cos(1)]
        assert np.linalg.norm(np.ravel(f(1, pt)) - want) < TOL
        assert list(idx) == [1, 2]

    def test_cellarg_chebmatrix_syntax(self):
        x = _x()
        f, idx, dom_out, coeffs, orders = to_first_order(
            lambda t, y: [
                y[3].diff(3) + y[2] + y[0].diff(),
                7 * y[2].diff() + y[1].diff(),
                5 * (y[0].diff(2) + 3 * y[1] + (t + 5).coth()),
                t.cos() * y[1].diff(2) + y[0].diff()
                - y[3].diff(2)],
            [x.exp(), x ** 2, x.tanh(), 2], DOM,
            num_args=4, cell_arg=True)
        pt = np.array([2, 1, 2.4, 2.3, 1.2, 3.2, 5.1, 4.6])
        t = 1.0
        want = [pt[1],
                np.tanh(t) / 5 - 3 * pt[2] - 1 / np.tanh(t + 5),
                pt[3], (2 - pt[1] + pt[7]) / np.cos(t),
                (t ** 2 - pt[3]) / 7,
                pt[6], pt[7], np.exp(t) - pt[4] - pt[1]]
        assert np.linalg.norm(np.ravel(f(t, pt)) - want) < 1e-12
        assert list(idx) == [1, 3, 5, 6]

    def test_breakpoints_in_rhs(self):
        # MATLAB problem 17: rhs = (abs(x/2-round(x/2)) < .05).
        x = _x()
        rhs = (x / 2 - (x / 2).round()).abs() < 0.05
        f, idx, dom_out, coeffs, orders = to_first_order(
            lambda u: x.cos() * u.diff(2)
            + 2 * (np.pi * x).sin().abs() * u, rhs, DOM)
        t, u = 0.5, [2, 1]
        rhs_t = float(rhs(np.asarray(t)))
        want = [u[1], (rhs_t - 2 * abs(np.sin(np.pi * t)) * u[0])
                / np.cos(t)]
        assert np.linalg.norm(np.ravel(f(t, u)) - want) < TOL
        expected_dom = sorted(set(np.arange(-1.0, 5.0))
                              | {float(v)
                                 for v in rhs.domain.breakpoints})
        assert np.linalg.norm(np.asarray(dom_out)
                              - np.asarray(expected_dom)) < 1e-10
        assert _cnorm(coeffs[0], x.cos()) < 1e-10

    def test_coupled_breakpoints_in_rhs(self):
        # MATLAB problem 18.
        x = _x()
        rhs2 = (x / 2 - (x / 2).round()).abs() < 0.05
        f, idx, dom_out, coeffs, orders = to_first_order(
            lambda t, u, v: [
                5 * (u.diff(2) + 3 * v),
                t.cos() * v.diff(2)
                + (np.pi * t).sin().abs() * u],
            [x.tanh(), rhs2], DOM)
        pt = [2, 1, 2.4, 2.3]
        t = 1.0
        rhs_t = float(rhs2(np.asarray(t)))
        want = [pt[1], np.tanh(t) / 5 - 3 * pt[2], pt[3],
                (rhs_t - abs(np.sin(np.pi * t)) * pt[0])
                / np.cos(t)]
        assert np.linalg.norm(np.ravel(f(t, pt)) - want) < TOL
        assert list(idx) == [1, 3]

    def test_error_same_equation_highest_order(self):
        with pytest.raises(TreeVarError) as exc:
            to_first_order(
                lambda t, u, v: [u.diff(2) + v.diff(2),
                                 u.diff() + v.sin()], [1, 2], DOM)
        assert exc.value.identifier == \
            "CHEBFUN:TREEVAR:toFirstOrder:diffOrders"

    def test_error_nonlinearity_highest_order(self):
        with pytest.raises(TreeVarError) as exc:
            to_first_order(
                lambda t, u, v: [u.diff(2) * v.diff(),
                                 v.diff(2) + u.sin()], [1, 2], DOM)
        assert exc.value.identifier == \
            "CHEBFUN:TREEVAR:expandTree:nonlinearity"
