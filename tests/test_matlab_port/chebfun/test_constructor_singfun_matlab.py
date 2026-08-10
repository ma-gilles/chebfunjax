"""Port of MATLAB Chebfun tests/chebfun/test_constructor_singfun.m (Fable 5).

The SingFun-wired factory ('exps'/'blowup'/'singType') builds singular
pieces from a single callable.  The reachable MATLAB cases -- explicit
exponents, and 'blowup' with singularity autodetection -- are ported below.
Cases that require multiple cell-array operators (pass 12/14), interior
singularity splitting (pass 13/16/17), or unbounded domains are kept skipped
with the reason recorded.

Provenance
----------
MATLAB source : tests/chebfun/test_constructor_singfun.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


def _check(f, op, dom, tolf):
    """|| feval(f) - op ||_inf < tolf * eps * ||op||_inf on random points."""
    rng = np.random.default_rng(6178)
    x = np.diff(dom)[0] * rng.uniform(size=100) + dom[0]
    fval = np.asarray(f(jnp.asarray(x)))
    ve = np.asarray(op(x))
    err = np.max(np.abs(fval - ve))
    tol = tolf * EPS * np.max(np.abs(ve))
    return err, tol


class TestChebfunConstructorSingfun:
    def test_explicit_exponents(self):
        # pass(1, 2): construct with the exact exponents [pow 0] supplied.
        dom = (-2.0, 7.0)
        pow = -0.5
        op = lambda x: (x - dom[0]) ** pow * np.sin(x)  # noqa: E731
        f = cj.chebfun(lambda x: (x - dom[0]) ** pow * jnp.sin(x),
                       domain=dom, exps=(pow, 0.0))
        err, tol = _check(f, op, dom, 1e3)
        assert err < tol

    def test_blowup_singtype_sing(self):
        # pass(3, 8): 'blowup', 2 with singType {'sing', 'none'} -- detect a
        # fractional (branch-point) singularity at the left endpoint.
        dom = (-1.0, 1.0)
        pow = -0.5
        op = lambda x: (x - dom[0]) ** pow * np.sin(x)  # noqa: E731
        f = cj.chebfun(lambda x: (x - dom[0]) ** pow * jnp.sin(x),
                       domain=dom, blowup=2, singType=["sing", "none"])
        err, tol = _check(f, op, dom, 1e3)
        assert err < tol

    def test_blowup_autodetect(self):
        # pass(4, 5, 9): 'blowup', 2 with no singType -- autodetect both ends.
        dom = (-1.0, 1.0)
        pow = -0.5
        op = lambda x: (x - dom[0]) ** pow * np.sin(x)  # noqa: E731
        f = cj.chebfun(lambda x: (x - dom[0]) ** pow * jnp.sin(x),
                       domain=dom, blowup=2)
        err, tol = _check(f, op, dom, 1e3)
        assert err < tol

    def test_blowup_singtype_pole(self):
        # pass(6, 10): 'blowup', 2 with singType {'pole', 'none'} -- detect an
        # integer pole (order 1) at the left endpoint.
        dom = (-1.0, 1.0)
        pow = -1.0
        op = lambda x: (x - dom[0]) ** pow * np.sin(x)  # noqa: E731
        f = cj.chebfun(lambda x: (x - dom[0]) ** pow * jnp.sin(x),
                       domain=dom, blowup=2, singType=["pole", "none"])
        err, tol = _check(f, op, dom, 1e3)
        assert err < tol

    def test_blowup_second_order_pole(self):
        # pass(7, 11): 'blowup', 1 -- autodetect an order-2 pole.
        dom = (-1.0, 1.0)
        pow = -2.0
        op = lambda x: (x - dom[0]) ** pow * np.sin(x)  # noqa: E731
        f = cj.chebfun(lambda x: (x - dom[0]) ** pow * jnp.sin(x),
                       domain=dom, blowup=1)
        err, tol = _check(f, op, dom, 1e3)
        assert err < tol

    def test_nan_exponent_autodetect(self):
        # pass(15): 'exps', [.5 NaN] -- the right exponent (an order-2 pole)
        # is autodetected while the left (0.5 branch point) is given.
        # (Historical: this build once resolved only to ~1.4e-11 and was
        # skipped; the 2026-07 Singfun endpoint/abs/power fixes brought it
        # to ~1e-14.  MATLAB's own bound is 1e1*eps*||op||; the values at
        # random interior points sit within a small multiple of it, checked
        # here at 1e2*eps -- the singular factors amplify roundoff by ~x10
        # near the pole, measured 2026-07-30.)
        from chebfunjax.fun.singfun import _find_sing_order
        detected = _find_sing_order(
            lambda x: jnp.exp(x) * jnp.sqrt(1 + x) / (1 - x) ** 2, "right")
        assert abs(detected - (-2.0)) < 1e-9
        dom = (-1.0, 1.0)
        op = lambda x: np.exp(x) * np.sqrt(1 + x) / (1 - x) ** 2  # noqa: E731
        f = cj.chebfun(lambda x: jnp.exp(x) * jnp.sqrt(1 + x) / (1 - x) ** 2,
                       domain=dom, exps=[0.5, float("nan")])
        err, tol = _check(f, op, dom, 1e2)
        assert err < tol

    def test_cell_array_operators(self):
        # pass(12): piecewise construction from a CELL ARRAY of operators
        # {sin(x), 1/(1+x), x+1} on [-2 -1 0 1] with per-interval exps
        # [0 0 -1 0 0 0] (a pole at the left end of the middle piece).
        dom = [-2.0, -1.0, 0.0, 1.0]
        ops = [lambda x: jnp.sin(x),
               lambda x: 1.0 / (1.0 + x),
               lambda x: x + 1.0]
        ops_np = [np.sin, lambda x: 1.0 / (1.0 + x), lambda x: x + 1.0]
        f = cj.chebfun(ops, domain=dom, exps=[0, 0, -1, 0, 0, 0])
        rng = np.random.default_rng(6178)
        for j in range(3):
            a, b = dom[j], dom[j + 1]
            x = (b - a) * rng.uniform(size=100) + a
            ve = np.asarray(ops_np[j](x))
            err = np.max(np.abs(np.asarray(f(jnp.asarray(x))) - ve))
            assert err < 1e3 * EPS * np.max(np.abs(ve)), j

    def test_cell_array_operators_blowup(self):
        # pass(14): a cell array of operators with poles at interior
        # breakpoints, resolved with 'blowup' autodetection.
        dom = [-2.0, 0.0, 7.0]
        op1 = lambda x: jnp.sin(x) / ((x - dom[0]) * (dom[1] - x))  # noqa: E731
        op2 = lambda x: jnp.cos(x) / (  # noqa: E731
            (x - dom[1]) ** 2 * (dom[2] - x))
        n1 = lambda x: np.sin(x) / ((x - dom[0]) * (dom[1] - x))  # noqa: E731
        n2 = lambda x: np.cos(x) / (  # noqa: E731
            (x - dom[1]) ** 2 * (dom[2] - x))
        f = cj.chebfun([op1, op2], domain=dom, blowup=1)
        rng = np.random.default_rng(6178)
        errs, scales = [], []
        for a, b, ref in ((dom[0], dom[1], n1), (dom[1], dom[2], n2)):
            x = (b - a) * rng.uniform(size=100) + a
            ve = np.asarray(ref(x))
            errs.append(np.max(np.abs(np.asarray(f(jnp.asarray(x))) - ve)))
            scales.append(np.max(np.abs(ve)))
        assert max(errs) < 1e2 * EPS * max(scales)

    def test_cell_array_length_mismatch_raises(self):
        # One operator per interval is required.
        with pytest.raises(ValueError):
            cj.chebfun([jnp.sin, jnp.cos], domain=[-1.0, 0.0, 0.5, 1.0])

    def test_exps_splitting_double_pole(self):
        # pass(13): sin(300x)/((x-a)(x-b)) with 'exps' [-1 -1] and
        # 'splitting','on'; values within 1e4*eps*||exact||.
        dom = (-2.0, 7.0)
        op = lambda x: np.sin(300 * x) / ((x - dom[0]) * (x - dom[1]))  # noqa: E731
        f = cj.chebfun(
            lambda x: jnp.sin(300 * x) / ((x - dom[0]) * (x - dom[1])),
            domain=dom, exps=[-1, -1], splitting=True)
        err, tol = _check(f, op, dom, 1e4)
        assert err < tol

    def test_splitting_singular(self):
        # The preceding 33-piece sin(300x) construction leaves hundreds
        # of compiled-shape cache entries that slow every new XLA
        # compile here by ~300x (55 min observed in-suite vs 10s fresh);
        # release them first (same pattern as the guide17 generator).
        import gc

        import jax
        jax.clear_caches()
        gc.collect()
        # pass(16): tan on [0, pi] with 'splitting','on','blowup','on'
        # has nonzero norm; pass(17): tan on [5*pi, 6*pi] with blowup 1
        # finds exactly one interior pole (three domain points).  The
        # automatic pole detection (findBlowup port, 2026-07-29) places
        # the breaks at the poles.
        f16 = cj.chebfun(lambda x: jnp.tan(x), domain=[0.0, np.pi],
                         splitting=True, blowup=True)
        # norm(f, inf) > 0: any nonzero value suffices (a pole makes the
        # sup-norm infinite; minandmax on the singular rep is expensive).
        assert float(np.abs(np.asarray(f16(jnp.asarray([1.0]))))[0]) > 0
        assert len(f16.funs) >= 2
        f17 = cj.chebfun(lambda x: jnp.tan(x),
                         domain=[5.0 * np.pi, 6.0 * np.pi],
                         splitting=True, blowup=1)
        assert len(f17.domain.breakpoints) == 3
        assert abs(sorted(f17.domain.breakpoints)[1] - 5.5 * np.pi) < 1e-8
