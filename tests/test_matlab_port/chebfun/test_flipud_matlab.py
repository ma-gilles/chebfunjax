"""Eleven native flipud predicates; MATLAB RNG stream identity unverified.
Source tests/chebfun/test_flipud.m, nativecommit7574c77. Python complex-power
promotion implements MATLAB implicit complex powers on negative real bases.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebfunPref

EPS = np.finfo(float).eps
XR = jnp.asarray(2 * np.random.RandomState(7681).rand(50) - 1)


def pref():
    p = ChebfunPref()
    p.splitting = True
    return p


def scalarop(x):
    return jnp.sin(x) * jnp.abs(x - 0.1)


def arrayop(x):
    return jnp.stack([scalarop(x), jnp.cos(x) * jnp.sign(x + 0.2)], axis=-1)


class TestChebfunFlipud:
    def test_native_empty(self):
        assert cj.chebfun().flipud().isempty()

    @pytest.mark.parametrize("array", [False, True])
    def test_native_column(self, array):
        op = arrayop if array else scalarop
        f = cj.chebfun(
            op, domain=(-1.0, -0.2, 0.1, 1.0) if array else (-1.0, 0.1, 1.0), pref=pref()
        )
        g = f.flipud()
        err = float(jnp.max(jnp.abs(g(XR) - op(-XR))))
        bound = (100 if array else 10) * g.vscale * EPS
        print(
            {"source_clauses": (4, 5) if array else (2, 3), "error": err, "source_bound": bound},
            flush=True,
        )
        assert err < bound
        assert g.flipud().isequal(f)

    @pytest.mark.parametrize("array", [False, True])
    def test_native_row(self, array):
        op = arrayop if array else scalarop
        f = cj.chebfun(
            op, domain=(-1.0, -0.2, 0.1, 1.0) if array else (-1.0, 0.1, 1.0), pref=pref()
        ).transpose()
        g = f.flipud()
        expected = op(XR)[:, ::-1].T if array else op(XR)
        err = float(jnp.max(jnp.abs(g(XR) - expected)))
        bound = (100 if array else 10) * g.vscale * EPS
        print(
            {"source_clauses": (8, 9) if array else (6, 7), "error": err, "source_bound": bound},
            flush=True,
        )
        assert err < bound
        assert g.flipud().isequal(f)

    @pytest.mark.parametrize("row", [True, False])
    def test_native_singular(self, row):
        dom = (-2.0, 7.0)
        power = -0.5
        op = lambda x: (x - dom[0] + 0j) ** power * jnp.sin(100 * x) * (x - dom[1] + 0j) ** power
        f = cj.chebfun(op, domain=dom, exps=(power, power), splitting=row)
        if row:
            f = f.transpose()
        g = f.flipud()
        x = jnp.asarray(9 * np.random.RandomState(6178).rand(100) - 2)
        expected = (
            f(x)
            if row
            else (dom[0] - x + 0j) ** power
            * jnp.sin(100 * (sum(dom) - x))
            * (dom[1] - x + 0j) ** power
        )
        # MATLAB treats 1x100 feval as a vector: infinity norm is maxabs.
        error = float(jnp.max(jnp.abs(g(x) - expected)))
        bound = (
            EPS * float(jnp.max(jnp.abs(expected)))
            if row
            else 1e4 * EPS * float(jnp.max(jnp.abs(expected)))
        )
        print(
            {"source_clause": 10 if row else 11, "error": error, "source_bound": bound}, flush=True
        )
        assert error < bound


class TestPreviousFlipudControls:
    def test_flip_reverses_argument(self):
        f = cj.chebfun(lambda x: jnp.exp(x) * jnp.sin(3 * x))
        g = f.flipud()
        xs = jnp.asarray(np.linspace(-0.95, 0.95, 60))
        err = jnp.abs(g(xs) - f(-xs))
        assert float(jnp.max(err)) < 100 * EPS * f.vscale

    def test_double_flip_is_identity(self):
        f = cj.chebfun(lambda x: jnp.cos(5 * x) + x)
        g = f.flipud().flipud()
        xs = jnp.asarray(np.linspace(-0.95, 0.95, 60))
        err = jnp.abs(g(xs) - f(xs))
        assert float(jnp.max(err)) < 100 * EPS * f.vscale


@pytest.mark.parametrize("row", [False, True])
def test_flipud_native_breakpoint_values(row):
    f = cj.chebfun(arrayop, domain=(-1.0, -0.2, 0.1, 1.0), pref=pref())
    if row:
        f = f.transpose()
    g = f.flipud()
    x = jnp.asarray([-0.2, 0.1]) if row else jnp.asarray([0.2, -0.1])
    expected = f(x)[::-1, :] if row else f(-x)
    np.testing.assert_array_equal(np.asarray(g(x)), np.asarray(expected))
