"""All18literalnative MEAN predicates; no MATLAB RNG stream identity claim.
Native tests/chebfun/test_mean.m commit7574c77. Matrix infinity norms retain
native matrix shapes; singleton dimensions use MATLAB vector maxabs.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebfunPref

EPS = np.finfo(float).eps


def matrix_inf(values, row_vector=False):
    a = np.asarray(values)
    if a.ndim == 0:
        return float(abs(a))
    if a.ndim == 1:
        a = a.reshape((1, -1) if row_vector else (-1, 1))
    if 1 in a.shape:
        return float(np.max(np.abs(a)))
    return float(np.max(np.sum(np.abs(a), axis=1)))


def normest(f):
    # Native seed reset each invocation and100uniformpoints.
    x = jnp.asarray(2 * np.random.RandomState(6178).rand(100) - 1)
    return matrix_inf(f(x))


class TestNativeMeanFinite:
    @pytest.mark.parametrize("array", [False, True])
    def test_two_inputs(self, array):
        p = ChebfunPref()
        if array:
            f = cj.chebfun(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1), pref=p)
            g = cj.chebfun(lambda x: jnp.stack([jnp.cos(x), 1j * jnp.exp(x)], axis=-1), pref=p)
            h = 0.5 * (f + g)
        else:
            f = cj.chebfun(jnp.sin, pref=p)
            g = cj.chebfun(jnp.cos, pref=p)
            h = cj.chebfun(lambda x: 0.5 * (jnp.sin(x) + jnp.cos(x)), pref=p)
        error = normest(f.mean(g) - h)
        print({"native_clause": 2 if array else 1, "error": error, "bound": 10 * EPS}, flush=True)
        assert error < 10 * EPS

    @pytest.mark.parametrize("row", [False, True])
    def test_odd(self, row):
        f = cj.chebfun(jnp.sin, pref=ChebfunPref())
        if row:
            f = f.transpose()
        assert abs(complex(np.asarray(f.mean()))) < EPS

    @pytest.mark.parametrize("row", [False, True])
    def test_array_odd(self, row):
        f = cj.chebfun(lambda x: jnp.stack([jnp.sin(x), x], axis=-1), pref=ChebfunPref())
        if row:
            f = f.transpose()
        error = matrix_inf(f.mean(), row_vector=not row)
        print({"native_clause": 6 if row else 5, "error": error, "bound": EPS}, flush=True)
        assert error < EPS

    @pytest.mark.parametrize("row", [False, True])
    def test_array_domain(self, row):
        f = cj.chebfun(
            lambda x: jnp.stack([jnp.sin(x), x], axis=-1), domain=(0.0, 6.0), pref=ChebfunPref()
        )
        expected = jnp.asarray(f.sum()) / 6
        if row:
            f = f.transpose()
            expected = expected.reshape((-1, 1))
        error = matrix_inf(f.mean() - expected, row_vector=not row)
        print(
            {"native_clause": 8 if row else 7, "error": error, "bound": f.vscale * EPS}, flush=True
        )
        assert error < f.vscale * EPS


class TestNativeMeanSingular:
    @pytest.mark.parametrize("clause", [9, 10, 11])
    def test_original(self, clause):
        dom = (-2.0, 7.0)
        if clause == 9:
            op = lambda x: jnp.sin(100 * x) / ((x - dom[0] + 0j) ** 0.5 * (x - dom[1] + 0j) ** 0.5)
            exps = (-0.5, -0.5)
        elif clause == 10:
            op = lambda x: jnp.sin(100 * x) / ((x - dom[0]) ** 1.5 * (dom[1] - x) ** 0.5)
            exps = (-1.5, -0.5)
        else:
            op = lambda x: jnp.sin(98 * x) / ((x - dom[0]) ** 1.5 * (dom[1] - x) ** 1.5)
            exps = (-1.5, -1.5)
        f = cj.chebfun(op, domain=dom, exps=exps, splitting="on")
        value = complex(np.asarray(f.mean()))
        print(
            {"native_clause": clause, "mean": repr(value), "length": len(f), "pieces": len(f.funs)},
            flush=True,
        )
        if clause == 9:
            exact = -0.01273522016443600j
            error = abs(value - exact)
            bound = 1e5 * EPS * abs(exact)
            print({"error": error, "source_bound": bound}, flush=True)
            assert error < bound
        elif clause == 10:
            assert np.isinf(value)
        else:
            assert np.isnan(value)


class TestNativeMeanUnbounded:
    def test_doubly_unbounded_nan(self):
        op1 = lambda x: x**2 * jnp.exp(-(x**2))
        op2 = lambda x: (1 - jnp.exp(-(x**2))) / x**2 + 2
        f = cj.chebfun([op1, op2], domain=(-jnp.inf, 2.0, jnp.inf))
        assert bool(jnp.isnan(f.mean()))

    def test_halfline_limit(self):
        op = lambda x: 0.75 + jnp.sin(10 * x) / jnp.exp(x)
        f = cj.chebfun(op, domain=(0.0, jnp.inf), splitting="on")
        error = abs(complex(np.asarray(f.mean())) - 0.75)
        bound = 100 * EPS * f.vscale
        print(
            {
                "native_clause": 13,
                "error": error,
                "bound": bound,
                "point_values": repr(np.asarray(f.point_values)),
            },
            flush=True,
        )
        assert error < bound


class TestNativeMeanDimensions:
    @pytest.mark.parametrize("clause", [14, 15, 16, 17, 18])
    def test_source(self, clause):
        x = cj.chebfun("x")
        if clause in (14, 15):
            f = type(x).horzcat(x, 3 * x)
            if clause == 14:
                assert float((f.mean(2) - 2 * x).norm()) < 100 * EPS
            else:
                assert float(np.linalg.norm(np.asarray(f.mean(1) - f.mean()))) == 0
        elif clause == 16:
            assert abs(complex(np.asarray(x.mean(1) - x.transpose().mean(2)))) == 0
        elif clause == 17:
            assert abs(complex(np.asarray(x.ctranspose().mean()))) < 100 * EPS
        else:
            assert float((x.ctranspose().mean(1) - x.ctranspose()).norm()) == 0


X = jnp.asarray(np.linspace(-0.97, 0.97, 60))


class TestPreviousMeanControls:
    def test_mean_of_two_functions(self):
        f = cj.chebfun(jnp.sin)
        g = cj.chebfun(jnp.cos)
        h = cj.chebfun(lambda x: 0.5 * (jnp.sin(x) + jnp.cos(x)))
        d = 0.5 * (f + g) - h
        assert float(jnp.max(jnp.abs(d(X)))) < 10 * EPS

    def test_mean_of_odd_function_is_zero(self):
        f = cj.chebfun(jnp.sin)
        assert abs(float(f.mean())) < EPS * 10

    def test_array_valued(self):
        # pass(5): mean of [sin(x), x] is ~0 per column.
        # pass(7): mean == sum / length on the domain [0, 6].
        # FIXED (Fable 5, Big-Three array-valued epic): array-valued mean -> (m,).
        f = cj.chebfun(lambda x: jnp.stack([jnp.sin(x), x], axis=-1))
        assert float(jnp.max(jnp.abs(f.mean()))) < EPS
        f2 = cj.chebfun(lambda x: jnp.stack([jnp.sin(x), x], axis=-1), domain=(0, 6))
        assert float(jnp.max(jnp.abs(f2.mean() - f2.sum() / 6))) < f2.vscale * EPS
