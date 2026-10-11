"""All ten literal native covariance predicates, Chebfun commit7574c77."""

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebfunPref

EPS = np.finfo(float).eps
DOM = (-1.0, -0.5, 0.0, 0.5, 1.0)


def array_f():
    return cj.chebfun(
        lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1),
        domain=DOM,
        pref=ChebfunPref(),
    )


def exact_cov():
    css = (1 - np.sin(2) / 2) / 2
    csc = 0
    cse = (np.sin(1) + np.cos(1) + np.exp(2) * (np.sin(1) - np.cos(1))) / (4 * np.exp(1))
    ccc = (1 + np.sin(1) * np.cos(1)) / 2 - np.sin(1) ** 2
    cce = (np.sin(1) - np.cos(1) + np.exp(2) * (np.sin(1) + np.cos(1))) / (4 * np.exp(1)) - np.sin(
        1
    ) * (np.exp(1) - np.exp(-1)) / 2
    cee = (1 - np.exp(-2)) / 2
    return np.array([[css, csc, cse], [csc, ccc, cce], [cse, cce, cee]])


class TestNativeCov:
    def test_empty(self):
        assert bool(jnp.isnan(cj.chebfun([]).cov()))

    @pytest.mark.parametrize("row", [False, True])
    def test_scalar_identity(self, row):
        f = cj.chebfun(
            [lambda x: jnp.exp(4 * jnp.pi * 1j * x), jnp.exp, jnp.exp],
            domain=(-1.0, 0.0, 0.5, 1.0),
            pref=ChebfunPref(),
        )
        if row:
            f = f.transpose()
        assert np.array_equal(np.asarray(f.cov()), np.asarray(f.var()))

    @pytest.mark.parametrize("clause", [4, 5, 6])
    def test_array(self, clause):
        f = array_f()
        exact = exact_cov()
        if clause == 5:
            f = f.transpose()
            exact = exact.T
        actual = f.cov(f) if clause == 6 else f.cov()
        error = np.linalg.norm((np.asarray(actual) - exact).reshape(-1))
        bound = 10 * f.vscale * EPS
        print({"clause": clause, "error": error, "bound": bound}, flush=True)
        assert error < bound

    @pytest.mark.parametrize("row", [False, True])
    def test_two_inputs(self, row):
        f = cj.chebfun(lambda x: jnp.stack([x, jnp.sin(x)], axis=-1), pref=ChebfunPref())
        g = cj.chebfun(
            lambda x: jnp.stack([jnp.exp(x), jnp.exp(2 * jnp.pi * 1j * x)], axis=-1),
            domain=DOM,
            pref=ChebfunPref(),
        )
        exact = np.array(
            [
                [1 / np.exp(1), -1 / (2 * np.pi * 1j)],
                [exact_cov()[0, 2], -2 * np.pi * 1j * np.sin(1) / (1 - 4 * np.pi**2)],
            ]
        )
        if row:
            f = f.transpose()
            g = g.transpose()
            exact = exact.T
        error = np.linalg.norm((np.asarray(f.cov(g)) - exact).reshape(-1))
        bound = 10 * max(f.vscale * EPS, g.vscale * EPS)
        print({"clause": 8 if row else 7, "error": error, "bound": bound}, flush=True)
        assert error < bound

    @pytest.mark.parametrize("normalization", [False, True])
    def test_errors(self, normalization):
        f = cj.chebfun(jnp.sin, pref=ChebfunPref())
        g = cj.chebfun(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1), pref=ChebfunPref())
        identifier = "CHEBFUN:CHEBFUN:cov:nargin" if normalization else "CHEBFUN:CHEBFUN:cov:size"
        with pytest.raises(ValueError, match=identifier):
            if normalization:
                f.cov(g, 1)
            else:
                f.cov(g)


class TestSupplementalComplexCov:
    @pytest.mark.parametrize("row", [False, True])
    def test_complex_single_input(self, row):
        f = cj.chebfun(lambda x: jnp.stack([x, 1j * x], axis=-1), pref=ChebfunPref())
        exact = np.array([[1 / 3, -1j / 3], [1j / 3, 1 / 3]])
        if row:
            f = f.transpose()
            exact = exact.T
        actual = np.asarray(f.cov())
        error = np.linalg.norm((actual - exact).reshape(-1))
        print(
            {
                "supplemental_complex_row": row,
                "dtype": str(actual.dtype),
                "error": error,
                "bound": 2 * EPS,
            },
            flush=True,
        )
        assert actual.shape == (2, 2)
        assert error < 2 * EPS
