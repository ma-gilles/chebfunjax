"""Three literal native VAR predicates; STD controls are diagnostic only."""

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebfunPref

EPS = np.finfo(float).eps


class TestNativeVar:
    @pytest.mark.parametrize("row", [False, True])
    def test_cell_ops(self, row):
        f = cj.chebfun(
            [lambda x: jnp.exp(4 * jnp.pi * 1j * x), jnp.exp, jnp.exp],
            domain=(-1.0, 0.0, 0.5, 1.0),
            pref=ChebfunPref(),
        )
        if row:
            f = f.transpose()
        error = abs(complex(np.asarray(f.var())) - np.exp(1) / 2)
        bound = 10 * f.vscale * EPS
        print({"clause": 2 if row else 1, "error": error, "bound": bound}, flush=True)
        assert error < bound

    def test_array(self):
        f = cj.chebfun(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1),
            domain=(-1.0, -0.5, 0.0, 0.5, 1.0),
            pref=ChebfunPref(),
        )
        exact = np.array(
            [
                (1 - np.sin(2) / 2) / 2,
                (1 + np.sin(1) * np.cos(1)) / 2 - np.sin(1) ** 2,
                (1 - np.exp(-2)) / 2,
            ]
        )
        error = np.max(np.abs(np.asarray(f.var()).reshape(-1) - exact))
        bound = 10 * f.vscale * EPS
        print({"clause": 3, "error": error, "bound": bound}, flush=True)
        assert error < bound


class TestDiagnosticStd:
    def test_array_row(self):
        f = cj.chebfun(lambda x: jnp.stack([x, 1j * x], axis=-1), pref=ChebfunPref())
        out = np.asarray(f.transpose().var())
        assert out.shape == (2, 1)
        assert np.max(np.abs(out - 1 / 3)) < 2 * EPS

    def test_std(self):
        f = cj.chebfun(lambda x: 1j * x, pref=ChebfunPref())
        assert abs(complex(np.asarray(f.std())) - np.sqrt(1 / 3)) < 2 * EPS
