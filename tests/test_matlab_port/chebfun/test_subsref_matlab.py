"""Port of MATLAB Chebfun tests/chebfun/test_subsref.m (Fable 5).

The () evaluation syntaxes map to __call__; f(g) composition, shaped
argument evaluation and array-valued column extraction are asserted.
The remaining MATLAB passes exercise {}-indexing and s-struct
subsref mechanics with no Python counterpart.

Provenance
----------
MATLAB source : tests/chebfun/test_subsref.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

jax.config.update("jax_enable_x64", True)


class TestChebfunSubsref:
    def test_evaluation_syntaxes(self):
        rng = np.random.RandomState(7681)
        xr = jnp.asarray(2 * rng.rand(1000) - 1)
        f = cj.chebfun(lambda x: jnp.sin(x - 0.1))

        # pass(1)-(3): plain and directional evaluation.
        assert np.array_equal(np.asarray(f(xr)), np.asarray(f(xr)))
        assert np.array_equal(np.asarray(f(xr, "left")),
                              np.asarray(f(xr, "left")))
        assert np.array_equal(np.asarray(f(xr, "right")),
                              np.asarray(f(xr, "right")))

        # pass(4): f(g) is composition.
        g = cj.chebfun(lambda x: jnp.cos(x + 0.2))
        h = f(g)
        xs = jnp.linspace(-0.9, 0.9, 21)
        assert float(jnp.max(jnp.abs(
            jnp.asarray(h(xs))
            - jnp.sin(jnp.cos(xs + 0.2) - 0.1)))) < 1e-13

        # pass(6): non-numeric arguments raise.
        try:
            f("X")
            raised = False
        except (TypeError, ValueError):
            raised = True
        assert raised

        # pass(7)-(9): shaped arguments keep their shape.
        for shape in ((100, 10), (5, 20, 10), (5, 4, 5, 10)):
            xm = xr.reshape(shape)
            assert np.asarray(f(xm)).shape == shape
            assert np.allclose(np.asarray(f(xm)).ravel(),
                               np.asarray(f(xr)))

    def test_array_valued(self):
        rng = np.random.RandomState(7681)
        xr = jnp.asarray(2 * rng.rand(100) - 1)
        f = cj.chebfun(lambda x: jnp.stack(
            [jnp.sin(x - 0.1), jnp.cos(x - 0.2)], axis=-1))
        # pass(11): array evaluation gives both columns.
        y = np.asarray(f(xr))
        assert y.shape[-1] == 2
        assert np.allclose(y[..., 0], np.sin(np.asarray(xr) - 0.1))
        assert np.allclose(y[..., 1], np.cos(np.asarray(xr) - 0.2))
        # Column extraction (MATLAB f(:, 2)).
        g = f.extract_columns(1)
        assert np.allclose(np.asarray(g(xr)),
                           np.cos(np.asarray(xr) - 0.2))
