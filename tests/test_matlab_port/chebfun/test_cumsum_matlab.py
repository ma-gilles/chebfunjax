"""Port of MATLAB Chebfun tests/chebfun/test_cumsum.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_cumsum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
# Native seedRNG(7681); xr = 2*rand(1000,1)-1, captured by the shared
# native repmat fixture whose MATLAB source resets to the identical stream.
XR = jnp.asarray(json.loads(Path(__file__).with_name('repmat_matlab_inputs.json').read_text())['xr'])[:, None]


class TestChebfunCumsum:
    def test_empty(self):
        from chebfunjax.chebfun1d.chebfun import chebfun
        assert chebfun().cumsum().isempty()

    def test_piecewise_cos(self):
        f1 = cj.chebfun(jnp.cos, domain=[-1.0, -0.5, 0.5, 1.0])
        If1 = f1.cumsum()
        exact = jnp.sin(XR) - np.sin(-1.0)
        values = If1(XR)
        assert values.shape == XR.shape == (1000, 1)
        err = jnp.abs(values - exact)
        assert float(jnp.max(err)) < 10 * If1.vscale * EPS

    def test_transpose_variant(self):
        # pass(3): cumsum of a row chebfun f1.' matches (sin - sin(-1)).'
        # -- and stays a row (isTransposed).
        f1 = cj.chebfun(jnp.cos, domain=[-1.0, -0.5, 0.5, 1.0])
        If1t = f1.T.cumsum()
        assert If1t.is_transposed
        exact = (jnp.sin(XR[:, 0]) - np.sin(-1.0))[None, :]
        values = If1t(XR)
        assert values.shape == exact.shape == (1, 1000)
        err = jnp.abs(values - exact)
        assert float(jnp.max(err)) < 10 * If1t.vscale * EPS

    def test_array_valued(self):
        # pass(4): piecewise array-valued cumsum on [-1 -0.5 0.5 1].
        # FIXED (Fable 5, Big-Three array-valued epic): per-column offset chain.
        f3 = cj.chebfun(
            lambda x: jnp.stack([jnp.cos(x), -jnp.sin(x), jnp.exp(x)], axis=-1),
            domain=(-1, -0.5, 0.5, 1),
        )
        If3 = f3.cumsum()
        t = XR[:, 0]
        exact = jnp.stack(
            [jnp.sin(t), jnp.cos(t), jnp.exp(t)], axis=-1
        ) - jnp.array([np.sin(-1.0), np.cos(-1.0), np.exp(-1.0)])
        values = If3(XR)
        assert values.shape == exact.shape == (1000, 3)
        err = jnp.abs(values - exact)
        assert float(jnp.max(err)) < 10 * If3.vscale * EPS

    def test_array_transpose_source_pass_05(self):
        f3 = cj.chebfun(
            lambda x: jnp.stack([jnp.cos(x), -jnp.sin(x), jnp.exp(x)], axis=-1),
            domain=(-1, -0.5, 0.5, 1),
        )
        If3t = f3.T.cumsum()
        t = XR[:, 0]
        exact = (jnp.stack([jnp.sin(t), jnp.cos(t), jnp.exp(t)], axis=-1)
                 - jnp.array([np.sin(-1.0), np.cos(-1.0), np.exp(-1.0)])).T
        values = If3t(XR)
        assert If3t.is_transposed and values.shape == exact.shape == (3, 1000)
        assert float(jnp.max(jnp.abs(values - exact))) < 10 * If3t.vscale * EPS

    def test_two_column_array_source_pass_06(self):
        f4 = cj.chebfun(
            lambda x: jnp.stack([jnp.cos(x), -jnp.sin(x)], axis=-1),
            domain=(-1, -0.5, 0.5, 1),
        )
        If4 = f4.cumsum()
        t = XR[:, 0]
        exact = jnp.stack([jnp.sin(t), jnp.cos(t)], axis=-1)
        exact = exact - jnp.array([np.sin(-1.0), np.cos(-1.0)])
        values = If4(XR)
        assert values.shape == exact.shape == (1000, 2)
        assert float(jnp.max(jnp.abs(values - exact))) < 10 * If4.vscale * EPS

    @pytest.mark.parametrize('source_clause', (7, 8))
    def test_second_scalar_antiderivative_source_pass_07_08(self, source_clause):
        # MATLAB repeats this exact predicate as both pass 7 and pass 8.
        f1 = cj.chebfun(jnp.cos, domain=(-1, -0.5, 0.5, 1))
        I2f1 = f1.cumsum(2)
        t = XR[:, 0]
        exact = (-jnp.cos(t) - np.sin(-1.0) * t
                 - (-np.cos(-1.0) - np.sin(-1.0) * -1.0))[:, None]
        values = I2f1(XR)
        assert values.shape == exact.shape == (1000, 1)
        assert float(jnp.max(jnp.abs(values - exact))) < 10 * I2f1.vscale * EPS

    def test_second_array_antiderivative_source_pass_09(self):
        f3 = cj.chebfun(
            lambda x: jnp.stack([jnp.cos(x), -jnp.sin(x), jnp.exp(x)], axis=-1),
            domain=(-1, -0.5, 0.5, 1),
        )
        I2f3 = f3.cumsum(2)
        t = XR[:, 0]
        exact = jnp.stack([
            -jnp.cos(t) - np.sin(-1.0) * t - (-np.cos(-1.0) - np.sin(-1.0) * -1.0),
            jnp.sin(t) - np.cos(-1.0) * t - (np.sin(-1.0) - np.cos(-1.0) * -1.0),
            jnp.exp(t) - np.exp(-1.0) * t - (np.exp(-1.0) - np.exp(-1.0) * -1.0),
        ], axis=-1)
        values = I2f3(XR)
        assert values.shape == exact.shape == (1000, 3)
        assert float(jnp.max(jnp.abs(values - exact))) < 10 * I2f3.vscale * EPS

    def test_cumsum_of_deriv_recovers(self):
        f = cj.chebfun(lambda x: jnp.exp(x) * jnp.sin(3 * x))
        g = f.diff().cumsum()
        # g(x) = f(x) - f(-1)
        exact = f(XR) - float(f(jnp.asarray(-1.0)))
        err = jnp.abs(g(XR) - exact)
        assert float(jnp.max(err)) < 1e3 * f.vscale * EPS
