"""Port of MATLAB Chebfun tests/chebfun/test_feval.m (Fable 5).

String endpoint syntaxes ('left','start','-') are skipped; numeric
evaluation incl. endpoint one-sided values ported.

Provenance
----------
MATLAB source : tests/chebfun/test_feval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import erf

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(7681)
XR = jnp.asarray(2 * RNG.uniform(size=1000) - 1)


class TestChebfunFeval:
    def test_empty_input_shape(self):
        f = cj.chebfun(lambda x: x)
        fx = f(jnp.zeros((0,)))
        assert np.asarray(fx).shape == (0,)

    def test_string_endpoint_syntax(self):
        pytest.skip("chebfunjax feval has no 'left'/'start'/'-' string "
                    "arguments")

    def test_endpoint_values(self):
        f = cj.chebfun(lambda x: jnp.asarray(erf(np.asarray(x))))
        lval = float(f(jnp.asarray(-1.0)))
        rval = float(f(jnp.asarray(1.0)))
        assert abs(lval - erf(-1.0)) < 10 * EPS * f.vscale
        assert abs(rval - erf(1.0)) < 10 * EPS * f.vscale

    def test_vectorized_evaluation(self):
        f = cj.chebfun(lambda x: jnp.exp(x) * jnp.sin(3 * x))
        err = jnp.abs(f(XR) - jnp.exp(XR) * jnp.sin(3 * XR))
        assert float(jnp.max(err)) < 100 * f.vscale * EPS

    def test_matrix_shaped_input(self):
        f = cj.chebfun(lambda x: x ** 2)
        xm = jnp.asarray(np.linspace(-1, 1, 12).reshape(3, 4))
        out = f(xm)
        assert np.asarray(out).shape == (3, 4)
        assert float(jnp.max(jnp.abs(out - xm ** 2))) < 100 * EPS
