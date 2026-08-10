"""Port of MATLAB Chebfun tests/chebfun/test_besselj.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_besselj.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import jv

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(6178)
XR = jnp.asarray(2 * RNG.uniform(size=100) - 1)


class TestChebfunBesselj:
    def test_besselj_of_exp(self):
        f = cj.chebfun(jnp.exp, domain=[-1.0, -0.5, 0.0, 0.5, 1.0])
        h = f.besselj(2)
        exact = jnp.asarray(jv(2, np.exp(np.asarray(XR))))
        err = jnp.abs(h(XR) - exact)
        assert float(jnp.max(err)) < 100 * EPS * max(h.vscale, 1.0)

    def test_scale_option(self):
        # MATLAB pass(2,3): besselj(nu, f, 0) matches the default, and
        # besselj(nu, f, 1) scales by exp(-|imag(f)|), which is 1 for the
        # real-valued f used here.
        f = cj.chebfun(jnp.exp, domain=[-1.0, -0.5, 0.0, 0.5, 1.0])
        h = f.besselj(2)
        h0 = f.besselj(2, 0)
        h1 = f.besselj(2, 1)
        exact = jnp.asarray(jv(2, np.exp(np.asarray(XR))))
        tol = 10 * EPS * max(h.vscale, h0.vscale, h1.vscale, 1.0)
        assert float(jnp.max(jnp.abs(h0(XR) - h(XR)))) < tol
        assert float(jnp.max(jnp.abs(h1(XR) - exact))) < 100 * tol

    def test_scale_option_complex(self):
        # For a complex-valued f the scale factor exp(-|imag(f)|) is
        # nontrivial (MATLAB besselj(nu, f, 1)).
        f = cj.chebfun(lambda x: x + 2j * x)
        h1 = f.besselj(2, 1)
        xr = np.asarray(XR)
        z = xr + 2j * xr
        exact = jv(2, z) * np.exp(-np.abs(np.imag(z)))
        err = np.max(np.abs(np.asarray(h1(XR)) - exact))
        assert float(err) < 1e3 * EPS * max(h1.vscale, 1.0)

    def test_complex_nu_raises(self):
        # MATLAB pass(8): 'CHEBFUN:CHEBFUN:besselj:nu'.
        f = cj.chebfun(lambda x: x)
        with pytest.raises(ValueError):
            f.besselj(3 + 1j)
