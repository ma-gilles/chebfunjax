"""Port of MATLAB Chebfun tests/chebfun/test_complex.m (Fable 5).

FIXED: complex_fun(f, g) = f + 1i*g added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_complex.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

XS = jnp.asarray(np.linspace(-0.95, 0.95, 50))


class TestChebfunComplex:
    def test_basic(self):
        f = cj.chebfun(jnp.sin)
        g = cj.chebfun(jnp.cos)
        z = cj.complex_fun(f, g)
        exact = np.sin(np.asarray(XS)) + 1j * np.cos(np.asarray(XS))
        assert np.max(np.abs(np.asarray(z(XS)) - exact)) < 1e-14

    def test_complex_input_rejected(self):
        f = cj.chebfun(lambda x: jnp.exp(1j * x))
        g = cj.chebfun(jnp.cos)
        with pytest.raises(ValueError):
            cj.complex_fun(f, g)
