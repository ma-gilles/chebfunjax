"""Port of MATLAB Chebfun tests/chebtech/test_flipud.m (Fable 5).

MATLAB ``flipud`` reflects a tech in x (x -> -x), i.e. ``g(x) = f(-x)``
(negates the odd-index Chebyshev coefficients).  chebfunjax now provides
``Tech.flipud()`` for both scalar and array-valued (and complex) techs, so
every assertion in this file is a real port at MATLAB's tolerances.

Provenance
----------
MATLAB source : tests/chebtech/test_flipud.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)


def _coeff_ninf(g, h):
    """max-norm of g.coeffs - h.coeffs, prolonging to a common length."""
    n = max(g.n, h.n)
    gc = g.prolong(n).coeffs
    hc = h.prolong(n).coeffs
    return float(jnp.max(jnp.abs(gc - hc)))


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechFlipud:
    def test_flipud_scalar_real(self, Tech):
        # pass(n,1): flipud(sin(x+.5)) == sin(-x+.5)
        # FIXED (Fable 5, Big-Three array-valued epic): Tech.flipud() added.
        f = Tech.from_function(lambda x: jnp.sin(x + 0.5))
        g = Tech.from_function(lambda x: jnp.sin(-x + 0.5))
        h = f.flipud()
        assert _coeff_ninf(g, h) < 10 * h.vscale * EPS

    def test_flipud_array_real(self, Tech):
        # pass(n,2): flipud([sin(x+.5), exp(x)]) == [sin(-x+.5), exp(-x)]
        # FIXED (Fable 5, Big-Three array-valued epic): array-valued flipud.
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x + 0.5), jnp.exp(x)], axis=-1)
        )
        g = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(-x + 0.5), jnp.exp(-x)], axis=-1)
        )
        h = f.flipud()
        assert _coeff_ninf(g, h) < 10 * h.vscale * EPS

    def test_flipud_scalar_complex(self, Tech):
        # pass(n,3): flipud(sin(1i*x+.5)) == sin(-1i*x+.5)
        # FIXED (Fable 5, Big-Three array-valued epic): complex flipud.
        f = Tech.from_function(lambda x: jnp.sin(1j * x + 0.5))
        g = Tech.from_function(lambda x: jnp.sin(-1j * x + 0.5))
        h = f.flipud()
        assert _coeff_ninf(g, h) < 10 * h.vscale * EPS

    def test_flipud_array_complex(self, Tech):
        # pass(n,4): flipud([sin(x+.5), exp(1i*x)]) == [sin(-x+.5), exp(-1i*x)]
        # FIXED (Fable 5, Big-Three array-valued epic): complex array flipud.
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x + 0.5), jnp.exp(1j * x)], axis=-1)
        )
        g = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(-x + 0.5), jnp.exp(-1j * x)], axis=-1)
        )
        h = f.flipud()
        assert _coeff_ninf(g, h) < 10 * h.vscale * EPS
