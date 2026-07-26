"""Port of MATLAB Chebfun tests/chebtech/test_isnan.m (Fable 5).

chebfunjax has no ``isnan()`` method, but MATLAB ``@chebtech/isnan.m`` is
exactly ``out = any(isnan(f.coeffs(:)))`` -- a global scalar reduction over
the coefficients.  These tests construct genuine scalar and array-valued
(n, m) techs and assert that equivalent.

MATLAB wraps the NaN-function *op* cases (pass 4, 5) in ``try/catch`` and
accepts EITHER outcome: a NaN-retaining tech (``isnan(f)`` true) OR the error
"Too many NaNs/Infs to handle.".  chebfunjax mirrors MATLAB faithfully -- the
numeric VALUES/scalar path (``make(NaN)`` -> ``from_coeffs``) retains NaN,
while the adaptive-op populate path (``from_function`` sampling an all-NaN
column) extrapolates and raises that exact error -- so the ported tests accept
both branches, exactly like the MATLAB ``try/catch``.

Provenance
----------
MATLAB source : tests/chebtech/test_isnan.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


def _isnan(f):
    """MATLAB @chebtech/isnan.m equivalent: any coeff NaN."""
    return bool(jnp.any(jnp.isnan(f.coeffs)))


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechIsnan:
    def test_scalar_not_nan(self, Tech):
        # pass(n,1): ~isnan(make(@(x) x))
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Tech.from_function(lambda x: x)
        assert not _isnan(f)

    def test_array_not_nan(self, Tech):
        # pass(n,2): ~isnan(make(@(x) [x, x.^2]))
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Tech.from_function(lambda x: jnp.stack([x, x**2], axis=-1))
        assert not _isnan(f)

    def test_constant_nan_is_nan(self, Tech):
        # pass(n,3): isnan(make(NaN))
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Tech.from_coeffs(jnp.array([jnp.nan], dtype=jnp.float64))
        assert _isnan(f)

    def test_scalar_nan_is_nan(self, Tech):
        # pass(n,4): make(@(x) x + NaN) -- MATLAB try/catch accepts either a
        # NaN tech (isnan) or the 'Too many NaNs/Infs to handle.' error.  The
        # op path samples all-NaN and the constructor raises that error
        # (@chebtech/populate.m -> extrapolate.m), which is one accepted branch.
        try:
            f = Tech.from_function(lambda x: x + jnp.nan)
            assert _isnan(f)
        except ValueError as exc:
            assert "Too many NaNs/Infs to handle." in str(exc)

    def test_array_nan_is_nan(self, Tech):
        # pass(n,5): make(@(x) [x, x + NaN]) -- same MATLAB try/catch split.
        # The all-NaN second column masks every row, so extrapolate raises.
        try:
            f = Tech.from_function(lambda x: jnp.stack([x, x + jnp.nan], axis=-1))
            assert _isnan(f)
        except ValueError as exc:
            assert "Too many NaNs/Infs to handle." in str(exc)
