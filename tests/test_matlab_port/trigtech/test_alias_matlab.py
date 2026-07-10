"""Port of MATLAB Chebfun tests/trigtech/test_alias.m (Opus 4.8).

``alias(coeffs, m)`` zero-pads (m > n) or *aliases* (frequency-folds, m < n)
Fourier coefficients.  chebfunjax implements only the zero-padding /
truncation branch (``_trig_prolong_coeffs`` / ``prolong``); it has no
frequency-folding ``alias`` and no array-valued trigtech, so the
downsampling-by-interpolation and array cases are xfailed.

Provenance
----------
MATLAB source : tests/trigtech/test_alias.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.trigtech import _trig_prolong_coeffs


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestTrigtechAlias:
    def test_padding(self):
        # alias((1:9)', 13) == [0; 0; (1:9)'; 0; 0]
        c0 = jnp.arange(1, 10, dtype=jnp.complex128)
        c1 = _trig_prolong_coeffs(c0, 13)
        expected = jnp.concatenate(
            [jnp.zeros(2, dtype=jnp.complex128), c0, jnp.zeros(2, dtype=jnp.complex128)]
        )
        assert _ninf(expected - c1) == 0.0

    @pytest.mark.xfail(
        reason="chebfunjax lacks frequency-folding alias (only zero-pad/truncate prolong)"
    )
    @pytest.mark.parametrize("n", [1, 2, 3, 7, 10, 12, 15])
    def test_alias_by_interpolation_real(self, n):
        raise AssertionError("alias (frequency folding) not implemented")

    @pytest.mark.xfail(
        reason="chebfunjax lacks frequency-folding alias (only zero-pad/truncate prolong)"
    )
    @pytest.mark.parametrize("n", [1, 2, 3, 7, 10, 12, 15])
    def test_alias_by_interpolation_complex(self, n):
        raise AssertionError("alias (frequency folding) not implemented")

    @pytest.mark.xfail(
        reason="chebfunjax lacks frequency-folding alias (only zero-pad/truncate prolong)"
    )
    @pytest.mark.parametrize("n", [8, 15])
    def test_alias_by_interpolation_larger(self, n):
        raise AssertionError("alias (frequency folding) not implemented")

    @pytest.mark.xfail(
        reason="chebfunjax lacks frequency-folding alias (only zero-pad/truncate prolong)"
    )
    @pytest.mark.parametrize("n", [8, 15])
    def test_alias_by_interpolation_larger_complex(self, n):
        raise AssertionError("alias (frequency folding) not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    @pytest.mark.parametrize("n", [1, 2, 3, 7, 10, 12, 15])
    def test_alias_array_valued(self, n):
        raise AssertionError("array-valued trigtech not implemented")
