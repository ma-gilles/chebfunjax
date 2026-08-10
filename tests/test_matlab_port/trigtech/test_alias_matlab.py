"""Port of MATLAB Chebfun tests/trigtech/test_alias.m (Opus 4.8).

``alias(coeffs, m)`` zero-pads (m > n) or *aliases* (frequency-folds, m < n)
Fourier coefficients.  chebfunjax now implements ``Trigtech.alias`` as an exact
port of ``@trigtech/alias.m``, so the padding and downsampling-by-interpolation
cases run; the array-valued case is skipped (no array-valued trigtech).

Provenance
----------
MATLAB source : tests/trigtech/test_alias.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech, trigpts

EPS = float(np.finfo(np.float64).eps)
_TOL = 10 * EPS


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _alias_by_interpolating(op, n):
    f = Trigtech.from_function(op)
    c = Trigtech.alias(f.coeffs, n)
    x = trigpts(c.shape[0])
    cexact = Trigtech.vals2coeffs(op(x))
    return _ninf(c - cexact)


class TestTrigtechAlias:
    def test_padding(self):
        # alias((1:9)', 13) == [0; 0; (1:9)'; 0; 0]
        c0 = jnp.arange(1, 10, dtype=jnp.complex128)
        c1 = Trigtech.alias(c0, 13)
        expected = jnp.concatenate(
            [jnp.zeros(2, dtype=jnp.complex128), c0, jnp.zeros(2, dtype=jnp.complex128)]
        )
        assert _ninf(expected - c1) == 0.0

    @pytest.mark.parametrize("n", [1, 2, 3, 7, 10, 12, 15])
    def test_alias_by_interpolation_real(self, n):
        assert _alias_by_interpolating(lambda x: jnp.cos(1 + jnp.sin(2 * jnp.pi * x)), n) < _TOL

    @pytest.mark.parametrize("n", [1, 2, 3, 7, 10, 12, 15])
    def test_alias_by_interpolation_complex(self, n):
        op = lambda x: jnp.cos(1 + jnp.sin(jnp.pi * x)) + 1j * jnp.exp(jnp.cos(jnp.pi * x))
        assert _alias_by_interpolating(op, n) < _TOL

    @pytest.mark.parametrize("n", [8, 15])
    def test_alias_by_interpolation_larger(self, n):
        assert _alias_by_interpolating(lambda x: 1 + jnp.cos(3 * jnp.pi * x), n) < _TOL

    @pytest.mark.parametrize("n", [8, 15])
    def test_alias_by_interpolation_larger_complex(self, n):
        op = lambda x: 1 + jnp.cos(3 * jnp.pi * x) + 1j * jnp.sin(2 * jnp.pi * x)
        assert _alias_by_interpolating(op, n) < _TOL

    @pytest.mark.parametrize("n", [1, 2, 3, 7, 10, 12, 15])
    def test_alias_array_valued(self, n):
        op = lambda x: jnp.stack(
            [jnp.cos(jnp.pi * x),
             jnp.sin(1 + jnp.cos(jnp.pi * x)),
             jnp.exp(jnp.cos(jnp.pi * x))], axis=-1)
        assert _alias_by_interpolating(op, n) < _TOL
