"""Port of MATLAB Chebfun tests/trigtech/test_iszero.m (Opus 4.8[1m]).

MATLAB ``@trigtech/iszero.m`` reports which columns are identically zero via
a value-space reduction::

    out = ~any(f.values, 1) & ~any(isnan(f.values), 1);

The MATLAB test drives this by assigning ``f.values`` to arbitrary matrices
(including NaN entries).  chebfunjax trigtechs are immutable and FFT-based,
but each column transforms independently, so a value matrix built through
``from_values`` reproduces every column's zero/NaN structure exactly: a NaN
in one column does not pollute the others.

Provenance
----------
MATLAB source : tests/trigtech/test_iszero.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.tech.trigtech import Trigtech


class TestTrigtechIszero:
    def test_mixed_columns(self):
        # values [0 1 0 ; 0 0 NaN]: iszero -> [1 0 0].
        vals = jnp.array([[0.0, 1.0, 0.0],
                          [0.0, 0.0, jnp.nan]], dtype=jnp.float64)
        f = Trigtech.from_values(vals)
        assert jnp.array_equal(f.iszero(), jnp.array([True, False, False]))

    def test_row_vector_values(self):
        # values [0 NaN 1] (one point, three columns): iszero -> [1 0 0].
        vals = jnp.array([[0.0, jnp.nan, 1.0]], dtype=jnp.float64)
        f = Trigtech.from_values(vals)
        assert jnp.array_equal(f.iszero(), jnp.array([True, False, False]))

    def test_col_vector_values(self):
        # values [0 NaN 1]' (three points, one column): iszero -> 0.
        vals = jnp.array([0.0, jnp.nan, 1.0], dtype=jnp.float64)
        f = Trigtech.from_values(vals)
        assert bool(f.iszero()) is False

    def test_all_zero(self):
        f = Trigtech.from_values(jnp.zeros(3, dtype=jnp.float64))
        assert bool(f.iszero()) is True

    def test_nan(self):
        f = Trigtech.from_values(jnp.array([jnp.nan], dtype=jnp.float64))
        assert bool(f.iszero()) is False
