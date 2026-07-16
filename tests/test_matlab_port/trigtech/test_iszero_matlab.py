"""Port of MATLAB Chebfun tests/trigtech/test_iszero.m (Fable 5).

MATLAB ``@trigtech/iszero.m`` reports which columns are identically zero via a
value-space reduction::

    out = ~any(f.values, 1);
    out = out & ~any(isnan(f.values), 1);   % a NaN column is not zero

The MATLAB test drives this by *assigning* ``f.values`` to arbitrary matrices
(including NaN entries).  chebfunjax trigtechs are immutable and FFT-based --
there is no way to inject an arbitrary (or NaN-bearing) value matrix, since
``from_values`` transforms to coefficients and a NaN pollutes every
coefficient.  Only the all-zeros case round-trips exactly, so it is ported;
the NaN-injection cases stay xfail with that precise reason.

Provenance
----------
MATLAB source : tests/trigtech/test_iszero.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.tech.trigtech import Trigtech

_INJECT_REASON = (
    "chebfunjax Trigtech is immutable and FFT-based; cannot inject an arbitrary/NaN "
    "value matrix (MATLAB sets f.values directly), so iszero's NaN-column cases are "
    "not reproducible"
)


def _iszero(f):
    """MATLAB @trigtech/iszero.m equivalent: per-column ``all zero, no NaN``."""
    v = f.values
    zero = ~jnp.any(v != 0, axis=0)
    return zero & ~jnp.any(jnp.isnan(v), axis=0)


class TestTrigtechIszero:
    @pytest.mark.xfail(reason=_INJECT_REASON)
    def test_mixed_columns(self):
        raise AssertionError("cannot inject values [0 1 0; 0 0 NaN]")

    @pytest.mark.xfail(reason=_INJECT_REASON)
    def test_row_vector_values(self):
        raise AssertionError("cannot inject values [0 NaN 1]")

    @pytest.mark.xfail(reason=_INJECT_REASON)
    def test_col_vector_values(self):
        raise AssertionError("cannot inject values [0 NaN 1]'")

    def test_all_zero(self):
        # pass(4): iszero(f) == 1 for values zeros(3, 1)
        # FIXED (Fable 5, Big-Three array-valued epic): zeros round-trip exactly.
        f = Trigtech.from_values(jnp.zeros(3, dtype=jnp.float64))
        assert int(_iszero(f)) == 1

    @pytest.mark.xfail(reason=_INJECT_REASON)
    def test_nan(self):
        raise AssertionError("cannot inject values NaN")
