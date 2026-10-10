"""Bitwise collation controls for source horzcat (Chebfun7574c77).

Compare the retained literal unblocked implementation; no tolerance change.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d._svd import _axis_panel, _axis_panel_unblocked
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize("count", [31, 32, 33, 64, 65])
@pytest.mark.parametrize(
    "kind",
    [
        "trig_real_odd",
        "trig_complex_odd",
        "trig_real_even",
        "trig_complex_even",
        "cheb_real",
        "cheb_complex",
    ],
)
def test_group_boundary_exact(count, kind):
    factors = []
    for k in range(count):
        n = (7 if "odd" in kind else 8) + 2 * (k % 2)
        x = jnp.arange(n, dtype=jnp.float64)
        vals = (x + 1) * (k + 1) / 128
        if "complex" in kind:
            vals = vals.astype(jnp.complex128) + 1j * (x + 2) / 64
        if kind.startswith("trig"):
            tech = Trigtech.from_values(vals)
            # Preserve authoritative cache rather than recomputing from coeffs.
            tech = Trigtech(
                coeffs=tech.coeffs,
                real_columns=tech.real_columns,
                _values=tech.values + 0.03125,
                ishappy=k != 0,
            )
        else:
            cls = Chebtech1 if k % 2 else Chebtech2
            tech = cls(coeffs=vals, ishappy=k != 0)
        factors.append(tech)
    actual = _axis_panel(tuple(factors))
    expected = _axis_panel_unblocked(tuple(factors))
    assert type(actual) is type(expected)
    assert actual.ishappy == expected.ishappy
    for a, b in [(actual.coeffs, expected.coeffs)]:
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype
        assert np.array_equal(a.view(np.uint8), b.view(np.uint8))
    if kind.startswith("trig"):
        assert actual.real_columns == expected.real_columns
        assert actual.is_real == expected.is_real
        a, b = np.asarray(actual.values), np.asarray(expected.values)
        assert a.shape == b.shape and a.dtype == b.dtype
        assert np.array_equal(a.view(np.uint8), b.view(np.uint8))
