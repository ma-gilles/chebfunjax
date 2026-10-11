"""MPOLES singleton exit and explicit Python no-progress error adapter.

Provenance
----------
Algorithm reference: licensed MATLAB R2025b mpoles.m.
Chebfun caller: @chebfun/residue.m, commit 7574c77.
No public nonfinite-residue numerical parity is asserted.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.utils._residue_source import _mpoles


@pytest.mark.parametrize("value", [5e-324, jnp.nan, jnp.inf, 0.0])
def test_unconditional_singleton(value):
    order, powers = _mpoles(jnp.array([value]), False)
    assert jnp.array_equal(order, jnp.array([0])) and powers == [1]


@pytest.mark.parametrize("values", [[2.2250738585072014e-308, 1.0], [jnp.nan, 1.0], [jnp.inf, 1.0]])
def test_multi_no_progress_error(values):
    with pytest.raises(ValueError, match="made no progress"):
        _mpoles(jnp.array(values), False)


def test_normal_relative_grouping():
    order, powers = _mpoles(jnp.array([100.0, 100.05, 1.0]), True)
    assert jnp.array_equal(order, jnp.array([1, 0, 2])) and powers == [1, 2, 1]


def test_normal_nonadjacent_grouping():
    order, powers = _mpoles(jnp.array([1.0, 2.0, 1.0]), False)
    assert jnp.array_equal(order, jnp.array([0, 2, 1])) and powers == [1, 2, 1]
