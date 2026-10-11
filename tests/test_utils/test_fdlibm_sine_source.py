"""Bounded fdlibm source arithmetic and Chebtech2 node contracts.

Provenance
----------
Netlib fdlibm s_sin.c, k_sin.c, k_cos.c, e_rem_pio2.c; original Sun notice
retained in the implementation. Chebfun commit: 7574c77 (@chebtech2/chebpts.m).
The fixture contains unmodified C reference outputs, never production tables.
The linked qualification record binds source files, compilation flags and fixture.
"""
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils._fdlibm_sine import _sin_cheb_angles
from chebfunjax.utils.quadrature import chebpts

FIXTURE = Path(__file__).parent / "fixtures/fdlibm_sine_source.npz"


@pytest.mark.parametrize("n", [17, 33, 65, 129, 65537])
def test_chebtech2_nodes_match_bound_c_source(n):
    with np.load(FIXTURE) as data:
        expected = data[f"{n}_fdlibm"]
    actual = np.asarray(chebpts(n, kind=2))
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_independent_angles_match_bound_c_source():
    with np.load(FIXTURE) as data:
        actual = np.asarray(_sin_cheb_angles(jnp.asarray(data["generic_angles"])))
        expected = data["generic_fdlibm"]
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("n", [14, 27, 48, 53, 84, 95, 100, 105, 801])
def test_rounded_endpoint_angles_remain_in_source_reduction_branch(n):
    nodes = np.asarray(chebpts(n, kind=2))
    assert np.all(np.isfinite(nodes))
    np.testing.assert_array_equal(nodes[[0, -1]], [-1., 1.])
    np.testing.assert_array_equal(nodes, -nodes[::-1])


def test_signed_zero_tiny_and_unsupported_range():
    values = np.asarray([0., -0., np.nextafter(0., 1.), -np.nextafter(0., 1.),
                         np.finfo(float).tiny, -np.finfo(float).tiny])
    actual = np.asarray(_sin_cheb_angles(jnp.asarray(values)))
    np.testing.assert_array_equal(actual.view(np.uint64), values.view(np.uint64))
    invalid = _sin_cheb_angles(jnp.asarray([3., -3., np.inf, -np.inf, np.nan]))
    assert bool(jnp.all(jnp.isnan(invalid)))
