"""Preserve actual sampled subdivision outputs while reusing recurrence states.

Provenance
----------
MATLAB source: @chebtech/roots.m, @chebtech/clenshaw.m; pin7574c77.
Fixtures supply actual retained inputs. Expected outputs are recomputed with
the untouched original provider on the same backend, retaining strict bitwise
identity without assuming captured compiler/hardware output portability.
"""
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech import _roots_sampled_cpu as helper
from chebfunjax.tech import chebtech as module


@pytest.mark.parametrize("n", [528, 539, 2091])
def test_retained_sampled_results(n):
    path = Path(__file__).parents[1] / "fixtures/roots_sampled_cpu_actual_inputs.npz"
    with np.load(path) as data:
        coeffs = data[f"n{n}_input"]
        expected = module._roots_sampled_subdivision(jnp.asarray(coeffs))
        actual = module._roots_subdivide(coeffs)
        for child, original in zip(actual, expected):
            np.testing.assert_array_equal(
                np.asarray(child).view(np.uint64),
                np.asarray(original).view(np.uint64))


@pytest.mark.parametrize("kind", ["complex", "float32", "device", "ndct"])
def test_ineligible_sampled_storage_keeps_generic_path(monkeypatch, kind):
    c = np.zeros(539)
    if kind == "complex":
        c = c.astype(complex)
    elif kind == "float32":
        c = c.astype(np.float32)
    elif kind == "device":
        c = jnp.asarray(c)
    else:
        c = np.zeros(4001)

    def forbidden(*args):
        raise AssertionError("ineligible storage reached recurrence adapter")

    monkeypatch.setattr(helper, "grid_and_states", forbidden)
    monkeypatch.setattr(module, "_roots_sampled_subdivision", lambda c: "generic")
    assert module._roots_subdivide(c) == "generic"


def test_nonfinite_sampled_storage_keeps_generic_path(monkeypatch):
    c = np.zeros(539)
    c[4] = np.nan
    monkeypatch.setattr(module, "_roots_sampled_subdivision", lambda c: "generic")
    assert module._roots_subdivide(c) == "generic"


@pytest.mark.parametrize("n", [528, 539, 2091])
def test_original_active_grid_raw_bits(n):
    capacity = 1 << (n - 1).bit_length()
    divisor = 2 * (n - 1)
    k = divisor.bit_length() - 1
    normalized = divisor << (52 - k)
    actual = helper.dynamic_points(jnp.asarray(n, jnp.int64),
                                   jnp.asarray(k, jnp.int64),
                                   jnp.asarray(normalized, jnp.uint64), capacity)
    expected = _original_grid(n)
    np.testing.assert_array_equal(np.asarray(actual[:2*n]).view(np.uint64),
                                  np.asarray(expected).view(np.uint64))


@partial(jax.jit, static_argnames=("n",))
def _original_grid(n):
    return jnp.concatenate((
        module._roots_mapped_points(n, -1., -0.004849834917525),
        module._roots_mapped_points(n, -0.004849834917525, 1.)))
