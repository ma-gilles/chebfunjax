"""Saved licensed native matrices; independent CPU binary64 regression oracles."""

from pathlib import Path

import jax
import numpy as np
import pytest

from chebfunjax.utils.diffmat import diffmat

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu" or not jax.config.x64_enabled,
    reason="Native arithmetic qualification is CPU binary64",
)
FIXTURE = Path(__file__).parents[1] / "references/pde_cpu_arithmetic_native.npz"


@pytest.mark.parametrize(
    "n,p", [(2, 1), (3, 2), (7, 3), (17, 1), (32, 2), (33, 1), (33, 2), (65, 1), (65, 2), (129, 5)]
)
def test_actual_native_matrix(n, p):
    with np.load(FIXTURE) as data:
        expected = data[f"D_{n}_{p}"]
    actual = np.asarray(diffmat(n, p=p))
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("n,p", [(33, 1), (33, 2), (65, 1), (65, 2)])
def test_actual_native_matrix_under_jit(n, p):
    with np.load(FIXTURE) as data:
        expected = data[f"D_{n}_{p}"]
    actual = np.asarray(jax.jit(lambda: diffmat(n, p=p))())
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_adapter_scope_keeps_generic_paths(monkeypatch):
    import importlib

    module = importlib.import_module("chebfunjax.utils.diffmat")

    def forbidden(*args):
        raise AssertionError("Bounded CPU sine adapter reached an excluded path")

    monkeypatch.setattr(module, "_sin_cheb_angles", forbidden)
    assert np.isfinite(np.asarray(diffmat(515, p=1))).all()
    assert np.isfinite(np.asarray(diffmat(17, p=1, kind=1))).all()
    # C2 node generation already requires binary64; preserve that rejection.
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="Binary64 software division requires JAX x64"):
            diffmat(17, p=1)
