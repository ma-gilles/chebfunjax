"""Independent retained MATLAB Hankel controls, CPU binary64 qualification.

Provenance: MATLAB R2025b Update 2, native Chebfun pin 7574c77.
Fixture provenance records the exact capture and original input arrays.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.besselh import besselh

FIXTURE = Path(__file__).parents[1] / "references/besselh_native_cpu_20261011.npz"


@pytest.mark.parametrize("row", range(11))
@pytest.mark.parametrize("kind", [1, 2])
def test_retained_native_integer_near_integer_and_complex(row, kind):
    with np.load(FIXTURE) as data:
        order = float(data["orders"][row])
        z = data["z"][row].reshape(1, -1)
        expected = data[f"h{kind}"][row].reshape(1, -1)
    actual = np.asarray(besselh(order, jnp.asarray(z), kind))
    assert actual.shape == z.shape
    # Relative-only checks include the exponentially small recessive values.
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0)


@pytest.mark.parametrize("kind", [1, 2])
def test_retained_native_source_inputs(kind):
    with np.load(FIXTURE) as data:
        z = data["source_z"]
        expected = data[f"source_h{kind}"]
    actual = np.asarray(besselh(1.663, jnp.asarray(z), kind))
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0)


@pytest.mark.parametrize("order", [float("nan"), float("inf"), 1 + 1j])
def test_rejects_nonreal_or_nonfinite_order(order):
    with pytest.raises(ValueError, match="order must be finite and real"):
        besselh(order, jnp.asarray(1.0), 1)


@pytest.mark.parametrize("kind", [0, 3])
def test_rejects_invalid_kind(kind):
    with pytest.raises(ValueError, match="k must be 1 or 2"):
        besselh(0.0, jnp.asarray(1.0), kind)
