# uses-numpy: independent QR reconstruction and physical-measure assertions.
"""Pure leaf staging controls; source trigtech/qr.m and bndfun/qr.m7574c77."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.spherefun._plus import real_trig_matrix_qr
from chebfunjax.tech.trigtech import _trig_coeffs2vals_impl, _trig_vals2coeffs_impl


@pytest.mark.parametrize("disabled", [True, False])
@pytest.mark.parametrize("size", [9, 10])
def test_nonconstant_qr_reconstruction_and_measure(disabled, size):
    t = -1 + 2 * jnp.arange(size) / size
    values = jnp.stack(
        [
            1 + 0.25 * jnp.cos(jnp.pi * t) + 0.125 * jnp.cos((size // 2) * jnp.pi * t),
            jnp.sin(2 * jnp.pi * t),
        ],
        axis=1,
    )
    coefficients = _trig_vals2coeffs_impl(values)
    with jax.disable_jit(disabled):
        qc, r = real_trig_matrix_qr(coefficients)
        rawq, rawr = real_trig_matrix_qr.__wrapped__(coefficients)
        qv = jnp.real(_trig_coeffs2vals_impl(qc))
    np.testing.assert_allclose(qc @ r, coefficients, rtol=0, atol=2e-14)
    np.testing.assert_allclose(
        np.asarray(qv).T @ np.asarray(qv) * (2 * np.pi / size), np.eye(2), rtol=0, atol=2e-14
    )
    np.testing.assert_allclose(qc, rawq, rtol=0, atol=2e-14)
    np.testing.assert_allclose(r, rawr, rtol=0, atol=2e-14)
