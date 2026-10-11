"""Supplemental branch controls; source cautions exclude interior crossings/zeros."""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_one_sided_endpoint_principal_sqrt(Tech, sign):
    # f=-1+sign*i*(1+x)*(1+T32/4), nonzero; imag has constant sign interior.
    # Endpoint lies on negative real axis, source explicitly supports it via extrapolation.
    c = np.zeros(34, dtype=complex)
    c[0] = -1 + sign * 1j
    c[1] = sign * 1j
    c[31] = sign * 0.125j
    c[32] = sign * 0.25j
    c[33] = sign * 0.125j
    f = Tech.from_coeffs(jnp.asarray(c))
    h = f**0.5
    x = np.concatenate([[-1.0], -1.0 + 2.0 ** (-np.arange(1, 41)), np.linspace(-0.9, 1.0, 101)])
    q = 1 + 0.25 * np.polynomial.chebyshev.chebval(x, np.eye(1, 33, 32).ravel())
    y = (1 + x) * q
    r = np.sqrt(1 + y * y)
    expected = y / np.sqrt(2 * (r + 1)) + sign * 1j * np.sqrt((r + 1) / 2)
    actual = np.asarray(h(jnp.asarray(x)))
    err = np.max(abs(actual - expected))
    bound = 100 * np.finfo(float).eps * np.max(abs(expected))
    print("BRANCH_DIAGNOSTIC", Tech.__name__, sign, float(err), float(bound), h.coeffs.shape)
    assert np.all(np.isfinite(actual)) and np.all(sign * actual.imag > 0)
    assert err < bound  # Supplemental diagnostic bound; no native POWER test threshold exists.
