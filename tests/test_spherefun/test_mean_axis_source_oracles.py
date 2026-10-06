# uses-numpy: Independent reference fixtures and numeric assertions in tests.
"""Pinned real-valued source fixtures for Spherefun.mean(dim).

MATLAB Chebfun 7574c77680d7e82b79626300bf255498271a72df, observed with
MATLAB R2025b in spherefun_mean_oracle_v3. The deliberately counterintuitive
zero for sin(theta)*cos(lambda) records source @spherefun/sum.m behavior
rather than an idealized surface integral. Draft only; no JAX numerical
execution was performed.
"""

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from chebfunjax.spherefun.spherefun import Spherefun


def _assert_source_close(actual, expected):
    """Absolute-only source bound for these low-degree real controls."""
    reference = np.asarray(expected)
    scale = max(1.0, float(np.max(np.abs(reference), initial=0.0)))
    npt.assert_allclose(
        np.asarray(actual), reference, rtol=0.0, atol=100.0 * np.finfo(np.float64).eps * scale
    )


def test_mean_real_source_fixture_values_and_orientation():
    constant = Spherefun.from_function(lambda lam, th: jnp.full_like(lam, 7.0))
    x = Spherefun.from_function(lambda lam, th: jnp.sin(th) * jnp.cos(lam))
    z = Spherefun.from_function(lambda lam, th: jnp.cos(th))
    lam = jnp.array([-jnp.pi, -0.7, 0.0, 1.3, jnp.pi])
    th = jnp.array([0.0, 0.21, jnp.pi / 2, 2.8, jnp.pi])

    _assert_source_close(constant.mean()(lam), 14.0 / jnp.pi)
    _assert_source_close(constant.mean(dim=2)(th), 7.0)
    _assert_source_close(x.mean(dim=1)(lam), 0.0)
    _assert_source_close(x.mean(dim=2)(th), 0.0)
    _assert_source_close(z.mean(dim=1)(lam), 0.0)
    _assert_source_close(z.mean(dim=2)(th), jnp.cos(th))
    assert constant.mean().is_transposed
    assert x.mean().is_transposed
    assert not constant.mean(dim=2).is_transposed


def test_mean_empty_and_invalid_dimension_source_order():
    assert Spherefun.empty().mean(dim=7).isempty()
    f = Spherefun.from_function(lambda lam, th: 1.0 + jnp.cos(th))
    try:
        f.mean(dim=7)
    except ValueError as exc:
        assert "CHEBFUN:SPHEREFUN:sum:unknown" in str(exc)
    else:
        raise AssertionError("nonempty invalid dim must raise")
