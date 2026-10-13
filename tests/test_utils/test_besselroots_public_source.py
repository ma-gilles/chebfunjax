"""Public besselroots wrapper preserves the source JAX kernel contract."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.bessel_roots_general import _bessel_roots_general
from chebfunjax.utils.specfun import besselroots


@pytest.mark.parametrize("nu", [-0.9, 0.0, 2.5, 6.0, 10.0])
def test_public_wrapper_matches_source_kernel(nu):
    np.testing.assert_array_equal(
        np.asarray(besselroots(nu, 18)), np.asarray(_bessel_roots_general(nu, 18))
    )


@pytest.mark.parametrize("nu", [0.0, 2.5, 6.0])
def test_public_wrapper_jits_with_static_order_and_count(nu):
    compiled = jax.jit(lambda: besselroots(nu, 12))
    np.testing.assert_array_equal(np.asarray(compiled()), np.asarray(besselroots(nu, 12)))


def test_zero_count_returns_before_order_or_count_validation():
    assert besselroots(object(), 0).shape == (0,)
    assert besselroots(0.0, 0.0).shape == (0,)
    assert besselroots(object(), [[0, 0], [0, 0]]).shape == (0,)


@pytest.mark.parametrize("n", [1.5, -1, float("nan"), float("inf"), [0, 1], []])
def test_invalid_count_uses_native_error_identifier_and_message(n):
    with pytest.raises(
        ValueError,
        match=r"CHEBFUN:besselroots:inputN: Input N must be a positive integer",
    ):
        besselroots(0.0, n)


def test_one_element_array_is_a_scalar_count():
    for n in ([1], jnp.array([1.0])):
        np.testing.assert_array_equal(np.asarray(besselroots(0.0, n)),
                                      np.asarray(besselroots(0.0, 1)))


def test_integral_float_count_is_accepted():
    np.testing.assert_array_equal(np.asarray(besselroots(2.5, 6.0)),
                                  np.asarray(_bessel_roots_general(2.5, 6)))
