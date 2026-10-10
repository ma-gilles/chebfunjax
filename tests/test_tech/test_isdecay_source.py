"""Native endpoint-decay controls, @chebtech/isdecay.m and @singfun/isdecay.m7574c77."""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_endpoint_multiplicity_and_cross_column_mask(Tech):
    f = Tech.from_coeffs(jnp.array([[1.5, 1.0 + 1e-10, 1.5],
                                  [2.0, 1.0, -2.0], [0.5, 0.0, 0.5]]))
    # Left double root, residual below relaxed threshold, right double root.
    np.testing.assert_array_equal(f.isdecay(), [[True, True, False],
                                              [False, False, True]])
    simple = Tech.from_coeffs(jnp.array([1., 1.]))
    np.testing.assert_array_equal(simple.isdecay(), [[False], [False]])


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_native_constant_comparison_including_complex(Tech):
    f = Tech.from_coeffs(jnp.array([[0, -1, 1, 2j, -1+3j, 1+3j]], dtype=jnp.complex128))
    # Literal native constant branch: comparison uses only the real part.
    expected = [[True, True, False, True, True, False]] * 2
    np.testing.assert_array_equal(f.isdecay(), expected)


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_empty_column_shapes(Tech):
    assert Tech.empty().isdecay().shape == (2, 0)
    assert Tech.from_coeffs(jnp.array([])).isdecay().shape == (2, 0)
    np.testing.assert_array_equal(Tech.from_coeffs(jnp.empty((0, 3))).isdecay(),
                                  np.zeros((2, 3), dtype=bool))


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_singfun_exponent_and_smooth_factor_routes(Tech):
    f = Singfun(Tech.from_coeffs(jnp.array([1.])), (1.0, 1.01))
    np.testing.assert_array_equal(f.isdecay(), [[False], [True]])
    g = Singfun(Tech.from_coeffs(jnp.array([1.5, 2., 0.5])), (0.2, 0.3))
    np.testing.assert_array_equal(g.isdecay(), [[True], [False]])
