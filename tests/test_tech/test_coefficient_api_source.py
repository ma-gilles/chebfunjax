"""Source coefficient API adapters with independent polynomial identities.

Provenance: @chebtech/{chebcoeffs,legcoeffs,jaccoeffs}.m, Chebfun7574c77.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_cheb_kind_and_count(Tech):
    # 2*T2 = U2-U0; the leading U coefficient needs the input's third row.
    f = Tech.from_coeffs(jnp.array([0., 0., 2.]))
    np.testing.assert_array_equal(f.chebcoeffs(1, 2), [-1.])
    np.testing.assert_array_equal(f.chebcoeffs(2), [0., 0.])
    np.testing.assert_array_equal(f.chebcoeffs(5, 2), [-1., 0., 1., 0., 0.])
    np.testing.assert_array_equal(f.chebcoeffs([], []), f.coeffs)
    assert f.chebcoeffs(0, 2).shape == (0,)
    with pytest.raises(ValueError, match="chebcoeffs:badKind"):
        f.chebcoeffs(kind=3)


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_legendre_convert_before_truncating_and_matrix(Tech):
    # T2 = (4*P2-P0)/3, including complex columns.
    f = Tech.from_coeffs(jnp.array([[0., 0j], [0., 0j], [1., 2j]]))
    np.testing.assert_allclose(f.legcoeffs(1), [[-1/3, -2j/3]], atol=2e-15, rtol=0)
    expected = np.array([[-1/3, -2j/3], [0, 0], [4/3, 8j/3], [0, 0], [0, 0]])
    np.testing.assert_allclose(f.legcoeffs(5), expected, atol=2e-15, rtol=0)
    assert f.legcoeffs(0).shape == (0, 2)


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_jacobi_overloads_and_asymmetry(Tech):
    # P1^(1,2) = (-1+5*x)/2, avoiding a symmetric/Legendre-only check.
    f = Tech.from_coeffs(jnp.array([[-0.5, -1j], [2.5, 5j]]))
    expected = np.array([[0, 0], [1, 2j]])
    np.testing.assert_allclose(f.jaccoeffs(1., 2.), expected, atol=2e-15, rtol=0)
    np.testing.assert_allclose(f.jaccoeffs(1, 1., 2.), expected[:1], atol=2e-15, rtol=0)
    np.testing.assert_allclose(f.jaccoeffs(4, 1., 2.), np.vstack([expected, np.zeros((2,2))]), atol=2e-15, rtol=0)
    with pytest.raises(TypeError):
        f.jaccoeffs(1.)


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_coefficient_api_jit_empty_and_lengths(Tech):
    coefficients = jnp.array([0., 0., 1.])
    f = Tech.from_coeffs(coefficients)
    compiled = jax.jit(lambda c: Tech.from_coeffs(c).legcoeffs(4))
    np.testing.assert_allclose(compiled(coefficients), [-1/3,0,4/3,0], atol=2e-15, rtol=0)
    for n in (-1, 1.5):
        with pytest.raises(ValueError):
            f.legcoeffs(n)
    for empty in (Tech.empty(), Tech.from_coeffs(jnp.array([]))):
        assert empty.chebcoeffs().size == 0
        assert empty.legcoeffs().size == 0
        assert empty.jaccoeffs(0.,0.).size == 0
