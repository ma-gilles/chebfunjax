"""Complex Tucker coefficients: @chebfun3/chebcoeffs3.m, Chebfun7574c77."""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_complex_tucker_factors_keep_coefficients(tech):
    # Explicit polynomial coefficients provide an independent tensor oracle.
    a = jnp.asarray([2.+3.j, .25-.5j])
    b = jnp.asarray([1.-2.j, .75+1.j, -.125j])
    c = jnp.asarray([3.+.25j])
    f = Chebfun3(cols=[tech.from_coeffs(a)], rows=[tech.from_coeffs(b)],
                 tubes=[tech.from_coeffs(c)], core=jnp.ones((1, 1, 1)),
                 domain=(-1., 1., -1., 1., -1., 1.))
    expected = a[:, None, None] * b[None, :, None] * c[None, None, :]
    np.testing.assert_allclose(f.chebcoeffs3(), expected, rtol=0, atol=0)
    core, aa, bb, cc = f.chebcoeffs3(low_rank=True)
    for got, want in ((aa[:, 0], a), (bb[:, 0], b), (cc[:, 0], c)):
        np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(core, jnp.ones((1, 1, 1)))


def test_complex_numeric_coefficient_roundtrip():
    coeffs = jnp.asarray([[[2.+3.j, .25-.5j]], [[1.-.125j, -.25+1.j]]])
    got = chebfun3(coeffs, coeffs=True).chebcoeffs3()
    # Original constructor2 uses norm(error) < 10 * (100 * chebfun3eps).
    assert got.shape == coeffs.shape
    assert jnp.iscomplexobj(got)
    assert float(jnp.linalg.norm((got-coeffs).ravel())) < 1000*jnp.finfo(jnp.float64).eps
