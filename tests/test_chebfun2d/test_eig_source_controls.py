"""Source continuous eig controls; @chebfun2/eig.m, Chebfun7574c77."""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun2d.chebfun2 import Chebfun2


def rank_one(domain=(-1., 1.)):
    c = chebfun(lambda y: 1+2*y, domain=domain)
    r = chebfun(lambda x: 3+x, domain=domain)
    return Chebfun2.from_cdr([c], jnp.ones(1), [r], domain=(*domain, *domain))


def test_nonsymmetric_orientation_and_continuous_normalization():
    f = rank_one()
    functions, diagonal = f.eig(full=True)
    np.testing.assert_allclose(diagonal, [[22/3]], rtol=0, atol=2e-13)
    assert abs(float(functions[0].norm())-1) < 1e-13
    # Native eigenfunction is proportional to 1+2*y, not 3+x.
    x = jnp.asarray([-.9, -.3, .2, .8])
    np.testing.assert_allclose(functions[0](x)/functions[0](0.), 1+2*x,
                               rtol=0, atol=2e-13)
    assert float((f @ functions-functions @ diagonal).norm()) < 2e-12


def test_shifted_square_domain():
    f = rank_one((2., 4.))
    functions, diagonal = f.eig(full=True)
    # Integral of (1+2*t)*(3+t) = 2*t^3/3 + 7*t^2/2 + 3*t.
    def primitive(t):
        return 2*t**3/3 + 7*t*t/2 + 3*t
    np.testing.assert_allclose(diagonal, [[primitive(4)-primitive(2)]],
                               rtol=0, atol=2e-12)
    assert abs(float(functions[0].norm())-1) < 1e-13


def test_legacy_triple_samples_continuous_eigenfunction():
    f = rank_one()
    values, sampled, grid = f.eig(return_functions=True)
    assert sampled.shape == (grid.size, 1)
    np.testing.assert_allclose(values, [22/3], rtol=0, atol=2e-13)
    expected = 1+2*grid
    # Compare shape independently of provider phase.
    np.testing.assert_allclose(sampled[:, 0]/sampled[-1, 0],
                               expected/expected[-1], rtol=0, atol=2e-13)


def test_nonsquare_exact_identifier():
    f = Chebfun2.from_function(lambda x, y: x+y, domain=(-1, 1, -2, 2))
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN2:eig:domainerr:'):
        f.eig()


def test_literal_complex_source_core_uses_conjugate_inner_product():
    # Source SVD returns literal Qright*V; native eig then uses V.H*U*S.
    # For these rank-one factors this is integral(conj(r)*c)=10/3.
    # This source convention is distinct from the bilinear kernel residual.
    c = chebfun(lambda y: 1+1j*y)
    r = chebfun(lambda x: 1+2j*x)
    f = Chebfun2.from_cdr([c], jnp.ones(1), [r])
    np.testing.assert_allclose(f.eig(), [10/3], rtol=0, atol=2e-13)


def test_empty_outputs_follow_source_empty_matrix_actions():
    f = Chebfun2.empty()
    assert f.eig().shape == (0,)
    functions, diagonal = f.eig(full=True)
    assert functions.shape == diagonal.shape == (0, 0)
    values, sampled, grid = f.eig(return_functions=True)
    assert values.shape == (0,) and sampled.shape == (grid.size, 0)


def test_zero_values_only():
    c = chebfun(lambda y: 1+y)
    r = chebfun(lambda x: 2-x)
    f = Chebfun2.from_cdr([c], jnp.zeros(1), [r])
    np.testing.assert_array_equal(f.eig(), [0.])


def test_constant_panel_inner_products_are_matrices():
    # Native mtimes returns one scalar per column pair, including columns
    # stored as (n, 1) tech panels by the full zero SVD branch.
    from chebfunjax.chebfun1d.linalg import Quasimatrix
    from chebfunjax.chebfun2d._svd import source_svd
    f = rank_one()
    cols, _, rows = f.cdr()
    zero = Chebfun2.from_cdr(cols, jnp.zeros(1), rows)
    left, _, right = source_svd(zero.approx, full=True)
    u = Quasimatrix(left, left[0].domain)
    v = Quasimatrix(right, right[0].domain)
    product = v.H @ u
    assert product.shape == (1, 1)
    np.testing.assert_allclose(product, [[1.]], atol=3e-15, rtol=0)


def test_zero_full_output_preserves_native_normalization_error():
    # eig.m normalizes the zero column; numeric rdivide rejects its zero norm.
    f = rank_one()
    cols, _, rows = f.cdr()
    zero = Chebfun2.from_cdr(cols, jnp.zeros(1), rows)
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:rdivide:columnRdivide:divisionByZero:'):
        zero.eig(full=True)
