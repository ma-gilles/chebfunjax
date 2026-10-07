"""Independent complex warning/trivial source controls; Chebfun7574c77 cf.m."""
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax import cf
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.domain import Domain


def test_nonzero_imaginary_warns_then_real_polynomial():
    f=Chebfun.from_coeffs(jnp.array([1+2j,2-3j,3+4j]),Domain((-1.,1.)))
    with pytest.warns(RuntimeWarning,match='CHEBFUN:CHEBFUN:cf:complex'):
        p,q,r,error=cf(f,1)
    np.testing.assert_array_equal(p.funs[0].coeffs,[1.,2.])
    assert error==3.


def test_complex_dtype_real_values_has_no_complex_warning():
    f=Chebfun.from_coeffs(jnp.array([1.,2.,3.],dtype=jnp.complex128),Domain((-1.,1.)))
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        p,q,r,error=cf(f,1)
    assert not any('CHEBFUN:CHEBFUN:cf:complex' in str(x.message) for x in records)
    np.testing.assert_array_equal(p.funs[0].coeffs,[1.,2.])


def test_trivial_return_precedes_complex_coefficient_warning():
    f=Chebfun.from_coeffs(jnp.array([1+2j,3-4j]),Domain((2.,6.)))
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        p,q,r,error=cf(f,1)
    assert p is f and error==0.
    assert not any('CHEBFUN:CHEBFUN:cf:complex' in str(x.message) for x in records)


@pytest.mark.parametrize("order", [[], np.empty((0, 2))])
def test_empty_denominator_order_matches_source_polynomial(order):
    # @chebfun/cf.m: isempty(n) || n == 0.
    f = Chebfun.from_coeffs(jnp.array([1., 2., 3.]), Domain((-1., 1.)))
    p, q, r, error = cf(f, 1, order)
    np.testing.assert_array_equal(p.funs[0].coeffs, [1., 2.])
    np.testing.assert_array_equal(q.funs[0].coeffs, [1.])
    assert error == 3.
    np.testing.assert_array_equal(r(jnp.array([-1., 0., 1.])), [-1., 1., 3.])


def test_scalar_handle_extrapolation_and_stored_endpoints():
    # Source cf trivial r=@(x)feval(p,x); feval extrapolates the polynomial.
    f = Chebfun.from_coeffs(jnp.array([1., 2.]), Domain((-1., 1.)))
    f = f.set_point_values(jnp.array([7., 9.]))
    p, q, r, error = cf(f, 1)
    np.testing.assert_array_equal(r(jnp.array([-2., -1., 0., 1., 2.])), [-3., 7., 1., 9., 5.])


def test_piecewise_evaluator_uses_default_breakpoint_value():
    from chebfunjax.chebfun1d.chebfun import _Piece
    from chebfunjax.utils._cf_public import _evaluate
    f = Chebfun(funs=[_Piece.from_coeffs(jnp.array([1.]), -1., 0.),
                     _Piece.from_coeffs(jnp.array([3.]), 0., 1.)],
                domain=Domain((-1., 0., 1.)))
    np.testing.assert_array_equal(_evaluate(f, jnp.array([-2., 0., 2.])), [1., 2., 3.])
