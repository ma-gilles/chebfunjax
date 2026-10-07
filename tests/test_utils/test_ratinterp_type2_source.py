"""Independent analytic controls, unrun; source assertions are not replaced.

Provenance
----------
MATLAB source : ratinterp.m/constructRatApproxCheb2 and ratbary
Chebfun commit: 7574c77
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils._ratbary import _source_type2_rational_handle
from chebfunjax.utils.ratapprox import _construct_rat_approx, ratinterp


def test_type2_linear_over_quadratic_analytic_shapes_and_nodes():
    # p(t)=1+t; q(t)=3+t+t^2 = 3.5+T1+.5*T2.
    r = _source_type2_rational_handle(jnp.array([1., 1.]), jnp.array([3.5, 1., .5]), (2., 6.))
    for x in (jnp.array(3.2), jnp.array([2., 4., 6., 3.2]),
              jnp.array([[2., 3.2], [4., 6.]]), jnp.array([3.2+.2j, 5.-.3j])):
        v = (x-4.)/2.
        expected = (1+v)/(3+v+v*v)
        actual = r(x)
        assert actual.shape == x.shape
        np.testing.assert_allclose(actual, expected, rtol=0, atol=100*np.finfo(float).eps)


def test_type2_singleton_numerator_literal_node_exception():
    # Source weights .5 then *.5 then (-2)^-1 / 0. At t=0 the
    # numerator node branch bypasses the infinite weight. Literal source
    # node products give -1, while the idealized ratio would be +1.
    r = _source_type2_rational_handle(jnp.array([2.]), jnp.array([2., 1.]), (-1., 1.))
    values = np.asarray(r(jnp.array([-.5, 0., .5])))
    assert np.isposinf(values.real[[0, 2]]).all()
    assert (values.imag == 0).all()
    assert values[1] == -1


def test_type2_complex_coefficients():
    a = jnp.array([1.+.25j, .2-.1j])
    b = jnp.array([3.+.5j, -.3+.2j])
    x = jnp.array([-.75+.1j, .3-.2j, 1.])
    r = _source_type2_rational_handle(a, b, (-1., 1.))
    expected = (a[0]+a[1]*x)/(b[0]+b[1]*x)
    np.testing.assert_allclose(r(x), expected, rtol=0, atol=100*np.finfo(float).eps)


def test_source_pole_retention_is_separate_from_type2_handle_contract():
    pole = .2+1e-12j
    _, _, _, mu, nu, poles, _ = ratinterp(lambda x: 1/(x-pole), 0, 1, NN=8, xi="type2")
    assert (mu, nu) == (0, 1)
    actual = np.asarray(poles).reshape(-1)[0]
    assert abs(actual.real-pole.real) <= 100*np.finfo(float).eps
    assert abs(actual.imag-pole.imag) <= 100*np.finfo(float).eps


def test_type1_constant_numerator_contract_retained():
    # TYPE1 dispatch deliberately retains its existing evaluator path.
    r = _construct_rat_approx("TYPE1", None, jnp.array([2.]), jnp.array([2., 1.]), 0, 1, -1., 1.)
    x = jnp.array([-.5, 0., .5])
    np.testing.assert_allclose(r(x), 2/(2+x), rtol=0, atol=100*np.finfo(float).eps)


def test_real_dtype_scalar_float_and_jit():
    r = _source_type2_rational_handle(jnp.array([1., 1.]), jnp.array([3.5, 1., .5]), (-1., 1.))
    result = r(jnp.array(.25))
    assert not jnp.iscomplexobj(result)
    assert abs(float(result) - 1.25/3.3125) < 100*np.finfo(float).eps
    compiled = jax.jit(r)(jnp.array(.25))
    assert not jnp.iscomplexobj(compiled)
    np.testing.assert_allclose(compiled, result, rtol=0, atol=100*np.finfo(float).eps)


def test_nan_query_raises_source_multiple_hit_error():
    r = _source_type2_rational_handle(jnp.array([1., 1.]), jnp.array([2., 1.]), (-1., 1.))
    with pytest.raises(ValueError, match="multiple indices"):
        r(jnp.array(float("nan")))


def test_nan_query_singleton_numerator_still_errors_in_denominator():
    r = _source_type2_rational_handle(jnp.array([2.]), jnp.array([2., 1.]), (-1., 1.))
    with pytest.raises(ValueError, match="denominator.*multiple indices"):
        r(jnp.array(float("nan")))


def test_jit_vmap_query_grad_at_regular_and_continuous_nodes():
    r = _source_type2_rational_handle(jnp.array([1., 1.]), jnp.array([3.5, 1., .5]), (-1., 1.))
    x = jnp.array([-1., -.3, 0., .4, 1.])
    expected = ((3+x+x*x)-(1+x)*(1+2*x))/(3+x+x*x)**2
    actual = jax.jit(jax.vmap(jax.grad(r)))(x)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=200*np.finfo(float).eps)


def test_complex_query_jvp_regular_and_shared_node():
    r = _source_type2_rational_handle(jnp.array([1.+.2j, .3]), jnp.array([2.+.1j, -.2]), (-1., 1.))
    x = jnp.array([.2+.1j, -1.+0j])
    direction = jnp.array([.3+.2j, .1-.2j])
    _, tangent = jax.jvp(r, (x,), (direction,))
    expected = (.3*(2+.1j-.2*x)+.2*(1+.2j+.3*x))/(2+.1j-.2*x)**2 * direction
    np.testing.assert_allclose(tangent, expected, rtol=0, atol=200*np.finfo(float).eps)


def test_source_nonshared_node_discontinuity_is_not_idealized():
    # mu2/nu1, p=1+x², q=2+x. At x=0 source gives -1/2,
    # while the analytic rational limit is +1/2.
    r = _source_type2_rational_handle(jnp.array([1.5, 0., .5]), jnp.array([2., 1.]), (-1., 1.))
    assert float(r(jnp.array(0.))) == -.5
    assert jnp.isnan(jax.grad(r)(jnp.array(0.)))


def test_traced_nan_query_is_explicitly_unqualified_nan():
    r = _source_type2_rational_handle(jnp.array([1., 1.]), jnp.array([2., 1.]), (-1., 1.))
    assert jnp.isnan(jax.jit(r)(jnp.array(float("nan"))))


@pytest.mark.parametrize("unsupported", [float("nan"), 1e200])
def test_traced_mixed_batch_preserves_valid_query(unsupported):
    r = _source_type2_rational_handle(jnp.array([1., 1.]), jnp.array([3.5, 1., .5]), (-1., 1.))
    # Large finite query overflows node products; NaN gives multiple hits.
    # Neither unsupported row may erase the ordinary finite row.
    values = jax.jit(r)(jnp.array([.25, unsupported]))
    assert not jnp.iscomplexobj(values)
    assert abs(float(values[0])-1.25/3.3125) < 100*np.finfo(float).eps
    assert jnp.isnan(values[1])


def test_python_float_jvp_and_grad_match_array_inputs():
    r = _source_type2_rational_handle(jnp.array([1., 1.]), jnp.array([3.5, 1., .5]), (-1., 1.))
    value, tangent = jax.jvp(r, (.25,), (1.,))
    avalue, atangent = jax.jvp(r, (jnp.array(.25),), (jnp.array(1.),))
    np.testing.assert_allclose(value, avalue, rtol=0, atol=100*np.finfo(float).eps)
    np.testing.assert_allclose(tangent, atangent, rtol=0, atol=100*np.finfo(float).eps)
    np.testing.assert_allclose(jax.grad(r)(.25), atangent, rtol=0, atol=100*np.finfo(float).eps)


def test_public_type2_nonconstant_numerator_real_jvp():
    r, _, _, mu, nu, _, _ = ratinterp(lambda x: (1+x)/(2+x), 1, 1, NN=8, xi="type2")
    assert (mu, nu) == (1, 1)
    point = .25
    np.testing.assert_allclose(r(point), (1+point)/(2+point), rtol=0, atol=100*np.finfo(float).eps)
    _, tangent = jax.jvp(r, (point,), (1.,))
    np.testing.assert_allclose(tangent, 1/(2+point)**2, rtol=0, atol=200*np.finfo(float).eps)
