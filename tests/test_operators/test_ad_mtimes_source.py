"""Source @adchebfun/adchebfun.m mtimes/scalar times, pinned 7574c77."""
import jax.numpy as jnp
import pytest

from chebfunjax import chebfun
from chebfunjax.autodiff.adchebfun import ADChebfun


@pytest.mark.parametrize("scalar", [0, -2.5, 1+2j])
def test_scalar_primal_jacobian_both_orders(scalar):
    f = chebfun(lambda x: 1+x+x*x)
    h = chebfun(lambda x: 2-x)
    u = ADChebfun(f).seed(1, (True,))
    x = jnp.linspace(-1, 1, 17)
    for result in (u @ scalar, scalar @ u, u.mtimes(scalar)):
        assert float(jnp.max(jnp.abs(result.func(x)-scalar*f(x)))) < 2e-13
        action = result.jacobian.apply(h)
        assert float(jnp.max(jnp.abs(action(x)-scalar*h(x)))) < 2e-13
        assert result.linearity == u.linearity
        assert result.domain == u.domain
        assert result.jacobian._coeff_fn is not None
        coefficients = result.jacobian.coeff_list()
        assert len(coefficients) == 1
        assert float(jnp.max(jnp.abs(coefficients[0](x)-scalar))) < 2e-13
        assert result.jacobian._coordinate_fn is not None
        coordinates = result.jacobian._coordinate_fn(8)
        assert float(jnp.max(jnp.abs(coordinates-scalar*jnp.eye(8)))) < 2e-13
    assert float(jnp.max(jnp.abs(u.func(x)-f(x)))) == 0

@pytest.mark.parametrize("zero", [False, True])
def test_genuine_functional_operand_rejection(zero):
    f = chebfun(0 if zero else lambda x: 1+x)
    u = ADChebfun(f).seed(1, (True,))
    for other in (u, u.diff(), f):
        with pytest.raises(ValueError, match="CHEBFUN:ADCHEBFUN:mtimes:dims"):
            u @ other
        with pytest.raises(ValueError, match="CHEBFUN:ADCHEBFUN:mtimes:dims"):
            other @ u

def test_array_expansion_explicitly_unsupported():
    u = ADChebfun(chebfun(1)).seed(1, (True,))
    with pytest.raises(NotImplementedError, match="array mtimes expansion"):
        u.mtimes(jnp.array([1., 2.]))


def test_translated_nonvectorized_source_6():
    from chebfunjax.operators.chebop import Chebop

    n = Chebop(domain=(0., 1.))
    n.op = lambda u: u.diff(2) - u @ u.diff()
    n.bc = lambda x, u: [u(0)-2, u(1)-3]
    with pytest.raises(ValueError, match="CHEBFUN:ADCHEBFUN:mtimes:dims"):
        n.solve(0.)


@pytest.mark.parametrize("scalar", [0, -2, -2.5, 1+2j])
@pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
def test_numeric_size_one_actual_operand_orders(scalar, shape):
    f = chebfun(lambda x: 1+x+x*x)
    h = chebfun(lambda x: 2-x)
    u = ADChebfun(f).seed(1, (True,))
    a = jnp.asarray(scalar).reshape(shape)
    x = jnp.linspace(-1, 1, 17)
    for result in (u @ a, a @ u):
        assert float(jnp.max(jnp.abs(result.func(x)-scalar*f(x)))) < 2e-13
        action = result.jacobian.apply(h)
        assert float(jnp.max(jnp.abs(action(x)-scalar*h(x)))) < 2e-13
        assert result.linearity == u.linearity
        assert result.domain == u.domain
        assert result.jacobian._coeff_fn is not None
        coefficients = result.jacobian.coeff_list()
        assert len(coefficients) == 1
        assert float(jnp.max(jnp.abs(coefficients[0](x)-scalar))) < 2e-13
        assert result.jacobian._coordinate_fn is not None
        coordinates = result.jacobian._coordinate_fn(8)
        assert float(jnp.max(jnp.abs(coordinates-scalar*jnp.eye(8)))) < 2e-13


@pytest.mark.parametrize("scalar", [-2., 1+2j])
def test_functional_scalar_coordinate_preservation(scalar):
    f = chebfun(lambda x: 1+x+x*x)
    h = chebfun(lambda x: 2-x)
    u = ADChebfun(f).seed(1, (True,))(0.)
    for a in (scalar, jnp.asarray(scalar), jnp.asarray([scalar])):
        for result in (u @ a, a @ u):
            assert float(jnp.abs(result.func-scalar)) < 2e-13
            assert float(jnp.abs(result.jacobian.apply(h)-2*scalar)) < 2e-13
            assert result.linearity == u.linearity
            assert result.jacobian._coordinate_fn is not None
            row = result.jacobian._coordinate_fn(8)
            expected = scalar*jnp.cos(jnp.arange(8)*jnp.pi/2)
            assert float(jnp.max(jnp.abs(row-expected))) < 2e-13


@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize("scalar", [0., -2., 1+2j])
def test_native_values_capability_preservation(functional, scalar):
    from chebfunjax.operators._native_values import Capability, FirstKindDisc

    f = chebfun(lambda x: 1+x+x*x)
    u = ADChebfun(f).seed(1, (True,))
    if functional:
        u = u(0.)
    disc = FirstKindDisc((8,), (-1., 1.))
    h = 2-disc.points
    for a in (scalar, jnp.asarray(scalar), jnp.asarray([[scalar]])):
        for result in (u @ a, a @ u):
            cap = result.jacobian._values_capability
            assert isinstance(cap, Capability)
            assert cap.domain == (-1., 1.)
            action = cap.realize(disc) @ h
            expected = 2*scalar if functional else scalar*h
            assert float(jnp.max(jnp.abs(action-expected))) < 2e-13
