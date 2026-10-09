"""Exact scalar equality branches of @chebfun/power.m, Chebfun7574c77.

Complex values exactly0/1/2 are equality cases, not general complex power.
No norm tolerance substitutes for identity, coefficients or pointValues.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun


@pytest.mark.parametrize("columns", [1, 2])
@pytest.mark.parametrize("row", [False, True])
def test_complex_scalar_equality_dispatch(columns, row):
    coefficients = jnp.array([2., .25]) if columns == 1 else jnp.array([[2., 3.], [.25, .5]])
    f = Chebfun.from_coeffs(coefficients, domain=(-2., 3.))
    values = jnp.array([7.+1j, 8.+2j]) if columns == 1 else jnp.array([[7.+1j, 9.+3j], [8.+2j, 10.+4j]])
    f = Chebfun._as_transposed(f.set_point_values(values), row)
    square = f * f
    for kind in [complex, lambda x: jnp.asarray(x, dtype=jnp.complex128),
                 lambda x: jnp.asarray([x], dtype=jnp.complex128)]:
        zero = f ** kind(0)
        # Native columnPower constructs chebfun(ones(...), ends), and
        # quasi2cheb returns a singleton unchanged: row.^0 is a column.
        assert zero.n_columns == columns and not zero.is_transposed
        assert tuple(zero.domain.breakpoints) == (-2., 3.)
        assert jnp.array_equal(zero._breakpoint_values(), jnp.ones_like(zero._breakpoint_values()))
        assert f ** kind(1) is f
        actual = f ** kind(2)
        assert actual.is_transposed == row
        assert jnp.array_equal(actual._breakpoint_values(), square._breakpoint_values())
        for a, b in zip(actual.funs, square.funs):
            assert type(a.tech) is type(b.tech)
            assert jnp.array_equal(a.tech.coeffs, b.tech.coeffs)


def test_chebfun_and_ad_exponents_dispatch_before_numeric_equality():
    base = chebfun(2.)
    exponent = chebfun(.5)
    grid = jnp.array([-.75, .25, .75])
    target = jnp.sqrt(2.)
    bound = 32*jnp.finfo(jnp.float64).eps
    assert jnp.max(jnp.abs((base ** exponent)(grid)-target)) < bound
    result = base ** ADChebfun(exponent)
    assert isinstance(result, ADChebfun)
    assert jnp.max(jnp.abs(result.func(grid)-target)) < bound
    assert jnp.max(jnp.abs(result.jacobian.apply(chebfun(1.))(grid)-jnp.log(2.)*target)) < bound


@pytest.mark.parametrize("powers, operation", [([0, 1, 2], "horzcat"), ([1, 0, 2], "vertcat")])
def test_row_mixed_power_vector_native_orientation_error(powers, operation):
    f = chebfun("x").T
    with pytest.raises(ValueError, match=f"CHEBFUN:CHEBFUN:{operation}:transpose"):
        _ = f ** jnp.asarray(powers)


def test_row_all_zero_and_nonzero_power_vector_orientations():
    f = chebfun("x").T
    zeros = f ** jnp.asarray([0, 0])
    assert not zeros.is_transposed and zeros.n_columns == 2
    assert jnp.array_equal(zeros(jnp.array([-.5, .5])), jnp.ones((2, 2)))
    nonzeros = f ** jnp.asarray([1, 2])
    assert nonzeros.is_transposed
    assert jnp.array_equal(nonzeros(jnp.array([-.5, .5])), jnp.array([[-.5, .5], [.25, .25]]))
