"""Independent analytic controls for adaptive scalar collocation output.

These manufactured solutions do not use MATLAB solution arrays, grid sizes,
cutoffs, or the production assembly as expected answers.

Provenance
----------
MATLAB source : @linop/linsolve.m, @opDiscretization/getDimAdjust.m,
    @chebcolloc/reduce.m, @chebcolloc2/toFunctionOut.m
Chebfun commit: 7574c77
"""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

import chebfunjax as cj
from chebfunjax.operators.blocks import D, I, eval_at
from chebfunjax.operators.chebop import _derivative_eval_at
from chebfunjax.operators.linop import Linop


def test_adaptive_first_order_manufactured_solution():
    domain = (-0.5, 1.5)
    a, b = domain
    def exact(x):
        return (1+x)*jnp.exp(x)

    def forcing(x):
        return (3+2*x)*jnp.exp(x)

    operator = D(domain, order=1) + I(domain)
    problem = Linop(operator, [eval_at(a, domain=domain)], domain,
                    [float(exact(a))])
    rhs = cj.chebfun(forcing, domain=domain)
    u = problem.solve(rhs)
    query = jnp.asarray([-0.413, -0.017, 0.237, 1.113, 1.421])
    npt.assert_allclose(u(query), exact(query), rtol=0, atol=1e-10)
    assert (u.diff()+u-rhs).norm() < 1e-10
    assert abs(u(a)-exact(a)) < 1e-10


def test_adaptive_third_order_polynomial_and_derivative_conditions():
    domain = (-2.0, 0.75)
    a, b = domain
    def exact(x):
        return x**3 + 2*x - 1

    problem = Linop(
        D(domain, order=3),
        [eval_at(a, domain=domain),
         _derivative_eval_at(b, domain=domain, order=1),
         _derivative_eval_at(a, domain=domain, order=2)],
        domain, [float(exact(a)), 3*b*b+2, 6*a])
    u = problem.solve(6.0)
    query = jnp.asarray([-1.913, -1.167, -0.319, 0.073, 0.681])
    npt.assert_allclose(u(query), exact(query), rtol=0, atol=1e-10)
    assert (u.diff(3)-6).norm() < 1e-10
    assert abs(u(a)-exact(a)) < 1e-10
    assert abs(u.diff()(b)-(3*b*b+2)) < 1e-10
    assert abs(u.diff(2)(a)-6*a) < 1e-10


@pytest.mark.parametrize("disable_jit", [False, True])
@pytest.mark.parametrize("amplitude", [1+2j, 2j, 1+0j])
def test_adaptive_complex_solution_survives_output_conversion(disable_jit, amplitude):
    domain = (-0.75, 1.25)
    a, b = domain
    reaction = 0.3+0.2j if amplitude == 1+2j else 0.3
    def exact(x):
        return amplitude*(x-a)*(b-x)

    def forcing(x):
        return -2*amplitude + reaction*exact(x)

    problem = Linop(D(domain, order=2)+reaction*I(domain),
                    [eval_at(a, domain=domain), eval_at(b, domain=domain)], domain,
                    [0.0, 0.0])
    with jax.disable_jit(disable_jit):
        rhs = cj.chebfun(forcing, domain=domain)
        u = problem.solve(rhs)
        query = jnp.asarray([-0.633, -0.219, 0.071, 0.677, 1.109])
        assert jnp.iscomplexobj(u.funs[0].coeffs)
        npt.assert_allclose(u(query), exact(query), rtol=0, atol=1e-10)
        assert (u.diff(2)+reaction*u-rhs).norm() < 1e-10
        assert abs(u(a)) < 1e-10
        assert abs(u(b)) < 1e-10


def test_adaptive_zero_order_complex_rhs_without_constraints():
    domain = (-1.5, 0.5)
    def exact(x):
        return jnp.exp(x)+1j*jnp.cos(2*x)

    rhs = cj.chebfun(exact, domain=domain)
    u = Linop(I(domain), domain=domain).solve(rhs)
    query = jnp.asarray([-1.421, -0.719, -0.113, 0.277, 0.431])
    assert jnp.iscomplexobj(u.funs[0].coeffs)
    npt.assert_allclose(u(query), exact(query), rtol=0, atol=1e-10)
    assert (u-rhs).norm() < 1e-10


def test_adaptive_zero_solution_without_constraints():
    u = Linop(I(), domain=(-1.0, 1.0)).solve(0.0)
    assert u.norm() == 0
    npt.assert_array_equal(u(jnp.asarray([-0.93, 0.13, 0.79])), 0.0)


@pytest.mark.parametrize("value", [2j, 1+2j])
def test_adaptive_singleton_preserves_complex_constant(value):
    # Source's trivial transform is the constant coefficient itself.
    # A one-point cap cannot satisfy cutoff < dimension; retain its warning.
    with pytest.warns(UserWarning, match="without convergence"):
        u = Linop(I(), domain=(-1.0, 1.0)).solve(
            value, n_min=1, n_max=1)
    assert jnp.iscomplexobj(u.funs[0].coeffs)
    npt.assert_array_equal(u(jnp.asarray([-0.93, 0.13, 0.79])), value)
