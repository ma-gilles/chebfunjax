"""Independent coefficient-space and typed-scalar ultraS controls.

Source: Chebfun7574c77 @coeffsDiscretization/instantiate,
@ultraS/{quasi2diffmat,rhs,toValues,reduce,toFunctionOut}.
"""
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.operators import _linear_altdisc
from chebfunjax.operators import _parameter_ultras as module
from chebfunjax.operators._linear_altdisc import _dimension_values
from chebfunjax.operators.chebop import Chebop
from chebfunjax.tech.chebtech import Chebtech2


def correction(size=4, order=2, domain=(-1., 1.), multiplier=1., coefficient=None):
    x = cj.chebfun(lambda t: t, domain=domain)
    zero = cj.chebfun(0., domain=domain)
    u = ADChebfun(zero).seed(1, (True, False))
    a = ADChebfun(zero).seed(2, (True, False))
    coefficient = multiplier*(2-x+x**2) if coefficient is None else coefficient
    out = (u.diff(2) if order == 2 else 0.)+2*u+coefficient*a
    bc = [(u+a)(domain[0]), u.diff()(domain[0]), u(domain[-1])]
    bc = bc if order == 2 else bc[:1]
    return module.UltrasParameterCorrection(out, bc, size, order, domain), bc


@pytest.mark.parametrize('domain', [(-1., 1.), (2., 5.)])
@pytest.mark.parametrize('multiplier', [1., 1+2j])
def test_independent_physical_polynomial_coefficient_action(domain, multiplier):
    disc, _ = correction(domain=domain, multiplier=multiplier)
    center, radius = sum(domain)/2, (domain[1]-domain[0])/2
    A, B = center, radius
    # Exact affine expansion of physical u=1+x+x^3 in Chebyshev T.
    coefficients = jnp.array([1+A+A**3+1.5*A*B**2,
                              B+3*A**2*B+.75*B**3,
                              1.5*A*B**2, .25*B**3, 0., 0.])
    parameter = .5*multiplier
    vector = jnp.concatenate([coefficients, jnp.asarray([parameter])])
    # Monomial coefficients in t of u''+2u+multiplier*(2-x+x^2)*a.
    g0 = 6*A+2*(1+A+A**3)+multiplier*(2-A+A**2)*parameter
    g1 = 6*B+2*(B+3*A**2*B)+multiplier*(-B+2*A*B)*parameter
    g2 = 6*A*B**2+multiplier*B**2*parameter
    g3 = 2*B**3
    # C2_0=1,C2_1=4t,C2_2=12t^2-2,C2_3=32t^3-12t.
    c3, c2 = g3/32, g2/12
    c1, c0 = (g1+12*c3)/4, g0+2*c2
    left, right = domain
    expected = jnp.asarray([1+left+left**3+parameter, 1+3*left**2,
                            1+right+right**3, c0, c1, c2, c3])
    actual = disc.matrix@vector
    budget = 32*jnp.finfo(jnp.float64).eps*7*(
        jnp.max(jnp.sum(jnp.abs(disc.matrix), axis=1))*jnp.max(jnp.abs(vector))
        +jnp.max(jnp.abs(expected)))
    assert float(jnp.max(jnp.abs(actual-expected))) <= float(budget)
    assert disc.output_space == 1  # Inclusive upper index -> actual C^(2).
    assert disc.matrix.shape == (7, 7)


def test_scalar_and_rhs_truncate_before_conversion(monkeypatch):
    # T4 is discarded by n=4 scalar/RHS extraction; converting at n+2 first
    # would move its content into lower C^(2) modes and give a different result.
    high_mode = cj.Chebfun.from_coeffs(jnp.array([0., 0., 0., 0., 1.]))
    disc, conditions = correction(coefficient=high_mode)
    assert jnp.array_equal(disc.matrix[3:, -1], jnp.zeros(4))
    captured = []
    original = module.jnp.linalg.solve
    def solve(matrix, rhs):
        captured.append(rhs)
        return original(matrix, rhs)
    monkeypatch.setattr(module.jnp.linalg, 'solve', solve)
    disc.solve(high_mode, conditions)
    assert jnp.array_equal(captured[0], jnp.zeros(7))


@pytest.mark.parametrize('order', [0, 2])
def test_plain_unscaled_solve_every_rhs_and_backward_residual(monkeypatch, order):
    disc, conditions = correction(order=order)
    original = module.jnp.linalg.solve
    calls = []
    def solve(matrix, rhs):
        assert jnp.array_equal(matrix, disc.matrix)
        calls.append((matrix, rhs))
        return original(matrix, rhs)
    monkeypatch.setattr(module.jnp.linalg, 'solve', solve)
    for residual in [1., 2.]:
        vector = disc.solve(residual, conditions)
        matrix, rhs = calls[-1]
        budget = 32*jnp.finfo(jnp.float64).eps*matrix.shape[0]*(
            jnp.max(jnp.sum(jnp.abs(matrix), axis=1))*jnp.max(jnp.abs(vector))
            +jnp.max(jnp.abs(rhs)))
        assert float(jnp.max(jnp.abs(matrix@vector+rhs))) <= float(budget)
    assert len(calls) == 2
    assert disc.is_factored is False


@pytest.mark.parametrize('complex_values', [False, True])
def test_projection_and_direct_coefficient_recovery(complex_values):
    disc, _ = correction(size=8)
    multiplier = 1+2j if complex_values else 1.
    vector = multiplier*jnp.arange(11.)
    coefficients, parameters = disc.project(vector)
    assert jnp.array_equal(coefficients, vector[:8])
    assert jnp.array_equal(parameters, vector[10:])
    function = module.to_function(coefficients, disc.domain, 4)
    assert isinstance(function.funs[0].tech, Chebtech2)
    assert jnp.array_equal(function.funs[0].tech.coeffs, coefficients[:4])


def test_native_coefficient_dimension_schedule():
    assert _dimension_values(300, 2048, 'ultraS') == (300, 600, 1200)
    assert _dimension_values(32, 4096, 'ultraS') == (32, 64, 128, 256, 512, 1024, 2048, 4096)


def test_manufactured_complex_numeric_parameter_uses_ultras(monkeypatch):
    x = cj.chebfun(lambda t: t)
    exact = (1+1j)*(x+2)
    op = Chebop(lambda x, u, a: u.diff(2)+u**2-((1+1j)*(x+2))**2+a-1j)
    op.lbc = lambda u, a: u-(1+1j)
    op.rbc = lambda u, a: [u-3*(1+1j), u.diff()-(1+1j)]
    op.init = [exact+.1*(1-x**2)**2, .25+.5j]
    def forbidden(*args, **kwargs):
        pytest.fail('Explicit ultraS used a legacy/default solve seed')
    monkeypatch.setattr(Chebop, '_solve_impl', forbidden)
    schedule_calls = []
    original_schedule = _linear_altdisc._dimension_values
    def schedule(minimum, maximum, backend):
        schedule_calls.append((minimum, maximum, backend))
        return original_schedule(minimum, maximum, backend)
    monkeypatch.setattr(_linear_altdisc, '_dimension_values', schedule)
    u, a = op.solve(0., n=8, discretization='ultraS')
    assert schedule_calls == [(32, 4096, 'ultraS')]
    assert isinstance(a, complex)
    assert abs(a-1j) < 1e-10
    assert float((u-exact).norm()) < 1e-10
    assert float(op.op(x, u, a).norm()) < 1e-10
    assert all(d['backend'] == 'ultraS' for d in op._last_info['linearDiscretizations'])


def test_nonlocal_field_rejects_unavailable_coefficient_realization():
    zero = cj.chebfun(0.)
    x = cj.chebfun(lambda t: t)
    u = ADChebfun(zero).seed(1, (True, False))
    a = ADChebfun(zero).seed(2, (True, False))
    differential = u.compose(x/2)+a
    with pytest.raises(TypeError, match='COEFFSDISCRETIZATION:instantiate:fail'):
        module.UltrasParameterCorrection(differential, [(u+a)(-1.)], 4, 0, (-1., 1.))
