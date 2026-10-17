"""Source C1 rectangular corrections, typed scalar columns and recovery.

Chebfun7574c77 @chebcolloc/{diff,feval,reduce}, @chebcolloc1/toFunctionOut,
@valsDiscretization/{instantiate,mldivide} and @linop/linsolve.
"""
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.operators import _parameter_first_kind as module
from chebfunjax.operators.chebop import Chebop
from chebfunjax.tech.chebtech import Chebtech1
from chebfunjax.utils.diffmat import diffmat


def correction(size=4, complex_column=False, domain=(-1., 1.), order=2):
    x = cj.chebfun(lambda t: t, domain=domain)
    zero = cj.chebfun(0., domain=domain)
    u = ADChebfun(zero).seed(1, (True, False))
    a = ADChebfun(zero).seed(2, (True, False))
    coefficient = x.exp()*(1+2j if complex_column else 1.)
    out = (u.diff(2) if order == 2 else 0.)+2*u+coefficient*a
    bc = [(u+a)(domain[0]), u.diff()(domain[0]), u(domain[-1])]
    bc = bc if order == 2 else bc[:1]
    disc = module.FirstKindParameterCorrection(out, bc, size, order, domain)
    return disc, coefficient, bc


def bound(matrix):
    # Normwise floating arithmetic budget: dimension times machine epsilon
    # and matrix infinity norm; no source mathematical tolerance is changed.
    return 32*jnp.finfo(jnp.float64).eps*max(matrix.shape)*jnp.maximum(1, jnp.max(jnp.sum(jnp.abs(matrix), axis=-1)))


@pytest.mark.parametrize('complex_column', [False, True])
def test_scalar_column_uses_own_equation_grid(complex_column):
    disc, coefficient, _ = correction(complex_column=complex_column)
    assert disc.matrix.shape == (7, 7)
    assert disc.function_disc.n == 6 and disc.equation_disc.n == 4
    assert jnp.array_equal(disc.matrix[3:, -1], coefficient(disc.points))
    # These are different interpolation operations on this nonpolynomial.
    assert not jnp.array_equal(disc.matrix[3:, -1],
                               disc.projection@coefficient(disc.function_disc.points))
    assert jnp.iscomplexobj(disc.matrix) is complex_column


@pytest.mark.parametrize('domain', [(-1., 1.), (2., 5.)])
def test_first_kind_diff_and_endpoint_rows(domain):
    disc, _, _ = correction(domain=domain)
    D = diffmat(6, 1, domain=domain, kind=1)
    D2 = diffmat(6, 2, domain=domain, kind=1)
    left = disc.function_disc.evaluation(domain[0], 1)
    right = disc.function_disc.evaluation(domain[-1], -1)
    expected = jnp.concatenate([
        jnp.stack([left, left@D, right]),
        disc.projection@(D2+2*jnp.eye(6)),
    ])
    assert float(jnp.max(jnp.abs(disc.matrix[:, :6]-expected))) <= float(bound(expected))
    assert jnp.array_equal(disc.matrix[:3, -1], jnp.array([1., 0., 0.]))


@pytest.mark.parametrize('order', [0, 2])
def test_source_factor_eligibility_and_backward_residual(monkeypatch, order):
    original = module.lu_factor
    calls = []
    def factor(matrix):
        calls.append(matrix.shape)
        return original(matrix)
    monkeypatch.setattr(module, 'lu_factor', factor)
    disc, _, conditions = correction(order=order)
    width = 5+order
    for value in [1., 2.]:
        vector = disc.solve(value, conditions)
        rhs = jnp.concatenate([jnp.zeros(1+order), jnp.full(4, value)])
        residual = disc.matrix@vector+rhs
        budget = 32*jnp.finfo(jnp.float64).eps*width*(
            jnp.max(jnp.sum(jnp.abs(disc.matrix), axis=1))*jnp.max(jnp.abs(vector))
            +jnp.max(jnp.abs(rhs)))
        assert float(jnp.max(jnp.abs(residual))) <= float(budget)
    assert calls == [(width, width)]*(2 if order else 1)
    assert disc.is_factored is (order == 0)


@pytest.mark.parametrize('complex_values', [False, True])
def test_projection_recovery_retains_c1_and_numeric_parameter(complex_values):
    disc, _, _ = correction(size=8)
    x = disc.function_disc.points
    multiplier = 1+2j if complex_values else 1.
    vector = jnp.concatenate([multiplier*(1+x+x**3), jnp.array([3.*multiplier])])
    values, parameters = disc.project(vector)
    function = module.to_function(values, disc.domain, 4)
    assert isinstance(function.funs[0].tech, Chebtech1)
    assert parameters.shape == (1,)
    assert jnp.array_equal(parameters, vector[-1:])
    probes = jnp.linspace(-1, 1, 13)
    error = jnp.max(jnp.abs(function(probes)-multiplier*(1+probes+probes**3)))
    assert float(error) <= float(32*jnp.finfo(jnp.float64).eps*10*3*jnp.abs(multiplier))


def test_manufactured_complex_numeric_parameter_uses_c1(monkeypatch):
    x = cj.chebfun(lambda t: t)
    exact = (1+1j)*(x+2)
    op = Chebop(lambda x, u, a: u.diff(2)+u**2-((1+1j)*(x+2))**2+a-1j)
    op.lbc = lambda u, a: u-(1+1j)
    op.rbc = lambda u, a: [u-3*(1+1j), u.diff()-(1+1j)]
    op.init = [exact+.1*(1-x**2)**2, .25+.5j]
    def forbidden(*args, **kwargs):
        pytest.fail('Explicit C1 used legacy/default continuation')
    monkeypatch.setattr(Chebop, '_solve_impl', forbidden)
    u, a = op.solve(0., n=8, discretization='chebcolloc1')
    assert isinstance(a, complex)
    assert abs(a-1j) < 1e-10
    assert float((u-exact).norm()) < 1e-10
    assert float(op.op(x, u, a).norm()) < 1e-10
    assert all(d['backend'] == 'chebcolloc1' for d in op._last_info['linearDiscretizations'])


@pytest.mark.parametrize('domain', [(-1., 1.), (2., 5.)])
@pytest.mark.parametrize('complex_column', [False, True])
def test_matrix_action_on_physical_polynomial(domain, complex_column):
    disc, _, _ = correction(domain=domain, complex_column=complex_column)
    multiplier = 1+2j if complex_column else 1.
    parameter = .5*multiplier
    x = disc.function_disc.points
    def polynomial(t):
        return 1+t+t**3
    vector = jnp.concatenate([polynomial(x), jnp.asarray([parameter])])
    left, right = domain
    points = disc.points
    expected = jnp.concatenate([
        jnp.asarray([polynomial(left)+parameter,
                     1+3*left**2, polynomial(right)]),
        6*points+2*polynomial(points)+multiplier*jnp.exp(points)*parameter,
    ])
    actual = disc.matrix@vector
    # Independent analytic polynomial/derivative values and native n-grid
    # scalar term. Standard normwise matrix-action floating error budget.
    budget = 32*jnp.finfo(jnp.float64).eps*disc.matrix.shape[0]*(
        jnp.max(jnp.sum(jnp.abs(disc.matrix), axis=1))*jnp.max(jnp.abs(vector))
        +jnp.max(jnp.abs(expected)))
    assert float(jnp.max(jnp.abs(actual-expected))) <= float(budget)
