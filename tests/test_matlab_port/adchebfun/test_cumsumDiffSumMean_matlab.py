"""All 27 clauses of MATLAB test_cumsumDiffSumMean.m.

Provenance: Chebfun commit 7574c77680d7e82b79626300bf255498271a72df,
@adchebfun/{valueTesting,taylorTesting}.m and test_cumsumDiffSumMean.m.
Nine operations each check exact primal equality, source Taylor bounds,
and linearity. Fixed degree-seven polynomials replace seeded rand(8,1);
this does not claim MATLAB RNG parity. Norms are function infinity norms
or MATLAB numeric matrix infinity norms (row-vector absolute row sum).
"""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
from chebfunjax.operators.blocks import ChebColloc2Disc

OPERATIONS = ('diff', 'diff2', 'diff4', 'sum', 'sum_subdomain',
              'cumsum', 'cumsum2', 'mean', 'deriv2')


def _input_pair():
    # Both functions have nonzero derivatives through order four. Degree7
    # matches the source's eight-point polynomial space, without RNG claims.
    u = chebfun(lambda x: .55+.012*x+.008*x**2+.003*x**4+.001*x**7)
    p = chebfun(lambda x: .055+.001*x+.0008*x**2+.0003*x**4+.0001*x**7)
    return u, p


def _operation(name, value):
    if name.startswith('diff'):
        return value.diff(int(name[4:] or '1'))
    if name == 'sum_subdomain':
        if isinstance(value, ADChebfun):
            return value.sum(-.25, .8)
        # Source @chebfun/sum uses these cumsum endpoint evaluations. The
        # current plain Chebfun.sum API has no subdomain arguments.
        integral = value.cumsum()
        return integral(.8)-integral(-.25)
    if name == 'deriv2':
        return value.deriv(jnp.arange(11, dtype=jnp.float64)*.01, 2)
    if name == 'cumsum2':
        return value.cumsum(2)
    return getattr(value, name)()


def _source_norm(value):
    if isinstance(value, Chebfun):
        return float(value.norm(jnp.inf))
    value = jnp.asarray(value)
    if value.ndim == 0:
        return float(jnp.abs(value))
    # Source deriv locations 0:0.01:0.1 are a MATLAB row vector.
    return float(jnp.linalg.norm(jnp.atleast_2d(value), ord=jnp.inf))


@pytest.mark.parametrize('operation', OPERATIONS)
def test_source_primal_clause(operation):
    u, _ = _input_pair()
    plain = _operation(operation, u)
    ad = _operation(operation, ADChebfun(u))
    assert _source_norm(plain-ad.func) == 0


@pytest.mark.parametrize('operation', OPERATIONS)
def test_source_taylor_clause(operation):
    u, p = _input_pair()
    ad = _operation(operation, ADChebfun(u))
    first, second = [], []
    for exponent in range(2, 6):
        perturbation = .2**exponent*p
        changed = _operation(operation, ADChebfun(u+perturbation))
        delta = changed.func-ad.func
        action = ad.jacobian.apply(perturbation)
        first.append(_source_norm(delta))
        second.append(_source_norm(delta-action))
    slopes = jnp.diff(jnp.log(jnp.asarray(first)))/jnp.log(.2)
    assert float(jnp.max(jnp.abs(slopes-1))) < 1e-2
    assert max(second) < 1e-12


@pytest.mark.parametrize('operation', OPERATIONS)
def test_source_linearity_clause(operation):
    u, _ = _input_pair()
    assert _operation(operation, ADChebfun(u)).is_linear


@pytest.mark.parametrize('domain', [(-1., 1.), (2., 5.), (-1., .2, 1.)])
@pytest.mark.parametrize('operation', ['sum', 'mean'])
def test_integral_functional_matrix_and_jax_action(domain, operation):
    u = chebfun(lambda x: 1+x*x, domain=domain)
    result = getattr(ADChebfun(u), operation)()
    disc = ChebColloc2Disc(12, domain)
    row = result.jacobian.matrix(disc)
    action = jax.jit(lambda v: row@v)
    actual, tangent = jax.jvp(action, (jnp.ones(disc.n),), (jnp.ones(disc.n),))
    expected = domain[-1]-domain[0] if operation == 'sum' else 1.
    assert abs(float(actual)-expected) < 1e-12
    assert abs(float(tangent)-expected) < 1e-12
    assert result.domain == domain


def test_vector_derivative_matrix_and_complex_primal():
    domain = (-1., .2, 1.)
    u = chebfun(lambda x: (1+2j)*x**3, domain=domain)
    points = jnp.arange(11, dtype=jnp.float64)*.01
    result = ADChebfun(u).deriv(points, 2)
    disc = ChebColloc2Disc(12, domain)
    # A cubic perturbation exercises the stacked evaluation Jacobian.
    actual = result.jacobian.matrix(disc) @ disc.points()**3
    assert float(jnp.max(jnp.abs(actual-6*points))) < 1e-12
    assert _source_norm(result.func-(1+2j)*6*points) < 1e-12


def test_subdomain_reversal_chain_and_source_errors():
    u = chebfun(lambda x: 1+x)
    ad = ADChebfun(u)
    result = (ad**2).sum(.8, -.25)
    assert not result.is_linear
    assert abs(float(result.jacobian.apply(chebfun(lambda x: jnp.ones_like(x))))
               - ((-.25+1)**2-(.8+1)**2)) < 1e-12
    with pytest.raises(ValueError, match='sum:nargin'):
        ad.sum(0.)
    with pytest.raises(ValueError, match='sum:domain'):
        ad.sum(-2., .5)
    with pytest.raises(ValueError, match='sum:domain'):
        ad.sum(-.5, 2.)
    ad.domain = (-float('inf'), 1.)
    with pytest.raises(ValueError, match='mean:domain'):
        ad.mean()
