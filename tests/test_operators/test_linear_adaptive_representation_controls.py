"""Check periodic conversion, unsupported data and coefficient-only breaks."""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece, chebfun
from chebfunjax.domain import Domain
from chebfunjax.fun.singfun import Singfun
from chebfunjax.operators._linear_altdisc import LinearDiscretization, solve_operator
from chebfunjax.operators.blocklinop import linop
from chebfunjax.operators.blocks import I, diag
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.tech.trigtech import Trigtech
from chebfunjax.utils.quadrature import chebpts


def _cosine_reference(backend):
    if backend == 'chebcolloc1':
        return jnp.cos(jnp.pi*chebpts(8, kind=1))
    return chebfun(lambda x: jnp.cos(jnp.pi*x)).funs[0].tech.coeffs[:8]


@pytest.mark.parametrize('backend', ['chebcolloc1', 'ultraS'])
def test_trig_operator_coefficient_converts_before_assembly(backend):
    coefficient = chebfun(lambda x: jnp.cos(jnp.pi*x), trig=True)
    assert isinstance(coefficient.funs[0].tech, Trigtech)
    discretization = LinearDiscretization(linop(diag(coefficient)), (8,), backend)
    constant = jnp.ones(8) if backend == 'chebcolloc1' else jnp.eye(8)[:, 0]
    assert jnp.max(jnp.abs(discretization.A@constant-_cosine_reference(backend))) < 1e-12


@pytest.mark.parametrize('backend', ['chebcolloc1', 'ultraS'])
def test_trig_rhs_converts_before_assembly(backend):
    forcing = chebfun(lambda x: jnp.cos(jnp.pi*x), trig=True)
    assert isinstance(forcing.funs[0].tech, Trigtech)
    discretization = LinearDiscretization(linop(I((-1., 1.))), (8,), backend)
    assert jnp.max(jnp.abs(discretization.rhs([forcing])-_cosine_reference(backend))) < 1e-12


@pytest.mark.parametrize('backend', ['chebcolloc1', 'ultraS'])
def test_singular_rhs_is_explicitly_unsupported(backend):
    technology = Singfun(Chebtech2.from_coeffs(jnp.asarray([1.])), (.5, 0.))
    forcing = Chebfun(funs=[_Piece(technology, (-1., 1.))], domain=Domain((-1., 1.)))
    discretization = LinearDiscretization(linop(I((-1., 1.))), (8,), backend)
    with pytest.raises(NotImplementedError, match='requires polynomial Chebtech1/Chebtech2'):
        discretization.rhs([forcing])


@pytest.mark.parametrize('backend', ['chebcolloc1', 'ultraS'])
def test_coefficient_breakpoints_merge_even_when_operator_domain_is_coarse(backend):
    coefficient = chebfun([2., 3.], domain=[-1., 0., 1.])
    operator = linop(diag(coefficient, domain=(-1., 1.)))
    assert operator.domain == (-1., 1.)
    solution, discretization, converged = solve_operator(operator, [1.], backend=backend, n=8)
    assert converged
    assert discretization.domain == (-1., 0., 1.)
    assert discretization.dimensions == (8, 8)
    expected = chebfun([.5, 1/3], domain=[-1., 0., 1.])
    assert (solution[0]-expected).norm(jnp.inf) < 1e-12
