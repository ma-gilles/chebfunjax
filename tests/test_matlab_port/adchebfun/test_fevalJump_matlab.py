"""All six test_fevalJump.m predicates, Chebfun7574c77.

Native seed6179 inputs are reused from valueTesting/taylorTesting captures.
No native fevalJump result capture is claimed. Taylor hMax=4 and all source
bounds are retained, including the exact-zero jump second remainders.
Numeric call syntax adapts MATLAB feval; side flags and char endpoints are
outside this source test's two operations.

Provenance
----------
MATLAB source: tests/adchebfun/test_fevalJump.m, @adchebfun/testUnary.m.
Chebfun commit: 7574c77
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun, jump
from chebfunjax.operators.blocks import ChebColloc2Disc


@pytest.fixture(scope='module')
def inputs():
    data = json.loads(Path(__file__).with_name('erf_matlab_inputs.json').read_text())
    u, p = jnp.asarray(data['raw_u']), jnp.asarray(data['raw_p'])
    return (.1*Chebfun.from_values(u)+.5, Chebfun.from_values(.1*u+.5),
            Chebfun.from_values(.01*p+.05))


def operation(u, kind):
    return u(.3) if kind == 'feval' else jump(u, .5, -1)


@pytest.mark.parametrize('kind', ['feval', 'jump'])
def test_source_value_clause(inputs, kind):
    u = inputs[0]
    assert float(jnp.abs(operation(ADChebfun(u), kind).func-operation(u, kind))) == 0


@pytest.mark.parametrize('kind', ['feval', 'jump'])
def test_source_taylor_clause(inputs, kind):
    _, u, p = inputs
    base = ADChebfun(u)
    result = operation(base, kind)
    first, second = [], []
    for exponent in range(2, 6):
        perturbation = .2**exponent*p
        delta = operation(base+perturbation, kind).func-result.func
        first.append(jnp.abs(delta))
        second.append(jnp.abs(delta-result.jacobian.apply(perturbation)))
    second = jnp.asarray(second)
    if kind == 'jump':
        assert not bool(jnp.any(second))
    else:
        order = jnp.diff(jnp.log(jnp.asarray(first)))/jnp.log(.2)
        assert float(jnp.max(jnp.abs(order-1))) < 1e-2
        assert float(jnp.max(second)) < 1e-12


@pytest.mark.parametrize('kind', ['feval', 'jump'])
def test_source_linearity_clause(inputs, kind):
    assert operation(ADChebfun(inputs[0]), kind).linearity == (True,)


def test_discontinuous_jump_chain_matrix_and_metadata():
    domain = (-1., 0., 1.)
    u = chebfun([lambda x: 1+x, lambda x: 3+2*x], domain=domain)
    h = chebfun([lambda x: 2+x, lambda x: 5-x], domain=domain)
    incoming = ADChebfun(u)**2
    result = jump(incoming, 0., 2.)
    assert float(result.func) == 6.
    assert result.domain == domain
    assert result.jumpLocations == (0.,)
    assert incoming.jumpLocations == ()
    assert not result.is_linear
    expected = jump(2*u*h, 0.)
    assert abs(float(result.jacobian.apply(h)-expected)) < 1e-12
    disc = ChebColloc2Disc(8, domain)
    # Piece endpoints are duplicated in the collocation vector.
    from chebfunjax.utils.quadrature import chebpts_ab
    values = jnp.concatenate([h.funs[k](chebpts_ab(8, *domain[k:k+2]))
                              for k in range(2)])
    matrix = result.jacobian.matrix(disc)
    assert abs(float(jax.jit(lambda v: matrix @ v)(values)-expected)) < 1e-11
    assert float(jump(ADChebfun(u), 0.).func) == float(jump(u, 0.)) == 2.


def test_jump_domain_union_and_copy_binary_metadata():
    u = ADChebfun(chebfun(lambda x: 2+x, domain=(-1., -.2, 1.)))
    left, right = u.jump(.3), u.jump(.7)
    assert left.domain == (-1., -.2, .3, 1.)
    assert left.jumpLocations == (.3,)
    assert (-left).jumpLocations == (.3,)
    assert (left+right).jumpLocations == (.3, .7)
    assert (left-right).jumpLocations == (.3, .7)
    assert (left*right).jumpLocations == (.3, .7)
    assert (left/(right+1)).jumpLocations == (.3, .7)
    assert u.domain == (-1., -.2, 1.)
    assert u.jumpLocations == ()


@pytest.mark.parametrize('points', [.3, [-.7, .3, .8]])
def test_complex_evaluation_action_and_matrix(points):
    u = chebfun(lambda x: 1+(2+3j)*x)
    h = chebfun(lambda x: (1-2j)*x**2)
    incoming = ADChebfun(u)**2
    result = incoming(points)
    expected = (2*u*h)(jnp.asarray(points))
    assert jnp.max(jnp.abs(result.func-(u**2)(jnp.asarray(points)))) < 1e-13
    assert jnp.max(jnp.abs(result.jacobian.apply(h)-expected)) < 1e-12
    from chebfunjax.utils.quadrature import chebpts
    matrix = result.jacobian.matrix(ChebColloc2Disc(12))
    values = h(chebpts(12))
    assert jnp.max(jnp.abs(jax.jit(lambda v: matrix @ v)(values)-expected)) < 1e-11
    assert result.func.shape == jnp.asarray(points).shape
