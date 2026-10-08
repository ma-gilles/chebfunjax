"""All original test_bessel.m clauses through testUnary.

Provenance: Chebfun7574c77680d7e82b79626300bf255498271a72df,
tests/adchebfun/test_bessel.m and @adchebfun/testUnary.m/taylorTesting.m.
Original50eps primal and1e-2 Taylor bounds, function infinity norms.
Inputs captured from MATLAB R2025b seedRNG(6179), pinned source7574c77.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun

OPERATIONS = [1, 2.5]


@pytest.fixture(scope='module')
def inputs():
    capture = json.loads(Path(__file__).with_name('erf_matlab_inputs.json').read_text())
    raw_u, raw_p = jnp.asarray(capture['raw_u']), jnp.asarray(capture['raw_p'])
    value_u = .1*Chebfun.from_values(raw_u)+.5
    taylor_u = Chebfun.from_values(.1*raw_u+.5)
    p = Chebfun.from_values(.01*raw_p+.05)
    return value_u, taylor_u, p


@pytest.mark.parametrize('operation', OPERATIONS)
def test_source_value_clause(inputs, operation):
    u, _, _ = inputs
    expected = u.besselj(operation)
    actual = ADChebfun(u).besselj(operation).func
    assert abs(float((expected-actual).norm(jnp.inf))) < 50*jnp.finfo(jnp.float64).eps


@pytest.mark.parametrize('operation', OPERATIONS)
def test_source_taylor_clause(inputs, operation):
    _, u, p = inputs
    result = ADChebfun(u).besselj(operation)
    first, second = [], []
    for exponent in range(2, 6):
        perturbation = .2**exponent*p
        changed = ADChebfun(u+perturbation).besselj(operation)
        delta = changed.func-result.func
        action = result.jacobian.apply(perturbation)
        first.append(delta.norm(jnp.inf))
        second.append((delta-action).norm(jnp.inf))
    for errors, expected in [(first, 1), (second, 2)]:
        orders = jnp.diff(jnp.log(jnp.asarray(errors)))/jnp.log(.2)
        assert float(jnp.max(jnp.abs(orders-expected))) < 1e-2


@pytest.mark.parametrize('operation', OPERATIONS)
def test_source_linearity_clause(inputs, operation):
    assert not ADChebfun(inputs[0]).besselj(operation).is_linear


@pytest.mark.parametrize('kind', [1, 2.5])
@pytest.mark.parametrize('domain', [(-1., 1.), (2., 5.), (-1., .2, 1.)])
def test_source_multiplier_preserves_incoming_chain_and_domain(kind, domain):
    u = chebfun(lambda x: .55+.002*x, domain=domain)
    h = chebfun(lambda x: .03+.001*x, domain=domain)
    argument = ADChebfun(u)**2
    result = argument.besselj(kind)
    expected = (-(u**2).besselj(kind+1)+kind*(u**2).besselj(kind)/(u**2))*(2*u*h)
    assert float((result.jacobian.apply(h)-expected).norm(jnp.inf)) < 1e-12
    assert result.domain == argument.domain
    assert not result.is_linear



def test_public_order_first_dispatch():
    import chebfunjax as cj
    u = chebfun(lambda x: .55+.002*x)
    result = cj.besselj(2.5, ADChebfun(u))
    assert float((result.func-cj.besselj(2.5, u)).norm(jnp.inf)) < 50*jnp.finfo(jnp.float64).eps


@pytest.mark.parametrize('order', [1., 2.5])
def test_independent_operator_matrix_action(order):
    import jax
    from scipy.special import jvp

    from chebfunjax.operators.blocks import ChebColloc2Disc

    domain = (2., 5.)
    u = chebfun(lambda x: .5+.002*x, domain=domain)
    h = chebfun(lambda x: .03+.001*x, domain=domain)
    result = (ADChebfun(u)**2).besselj(order)
    disc = ChebColloc2Disc(12, domain)
    points = disc.points()
    expected = jnp.asarray(jvp(order, u(points)**2))*2*u(points)*h(points)
    action = result.jacobian.apply(h)(points)
    matrix = result.jacobian.matrix(disc)
    matrix_action = jax.jit(lambda v: matrix @ v)(h(points))
    assert float(jnp.max(jnp.abs(action-expected))) < 1e-13
    assert float(jnp.max(jnp.abs(matrix_action-expected))) < 1e-13
