"""Principal-branch controls for source elementary AD prerequisites.

Provenance: @chebfun/power.m, @chebtech/power.m, @chebfun/sinc.m,
@adchebfun/adchebfun.m at Chebfun 7574c77680d7e82b79626300bf255498271a72df.
These analytic controls supplement, rather than replace, original unary clauses.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun


@pytest.mark.parametrize('kind', ['positive', 'negative', 'crossing', 'complex'])
def test_principal_sqrt_branches(kind):
    functions = {
        'positive': lambda x: 2+x,
        'negative': lambda x: -2-x,
        'crossing': lambda x: x,
        'complex': lambda x: 2+x+1j,
    }
    f = functions[kind]
    result = chebfun(f).sqrt()
    x = jnp.asarray([-.91, -.43, .19, .73])
    expected = jnp.sqrt(jnp.asarray(f(x), dtype=jnp.complex128))
    assert float(jnp.max(jnp.abs(result(x)-expected))) < 50*jnp.finfo(jnp.float64).eps


@pytest.mark.parametrize('name', ['acsc', 'acscd', 'asec', 'asecd'])
@pytest.mark.parametrize('sign', [-1, 1])
def test_real_reciprocal_inverse_branch(name, sign):
    x = jnp.asarray([-.8, .2, .7])
    f = chebfun(lambda t: sign*(.55+.01*t))
    expected = sign*(jnp.pi/2 - 1j*jnp.arccosh(1/jnp.abs(f(x))))
    if name.startswith('asec'):
        expected = jnp.pi/2-expected
    if name.endswith('d'):
        expected = expected*180/jnp.pi
    error = jnp.max(jnp.abs(getattr(f, name)()(x)-expected))
    assert float(error) < 50*jnp.finfo(jnp.float64).eps*max(1., float(jnp.max(jnp.abs(expected))))


def test_unnormalized_sinc_zero():
    f = chebfun(lambda x: x).sinc()
    assert abs(complex(f(0))-1) < 50*jnp.finfo(jnp.float64).eps
    assert abs(complex(f(.5))-jnp.sin(.5)/.5) < 50*jnp.finfo(jnp.float64).eps


@pytest.mark.parametrize('operation', ['exp', 'log', 'sin', 'cosh'])
def test_unary_retains_incoming_ad_domain(operation):
    from chebfunjax.autodiff.adchebfun import ADChebfun

    u = ADChebfun(chebfun(lambda x: .55+.01*x, domain=[-1, 0, 1]))
    result = getattr(u, operation)()
    assert result.domain == u.domain
    assert result.jacobian.domain == u.jacobian.domain
