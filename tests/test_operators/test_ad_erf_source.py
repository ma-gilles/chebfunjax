"""Independent directional and seeded controls for source AD error functions.

Provenance: @adchebfun/adchebfun.m, Chebfun7574c77680d7e82b79626300bf255498271a72df.
"""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import chebfun

OPERATIONS = ['erf', 'erfc', 'erfcinv', 'erfcx', 'erfinv']


@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize('seed', ['operator', 'scalar'])
def test_jacobian_composes_seed_and_preserves_domain(operation, seed):
    u = ADChebfun(chebfun(lambda x: .55+.01*x, domain=[-1., 0., 1.]))
    if seed == 'scalar':
        u = u.seed(1, 0)
    # A nonlinear incoming map ensures the outer Jacobian must compose.
    mapped = u**2
    result = getattr(mapped, operation)()
    original_domain = mapped.domain
    x = jnp.linspace(-1, 1, 31)
    p = chebfun(lambda x: .05+.003*x, domain=[-1., 0., 1.])
    action = result.jacobian.apply(p) if seed == 'operator' else result.jacobian
    raw = (lambda z: jax.scipy.special.erfinv(1-z)) if operation == 'erfcinv' else getattr(jax.scipy.special, operation)
    expected = jax.vmap(jax.grad(lambda z: raw(z**2)))(u.func(x))
    if seed == 'operator':
        expected = expected*p(x)
    assert float(jnp.max(jnp.abs(action(x)-expected))) < 1e-12
    assert result.domain == original_domain == (-1., 0., 1.)
    assert not result.is_linear


@pytest.mark.parametrize('operation', OPERATIONS)
def test_zero_jacobian_source_linearity(operation):
    u = .55*(ADChebfun(chebfun(lambda x: .55+.01*x))**0)
    result = getattr(u, operation)()
    assert result.is_linear
    assert float(result.jacobian.apply(chebfun(1.)).norm(jnp.inf)) == 0
