"""Domain, complex, matrix and incoming-chain integral AD controls."""
import jax
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.operators.blocks import ChebColloc2Disc, fred_op, volt_op


@pytest.mark.parametrize('operation', ['fred', 'volt'])
@pytest.mark.parametrize('domain', [(-1., 1.), (2., 5.), (-1., .2, 1.)])
@pytest.mark.parametrize('factor', [1., 1+2j])
def test_incoming_chain_matrix_and_complex_values(operation, domain, factor):
    u = cj.chebfun(lambda x: 1+x, domain=domain)
    h = cj.chebfun(lambda x: 1+0*x, domain=domain)
    ad = ADChebfun(u)**2
    result = getattr(cj, operation)(lambda x, y: factor+0*x*y, ad)
    a, b = domain[0], domain[-1]
    upper = (lambda x: b+0*x) if operation == 'fred' else (lambda x: x)
    points = jnp.linspace(a, b, 21)
    expected = factor*((1+upper(points))**3-(1+a)**3)/3
    derivative = factor*((1+upper(points))**2-(1+a)**2)
    assert float(jnp.max(jnp.abs(result.func(points)-expected))) < 1e-11
    assert float(jnp.max(jnp.abs(result.jacobian.apply(h)(points)-derivative))) < 1e-11
    assert result.domain == ad.domain
    assert tuple(result.func.domain.breakpoints) == domain
    assert not result.is_linear
    disc = ChebColloc2Disc(16, domain)
    matrix = result.jacobian.matrix(disc)
    actual, tangent = jax.jvp(jax.jit(lambda v: matrix@v),
                              (jnp.ones(disc.n),), (jnp.ones(disc.n),))
    expected_nodes = factor*((1+upper(disc.points()))**2-(1+a)**2)
    assert float(jnp.max(jnp.abs(actual-expected_nodes))) < 1e-11
    assert float(jnp.max(jnp.abs(tangent-expected_nodes))) < 1e-11


@pytest.mark.parametrize('operation', ['fred', 'volt'])
def test_piecewise_nonsmooth_input_and_reversed_call(operation):
    f = cj.chebfun(lambda x: (1+2j)*jnp.abs(x), domain=(-1., 0., 1.))
    def kernel(x, y):
        return 1.+0*x*y
    result = getattr(cj, operation)(kernel, f)
    reversed_result = getattr(cj, operation)(f, kernel)
    xx = jnp.linspace(-1., 1., 31)
    expected = (jnp.ones_like(xx) if operation == 'fred' else
                .5+jnp.sign(xx)*xx**2/2)*(1+2j)
    assert float(jnp.max(jnp.abs(result(xx)-expected))) < 1e-12
    assert float((result-reversed_result).norm(jnp.inf)) == 0
    assert tuple(result.domain.breakpoints) == (-1., 0., 1.)


@pytest.mark.parametrize('operation', ['fred', 'volt'])
def test_array_values_and_row_orientation(operation):
    f = cj.chebfun(lambda x: jnp.stack((1+0*x, (1+1j)*x), axis=-1))
    result = getattr(cj, operation)(lambda x, y: 1.+0*x*y, f)
    xx = jnp.linspace(-1., 1., 11)
    expected = (jnp.stack((2+0*xx, 0*xx), axis=-1) if operation == 'fred' else
                jnp.stack((xx+1, (1+1j)*(xx**2-1)/2), axis=-1))
    assert float(jnp.max(jnp.abs(result(xx)-expected))) < 1e-12
    f = f.transpose()
    row = getattr(cj, operation)(lambda x, y: 1.+0*x*y, f)
    assert row.is_transposed
    assert float(jnp.max(jnp.abs(row(xx)-expected.T))) < 1e-12


@pytest.mark.parametrize('builder', [fred_op, volt_op])
def test_source_onevar_matrix_option(builder):
    def kernel(x, y=None):
        return ((1+1j)*jnp.exp(x[:, None]-x[None, :]) if y is None
                else (1+1j)*jnp.exp(x-y))
    disc = ChebColloc2Disc(12, (-1., 1.))
    normal = builder(kernel).matrix(disc)
    onevar = builder(kernel, (-1., 1.), 'onevar').matrix(disc)
    assert float(jnp.max(jnp.abs(normal-onevar))) == 0
