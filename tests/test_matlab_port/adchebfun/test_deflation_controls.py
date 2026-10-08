"""Independent real derivative, chain, block and primal controls."""
import jax.numpy as jnp
import pytest

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.blocks import ChebColloc2Disc
from chebfunjax.operators.chebmatrix import ChebMatrix
from chebfunjax.operators.deflation import deflation_fun


@pytest.mark.parametrize("kind", ["L2", "H1"])
@pytest.mark.parametrize("count,power,shift", [(1, 2, .5), (2, 2, .5), (3, 2, .5),
                                             (3, 3, 2), (2, 1, 0), (2, 1, 4)])
def test_independent_chain_derivative(kind, count, power, shift):
    domain = (2., 5.)
    u = chebfun(lambda x: .5+.02*x, domain=domain)
    h = chebfun(lambda x: .03+.001*x**2, domain=domain)
    roots = [chebfun(c, domain=domain) for c in [-.1, .1, .2]][:count]
    argument = ADChebfun(u)**2
    residual = argument.diff(2)+argument.sin()
    actual = deflation_fun(residual, argument, roots, power, shift, kind)
    w, dw = u**2, 2*u*h
    f = w.diff(2)+w.sin()
    df = dw.diff(2)+w.cos()*dw
    product, logarithmic = 1., 0.
    for r in roots:
        delta = w-r
        # Independent squared-norm derivative, with no AD norm or Jacobian.
        square = (delta*delta).sum()
        inner = (delta*dw).sum()
        if kind == "H1":
            square += (delta.diff()**2).sum()
            inner += (delta.diff()*dw.diff()).sum()
        product *= square
        logarithmic += inner/square
    factor = product**(-power/2)
    expected = df*(factor+shift)-f*(power*factor*logarithmic)
    scale = max(1., float(expected.norm(jnp.inf)))
    assert float((actual.jacobian.apply(h)-expected).norm(jnp.inf)) < 5e-12*scale
    disc = ChebColloc2Disc(24, domain)
    nodes = disc.points()
    assert float(jnp.max(jnp.abs(actual.jacobian.matrix(disc)@h(nodes)-expected(nodes)))) < 5e-10*scale
    assert actual.domain == domain


@pytest.mark.parametrize("kind,expected_square", [("L2", 8/3), ("H1", 14/3)])
def test_plain_constant_residual_exact_integral(kind, expected_square):
    u = chebfun(lambda x: 1+x)
    residual = chebfun(3.)
    root = chebfun(0.)
    value = deflation_fun(residual, u, root, 2., .5, kind)
    # Integral (1+x)^2 is 8/3 on [-1,1], plus derivative integral 2 for H1.
    square = expected_square
    assert float((value-chebfun(3*(1/square+.5))).norm(jnp.inf)) < 1e-13


def test_root_block_and_column_layouts():
    u, residual = chebfun(lambda x: .6+.01*x), chebfun(2.)
    roots = [chebfun(.1), chebfun(.2)]
    block = ChebMatrix([roots])
    columns = chebfun(lambda x: jnp.stack([jnp.zeros_like(x)+.1, jnp.zeros_like(x)+.2], axis=-1))
    reference = deflation_fun(residual, u, roots, 2, .5)
    for container in (block, columns):
        assert float((deflation_fun(residual, u, container, 2, .5)-reference).norm(jnp.inf)) < 1e-12
