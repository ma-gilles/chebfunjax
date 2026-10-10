"""Independent polynomial controls for native AD composition,7574c77."""

import jax.numpy as jnp
import pytest

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.blocks import ChebColloc2Disc, compose_op


@pytest.mark.parametrize("kind", ["fixed", "self", "derivative", "nonlinear_outer"])
def test_polynomial_primal_and_jacobian(kind):
    domain = (-1.0, 1.0)
    u = chebfun(lambda x: 0.2 + 0.1 * x * x, domain=domain)
    h = chebfun(lambda x: 1 + x + x * x, domain=domain)
    ad = ADChebfun(u)
    disc = ChebColloc2Disc(14, domain)
    x = disc.points()
    if kind == "fixed":
        g = chebfun(lambda x: 0.3 * x, domain=domain)
        result = ad(g)
        gx = 0.3 * x
        expected = 1 + gx + gx * gx
    elif kind == "self":
        result = ad(ad)
        gx = 0.2 + 0.1 * x * x
        expected = 1 + gx + gx * gx + 0.2 * gx * (1 + x + x * x)
    elif kind == "derivative":
        result = ad(0.5 * ad.diff())
        gx = 0.1 * x
        expected = 1 + gx + gx * gx + 0.2 * gx * 0.5 * (1 + 2 * x)
    else:
        g = chebfun(lambda x: 0.3 * x, domain=domain)
        result = (ad**2)(g)
        gx = 0.3 * x
        expected = 2 * (0.2 + 0.1 * gx * gx) * (1 + gx + gx * gx)
    primal = (0.2 + 0.1 * gx * gx) ** (2 if kind == "nonlinear_outer" else 1)
    assert float(jnp.max(jnp.abs(result.func(x) - primal))) < 1e-12
    assert float(jnp.max(jnp.abs(result.jacobian.matrix(disc) @ h(x) - expected))) < 1e-11
    assert float(jnp.max(jnp.abs(result.jacobian.apply(h)(x) - expected))) < 1e-11
    assert result.is_linear == (kind == "fixed")


@pytest.mark.parametrize("location", [-1.2, 0.0, 1.2])
def test_piecewise_matrix_routing_and_clipping(location):
    disc = ChebColloc2Disc([6, 8], (-1.0, 0.0, 1.0))
    g = chebfun(lambda x: 0 * x + location)
    matrix = compose_op(g, disc.domain).matrix(disc)
    x = disc.points()
    values = jnp.concatenate([3 + x[:6] + x[:6] ** 2, 7 + 2 * x[6:] + x[6:] ** 2])
    expected = 3 + location + location**2 if location <= 0 else 7 + 2 * location + location**2
    assert matrix.shape == (14, 14)
    assert float(jnp.max(jnp.abs(matrix @ values - expected))) < 1e-11
    if location == 0:
        assert bool(jnp.all(matrix[:, 5] == 1))
        assert bool(jnp.all(matrix[:, 6:] == 0))


def test_fixed_composition_operator_action():
    u = chebfun(lambda x: 1 + x + 2 * x * x)
    g = chebfun(lambda x: 0.1 + 0.2 * x)
    x = jnp.linspace(-1, 1, 31)
    result = compose_op(g) * u
    assert (
        float(jnp.max(jnp.abs(result(x) - (1 + (0.1 + 0.2 * x) + 2 * (0.1 + 0.2 * x) ** 2))))
        < 1e-12
    )
