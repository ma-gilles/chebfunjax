"""Piecewise numeric endpoint normalization, Chebfun commit: 7574c77.

Source: @chebop/parseBC.m setupNumericalConditions, same vector contract
on single and multiple intervals. Added focused API controls.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.operators.chebop import Chebop


@pytest.mark.parametrize('kind', ['list', 'jax', 'column', 'callable'])
def test_piecewise_system_endpoint_vector(kind):
    operator = Chebop(lambda t, u, v: [u.diff(), v.diff()], domain=[0., .5, 1.])
    operator.lbc = {'list': [1., 2.], 'jax': jnp.array([1., 2.]),
                    'column': jnp.array([[1.], [2.]]),
                    'callable': lambda u, v: [u - 1., v - 2.]}[kind]
    u, v = operator.solve(0., n=8)
    points = jnp.linspace(0., 1., 11)
    assert jnp.max(jnp.abs(u(points) - 1.)) < 100*jnp.finfo(jnp.float64).eps
    assert jnp.max(jnp.abs(v(points) - 2.)) < 100*jnp.finfo(jnp.float64).eps


def test_piecewise_system_rejects_wrong_count():
    operator = Chebop(lambda t, u, v: [u.diff(), v.diff()], domain=[0., .5, 1.])
    operator.lbc = [1.]
    with pytest.raises(ValueError, match='parseBC:numberOfConditions'):
        operator.solve(0., n=8)
