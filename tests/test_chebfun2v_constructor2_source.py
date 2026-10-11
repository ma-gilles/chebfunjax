"""Literal native tests/chebfun2v/test_constructor2.m six predicates.

Python keyword domains and left-associated vertcat adapt MATLAB syntax.
Native Chebfun pin7574c77; all defaults and continuous norm bounds retained.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebpref import ChebfunPref


@pytest.mark.parametrize("clause", [1, 2, 3, 4])
def test_literal_vectorize(clause):
    domain = (-1.0, 1.0, -1.0, 1.0) if clause in (1, 3) else (-2.0, 3.0, -1.0, 0.0)
    count = 2 if clause < 3 else 3
    f1 = lambda x, y: x * y  # noqa: E731 - literal native function handle
    f2 = lambda x, y: x * y  # noqa: E731 - native scalar callback vectorize adapter
    h1 = Chebfun2v.from_functions(*([f1] * count), domain=domain)
    h2 = Chebfun2v.from_functions(*([f2] * count), vectorize=True, domain=domain)
    tol = 100 * ChebfunPref().cheb2Prefs.chebfun2eps
    assert (h1 - h2).norm() < tol


@pytest.fixture(scope="module")
def source_coordinates():
    d = (0.0, 2 * jnp.pi, -1.0, 1.0)
    return (
        Chebfun2.from_function(lambda u, v: u, domain=d),
        Chebfun2.from_function(lambda u, v: v, domain=d),
    )


def test_literal_surface5(source_coordinates):
    u, v = source_coordinates
    x = (1 + 0.5 * v * (u / 2).cos()) * u.cos()
    y = (1 + 0.5 * v * (u / 2).cos()) * u.sin()
    z = 0.5 * v * (u / 2).sin()
    r = x.vertcat(y, z)
    ru = r.diff(1, 1)
    rv = r.diff(1, 2)
    tol = 100 * ChebfunPref().cheb2Prefs.chebfun2eps
    assert (ru.ctranspose() @ rv).norm(jnp.inf) < 10 * tol


def test_literal_component_count6(source_coordinates):
    u, v = source_coordinates
    field = (5 * u).sin().vertcat((5 * v).cos()).vertcat(0)
    assert field.n_components == 3
