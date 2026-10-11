"""Native inv.m explicitly bypasses vectorCheck for every algorithm."""
import importlib

import jax.numpy as jnp
import pytest

from chebfunjax import chebfun


@pytest.mark.parametrize("algorithm", ["roots", "newton", "bisection", "regulafalsi", "illinois", "brent"])
def test_inverse_skips_constructor_probe(monkeypatch, algorithm):
    f = chebfun(lambda x: x + .125*x**3)
    context = importlib.import_module("chebfunjax.chebfun1d._construction_context")
    def forbidden(*args, **kwargs):
        raise AssertionError("inverse reached vectorCheck despite native noVectorCheck")
    monkeypatch.setattr(context, "vector_check", forbidden)
    g = f.inv(algorithm=algorithm)
    points = jnp.linspace(-1., 1., 17)
    assert float(jnp.max(jnp.abs(g(f(points)) - points))) < 100*jnp.finfo(jnp.float64).eps*g.vscale
