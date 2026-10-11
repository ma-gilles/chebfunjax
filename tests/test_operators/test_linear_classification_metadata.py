"""Per-solve source AD classification metadata; Chebfun commit 7574c77."""

import jax.numpy as jnp
import pytest

import chebfunjax.operators.chebop_altdisc as alternative
from chebfunjax import chebfun
from chebfunjax.autodiff.adchebfun import (
    _detect_linearity_metadata,
    detect_linearity,
)
from chebfunjax.operators.chebop import Chebop


@pytest.mark.parametrize(
    "operator, expected",
    [
        (lambda u: u.diff() - u, (True, True)),
        (lambda u: u.diff() + u.cumsum(), (True, False)),
        (lambda u: u * u, (False, True)),
        (lambda u: 2.0, (True, False)),
    ],
)
def test_classification_decision_and_metadata(operator, expected):
    probe = chebfun(lambda x: jnp.sin(jnp.pi * x), domain=(0.0, 1.0), n=8)
    assert _detect_linearity_metadata(operator, probe) == expected
    assert detect_linearity(operator, probe) == expected[0]


def test_local_solve_skips_second_extraction(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Local solve repeated AD extraction")

    monkeypatch.setattr(alternative, "_linearize_scalar_ad", forbidden)
    N = Chebop(lambda u: u.diff() - u, domain=(0.0, 1.0))
    N.lbc = 1.0
    u = N.solve(0.0, n=19)
    x = jnp.linspace(0.0, 1.0, 31)
    assert float(jnp.max(jnp.abs(u(x) - jnp.exp(x)))) < 1e-11


def test_mutation_does_not_reuse_old_classification(monkeypatch):
    N = Chebop(lambda u: u.diff() - u, domain=(0.0, 1.0))
    assert N._is_linear(_with_local=True) == (True, True)
    N.op = lambda u: u.diff() + u.cumsum() + 3
    assert N._is_linear(_with_local=True) == (True, False)
    N.lbc = 1.0
    N.init = chebfun(lambda x: 0.2 + 0.1 * x, domain=(0.0, 1.0))
    original = alternative._linearize_scalar_ad
    calls = []

    def observed(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(alternative, "_linearize_scalar_ad", observed)
    rhs = chebfun(lambda x: 4 + x + 0.5 * x * x, domain=(0.0, 1.0))
    u = N.solve(rhs, n=19)
    assert calls == [True]
    x = jnp.linspace(0.0, 1.0, 31)
    assert float(jnp.max(jnp.abs(u(x) - (1 + x)))) < 1e-11
