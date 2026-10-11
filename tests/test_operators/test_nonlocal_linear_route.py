"""Source AD linear routing controls, Chebfun commit 7574c77.

The analytic1e-11 bound is an added control, not a DDE page assertion.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop


@pytest.mark.parametrize("initial", [False, True])
def test_nonlocal_affine_rhs_and_initial_correction(monkeypatch, initial):
    def forbidden(*args, **kwargs):
        raise AssertionError("Nonlocal linear solve must use extracted AD blocks")

    monkeypatch.setattr(Chebop, "_build_linop", forbidden)
    N = Chebop(lambda x, u: u.diff() + u.cumsum() + 3, domain=(0.0, 1.0))
    N.lbc = 1.0
    if initial:
        N.init = chebfun(lambda x: 0.2 + 0.1 * x, domain=(0.0, 1.0))
    rhs = chebfun(lambda x: 4 + x + 0.5 * x * x, domain=(0.0, 1.0))
    u = N.solve(rhs, n=19)
    x = jnp.linspace(0.0, 1.0, 31)
    assert float(jnp.max(jnp.abs(u(x) - (1 + x)))) < 1e-11


def test_local_linear_assembly_is_retained(monkeypatch):
    original = Chebop._build_linop
    calls = []

    def recorded(self):
        calls.append(True)
        return original(self)

    monkeypatch.setattr(Chebop, "_build_linop", recorded)
    N = Chebop(lambda u: u.diff() - u, domain=(0.0, 1.0))
    N.lbc = 1.0
    u = N.solve(0.0, n=19)
    assert calls == [True]
    x = jnp.linspace(0.0, 1.0, 31)
    assert float(jnp.max(jnp.abs(u(x) - jnp.exp(x)))) < 1e-11
