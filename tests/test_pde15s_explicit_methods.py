"""Retained explicit legacy dispatch; this is not JAX-only qualification."""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun1d.pde15s import pde15s


@pytest.mark.parametrize("method", ["Radau", "BDF", "RK45", "DOP853"])
def test_explicit_legacy_method(method):
    u0 = chebfun(lambda x: 1 + 0 * x)
    out = pde15s(lambda u: -u, [0.0, 0.05, 0.1], u0, n=8, method=method, rtol=1e-8, atol=1e-10)
    assert len(out) == 3
    for t, u in zip([0.0, 0.05, 0.1], out):
        assert float(jnp.max(jnp.abs(u(jnp.array([-0.7, 0.2])) - jnp.exp(-t)))) < 1e-7
