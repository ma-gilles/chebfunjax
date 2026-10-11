"""Source-required complex preference forwarding controls."""

import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_complex_cube_forwards_native_extrapolate(Tech, monkeypatch):
    # Native power.m sets pref.extrapolate=1 when f is complex.
    f = Tech.from_coeffs(jnp.array([-1.0 + 1j, 1j]))
    seen = {}
    sentinel = object()

    def spy(self, op, *args, **kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(Tech, "compose", spy)
    assert f**3 is sentinel
    assert seen.get("extrapolate") is True
