"""Preserve pre-existing Python rational max_iter=0 failure adapter."""

import jax.numpy as jnp
import pytest

import chebfunjax as cj


def test_rational_zero_iteration_preserves_warning_and_failure():
    f = cj.chebfun(lambda x: jnp.cos(jnp.pi * x), trig=True)
    with pytest.warns(UserWarning, match="after 0 iterations"):
        with pytest.raises(RuntimeError, match="trigremez: rational iteration failed"):
            cj.trigremez(f, 1, 1, max_iter=0)
