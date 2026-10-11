"""Source controls for separableApprox real/imag/complex, commit 7574c77.

Tiny-component bounds are analytic regressions, not invented native tests.
Error cases preserve native complex.m order and literal messages.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2


@pytest.mark.parametrize("part", ["real", "imag"])
def test_tiny_nonzero_constant_part(part):
    delta = 1e-10
    value = delta + 1j if part == "real" else 1 + 1j * delta
    f = Chebfun2.from_function(lambda x, y: jnp.asarray(value))
    g = getattr(f, part)()
    assert jnp.abs(g(0.0, 0.0) - delta) < 1000 * jnp.finfo(jnp.float64).eps * delta


def test_complex_rejects_nonreal_one_input():
    f = Chebfun2.from_function(lambda x, y: jnp.asarray(1 + 1j))
    with pytest.raises(ValueError, match="Input must be real valued"):
        Chebfun2.complex(f)


@pytest.mark.parametrize("position", [0, 1])
def test_complex_rejects_nonreal_two_inputs(position):
    real = Chebfun2.from_function(lambda x, y: jnp.asarray(1.0))
    nonreal = Chebfun2.from_function(lambda x, y: jnp.asarray(1 + 1j))
    args = (nonreal, real) if position == 0 else (real, nonreal)
    with pytest.raises(ValueError, match="Inputs must be real valued"):
        Chebfun2.complex(*args)


def test_complex_second_type_guard_precedes_real_guard():
    nonreal = Chebfun2.from_function(lambda x, y: jnp.asarray(1 + 1j))
    with pytest.raises(TypeError, match="Second input must be a CHEBFUN2"):
        Chebfun2.complex(nonreal, 1)
