"""Complex adaptive GE controls, @chebfun2/constructor.m, commit 7574c77.

Analytic regressions cover Chebyshev/Fourier real and complex callbacks.
The bound is supplemental; existing native mean/std/conj bounds are unchanged.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2


@pytest.mark.parametrize("trig", [False, True])
@pytest.mark.parametrize("complex_output", [False, True])
def test_joint_complex_adaptive_constructor(trig, complex_output):
    def callback(x, y):
        a = jnp.cos(jnp.pi * x) if trig else x
        b = jnp.sin(jnp.pi * y) if trig else y
        return a + (1j if complex_output else 1) * b

    f = Chebfun2.from_function(callback, trig=trig)
    t = jnp.linspace(-0.93, 0.87, 17)
    xx, yy = jnp.meshgrid(t, t)
    assert jnp.max(jnp.abs(f(xx, yy) - callback(xx, yy))) < 1000 * jnp.finfo(jnp.float64).eps
    assert bool(jnp.iscomplexobj(f(xx, yy))) == complex_output
