"""Additional analytic controls; distinct from unchanged native predicates."""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.utils.conformal import _kerzman_stein, _poly_method, conformal


def test_circle_continuous_ks():
    C = chebfun(lambda t: jnp.exp(1j * jnp.pi * t), trig=True)
    g, Z, W = _kerzman_stein(C, 32)
    expected = jnp.exp(1j * (-jnp.pi + 2 * jnp.pi * jnp.arange(32) / 32))
    # Independent circle solution; tolerance covers the composed spectral
    # inverse, derivative and linear solve, not a replacement source bound.
    assert float(jnp.max(jnp.abs(Z - expected))) < 1e-11
    assert float(jnp.max(jnp.abs(W - Z))) < 1e-11
    assert float(jnp.max(jnp.abs(g(jnp.arange(32) * (2 * jnp.pi / 32)) - W))) < 1e-11


def test_circle_polynomial_source_samples():
    C = chebfun(lambda t: jnp.exp(1j * jnp.pi * t), trig=True)
    W, Z = _poly_method(C, 0, 1, tol=1e-5)
    # Native first degree16 samples8*n, excluding left/including right.
    assert Z.shape == (128,)
    expected = jnp.exp(1j * jnp.pi * (-1 + jnp.arange(1, 129) * 2 / 128))
    assert float(jnp.max(jnp.abs(Z - expected))) < 1e-12
    assert float(jnp.max(jnp.abs(W - expected))) < 1e-12


@pytest.mark.parametrize('frequency', [-1, 2])
def test_native_winding_rejection(frequency):
    C = chebfun(lambda t: jnp.exp(frequency * 1j * jnp.pi * t), trig=True)
    with pytest.raises(ValueError, match='wind once counterclockwise'):
        conformal(C)


def test_array_adapter_circle():
    boundary = jnp.exp(2j * jnp.pi * jnp.arange(32) / 32)
    f, finv, *_ = conformal(boundary)
    z = jnp.asarray([0.0, 0.2 + 0.1j, -0.3j])
    assert float(jnp.max(jnp.abs(f(z) - z))) < 1e-11
    assert float(jnp.max(jnp.abs(finv(z) - z))) < 1e-11
