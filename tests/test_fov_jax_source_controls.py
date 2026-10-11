"""Supplemental JAX and preference controls for native fov.m, pin7574c77."""
import jax
import jax.numpy as jnp

from chebfunjax.chebfun1d.fov import _fov_curve, fov
from chebfunjax.chebpref import ChebfunPref


def test_nilpotent_circle_jit_and_derivative():
    matrix = jnp.array([[0., 2.], [0., 0.]])
    theta = jnp.array([.2, .7, 1.3])
    z = jax.jit(_fov_curve)(theta, matrix)
    assert jnp.max(jnp.abs(z - jnp.exp(-1j * theta))) < 10 * jnp.finfo(float).eps
    derivative = jax.grad(lambda t: _fov_curve(t, matrix)[0].real)(.7)
    assert abs(derivative + jnp.sin(.7)) < 10 * jnp.finfo(float).eps


def test_preferences_override_domain_and_splitting_without_mutation():
    pref = ChebfunPref(domain=[10., 20.], splitting=False)
    f = fov(7., pref=pref)
    assert f.domain.breakpoints == (0., float(2 * jnp.pi))
    assert jnp.max(jnp.abs(f(jnp.array([0., jnp.pi, 2*jnp.pi])) - 7)) < 70*jnp.finfo(float).eps
    assert list(pref.domain) == [10., 20.]
    assert not pref.splitting
