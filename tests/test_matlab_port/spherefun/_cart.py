"""Cartesian-handle helpers for the spherefun MATLAB-suite ports (Fable 5):
MATLAB's ``spherefun(@(x,y,z) ...)`` is ``Spherefun.from_function`` of the
handle composed with ``x = cos(lam) sin(th)``, ``y = sin(lam) sin(th)``,
``z = cos(th)``."""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.spherefun.spherefun import Spherefun


def sph_xyz(f):
    """Spherefun of a Cartesian handle f(x, y, z)."""
    return Spherefun.from_function(
        lambda lam, th: f(jnp.cos(lam) * jnp.sin(th),
                          jnp.sin(lam) * jnp.sin(th), jnp.cos(th)))


def sph_lt(f):
    """Spherefun of a (lambda, theta) handle."""
    return Spherefun.from_function(f)
