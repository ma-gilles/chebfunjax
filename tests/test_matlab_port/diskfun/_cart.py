"""Cartesian-handle helpers for the diskfun MATLAB-suite ports (Fable 5):
MATLAB's ``diskfun(@(x,y) ...)`` is ``Diskfun.from_function`` of the handle
composed with ``x = r cos(t)``, ``y = r sin(t)``."""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.diskfun.diskfun import Diskfun


def disk_xy(f):
    """Diskfun of a Cartesian handle f(x, y)."""
    return Diskfun.from_function(
        lambda t, r: f(r * jnp.cos(t), r * jnp.sin(t)))


def disk_polar(f):
    """Diskfun of a polar handle f(theta, r)."""
    return Diskfun.from_function(f)
