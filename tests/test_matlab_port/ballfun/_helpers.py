"""Shared helpers for the ballfun MATLAB-suite ports (Fable 5)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

EPS = float(np.finfo(np.float64).eps)
R0 = jnp.asarray(0.6)
L0 = jnp.asarray(0.7)
T0 = jnp.asarray(1.1)
X0 = float(R0 * jnp.cos(L0) * jnp.sin(T0))
Y0 = float(R0 * jnp.sin(L0) * jnp.sin(T0))
Z0 = float(R0 * jnp.cos(T0))


def val(f):
    return float(f(R0, L0, T0))
