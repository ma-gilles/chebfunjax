"""Exponential-integrator helpers (MATLAB ``expinteg`` class, Fable 5).

Provenance
----------
MATLAB source : @expinteg/phiFun.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and The
    Chebfun Developers.
"""

from __future__ import annotations

import math

import jax.numpy as jnp


def phi_fun(l: int):
    """The phi-function ``phi_l`` of exponential integrators as a callable,
    ``phi_0(z) = exp(z)`` and ``phi_l(z) = (phi_{l-1}(z) - 1/(l-1)!) / z``
    (MATLAB ``expinteg.phiFun(l)``).

    Provenance
    ----------
    MATLAB source : @expinteg/phiFun.m
    Chebfun commit: 7574c77
    """
    if int(l) == 0:
        return lambda z: jnp.exp(jnp.asarray(z))
    f = phi_fun(int(l) - 1)
    c = 1.0 / math.factorial(int(l) - 1)
    return lambda z: (f(z) - c) / jnp.asarray(z)
