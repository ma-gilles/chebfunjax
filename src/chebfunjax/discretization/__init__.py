"""Spectral discretization classes.

Provides discretizations for differential operators used in BVP/ODE solving:

- :class:`ChebColloc2` — Chebyshev collocation on 2nd-kind points (default)
- :class:`ChebColloc1` — Chebyshev collocation on 1st-kind points
- :class:`TrigColloc` — Trigonometric collocation on equidistant points
"""

from chebfunjax.discretization.chebcolloc import ChebColloc1, ChebColloc2
from chebfunjax.discretization.trigcolloc import (
    TrigColloc,
    trig_cumsummat,
    trig_diffmat,
)

__all__ = [
    "ChebColloc1",
    "ChebColloc2",
    "TrigColloc",
    "trig_diffmat",
    "trig_cumsummat",
]
