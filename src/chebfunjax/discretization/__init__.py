"""Spectral discretization classes.

Provides discretizations for differential operators used in BVP/ODE solving:

- :class:`ChebColloc2` — Chebyshev collocation on 2nd-kind points (default)
- :class:`ChebColloc1` — Chebyshev collocation on 1st-kind points
"""

from chebfunjax.discretization.chebcolloc import ChebColloc1, ChebColloc2

__all__ = ["ChebColloc1", "ChebColloc2"]
