"""chebfunjax.chebfun2d — 2D function approximation on rectangles."""

from chebfunjax.chebfun2d.chebfun2 import Chebfun2, chebfun2
from chebfunjax.chebfun2d.separable_approx import SeparableApprox

__all__ = [
    "SeparableApprox",
    "Chebfun2",
    "chebfun2",
]
