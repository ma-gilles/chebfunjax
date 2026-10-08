"""Spherefun — functions on the unit sphere."""

from chebfunjax.spherefun._constructor import spherefun as _spherefun_factory
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

spherefun = _spherefun_factory

__all__ = ["Spherefun", "Spherefunv", "spherefun"]
