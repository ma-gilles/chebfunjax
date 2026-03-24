"""Utility functions — quadrature, transforms, interpolation, polynomials, etc."""

from chebfunjax.utils.aaa import aaa
from chebfunjax.utils.gallery import gallery, list_gallery
from chebfunjax.utils.lebesgue import lebesgue_constant, lebesgue_function
from chebfunjax.utils.sing import find_pole_order, find_sing_exponents, find_sing_order
from chebfunjax.utils.polynomials import (
    chebeval,
    chebpoly,
    hermeval,
    jaceval,
    jacpoly,
    lageval,
    legeval,
    legpoly,
    ultraeval,
    ultrapoly,
)
from chebfunjax.utils.quadrature import chebpts, chebpts_ab, chebweights
from chebfunjax.utils.ratapprox import padeapprox, ratinterp, trigratinterp
from chebfunjax.utils.transforms import (
    cheb2jac,
    cheb2leg,
    chebcoeffs2legcoeffs,
    chebvals2legcoeffs,
    coeffs2vals,
    jac2cheb,
    leg2cheb,
    legcoeffs2chebcoeffs,
    vals2coeffs,
)

__all__ = [
    "aaa",
    "gallery",
    "list_gallery",
    "lebesgue_function",
    "lebesgue_constant",
    "find_pole_order",
    "find_sing_order",
    "find_sing_exponents",
    "padeapprox",
    "ratinterp",
    "trigratinterp",
    "chebpts",
    "chebpts_ab",
    "chebweights",
    "cheb2leg",
    "leg2cheb",
    "cheb2jac",
    "jac2cheb",
    "chebcoeffs2legcoeffs",
    "legcoeffs2chebcoeffs",
    "chebvals2legcoeffs",
    "coeffs2vals",
    "vals2coeffs",
    "chebpoly",
    "chebeval",
    "legpoly",
    "legeval",
    "jacpoly",
    "jaceval",
    "ultrapoly",
    "ultraeval",
    "hermeval",
    "lageval",
]
