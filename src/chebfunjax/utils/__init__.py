"""Utility functions — quadrature, transforms, interpolation, polynomials, etc."""

from chebfunjax.utils.aaa import aaa, aaatrig
from chebfunjax.utils.conformal import conformal
from chebfunjax.utils.conformal2 import conformal2
from chebfunjax.utils.fov import fov
from chebfunjax.utils.gallery import gallery, list_gallery
from chebfunjax.utils.gpr import gpr
from chebfunjax.utils.phaseplot import phaseplot
from chebfunjax.utils.pswf import pswf, pswfpts
from chebfunjax.utils.lebesgue import lebesgue_constant, lebesgue_function
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
from chebfunjax.utils.quadrature import (
    chebpts,
    chebpts2,
    chebpts3,
    chebpts_ab,
    chebweights,
    paduapts,
)
from chebfunjax.utils.random import randnfundisk, randnfunsphere, smoothie
from chebfunjax.utils.ratapprox import padeapprox, ratinterp, trigratinterp
from chebfunjax.utils.sing import find_pole_order, find_sing_exponents, find_sing_order
from chebfunjax.utils.specfun import besselroots, gammaratio
from chebfunjax.utils.transforms import (
    cheb2jac,
    cheb2leg,
    chebcoeffs2chebvals,
    chebcoeffs2legcoeffs,
    chebvals2chebcoeffs,
    chebvals2chebvals,
    chebvals2legcoeffs,
    chebvals2legvals,
    coeffs2vals,
    jac2cheb,
    jac2jac,
    leg2cheb,
    legcoeffs2chebcoeffs,
    legcoeffs2chebvals,
    legcoeffs2legvals,
    legvals2chebcoeffs,
    legvals2chebvals,
    legvals2legcoeffs,
    ultra2ultra,
    ultracoeffs,
    vals2coeffs,
)
from chebfunjax.utils.trigutils import diffbarytrig, trigpoly

__all__ = [
    # AAA
    "aaa",
    "aaatrig",
    # conformal
    "conformal",
    "conformal2",
    # fov
    "fov",
    # gpr
    "gpr",
    # phaseplot
    "phaseplot",
    # pswf
    "pswf",
    "pswfpts",
    # gallery
    "gallery",
    "list_gallery",
    # lebesgue
    "lebesgue_function",
    "lebesgue_constant",
    # sing
    "find_pole_order",
    "find_sing_order",
    "find_sing_exponents",
    # ratapprox
    "padeapprox",
    "ratinterp",
    "trigratinterp",
    # quadrature
    "chebpts",
    "chebpts2",
    "chebpts3",
    "chebpts_ab",
    "chebweights",
    "paduapts",
    # transforms
    "cheb2leg",
    "leg2cheb",
    "cheb2jac",
    "jac2cheb",
    "jac2jac",
    "ultra2ultra",
    "ultracoeffs",
    "chebcoeffs2legcoeffs",
    "legcoeffs2chebcoeffs",
    "chebvals2legcoeffs",
    "legvals2chebcoeffs",
    "legvals2chebvals",
    "legvals2legcoeffs",
    "legcoeffs2chebvals",
    "legcoeffs2legvals",
    "chebvals2legvals",
    "chebvals2chebvals",
    "chebcoeffs2chebvals",
    "chebvals2chebcoeffs",
    "coeffs2vals",
    "vals2coeffs",
    # polynomials
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
    # trigutils
    "trigpoly",
    "diffbarytrig",
    # specfun
    "besselroots",
    "gammaratio",
    # random
    "smoothie",
    "randnfundisk",
    "randnfunsphere",
]
