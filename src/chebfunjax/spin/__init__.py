"""Spin — exponential-integrator time-stepping for 1D and 2D periodic PDEs.

Translated from MATLAB Chebfun classes @spinop, @spinop2, and @expinteg
(commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from chebfunjax.spin.solver import spin
from chebfunjax.spin.solver2d import spin2
from chebfunjax.spin.spinop import SpinOp
from chebfunjax.spin.spinop2 import SpinOp2

__all__ = ["SpinOp", "spin", "SpinOp2", "spin2"]
