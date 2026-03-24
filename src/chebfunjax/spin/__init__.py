"""Spin — exponential-integrator time-stepping for 1D periodic PDEs.

Translated from MATLAB Chebfun classes @spinop and @expinteg (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from chebfunjax.spin.solver import spin
from chebfunjax.spin.spinop import SpinOp

__all__ = ["SpinOp", "spin"]
