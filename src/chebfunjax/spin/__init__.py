"""Spin — exponential-integrator time-stepping for 1D, 2D, 3D periodic and sphere PDEs.

Translated from MATLAB Chebfun classes @spinop, @spinop2, @spinop3, @spinopsphere,
and @expinteg (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from chebfunjax.spin.solver import spin
from chebfunjax.spin.solver2d import spin2
from chebfunjax.spin.solver3 import spin3, spinsphere
from chebfunjax.spin.spinop import SpinOp
from chebfunjax.spin.spinop2 import SpinOp2
from chebfunjax.spin.spinop3 import SpinOp3
from chebfunjax.spin.spinopsphere import SpinOpSphere

__all__ = [
    "SpinOp", "spin",
    "SpinOp2", "spin2",
    "SpinOp3", "spin3",
    "SpinOpSphere", "spinsphere",
]
