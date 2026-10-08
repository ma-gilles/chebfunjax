"""Reduced-step control for Chebfun7574c77 dampingErrorBased.

The manufactured solution is u=x+2,a=1. Starting a=.1 exercises the
predictor/corrector step reductions that the source C2 example does not need.
"""
import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop


def test_parameter_cubic_requires_reduced_steps():
    x = cj.chebfun(lambda t: t)
    op = Chebop(lambda x, u, a: u.diff(2) + a**3 - 1)
    op.lbc = lambda u, a: [u - 1, u.diff() - 1]
    op.rbc = lambda u, a: u - 3
    op.init = [x + 2, .1]
    u, a = op.solve(0, n=8)
    assert op._last_info['converged']
    assert abs(a - 1) < 1e-10
    assert float((u - x - 2).norm()) < 1e-10
    assert any(row['lambda'] < 1 for row in op._last_info['dampingHistory'])
