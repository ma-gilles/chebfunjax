"""Literal natural adaptive ultraS predicate, separate from augmented n96.

MATLAB tests/chebop/test_paramODE_nonlin_US.m at7574c77.
No parameter differential equation or prescribed dimension.
"""
import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop


def test_original_natural_adaptive_ultras_predicate():
    x = cj.chebfun(lambda t: t, domain=(-1., 1.))
    operator = Chebop(lambda x, u, a: (1-x**2)*u+.1*u.diff(2)+a*u.exp())
    operator.lbc = lambda u, a: [u+a+1, u.diff()]
    operator.rbc = lambda u, a: u-1
    operator.init = [x, -1.]
    u, a = operator.solve(cj.chebfun(0.), discretization='ultraS')
    residual = operator.op(x, u, a)
    left, right = operator.lbc(u, a), operator.rbc(u, a)
    error = residual.norm()+left[0](-1.)+left[1](-1.)+right(1.)
    print('Native US predicate:', float(error), 'residual:', float(residual.norm()),
          'parameter:', a, 'metadata:', operator._last_info)
    assert float(error) < 1e3*5e-13
