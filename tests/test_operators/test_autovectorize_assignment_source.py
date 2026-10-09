"""Small constructor/action controls for source autoVectorize slots2/4."""
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop
from tests.test_matlab_port.chebop.test_autoVectorize_matlab import (
    DOMAIN,
    _coefficient_operator,
)


@pytest.mark.parametrize('scale', [1., .2])
def test_late_assignment_action_and_capture(scale):
    fun = _coefficient_operator(scale)
    direct = Chebop(fun, DOMAIN)
    assigned = Chebop(domain=DOMAIN)
    assert assigned.op is None
    assigned.op = fun
    x = chebfun(lambda x: x, domain=DOMAIN)
    u = (5*x**2).sin()
    assert tuple(direct.domain) == tuple(assigned.domain) == DOMAIN
    assert direct._n_vars() == assigned._n_vars() == 1
    assert float((direct(x, u)-assigned(x, u)).norm()) == 0.
    f = (4*x).sin().exp()
    expected = u.diff(2)-scale*f*(1-u**2)*u.diff()+u
    assert float((assigned(x, u)-expected).norm()) == 0.
