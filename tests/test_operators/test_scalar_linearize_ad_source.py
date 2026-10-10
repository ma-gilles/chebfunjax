"""Exact scalar AD state/BC controls for native linearize.m; added regressions."""
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
from chebfunjax.operators.chebop import Chebop
from chebfunjax.operators.chebop_altdisc import LinearizedChebop, linearize_about


def test_scalar_nonlinear_nonlocal_action_and_shape():
    dom = (-1., .25, 1.)
    u = chebfun(lambda x: 2+x, domain=dom)
    h = chebfun(lambda x: 1-x*x, domain=dom)
    op = Chebop(lambda u: u**3 + u*u.sum(), domain=dom)
    L = op.linearize(u)
    result = L*h
    expected = 3*u**2*h + h*u.sum() + u*h.sum()
    assert isinstance(L, LinearizedChebop)
    assert isinstance(result, Chebfun)
    assert len(L.blocks) == len(L.blocks[0]) == 1
    assert float((result-expected).norm()) < 1e-12


@pytest.mark.parametrize('boundary', ['nonlinear', 'numeric'])
def test_scalar_homogeneous_correction_boundaries(boundary):
    dom = (-1., 1.)
    u = chebfun(lambda x: 2+x, domain=dom)
    op = Chebop(lambda u: u.diff(2), domain=dom)
    if boundary == 'nonlinear':
        op.lbc = lambda u: u**2-9
        op.rbc = lambda u: u**2-16
    else:
        op.lbc = 7
        op.rbc = -3
    L = linearize_about(op, u, n=33)
    result = L.solve(chebfun(-2., domain=dom))
    expected = chebfun(lambda x: 1-x*x, domain=dom)
    assert isinstance(result, Chebfun)
    assert float((result-expected).norm()) < 1e-10
    assert abs(float(result(-1.))) < 1e-12
    assert abs(float(result(1.))) < 1e-12
    assert L._n_bc == 2


@pytest.mark.parametrize('message,expected', [
    ('CHEBFUN:CHEBTECH:extrapolate:nansInfs: detail', 'CHEBFUN:CHEBOP:linearize:invalidInitialGuess'),
    ('deliberate callback error', 'deliberate callback error'),
])
def test_scalar_native_operator_error_classification(message, expected):
    def operator(u):
        raise ValueError(message)
    op = Chebop(operator, domain=(-1., 1.))
    with pytest.raises(ValueError, match=expected):
        op.linearize(chebfun(1., domain=(-1., 1.)))
