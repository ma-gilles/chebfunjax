"""Actual solve controls for native dampingErrorBased.m93 (7574c77).

F(z)=z**3-2*z+2, z0=1: delta=-1; full trial deltaBar=-2;
contraction=2 and muPrime=1/4. Threshold1/2 forces full fallback0;
threshold1/8 permits reduced trial3/4, whose F=59/64 and muPrime=2/11.
max_iter=0 deliberately stops after that first accepted source iteration.
These are manufactured controls, not native MATLAB execution captures.
"""
from fractions import Fraction

import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators import _coupled_newton
from chebfunjax.operators.chebop import Chebop

# Same bound as the existing parameter cubic reduced-step control. Small
# fixed polynomial systems avoid approximation error; this is not a claim
# of a universal128eps linear-solve forward-error bound.
BOUND = 1e-10


@pytest.fixture(autouse=True)
def restore_session():
    previous = ChebopPref._defaults
    ChebopPref.setDefaults("factory")
    try:
        yield
    finally:
        ChebopPref._defaults = previous


def _oracle(threshold):
    full_delta = Fraction(-1)
    trial_delta = Fraction(-2)
    contraction = abs(trial_delta/full_delta)
    mu = Fraction(1, 2)*abs(full_delta)/abs(trial_delta)
    assert contraction == 2 and mu == Fraction(1, 4)
    step = min(mu, Fraction(1, 2))
    if step < threshold:
        return Fraction(0), Fraction(1), True
    value = 1-step
    residual = value**3-2*value+2
    assert residual == Fraction(59, 64)
    mu_prime = Fraction(1, 2)*step**2/abs(-residual+(1-step))
    assert mu_prime == Fraction(2, 11)
    assert residual < 1 and mu_prime < 4*step
    return value, step, False


def _check_history(op, threshold):
    expected, step, fallback = _oracle(threshold)
    info = op._last_info
    assert not info['converged']
    assert len(info['normDelta']) == len(info['dampingHistory']) == 1
    row = info['dampingHistory'][0]
    assert abs(row['lambda']-float(step)) < BOUND
    if fallback:
        assert row['contraction'] != row['contraction']  # Native cFactor=NaN.
    else:
        assert abs(row['contraction']-59/64) < BOUND
    # Fixed n is equation size8; record actual assembler metadata unchanged.
    dimensions = info['linearDimensions']
    assert dimensions
    assert all(d == 8 or d == [8] for d in dimensions)
    print({'threshold': float(threshold), 'expected': str(expected),
           'history': info['dampingHistory'],
           'linearDiscretizations': info['linearDiscretizations']}, flush=True)
    return float(expected)


@pytest.mark.parametrize('threshold', [Fraction(1, 2), Fraction(1, 8)],
                         ids=['fallback', 'reduced'])
def test_scalar_threshold_actual_solve(threshold):
    ChebopPref.setDefaults('lambdaMin', float(threshold))
    op = Chebop(lambda x, u: u**3-2*u+2)
    op.init = cj.chebfun(1)
    with pytest.warns(RuntimeWarning, match='maximum number of iterations|damping did not converge'):
        u = op.solve(0, n=8, max_iter=0)
    expected = _check_history(op, threshold)
    assert float((u-expected).norm()) < BOUND


@pytest.mark.parametrize('threshold', [Fraction(1, 2), Fraction(1, 8)],
                         ids=['fallback', 'reduced'])
def test_parameter_threshold_actual_solve(threshold):
    ChebopPref.setDefaults('lambdaMin', float(threshold))
    x = cj.chebfun(lambda t: t)
    op = Chebop(lambda x, u, a: u.diff(2)+a**3-2*a+2)
    op.lbc = lambda u, a: [u-1, u.diff()-1]
    op.rbc = lambda u, a: u-3
    op.init = [x+2, 1.0]
    assert op._has_explicit_scalar_parameters()
    with pytest.warns(RuntimeWarning, match='maximum number of iterations|damping did not converge'):
        u, a = op.solve(0, n=8, max_iter=0)
    expected = _check_history(op, threshold)
    assert abs(a-expected) < BOUND
    assert float((u-x-2).norm()) < BOUND


@pytest.mark.parametrize('threshold', [Fraction(1, 2), Fraction(1, 8)],
                         ids=['fallback', 'reduced'])
def test_coupled_threshold_actual_solve(threshold):
    ChebopPref.setDefaults('lambdaMin', float(threshold))
    op = Chebop(lambda x, u, v: [u**3-2*u+2, v-1])
    op.bc = lambda u, v: []
    op.init = [cj.chebfun(1), cj.chebfun(1)]
    assert op._bc_general is not None
    assert _coupled_newton.supported(op)
    with pytest.warns(RuntimeWarning, match='maximum number of iterations|damping did not converge'):
        u, v = op.solve(0, n=8, max_iter=0)
    expected = _check_history(op, threshold)
    assert float((u-expected).norm()) < BOUND
    assert float((v-1).norm()) < BOUND
