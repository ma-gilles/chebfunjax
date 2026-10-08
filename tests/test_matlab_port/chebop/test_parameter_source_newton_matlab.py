"""Source C2 tolerance and mixed-block damping/refinement regression controls.

Source: Chebfun7574c77 tests/chebop/test_paramODE_nonlin_C2.m;
@chebop/solvebvpNonlinear.m; @linop/linsolve.m.
Manufactured complex and rank-deficient cases are independent controls.
"""
import pytest

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop


def test_adaptive_source_c2_bound():
    x = cj.chebfun(lambda t: t)
    op = Chebop(lambda x,u,a: (1-x**2)*u + .1*u.diff(2) + a*u.exp())
    op.lbc = lambda u,a: [u+a+1, u.diff()]
    op.rbc = lambda u,a: u-1
    op.init = [x,-1.]
    u,a = op.solve(0)
    residual = op.op(x,u,a).norm()
    error = residual + (u+a+1)(-1) + u.diff()(-1) + u(1)-1
    assert float(error) < 5e-10  # Literal source 1e3*factory bvpTol.
    assert float(residual) < 5e-10
    info = op._last_info
    print('C2 capture:', float(residual), a, info)
    assert info['converged']
    assert info['linearDimensions'][0] == 32
    assert info['linearDiscretizations'][0] == {
        'dimension': 32, 'functionPoints': 34, 'equationPoints': 32, 'diffOrder': 2}
    assert info['bvpTol'] == 5e-13
    assert info['newtonTolerance'] == 1e-10
    assert len(info['normDelta']) == len(info['dampingHistory'])


def test_complex_initial_function_and_parameter():
    x = cj.chebfun(lambda t:t)
    exact = (1+1j)*(x+2)
    op = Chebop(lambda x,u,a: u.diff(2)+u**2-((1+1j)*(x+2))**2+a-1j)
    op.lbc = lambda u,a: u-(1+1j)
    op.rbc = lambda u,a: [u-3*(1+1j), u.diff()-(1+1j)]
    op.init = [exact+.1*(1-x**2)**2, .25+.5j]
    assert op._has_explicit_scalar_parameters()
    u,a = op.solve(0)
    assert op._last_info['converged']
    assert abs(a-1j) < 1e-10
    assert float((u-exact).norm()) < 1e-10
    assert float(op.op(x,u,a).norm()) < 1e-10


def test_constant_boundary_row_retains_rank_failure():
    x = cj.chebfun(lambda t:t)
    op = Chebop(lambda x,u,a: u.diff(2)+u**2+a)
    op.lbc = lambda u,a:[u-1,0.]
    op.rbc = lambda u,a:u-3
    op.init = [x+2,0.]
    with pytest.warns(RuntimeWarning,match='did not converge'):
        op.solve(0,n=8)
    assert not op._last_info['converged']


def test_numeric_system_boundary_vector_and_length_error():
    x = cj.chebfun(lambda t:t)
    exact = (1+1j)*(x+2)
    op = Chebop(lambda x,u,a: u.diff(2)+u**2-((1+1j)*(x+2))**2+a-1j)
    op.lbc = [1+1j, 1j]
    op.rbc = lambda u,a:u-3*(1+1j)
    op.init = [exact+.1*(1-x**2), 1j]
    u,a = op.solve(0)
    assert op._last_info['converged']
    assert abs(a-1j) < 1e-10
    assert float((u-exact).norm()) < 1e-10
    op.lbc = 1+1j
    with pytest.raises(ValueError,match='one value per unknown'):
        op.solve(0)


def test_undamped_source_initial_iteration():
    x = cj.chebfun(lambda t:t)
    op = Chebop(lambda x,u,a:u.diff(2)+u**2-(x+2)**2+a-1j)
    op.lbc = lambda u,a:[u-1,u.diff()-1]
    op.rbc = lambda u,a:u-3
    op.init = [x+2,0.]
    op.damping = False
    u,a = op.solve(0,n=8)
    assert abs(a-1j) < 1e-10
    assert float(op.op(x,u,a).norm()) < 1e-10
    # Source's first undamped iteration computes the update but does not apply it.
    history = op._last_info['normDelta']
    assert history[0] == history[1]
    assert op._last_info['converged']
