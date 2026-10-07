"""Independent controls for initial operator error translation boundaries.

Provenance
----------
MATLAB source : @chebop/linearize.m:144-163
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Original: Copyright 2017 by The University of Oxford
    and The Chebfun Developers. See https://www.chebfun.org/.
"""
import pytest

from chebfunjax.operators.chebop import Chebop

INVALID='CHEBFUN:CHEBOP:linearize:invalidInitialGuess'
DIVZERO='CHEBFUN:CHEBFUN:rdivide:columnRdivide:divisionByZeroChebfun'
NANINF='CHEBFUN:CHEBTECH:extrapolate:nansInfs'


def test_zero_system_initial_guess_reciprocal_translation():
    N=Chebop(lambda x,u,v:[u.diff()+1/u,v.diff()],domain=(0.,1.))
    N.bc=lambda u,v:[u(0)-1,v(0)-2]
    with pytest.raises(ValueError) as caught:
        N._solve_nonlinear_system_fixed([1.,0.],n=4,max_iter=1)
    assert str(caught.value).startswith(INVALID+':')
    assert str(caught.value.__cause__).startswith(DIVZERO+':')


def test_valid_explicit_initial_guess_reciprocal_is_preserved():
    N=Chebop(lambda x,u,v:[u.diff()+1/u,v.diff()],domain=(0.,1.))
    N.bc=lambda u,v:[u(0)-1,v(0)-2]
    N.init=[1.,2.]
    solution=N._solve_nonlinear_system_fixed([1.,0.],n=4,max_iter=1)
    assert abs(solution[0](.37)-1)<1e-12
    assert abs(solution[1](.37)-2)<1e-12


@pytest.mark.parametrize('identifier',[DIVZERO,NANINF])
def test_exact_source_identifier_at_initial_operator_is_translated(identifier):
    error=ValueError(identifier+': diagnostic initial operator failure')
    def op(x,u,v):
        raise error
    N=Chebop(op,domain=(0.,1.))
    with pytest.raises(ValueError) as caught:
        N._solve_nonlinear_system_fixed(0.,n=4,max_iter=1)
    assert str(caught.value).startswith(INVALID+':')
    assert caught.value.__cause__ is error


def test_unrelated_initial_operator_error_is_unchanged():
    error=ValueError('user:domain: NaN or divisionByZero in unrelated message')
    def op(x,u,v):
        raise error
    N=Chebop(op,domain=(0.,1.))
    with pytest.raises(ValueError) as caught:
        N._solve_nonlinear_system_fixed(0.,n=4,max_iter=1)
    assert caught.value is error


def test_later_jacobian_probe_error_is_not_initial_guess_translation():
    error=ValueError(DIVZERO+': later operator-domain failure')
    def op(x,u,v):
        if float(u(0))!=0.:
            raise error
        return [u,v]
    N=Chebop(op,domain=(0.,1.))
    with pytest.raises(ValueError) as caught:
        N._solve_nonlinear_system_fixed(1.,n=4,max_iter=1)
    assert caught.value is error


def test_boundary_callback_error_is_not_operator_translation():
    error=ValueError(DIVZERO+': boundary callback failure')
    N=Chebop(lambda x,u,v:[u,v],domain=(0.,1.))
    def bc(u,v):
        raise error
    N.bc=bc
    with pytest.raises(ValueError) as caught:
        N._solve_nonlinear_system_fixed(0.,n=4,max_iter=1)
    assert caught.value is error



def test_actual_residual_boundary_error_is_not_operator_translation():
    error=ValueError(DIVZERO+': residual boundary callback failure')
    calls=[]
    N=Chebop(lambda x,u,v:[u,v],domain=(0.,1.))
    def bc(u,v):
        calls.append(1)
        if len(calls)==1:
            return [u(0),v(0)]  # Successful sizing call before residual(U).
        raise error
    N.bc=bc
    with pytest.raises(ValueError) as caught:
        N._solve_nonlinear_system_fixed(0.,n=4,max_iter=1)
    assert len(calls)==2
    assert caught.value is error
