"""Independent analytic controls for source ODE113 terminal-history rebasing.

Provenance: installed R2025b ode113.m and private/odezero.m, not MATLAB output
fixtures. Bounds are inherited from native45 event and native113 polynomial controls.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.native_ode113 import native_ode113

EPS=np.finfo(float).eps

def unit(t,y):
    return jnp.ones_like(y)

@pytest.mark.parametrize('direction,end',[(0,.7),(1,.7),(-1,2.)])
def test_terminal_direction(direction,end):
    sol=native_ode113(unit,[0.,2.],[0.],{'Events':lambda t,y:(y[0]-.7,1,direction),'MaxStep':.4})
    assert abs(float(sol['x'][-1])-end)<256*EPS
    assert sol['xe'].size==(0 if direction==-1 else 1)
    if sol['xe'].size:
        np.testing.assert_array_equal(sol['y'][:,-1],sol['ye'][:,-1])
        np.testing.assert_array_equal(sol['ie'],[1])
        assert abs(float(sol['ye'][0,0])-.7)<256*EPS


def test_backward_crossing():
    sol=native_ode113(unit,[2.,0.],[2.],{'Events':lambda t,y:(y[0]-.7,1,-1),'MaxStep':.4})
    assert abs(float(sol['x'][-1])-.7)<256*EPS
    assert bool(jnp.all(jnp.diff(sol['x'])<0))


def test_initial_zero_does_not_stop():
    sol=native_ode113(unit,[0.,1.],[0.],{'Events':lambda t,y:(y[0],1,0)})
    assert float(sol['x'][-1])==1.
    assert sol['xe'].size==1
    assert float(sol['xe'][0])<256*EPS


def test_vector_nonterminal_then_terminal():
    def event(t,y):
        return jnp.array([y[0]-.3,y[0]-.7]),jnp.array([0,1]),jnp.array([0,0])
    sol=native_ode113(unit,[0.,2.],[0.],{'Events':event,'InitialStep':1.,'MaxStep':1.})
    np.testing.assert_array_equal(sol['ie'],[1,2])
    np.testing.assert_allclose(sol['xe'],[.3,.7],rtol=0,atol=256*EPS)
    assert abs(float(sol['x'][-1])-.7)<256*EPS

@pytest.mark.parametrize('reverse',[False,True])
def test_quadratic_terminal_dense_history(reverse):
    span=(1.,0.) if reverse else (0.,1.)
    initial=jnp.array([span[0],span[0]**2])
    sol=native_ode113(lambda t,y:jnp.array([1.,2*t]),span,initial,
                     {'Events':lambda t,y:(y[0]-.7,1,0),'RelTol':1e-8,'AbsTol':1e-10,'MaxStep':.4})
    assert abs(float(sol['x'][-1])-.7)<256*EPS
    assert int(sol['idata']['klastvec'][-1])>=2
    x=jnp.linspace(sol['x'][-2],sol['x'][-1],31)
    values,derivatives=sol['sol'](x,return_derivative=True)
    np.testing.assert_allclose(values,jnp.stack([x,x*x]),rtol=0,atol=100*EPS)
    np.testing.assert_allclose(derivatives,jnp.stack([jnp.ones_like(x),2*x]),rtol=0,atol=100*EPS)
    np.testing.assert_array_equal(sol['sol'](sol['x']),sol['y'])
    with pytest.raises(ValueError,match='outside integration interval'):
        sol['sol'](jnp.array([span[-1]]))

@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("reverse", [False, True])
def test_terminal_history_rebase_independent_polynomial(order, reverse):
    from math import comb

    from chebfunjax.utils.native_ode113 import _ntrp113, _State, _truncate_at_event

    step = -.125 if reverse else .125
    previous_time = .5
    accepted_time = previous_time + step
    event_time = previous_time + .6*step
    factor = 1.+.5j
    degree = order+1

    def polynomial(t):
        return factor*t**degree

    def derivative(t):
        return factor*degree*t**(degree-1)

    def phi_at(t):
        # Backward differences of the analytic derivative on an equal-step
        # grid give the Newton basis used by the Adams dense interpolant.
        differences = [sum((-1)**j*comb(i,j)*derivative(t-j*step)
                           for j in range(i+1)) for i in range(order+1)]
        return jnp.zeros((1,14), dtype=jnp.complex128).at[0,:order+1].set(jnp.array(differences))

    blank = _State(*(jnp.asarray(0.) for _ in _State._fields))
    def state(t):
        return blank._replace(t=jnp.asarray(t), y=jnp.array([polynomial(t)]),
                              phi=phi_at(t), psi=step*jnp.arange(1,13),
                              beta=jnp.ones(12),klast=jnp.asarray(order))
    previous,accepted=state(previous_time),state(accepted_time)
    shifted=_truncate_at_event(previous,accepted,jnp.asarray(event_time),jnp.array([polynomial(event_time)]))
    query=jnp.linspace(previous_time,event_time,19)
    values,derivatives=_ntrp113(query,shifted.t,shifted.y,shifted.klast,
                               shifted.phi,shifted.psi,return_derivative=True)
    np.testing.assert_allclose(values[0],polynomial(query),rtol=0,atol=100*EPS)
    np.testing.assert_allclose(derivatives[0],derivative(query),rtol=0,atol=100*EPS)
    assert bool(shifted.done)
    assert float(shifted.t)==event_time
