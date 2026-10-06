"""Whole source ODE orchestration controls with independent analytic trajectories.

Provenance
----------
MATLAB source : @chebfun/constructODEsol.m, @chebfun/odesol.m, @chebfun/join.m
Chebfun commit: 7574c77
"""
from types import SimpleNamespace

import jax.numpy as jnp

# uses-numpy: diagnostic host comparisons of JAX Chebfun outputs.
import numpy as np
import numpy.testing as npt
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import ode15s, ode45, ode78, ode89, ode113

EPS = np.finfo(float).eps
SOLVERS = [ode45, ode113, ode15s, ode78, ode89]


def linear_solver(calls, *, event=None):
    def solve(fun, span, initial, *args):
        del fun
        initial = jnp.atleast_1d(initial)
        calls.append((tuple(span), np.asarray(initial), args))
        end = span[-1]
        triggered = event is not None and span[0] < event < end
        if triggered:
            end = event
        slope = jnp.arange(1, initial.size+1, dtype=jnp.float64)

        def dense(x):
            return initial[:, None] + slope[:, None]*(jnp.atleast_1d(x)-span[0])

        return SimpleNamespace(y=dense(jnp.array([span[0], end])), sol=dense,
                               extdata={'options':args[0] if args else {}},
                               ie=[1] if triggered else [], xe=[end] if triggered else [])
    return solve


@pytest.mark.parametrize('entry', ['top', 'class', 'factory'])
@pytest.mark.parametrize('reverse', [False, True])
def test_source_restart_carries_state_and_returns_coupled_solution(entry, reverse):
    constructor = {'top':cj.constructODEsol, 'class':cj.Chebfun.constructODEsol,
                   'factory':cj.chebfun.constructODEsol}[entry]
    span = (1., .4, 0.) if reverse else (0., .4, 1.)
    initial = jnp.array([span[0], 2*span[0]])
    calls = []
    t, y = constructor(linear_solver(calls), lambda t,y:y, span, initial,
                       return_time=True)
    assert [c[0] for c in calls] == [span[:2], span[1:]]
    npt.assert_allclose(calls[1][1], [span[1], 2*span[1]], atol=100*EPS, rtol=0)
    x = jnp.linspace(0, 1, 17)
    npt.assert_allclose(y(x), np.stack([x,2*x],axis=-1), atol=100*EPS, rtol=0)
    npt.assert_allclose(t(x), x, atol=100*EPS, rtol=0)
    assert y.n_columns == 2 and y.domain.breakpoints == (0., .4, 1.)


def test_no_restart_passes_full_span_to_solver_and_keeps_domain_knots():
    calls = []
    options = {'restartSolver':False}
    y = cj.constructODEsol(linear_solver(calls), lambda t,y:y,
                          (0., .2, .6, 1.), [0.], options)
    assert len(calls) == 1 and calls[0][0] == (0., .2, .6, 1.)
    assert calls[0][2] == (options,)
    assert y.domain.breakpoints == (0., .2, .6, 1.)
    npt.assert_allclose(y(jnp.array([.1,.3,.8])), [.1,.3,.8], atol=100*EPS,rtol=0)


@pytest.mark.parametrize('return_time', [False, True])
def test_source_single_span_event_pads_nan_tail(return_time):
    calls = []
    result = cj.constructODEsol(linear_solver(calls,event=.6), lambda t,y:y,
                               (0.,1.), [0.,0.], return_time=return_time)
    y = result[1] if return_time else result
    npt.assert_allclose(y(jnp.array([.1,.5])), [[.1,.2],[.5,1.]],atol=100*EPS,rtol=0)
    assert bool(jnp.all(jnp.isnan(y(jnp.array([.8,1.])))))
    assert y.domain.breakpoints == (0.,.6,1.)
    if return_time:
        npt.assert_allclose(result[0](jnp.array([.1,.8])), [.1,.8],atol=100*EPS,rtol=0)


def test_source_later_restart_event_keeps_literal_time_tail_remapping():
    calls = []
    options = {'Events':lambda t,y:(y[0]-.8,1,1)}
    t,y = cj.constructODEsol(linear_solver(calls,event=.8),lambda t,y:y,
                             (0.,.4,.7,1.),[0.],options,return_time=True)
    assert len(calls)==3 and y.domain.breakpoints==(0.,.4,.7,.8,1.)
    assert bool(jnp.all(jnp.isnan(y(jnp.array([.9,1.])))))
    # Literal source tail domain/value [.4,1] is joined at event .8:
    # its interval length .6 makes time end1.4, while solution still ends1.
    assert t.domain.breakpoints==(0.,.4,.7,.8,1.4)
    npt.assert_allclose(t(jnp.array([.9,1.3])), [.5,.9],atol=100*EPS,rtol=0)


def test_source_no_restart_event_uses_second_knot_as_old_end():
    calls = []
    y = cj.constructODEsol(linear_solver(calls,event=.2),lambda t,y:y,
                          (0.,.4,1.),[0.],{'restartSolver':False})
    assert calls[0][0]==(0.,.4,1.)
    # Source replaces second knot, fits original later knot, then joins
    # [.2,.4] after the valid-domain end1, producing end1.2 literally.
    assert y.domain.breakpoints==(0.,.2,1.,1.2)
    assert bool(jnp.isnan(y(jnp.asarray(1.1))))


@pytest.mark.parametrize('span',[(1.,0.),(1.,.7,0.)])
def test_source_reverse_event_tail_keeps_increasing_domain_error(span):
    def reverse_event(fun,domain,initial,*args):
        del fun,initial,args
        event=.8
        return {'y':jnp.array([[domain[0],event]]),'sol':lambda x:jnp.atleast_1d(x)[None,:],
                'ie':[1],'xe':[event]}
    with pytest.raises(ValueError,match='increas'):
        cj.constructODEsol(reverse_event,lambda t,y:y,span,[1.],{'Events':lambda t,y:y})


def test_source_multiple_event_times_do_not_silently_pick_one():
    def multiple(fun,span,initial):
        del fun,initial
        return {'y':jnp.array([[0.,1.]]),'sol':lambda x:jnp.atleast_1d(x)[None,:],
                'ie':[1,1],'xe':[.3,.6]}
    with pytest.raises(ValueError,match='one scalar event time'):
        cj.constructODEsol(multiple,lambda t,y:y,(0.,1.),[0.])


@pytest.mark.parametrize('solver',SOLVERS)
def test_public_wrappers_restart_backward_and_preserve_coupled_complex_dynamics(solver):
    def rhs(t,y):
        return jnp.array([1j*y[0],-y[1]])
    t,y=solver(rhs,(1.,.4,0.),[np.exp(1j),np.exp(-1.)],
                {'RelTol':1e-10,'AbsTol':1e-12},return_time=True)
    x=jnp.linspace(0,1,29)
    exact=np.stack([np.exp(1j*np.asarray(x)),np.exp(-np.asarray(x))],axis=-1)
    npt.assert_allclose(y(x),exact,atol=2e-8,rtol=0)
    npt.assert_allclose(t(x),x,atol=100*EPS,rtol=0)
    assert y.n_columns==2 and y.domain.breakpoints==(0.,.4,1.)


@pytest.mark.parametrize('solver',SOLVERS)
def test_public_wrappers_translate_matlab_events_and_stop_restarting(solver):
    t,y=solver(lambda t,y:jnp.ones_like(y),(0.,.3,.5,1.),[0.],
                {'Events':lambda t,y:(y[0]-.7,1,1),
                 'RelTol':1e-10,'AbsTol':1e-12},return_time=True, backend='scipy')
    npt.assert_allclose(y(jnp.array([.1,.4,.6])), [.1,.4,.6],atol=100*EPS,rtol=0)
    assert bool(jnp.all(jnp.isnan(y(jnp.array([.8,1.])))))
    npt.assert_allclose(y.domain.breakpoints,(0.,.3,.5,.7,1.),atol=100*EPS,rtol=0)
    # Source time-tail [.3,1] is remapped to start at event .7.
    npt.assert_allclose(t.domain.b,1.4,atol=100*EPS,rtol=0)
    npt.assert_allclose(t(jnp.asarray(.9)),.5,atol=100*EPS,rtol=0)


@pytest.mark.parametrize('entry',['top','class','factory'])
def test_public_ode45_aliases_and_python_scalar_event_route(entry):
    def event(t,y):
        return y[0]-.6
    event.terminal=True
    event.direction=1
    solver={'top':cj.ode45,'class':cj.Chebfun.ode45,'factory':cj.chebfun.ode45}[entry]
    y=solver(lambda t,y:jnp.ones_like(y),(0.,.3,1.),[0.],events=event)
    npt.assert_allclose(y(jnp.asarray(.5)),.5,atol=100*EPS,rtol=0)
    assert bool(jnp.isnan(y(jnp.asarray(.8))))


def test_unsupported_native_option_is_rejected_instead_of_ignored():
    with pytest.raises(NotImplementedError,match='Mass'):
        ode45(lambda t,y:y,(0.,1.),[1.],{'Mass':jnp.eye(1)})


def test_wrapper_native_and_fit_default_tolerances_are_distinct(monkeypatch):
    import scipy.integrate

    from chebfunjax.utils import ode_solution
    solve = scipy.integrate.solve_ivp
    fit = ode_solution._ode_fit_parameters
    native=[]
    fitting=[]

    def record_native(*args,**kwargs):
        native.append((kwargs['rtol'],kwargs['atol'],kwargs['max_step']))
        return solve(*args,**kwargs)

    def record_fit(states,rel,abs_):
        fitting.append((rel,abs_))
        return fit(states,rel,abs_)

    monkeypatch.setattr(scipy.integrate,'solve_ivp',record_native)
    monkeypatch.setattr(ode_solution,'_ode_fit_parameters',record_fit)
    ode45(lambda t,y:jnp.ones_like(y),(0.,1.),[0.])
    assert native==[(1e-3,1e-6,.1)] and fitting==[(1e-6,1e-6)]
