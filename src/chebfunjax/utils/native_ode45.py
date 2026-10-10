"""JAX Dormand--Prince 5(4) mesh for the Chebfun2v ODE wrapper.

Provenance
----------
Algorithm: installed MATLAB R2017a ode45.m and private/odearguments.m.
Chebfun context: @chebfun2v/ode45.m, commit 7574c77.
Finite binary64 one-output mesh branch; mass matrices, output callbacks,
NonNegative and single precision are not yet implemented. This module returns
accepted mesh values, not a SciPy dense-output substitution. Native executable
capture remains unavailable; qualification uses independent source controls.
"""
import warnings
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from chebfunjax.utils.native_ode_events import _event_values

_A = (1/5, 3/10, 4/5, 8/9, 1., 1.)
_B = ((1/5, 3/40, 44/45, 19372/6561, 9017/3168, 35/384),
      (0., 9/40, -56/15, -25360/2187, -355/33, 0.),
      (0., 0., 32/9, 64448/6561, 46732/5247, 500/1113),
      (0., 0., 0., -212/729, 49/176, 125/192),
      (0., 0., 0., 0., -5103/18656, -2187/6784),
      (0., 0., 0., 0., 0., 11/84),
      (0., 0., 0., 0., 0., 0.))
_E = (71/57600, 0., -71/16695, 71/1920, -17253/339200, 22/525, -1/40)


class State(NamedTuple):
    t: jax.Array
    y: jax.Array
    f: jax.Array
    absh: jax.Array
    nfevals: jax.Array
    nfailed: jax.Array


class Trial(NamedTuple):
    absh: jax.Array
    h: jax.Array
    done: jax.Array
    nofailed: jax.Array
    accepted: jax.Array
    failed_at_minimum: jax.Array
    invalid: jax.Array
    nfevals: jax.Array
    nfailed: jax.Array
    tnew: jax.Array
    ynew: jax.Array
    f: jax.Array
    error: jax.Array


@partial(jax.jit, static_argnames=('fun',))
def _rhs(fun, t, y, args):
    return jnp.asarray(fun(t, y, *args)).reshape(-1)


@partial(jax.jit, static_argnames=('fun',))
def _stages(fun, t, y, h, f, tfinal, done, args):
    hA = h*jnp.asarray(_A, dtype=jnp.float64)
    hB = h*jnp.asarray(_B, dtype=jnp.float64)
    for k in range(5):
        f = f.at[:, k+1].set(fun(t+hA[k], y+f@hB[:, k], *args))
    tnew = jnp.where(done, tfinal, t+hA[5])
    purified_h = tnew-t
    ynew = y+f@hB[:, 5]
    f = f.at[:, 6].set(fun(tnew, ynew, *args))
    return tnew, ynew, purified_h, f


@partial(jax.jit, static_argnames=('fun', 'norm_control'))
def _step(fun, state, tfinal, direction, threshold, rtol, hmax, args, *, norm_control):
    hmin = 16*jnp.spacing(jnp.abs(state.t))
    absh = jnp.minimum(hmax, jnp.maximum(hmin, state.absh))
    done = 1.1*absh >= jnp.abs(tfinal-state.t)
    h = jnp.where(done, tfinal-state.t, direction*absh)
    trial = Trial(jnp.abs(h), h, done, jnp.asarray(True), jnp.asarray(False),
                  jnp.asarray(False), jnp.asarray(False), state.nfevals,
                  state.nfailed, state.t, state.y, state.f, jnp.asarray(0., dtype=jnp.float64))

    def attempt(a):
        tnew, ynew, h, f = _stages(fun, state.t, state.y, a.h, a.f, tfinal, a.done, args)
        error_vector = f@jnp.asarray(_E, dtype=jnp.float64)
        if norm_control:
            weight = jnp.maximum(jnp.maximum(jnp.linalg.norm(state.y),
                                             jnp.linalg.norm(ynew)), threshold)
            error = a.absh*(jnp.linalg.norm(error_vector)/weight)
        else:
            weight = jnp.maximum(jnp.maximum(jnp.abs(state.y), jnp.abs(ynew)), threshold)
            error = a.absh*jnp.max(jnp.abs(error_vector/weight))
        invalid = ~(jnp.all(jnp.isfinite(f)) & jnp.all(jnp.isfinite(ynew)) & jnp.isfinite(error))
        failed = error > rtol
        first = jnp.maximum(hmin, a.absh*jnp.maximum(.1, .8*(rtol/error)**(1/5)))
        again = jnp.maximum(hmin, .5*a.absh)
        reduced = jnp.where(a.nofailed, first, again)
        return Trial(jnp.where(failed, reduced, a.absh),
                     jnp.where(failed, direction*reduced, h), a.done & ~failed,
                     a.nofailed & ~failed, ~failed & ~invalid,
                     failed & (a.absh <= hmin), invalid,
                     a.nfevals+6, a.nfailed+failed.astype(jnp.int32), tnew, ynew, f, error)

    trial = jax.lax.while_loop(
        lambda a: ~(a.accepted | a.failed_at_minimum | a.invalid), attempt, trial)
    growth = 1.25*(trial.error/rtol)**(1/5)
    next_h = jnp.where(growth > .2, trial.absh/growth, 5*trial.absh)
    next_h = jnp.where(trial.nofailed, next_h, trial.absh)
    # Native FSAL: reuse the accepted seventh stage, without a new RHS call.
    f = trial.f.at[:, 0].set(trial.f[:, 6])
    return State(trial.tnew, trial.ynew, f, next_h, trial.nfevals, trial.nfailed), trial


# Dense extension coefficients for Dormand--Prince, R2017a ntrp45.m.
_BI = ((1, -183/64, 37/12, -145/128), (0, 0, 0, 0),
       (0, 1500/371, -1000/159, 1000/371),
       (0, -125/32, 125/12, -375/64),
       (0, 9477/3392, -729/106, 25515/6784),
       (0, -11/7, 11/3, -55/28), (0, 3/2, -4, 5/2))


@jax.jit
def _interpolate(tinterp, t, y, h, f):
    """Native quartic extension and its derivative, with ordered products."""
    s = (jnp.atleast_1d(tinterp)-t)/h
    bi = jnp.asarray(_BI, dtype=jnp.float64)
    powers = jnp.cumprod(jnp.stack((s, s, s, s)), axis=0)
    values = y[:, None] + (f@(h*bi))@powers
    derivatives = (f@bi)@jnp.concatenate(
        (jnp.ones_like(s)[None, :],
         jnp.cumprod(jnp.stack((2*s, 3/2*s, 4/3*s)), axis=0)), axis=0)
    return values, derivatives




def _locate_events(event, v, t, y, tnew, ynew, t0, h, f):
    """Preserve the ODE45 event API using its native dense polynomial."""
    from chebfunjax.utils.native_ode_events import locate_events

    return locate_events(event, v, t, y, tnew, ynew, t0,
                         lambda query: _interpolate(query, t, y, h, f)[0][:, 0])


def native_ode45(fun, tspan, y0, options=None, *, args=(), max_steps=100000):
    """Return accepted ``x``/``y`` mesh and statistics using JAX arithmetic.

    Python owns adaptive mesh storage; each complete attempted step, including
    rejection logic, uses a persistent JAX kernel. ``args`` carries dynamic
    field coefficients so different trajectories do not create new closures.
    """
    if not jax.config.x64_enabled:
        raise ValueError('native ode45 requires JAX x64')
    options = {} if options is None else dict(options)
    # ODESET option names are case-insensitive.
    names = ('RelTol', 'AbsTol', 'InitialStep', 'MaxStep', 'NormControl', 'Events')
    canonical = {name.lower(): name for name in names}
    options = {canonical.get(key.lower(), key): value for key, value in options.items()}
    supported = {'RelTol', 'AbsTol', 'InitialStep', 'MaxStep', 'NormControl', 'Events'}
    def empty(v):
        return v is None or (isinstance(v, (list, tuple, dict, str)) and not v) or getattr(v, 'size', None) == 0
    for key, value in options.items():
        if key not in supported and not empty(value):
            raise NotImplementedError(f'native ode45 option {key} is not yet implemented')
    def opt(key, default):
        value = options.get(key)
        return default if empty(value) else value
    span = jnp.asarray(tspan, dtype=jnp.float64).reshape(-1, order='F')
    if span.size < 2 or not bool(jnp.all(jnp.isfinite(span))):
        raise ValueError('ode45 requires at least two finite times')
    direction = 1. if float(span[-1]) > float(span[0]) else -1.
    if not bool(jnp.all(direction*jnp.diff(span) > 0)):
        raise ValueError('ode45 times must be strictly monotone')
    y = jnp.asarray(y0).reshape(-1, order='F')
    y = y.astype(jnp.complex128 if jnp.iscomplexobj(y) else jnp.float64)
    if not y.size or not bool(jnp.all(jnp.isfinite(y))):
        raise ValueError('ode45 requires a finite initial vector')
    f0 = _rhs(fun, span[0], y, args)
    if f0.shape != y.shape or not bool(jnp.all(jnp.isfinite(f0))):
        raise ValueError('ode45 RHS must return one finite value per component')
    y = y.astype(jnp.result_type(y, f0))
    f0 = f0.astype(y.dtype)
    rtol = jnp.asarray(opt('RelTol', 1e-3), dtype=jnp.float64)
    if rtol.ndim or not bool(jnp.isfinite(rtol) & (rtol > 0)):
        raise ValueError('RelTol must be a positive finite scalar')
    floor = 100*jnp.finfo(jnp.float64).eps
    if float(rtol) < floor:
        warnings.warn('RelTol increased to native 100*eps floor', stacklevel=2)
        rtol = jnp.asarray(floor, dtype=jnp.float64)
    atol = jnp.asarray(opt('AbsTol', 1e-6), dtype=jnp.float64).reshape(-1, order='F')
    norm_control = opt('NormControl', 'off')
    if norm_control not in ('on', 'off'):
        raise ValueError('NormControl must be on or off')
    norm_control = norm_control == 'on'
    if atol.size not in (1, y.size) or (norm_control and atol.size != 1):
        raise ValueError('AbsTol size does not match the state or NormControl')
    if not bool(jnp.all(jnp.isfinite(atol) & (atol > 0))):
        raise ValueError('AbsTol must be finite and positive')
    threshold = atol[0]/rtol if norm_control else jnp.broadcast_to(atol, y.shape)/rtol
    length = jnp.abs(span[-1]-span[0])
    hmax = jnp.minimum(length, jnp.abs(jnp.asarray(opt('MaxStep', .1*length), dtype=jnp.float64)))
    if hmax.ndim or not bool(jnp.isfinite(hmax) & (hmax > 0)):
        raise ValueError('MaxStep must have positive finite magnitude')
    hmin = 16*jnp.spacing(jnp.abs(span[0]))
    initial = opt('InitialStep', None)
    if initial is None:
        absh = jnp.minimum(hmax, jnp.abs(span[1]-span[0]))
        if norm_control:
            ratio = (jnp.linalg.norm(f0)/jnp.maximum(jnp.linalg.norm(y), threshold))/(.8*rtol**(1/5))
        else:
            ratio = jnp.max(jnp.abs(f0/jnp.maximum(jnp.abs(y), threshold)))/(.8*rtol**(1/5))
        absh = jnp.where(absh*ratio > 1, 1/ratio, absh)
        absh = jnp.maximum(absh, hmin)
    else:
        initial = jnp.abs(jnp.asarray(initial, dtype=jnp.float64))
        if initial.ndim or not bool(jnp.isfinite(initial) & (initial > 0)):
            raise ValueError('InitialStep must have positive finite magnitude')
        absh = jnp.minimum(hmax, jnp.maximum(hmin, initial))
    f = jnp.zeros((y.size, 7), dtype=y.dtype).at[:, 0].set(f0)
    state = State(span[0], y, f, absh, jnp.asarray(1, jnp.int32), jnp.asarray(0, jnp.int32))
    history = [state]
    event = opt('Events', None)
    event_times, event_states, event_indices = [], [], []
    if event is not None:
        if not callable(event):
            raise ValueError('Events must be callable')
        event_value = _event_values(event, state.t, state.y)[0]
    for _ in range(max_steps):
        candidate, trial = _step(fun, state, span[-1], jnp.asarray(direction), threshold,
                                 rtol, hmax, args, norm_control=norm_control)
        if bool(trial.invalid):
            raise RuntimeError('native ode45 encountered nonfinite state or derivative')
        if bool(trial.failed_at_minimum):
            warnings.warn('ode45 tolerance not met at minimum step; returning accepted partial mesh', stacklevel=2)
            break
        stopped = False
        if event is not None:
            te, ye, ie, event_value, stopped = _locate_events(
                event, event_value, state.t, state.y, trial.tnew, trial.ynew,
                span[0], trial.h, trial.f)
            event_times.extend(te)
            event_states.extend(ye)
            event_indices.extend(ie)
            if stopped:
                taux = state.t + (te[-1]-state.t)*jnp.asarray(_A)
                derivatives = _interpolate(taux, state.t, state.y, trial.h, trial.f)[1]
                adjusted = trial.f.at[:, 1:].set(derivatives)
                candidate = candidate._replace(t=te[-1], y=ye[-1], f=adjusted)
        state = candidate
        history.append(state)
        if stopped or bool(trial.done):
            break
    else:
        raise RuntimeError('native ode45 exceeded max_steps resource cap')
    result = {'solver': 'ode45', 'x': jnp.stack([s.t for s in history]),
            'y': jnp.stack([s.y for s in history], axis=1),
            'stats': {'nsteps': len(history)-1, 'nfailed': int(trial.nfailed),
                      'nfevals': int(trial.nfevals)},
            'extdata': {'options': options},
            'scope': 'R2017a finite one-output mesh; native executable capture unavailable'}
    if event is not None:
        result.update(xe=jnp.stack(event_times) if event_times else jnp.empty((0,)),
                      ye=jnp.stack(event_states, axis=1) if event_states else jnp.empty((y.size, 0)),
                      ie=jnp.asarray(event_indices, dtype=jnp.int32))
    return result
