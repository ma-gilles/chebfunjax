"""Constant-positive-delay DDESD with JAX RK4 and Hermite history.

Provenance
----------
Algorithm: installed MATLAB R2025b Update2 funfun/ddesd.m, private/ntrp3h.m,
+matlab/+ode/+internal/+dde/lagvals.m (MathWorks, 1984-2024).
Chebfun context: DelayDifferentialEquations.m example9, Chebfun commit: 7574c77.
Finite real binary64, forward two-endpoint natural-mesh subset. Events,
functional delays, continuation histories, output callbacks/refinement,
norm control and single/complex arithmetic are explicitly unsupported.
"""
import warnings
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


def _vector(value):
    array = jnp.asarray(value)
    if (jnp.iscomplexobj(array)
            or (jnp.issubdtype(array.dtype, jnp.floating)
                and array.dtype != jnp.float64)):
        raise NotImplementedError('DDESD callbacks require real binary64 output')
    if array.ndim > 2 or (array.ndim == 2 and min(array.shape) != 1):
        raise NotImplementedError('DDESD states and callbacks must be vectors')
    return jnp.asarray(array, dtype=jnp.float64).reshape(-1)


def _bar(value):
    return jax.lax.optimization_barrier(value)


@jax.jit
def _hermite(query, left, y, right, ynew, yp, ypnew):
    """Source ntrp3h scalar query, including its derivative."""
    h = _bar(right-left)
    s = _bar((query-left)/h)
    s2 = _bar(s*s)
    s3 = _bar(s*s2)
    slope = _bar((ynew-y)/h)
    c = _bar(_bar(3*slope)-_bar(2*yp))-ypnew
    d = _bar(yp+ypnew)-_bar(2*slope)
    value = y+_bar(_bar(_bar(_bar(h*d)*s3)+_bar(_bar(h*c)*s2))
                  +_bar(_bar(h*yp)*s))
    derivative = yp+_bar(_bar(3*d)*s2+_bar(2*c)*s)
    return value, derivative


@partial(jax.jit, static_argnames=('history',))
def _lag_values(history, args, query, delays, times, values, slopes, count):
    """Source lagvals ownership; inactive capacity does not enter the search."""
    queries = jnp.minimum(query, query-delays)
    search_times = jnp.where(jnp.arange(times.size) < count, times, jnp.inf)

    def at(point):
        def from_mesh(_):
            index = jnp.searchsorted(search_times, point, side='right')-1
            index = jnp.clip(index, 0, jnp.maximum(0, count-2))
            value, _ = _hermite(point, times[index], values[:, index],
                                times[index+1], values[:, index+1],
                                slopes[:, index], slopes[:, index+1])
            return value
        return jax.lax.cond(
            point < times[0], lambda _: _vector(history(point, *args)),
            lambda _: jax.lax.cond(
                point == times[0], lambda _: values[:, 0],
                lambda _: jax.lax.cond(point == times[count-1],
                                      lambda _: values[:, count-1], from_mesh, None),
                None), None)
    result = jax.vmap(at)(queries).T
    implicit = (count > 1) & jnp.any(queries > times[jnp.maximum(0, count-2)])
    return result, implicit


@partial(jax.jit, static_argnames=('fun', 'history'))
def _trial(fun, history, args, delays, t, y, yp, h, tnew, first,
           threshold, rtol, nofailed, times, values, slopes, count,
           previous_new, previous_slope):
    """One native attempted step; accepted history has count entries."""
    predicted, predicted_slope = jax.lax.cond(
        first, lambda _: (y+h*yp, yp),
        lambda _: _hermite(tnew, t, y, tnew, previous_new, yp, previous_slope), None)
    times = times.at[count].set(tnew)
    values = values.at[:, count].set(predicted)
    slopes = slopes.at[:, count].set(predicted_slope)
    hB = h*jnp.asarray([0., .5, .5, 1.])
    hC = h*jnp.asarray([1/6, 1/3, 1/3, 1/6])
    stages = jnp.zeros((y.size, 4), dtype=y.dtype).at[:, 0].set(yp)

    def iteration(state):
        stages, values, slopes, implicit = state
        for j in range(1, 4):
            temp = y+hB[j]*stages[:, j-1]
            lagged, imp = _lag_values(history, args, t+hB[j], delays,
                                      times, values, slopes, count+1)
            stages = stages.at[:, j].set(_vector(fun(t+hB[j], temp, lagged, *args)))
            implicit = implicit | imp
        new = y+stages@hC
        lagged, imp = _lag_values(history, args, tnew, delays,
                                  times, values, slopes, count+1)
        slope = _vector(fun(tnew, new, lagged, *args))
        return stages, values.at[:, count].set(new), slopes.at[:, count].set(slope), implicit | imp

    state = iteration((stages, values, slopes, jnp.asarray(False)))
    implicit = state[3]
    state = jax.lax.cond(implicit, iteration, lambda state: state, state)
    _, values, slopes, _ = state
    new, slope = values[:, count], slopes[:, count]
    weights = jnp.maximum(jnp.maximum(jnp.abs(y), jnp.abs(new)), threshold)
    sm = jnp.asarray([.5-jnp.sqrt(3.)/6, .5+jnp.sqrt(3.)/6])

    def residual(i, error):
        point = t+h*sm[i]
        ym, ypm = _hermite(point, t, y, tnew, new, yp, slope)
        lagged, _ = _lag_values(history, args, point, delays,
                                times, values, slopes, count+1)
        res = ypm-_vector(fun(point, ym, lagged, *args))
        return 2.1342*jnp.maximum(error, h*jnp.max(jnp.abs(res/weights)))
    error = residual(0, jnp.asarray(0.))
    skip = (error > rtol) & ~nofailed
    error = jax.lax.cond(skip, lambda e: e, lambda e: residual(1, e), error)
    calls = 4+4*implicit.astype(jnp.int64)+1+(~skip).astype(jnp.int64)
    return new, slope, error, calls, times, values, slopes


@jax.jit
def _next_step(h, error, rtol, nofailed, accepted, hmin, hmax):
    """Native fourth-order acceptance/rejection step-size arithmetic."""
    reduced = jnp.where(nofailed, h*jnp.maximum(.5, .8*(rtol/error)**.25), h*.5)
    grown = h/jnp.maximum(.5, 1.25*(error/rtol)**.25)
    return jnp.where(accepted,
                     jnp.where(nofailed, jnp.minimum(jnp.maximum(hmin, grown), hmax), h),
                     jnp.maximum(hmin, reduced))


@dataclass
class DDESolution:
    """Natural mesh, states and Hermite slopes from the supported DDESD path.

    Provenance
    ----------
    MATLAB ddesd.m/odefinalize output; Chebfun commit: 7574c77.
    """
    x: jax.Array
    y: jax.Array
    yp: jax.Array
    nsteps: int
    nfailed: int
    nfevals: int
    success: bool
    solver: str = 'ddesd'


def ddesd(fun, delays, history, tspan, options=None, *args):
    """Solve a forward real system with positive constant delays.

    Supported options: RelTol, AbsTol, InitialStep, MaxStep, MinStep, InitialY.
    Options using events, functional delays, continuation, callbacks, norm
    control or output refinement are rejected. Natural mesh arrays use the
    source orientation: y and yp are component by time. History callables
    and the right-hand side must accept JAX arrays during tracing.

    Provenance
    ----------
    MATLAB R2025b Update2 ddesd.m, ntrp3h.m and lagvals.m; MathWorks.
    Chebfun commit: 7574c77, example DelayDifferentialEquations.m.
    """
    if not jax.config.x64_enabled:
        raise ValueError('ddesd requires JAX binary64 mode')
    if not callable(fun):
        raise TypeError('MATLAB:ddesd:DdefunNotFunctionHandle')
    if callable(delays):
        raise NotImplementedError('Functional/state-dependent delays are not implemented')
    if isinstance(history, (dict, DDESolution)):
        raise NotImplementedError('Continuation history is not implemented')
    options = {} if options is None else dict(options)
    supported = {'RelTol', 'AbsTol', 'InitialStep', 'MaxStep', 'MinStep', 'InitialY'}
    defaults = {'NormControl': 'off', 'Stats': 'off', 'Refine': 1}
    for name, value in options.items():
        if name not in supported and not (name in defaults and value == defaults[name]):
            raise NotImplementedError(f'ddesd option {name} is not implemented')

    def real_array(value, name):
        array = jnp.asarray(value)
        if (jnp.iscomplexobj(array)
            or (jnp.issubdtype(array.dtype, jnp.floating)
                and array.dtype != jnp.float64)):
            raise NotImplementedError(f'{name}: only real binary64 is implemented')
        array = jnp.asarray(array, dtype=jnp.float64)
        if not bool(jnp.all(jnp.isfinite(array))):
            raise ValueError(f'{name} must be finite')
        return array

    span = real_array(tspan, 'tspan').reshape(-1)
    if span.size != 2:
        raise NotImplementedError('Only two-endpoint natural-mesh output is implemented')
    t, final = span[0], span[1]
    if bool(final <= t):
        raise ValueError('MATLAB:ddesd:TspandEndLTtspan1')
    delays = _vector(real_array(delays, 'delays'))
    if not delays.size or bool(jnp.any(delays <= 0)):
        raise NotImplementedError('Only nonempty positive constant delays are implemented')
    if callable(history):
        history_fun = history
        initial = history(t, *args)
    else:
        initial = _vector(real_array(history, 'history'))
        def history_fun(time, *unused):
            return initial
    y = _vector(real_array(options.get('InitialY', initial), 'initial state'))
    if not 0 < y.size <= 8192:
        raise ValueError('Supported history dimension is1 through8192')
    rtol = real_array(options.get('RelTol', 1e-3), 'RelTol')
    if rtol.size != 1 or bool(jnp.any(rtol <= 0)):
        raise ValueError('MATLAB:ddesd:OptRelTolNotPosScalar')
    rtol = rtol.reshape(())
    minimum = 100*jnp.finfo(jnp.float64).eps
    if bool(rtol < minimum):
        rtol = jnp.asarray(minimum)
        warnings.warn('MATLAB:ddesd:RelTolIncrease', RuntimeWarning, stacklevel=2)
    atol = _vector(real_array(options.get('AbsTol', 1e-6), 'AbsTol'))
    if bool(jnp.any(atol <= 0)) or atol.size not in (1, y.size):
        raise ValueError('MATLAB:ddesd:AbsTolSize or OptAbsTolNotPos')
    threshold = atol/rtol
    length = final-t
    hmax0 = jnp.minimum(length, real_array(options.get('MaxStep', .1*length), 'MaxStep'))
    hmin_user = real_array(options.get('MinStep', 0.), 'MinStep')
    if hmax0.size != 1 or bool(hmax0 <= 0):
        raise ValueError('MATLAB:ddesd:OptMaxStepNotPos')
    if hmin_user.size != 1 or bool(hmin_user < 0) or ('MinStep' in options and bool(hmin_user == 0)):
        raise ValueError('MATLAB:odearguments:MinStepLEzero')
    hmax0 = hmax0.reshape(())
    hmin_user = jnp.minimum(length, hmin_user.reshape(()))
    if 'MaxStep' not in options:
        hmax0 = jnp.maximum(hmax0, hmin_user)
    elif bool(hmax0 < hmin_user):
        raise ValueError('MATLAB:odearguments:InconsistentMinStep')
    chunk = min(100, 8192//y.size)
    # Two slots are needed for tracing interpolation, even before step1.
    capacity = max(2, chunk)
    times = jnp.zeros(capacity).at[0].set(t)
    values = jnp.zeros((y.size, capacity)).at[:, 0].set(y)
    slopes = jnp.zeros_like(values)
    lagged, _ = _lag_values(history_fun, args, t, delays, times, values, slopes, 1)
    raw = real_array(fun(t, y, lagged, *args), 'right-hand side')
    if raw.size != y.size or (raw.ndim > 1 and raw.shape != (y.size, 1)):
        raise ValueError('MATLAB:ddesd:DDELengthMismatchHistory or DDEOutputNotCol')
    yp = raw.reshape(-1)
    slopes = slopes.at[:, 0].set(yp)
    tiny = 16*jnp.spacing(jnp.abs(t))
    hmin, hmax = jnp.maximum(tiny, hmin_user), jnp.maximum(tiny, hmax0)
    if 'InitialStep' in options:
        htry = real_array(options['InitialStep'], 'InitialStep')
        if htry.size != 1 or bool(htry <= 0):
            raise ValueError('MATLAB:ddesd:OptInitialStepNotPos')
        h = jnp.minimum(hmax, jnp.maximum(hmin, htry.reshape(())))
    elif bool(hmax0 == hmin_user):
        h = hmin_user
    else:
        h = jnp.minimum(hmax, length)
        rh = jnp.max(jnp.abs(yp/jnp.maximum(jnp.abs(y), threshold)))/(.8*rtol**.25)
        h = jnp.where(h*rh > 1, 1/rh, h)
        h = jnp.maximum(h, hmin)
    count, failed, calls = 1, 0, 1
    first, success = True, True
    previous_new, previous_slope = y, yp
    while bool(t < final):
        tiny = 16*jnp.spacing(jnp.abs(t))
        hmin, hmax = jnp.maximum(tiny, hmin_user), jnp.maximum(tiny, hmax0)
        h = jnp.minimum(hmax, jnp.maximum(hmin, h))
        distance = final-t
        last = bool(jnp.minimum(1.1*h, hmax) >= distance)
        h = distance if last else (distance/2 if bool(2*h >= distance) else h)
        nofailed = True
        if count == times.size:
            times = jnp.pad(times, (0, chunk))
            values = jnp.pad(values, ((0, 0), (0, chunk)))
            slopes = jnp.pad(slopes, ((0, 0), (0, chunk)))
        while True:
            tnew = final if last else t+h
            new, slope, error, ncall, times, values, slopes = _trial(
                fun, history_fun, args, delays, t, y, yp, h, tnew, first,
                threshold, rtol, nofailed, times, values, slopes, count,
                previous_new, previous_slope)
            # Source retains failed trial states for the next Hermite prediction.
            previous_new, previous_slope = new, slope
            calls += int(ncall)
            if not bool(jnp.isfinite(error) & jnp.all(jnp.isfinite(new)) & jnp.all(jnp.isfinite(slope))):
                raise FloatingPointError('ddesd nonfinite stage or error (adapter validation)')
            if bool(error <= rtol):
                break
            failed += 1
            if bool(h <= hmin):
                warnings.warn('MATLAB:ddesd:IntegrationTolNotMet', RuntimeWarning, stacklevel=2)
                success = False
                break
            h = _next_step(h, error, rtol, nofailed, False, hmin, hmax)
            nofailed, last = False, False
        if not success:
            break
        count += 1
        t, y, yp = tnew, new, slope
        if last:
            break
        first = False
        h = _next_step(h, error, rtol, nofailed, True, hmin, hmax)
    return DDESolution(times[:count], values[:, :count], slopes[:, :count],
                       count-1, failed, calls, success)
