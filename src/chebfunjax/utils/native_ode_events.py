"""Shared native event brackets with solver-specific dense interpolation.

Provenance: installed R2025b private/odezero.m. Extracted from the existing
ode45 locator without changing event arithmetic or bracket decisions.
"""
import jax.numpy as jnp


def _event_values(event, t, y, size=None):
    value, terminal, direction = event(t, y)
    value = jnp.asarray(value, dtype=jnp.float64).reshape(-1, order='F')
    terminal = jnp.asarray(terminal).reshape(-1, order='F')
    direction = jnp.asarray(direction).reshape(-1, order='F')
    if direction.size == 0:
        direction = jnp.zeros_like(value)
    elif direction.size == 1:
        # odezero uses direction .* (vR-vL): MATLAB expands a scalar here.
        direction = jnp.broadcast_to(direction, value.shape)
    if (not value.size or direction.size != value.size
            or (size is not None and value.size != size)):
        raise ValueError('Events values and directions must have matching, fixed nonzero lengths')
    valid_terminal = (terminal == 0) | (terminal == 1)
    if terminal.size == value.size:
        # Chebop maxnorm produces +/-Inf values and 1+0*Inf == NaN flags
        # for disabled components. odezero never indexes an uncrossed flag.
        valid_terminal = valid_terminal | (jnp.isnan(terminal) & jnp.isinf(value))
    if not bool(jnp.all(~jnp.isnan(value)) & jnp.all(valid_terminal)
                & jnp.all((direction == -1) | (direction == 0) | (direction == 1))):
        raise ValueError('Events requires non-NaN values, binary active terminal flags '
                         'and directions -1/0/1')
    return value, terminal, direction


def locate_events(event, v, t, y, tnew, ynew, t0, interpolate):
    """Directional Illinois brackets, following installed R2025b odezero semantics.

    Host control owns bracket decisions; interpolation and event arithmetic
    use JAX arrays. The first-step terminal exception and right bracket
    endpoint are intentional native rules. Indices are MATLAB one-based.
    """
    tol = jnp.minimum(128*jnp.maximum(jnp.spacing(jnp.abs(t)),
                                     jnp.spacing(jnp.abs(tnew))), jnp.abs(tnew-t))
    tdir = jnp.sign(tnew-t)
    vnew, terminal, direction = _event_values(event, tnew, ynew, v.size)
    left, yl, vl = t, y, v
    right, yr, vr = tnew, ynew, vnew
    trial = right
    vt = vr
    times, values, indices = [], [], []
    def crossing(a, b):
        return [i for i in range(a.size)
                if bool((jnp.sign(a[i]) != jnp.sign(b[i])) & (direction[i]*(b[i]-a[i]) >= 0))]
    for _ in range(10000):
        moved = 0
        for _ in range(10000):
            active = crossing(vl, vr)
            if not active:
                if moved:
                    raise RuntimeError('ODE event bracket lost its crossing')
                return times, values, indices, vnew, False
            delta = right-left
            if bool(jnp.abs(delta) <= tol):
                break
            if bool(left == t) and any(bool((vl[i] == 0) & (vr[i] != 0)) for i in active):
                trial = left + tdir*.5*tol
            else:
                fraction = jnp.asarray(1.)
                for i in active:
                    if bool(vl[i] == 0):
                        maybe = (1-vr[i]*(trial-right)/((vt[i]-vr[i])*delta)
                                 if bool((tdir*trial > tdir*right) & (vt[i] != vr[i]))
                                 else jnp.asarray(.5))
                        if bool((maybe < 0) | (maybe > 1)):
                            maybe = jnp.asarray(.5)
                    elif bool(vr[i] == 0):
                        maybe = (vl[i]*(left-trial)/((vt[i]-vl[i])*delta)
                                 if bool((tdir*trial < tdir*left) & (vt[i] != vl[i]))
                                 else jnp.asarray(.5))
                        if bool((maybe < 0) | (maybe > 1)):
                            maybe = jnp.asarray(.5)
                    else:
                        maybe = -vl[i]/(vr[i]-vl[i])
                    fraction = jnp.minimum(fraction, maybe)
                change = jnp.maximum(.5*tol, jnp.minimum(fraction*jnp.abs(delta),
                                                        jnp.abs(delta)-.5*tol))
                trial = left + tdir*change
            yt = interpolate(trial)
            vt = _event_values(event, trial, yt, v.size)[0]
            if crossing(vl, vt):
                right, trial = trial, right
                yr, yt = yt, yr
                vr, vt = vt, vr
                if moved == 2:
                    half = .5*vl
                    vl = jnp.where(jnp.abs(half) >= jnp.finfo(jnp.float64).tiny, half, vl)
                moved = 2
            else:
                left, trial = trial, left
                yl, yt = yt, yl
                vl, vt = vt, vl
                if moved == 1:
                    half = .5*vr
                    vr = jnp.where(jnp.abs(half) >= jnp.finfo(jnp.float64).tiny, half, vr)
                moved = 1
        else:
            raise RuntimeError('ODE event bracket resource cap exceeded')
        for i in active:
            times.append(right)
            values.append(yr)
            indices.append(i+1)
        # odezero evaluates isterminal(indzc) before any(): a scalar flag
        # is not expanded. Check every index before reducing so an earlier
        # true flag cannot hide a later out-of-range index (JAX would clamp).
        if any(i >= terminal.size for i in active):
            raise IndexError('Events terminal flag index exceeds its output length')
        if any(bool(terminal[i]) for i in active):
            return times, values, indices, vnew, bool(left != t0)
        if bool(jnp.abs(tnew-right) <= tol):
            return times, values, indices, vnew, False
        trial, yt, vt = right, yr, vr
        left = right + tdir*.5*tol
        yl = interpolate(left)
        vl = _event_values(event, left, yl, v.size)[0]
        right, yr, vr = tnew, ynew, vnew
    raise RuntimeError('ODE event count resource cap exceeded')

