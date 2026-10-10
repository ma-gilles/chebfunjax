"""JAX Adams PECE controller and native ode113 continuous output.

This initial port implements the ordinary finite one-output solver-structure
path. Mass matrices, NonNegative, output callbacks/refinement, single
precision and non-JAX-traceable callbacks require separate ports. Unsupported
nonempty options reject explicitly. This is not a MATLAB parity receipt.

Provenance
----------
MATLAB source : @chebfun/ode113.m and @chebfun/odesol.m (integration context).
Chebfun commit: 7574c77
Native algorithm: installed MATLAB R2025b ode113.m SHA256
8b137fe3a836725e5503829bb3a9636c830919944f3b637ab826f194cdbcaee9;
private/ntrp113.m SHA256
69da58404189f03f3ae608f9cc02ac5c3ecc1593defd951714265f8e789fd560.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

_GSTAR = (
    0.5000,
    0.0833,
    0.0417,
    0.0264,
    0.0188,
    0.0143,
    0.0114,
    0.00936,
    0.00789,
    0.00679,
    0.00592,
    0.00524,
    0.00468,
)


class _State(NamedTuple):
    t: jax.Array
    y: jax.Array
    phi: jax.Array
    psi: jax.Array
    alpha: jax.Array
    beta: jax.Array
    sig: jax.Array
    w: jax.Array
    v: jax.Array
    g: jax.Array
    k: jax.Array
    klast: jax.Array
    hlast: jax.Array
    ns: jax.Array
    phase1: jax.Array
    absh: jax.Array
    nfev: jax.Array
    nfailed: jax.Array
    done: jax.Array


class _Trial(NamedTuple):
    state: _State
    h: jax.Array
    done: jax.Array
    failed: jax.Array
    accepted: jax.Array
    tolerance_failed: jax.Array
    invalid: jax.Array
    prediction: jax.Array
    delta: jax.Array
    erk: jax.Array
    erkm1: jax.Array
    erkm2: jax.Array
    knew: jax.Array
    tnew: jax.Array


def _stack_history_entries(entries, block_size=256):
    """Stack native history with bounded compiler operand counts.

    This only copies entries in order.  The native ODE113 step/controller and
    dense interpolation equations are unchanged.  Native history fields have
    homogeneous shape, dtype and weak-type metadata; retain JAX's original
    promotion/error behavior for any nonhomogeneous private-helper input.
    """
    if block_size < 2:
        raise ValueError("History block size must be at least two.")
    if not entries:
        return jnp.stack(entries)
    first = entries[0]
    metadata = (first.shape, first.dtype, first.weak_type)
    if any((entry.shape, entry.dtype, entry.weak_type) != metadata
           for entry in entries[1:]):
        return jnp.stack(entries)
    blocks = [jnp.stack(entries[start:start + block_size])
              for start in range(0, len(entries), block_size)]
    while len(blocks) > 1:
        blocks = [jnp.concatenate(blocks[start:start + block_size], axis=0)
                  if len(blocks[start:start + block_size]) > 1 else blocks[start]
                  for start in range(0, len(blocks), block_size)]
    return blocks[0]


def _assemble_history(history):
    """Preserve the five native history layouts used by dense output.

    Native ODE113 collects t, y, klast, phi and psi at accepted steps.  Grouping
    stack/concatenate copies bounds JAX compiler graph width without changing
    any values, ordering, dtype, controller operation or interpolation formula.
    """
    return tuple(_stack_history_entries([getattr(state, field) for state in history])
                 for field in ("t", "y", "klast", "phi", "psi"))


def _coefficients(s, h):
    """Native variable-step Adams coefficients, source lines281–344."""
    ns = jnp.where(h != s.hlast, 0, s.ns)
    ns = jnp.where(ns <= s.klast, ns + 1, ns)
    s = s._replace(ns=ns)

    def update(s):
        k, ns = s.k, s.ns
        beta = s.beta.at[ns - 1].set(1.0)
        alpha = s.alpha.at[ns - 1].set(1.0 / ns)
        sig = s.sig.at[ns].set(1.0)

        def update_history(i, carry):
            psi, alpha, beta, sig, temp1 = carry
            temp2 = psi[i - 1]
            psi = psi.at[i - 1].set(temp1)
            temp1 = temp2 + h
            beta = beta.at[i].set(beta[i - 1] * psi[i - 1] / temp2)
            alpha = alpha.at[i].set(h / temp1)
            sig = sig.at[i + 1].set((i + 1) * alpha[i] * sig[i])
            return psi, alpha, beta, sig, temp1

        psi, alpha, beta, sig, temp1 = jax.lax.fori_loop(
            ns, k, update_history, (s.psi, alpha, beta, sig, h * ns)
        )
        psi = psi.at[k - 1].set(temp1)

        def first(_):
            indices = jnp.arange(1, 13, dtype=jnp.float64)
            v = jnp.where(indices <= k, 1.0 / (indices * (indices + 1)), s.v)
            return v, v, s.g

        def repeated(_):
            def raised(v):
                v = v.at[k - 1].set(1.0 / (k * (k + 1)))

                def diagonal(j, v):
                    i = k - j - 1
                    return v.at[i].set(v[i] - alpha[j] * v[i + 1])

                return jax.lax.fori_loop(1, ns - 1, diagonal, v)

            v = jax.lax.cond(k > s.klast, raised, lambda v: v, s.v)

            def update_v(i, carry):
                v, w = carry
                v = v.at[i].set(v[i] - alpha[ns - 1] * v[i + 1])
                return v, w.at[i].set(v[i])

            v, w = jax.lax.fori_loop(0, k + 1 - ns, update_v, (v, s.w))
            return v, w, s.g.at[ns].set(w[0])

        v, w, g = jax.lax.cond(ns == 1, first, repeated, operand=None)

        def update_g(i, carry):
            w, g = carry

            def update_w(iq, w):
                return w.at[iq].set(w[iq] - alpha[i - 1] * w[iq + 1])

            w = jax.lax.fori_loop(0, k + 1 - i, update_w, w)
            return w, g.at[i].set(w[0])

        w, g = jax.lax.fori_loop(ns + 1, k + 1, update_g, (w, g))
        return s._replace(psi=psi, alpha=alpha, beta=beta, sig=sig, v=v, w=w, g=g)

    return jax.lax.cond(s.k >= ns, update, lambda s: s, s)


@partial(jax.jit, static_argnames=("fun", "norm_control"))
def _advance_step(fun, s, tfinal, tdir, threshold, rtol, userhmin, userhmax, *, norm_control=False):
    """Advance one accepted native Adams step including all rejected trials."""
    gstar = jnp.asarray(_GSTAR, dtype=jnp.float64)
    tiny = 16 * jnp.spacing(jnp.abs(s.t))
    hmin, hmax = jnp.maximum(tiny, userhmin), jnp.maximum(tiny, userhmax)
    absh = jnp.minimum(hmax, jnp.maximum(hmin, s.absh))
    h = tdir * absh
    done = 1.1 * absh >= jnp.abs(tfinal - s.t)
    h = jnp.where(done, tfinal - s.t, h)
    s = s._replace(absh=jnp.abs(h))
    if norm_control:
        invwt = 1.0 / jnp.maximum(jnp.linalg.norm(s.y), threshold)

        def weighted_norm(v):
            return jnp.linalg.norm(v) * invwt
    else:
        invwt = 1.0 / jnp.maximum(jnp.abs(s.y), threshold)

        def weighted_norm(v):
            return jnp.max(jnp.abs(v * invwt))

    trial = _Trial(
        s,
        h,
        done,
        jnp.asarray(0, jnp.int32),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.zeros_like(s.y),
        jnp.zeros_like(s.y),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        s.k,
        s.t,
    )

    def attempt(trial):
        s, h = _coefficients(trial.state, trial.h), trial.h
        k, ns = s.k, s.ns
        columns = jnp.arange(14)
        beta = jnp.pad(s.beta, (0, 2), constant_values=1.0)
        phi = jnp.where((columns >= ns) & (columns < k), s.phi * beta, s.phi)
        phi = phi.at[:, k + 1].set(phi[:, k]).at[:, k].set(0.0)

        def predict(i, carry):
            p, phi = carry
            index = k - 1 - i
            p = p + s.g[index] * phi[:, index]
            phi = phi.at[:, index].set(phi[:, index] + phi[:, index + 1])
            return p, phi

        p, phi = jax.lax.fori_loop(0, k, predict, (jnp.zeros_like(s.y), phi))
        p = s.y + h * p
        tnew = jnp.where(trial.done, tfinal, s.t + h)
        yp = fun(tnew, p)
        delta = yp - phi[:, 0]
        temp3 = weighted_norm(delta)
        err = s.absh * (s.g[k - 1] - s.g[k]) * temp3
        erk = s.absh * s.sig[k] * gstar[k - 1] * temp3
        erkm1 = jnp.where(
            k >= 2, s.absh * s.sig[k - 1] * gstar[k - 2] * weighted_norm(phi[:, k - 1] + delta), 0.0
        )
        erkm2 = jnp.where(
            k >= 3, s.absh * s.sig[k - 2] * gstar[k - 3] * weighted_norm(phi[:, k - 2] + delta), 0.0
        )
        lower = ((k == 2) & (erkm1 <= 0.5 * erk)) | ((k > 2) & (jnp.maximum(erkm1, erkm2) <= erk))
        knew = jnp.where(lower, k - 1, k)
        rejected = err > rtol
        invalid = ~jnp.all(jnp.isfinite(yp)) | ~jnp.isfinite(err)
        tolerance_failed = rejected & (s.absh <= hmin)
        s = s._replace(phi=phi, nfev=s.nfev + 1, nfailed=s.nfailed + rejected.astype(jnp.int32))

        def restore(s):
            def undo(i, phi):
                return phi.at[:, i].set((phi[:, i] - phi[:, i + 1]) / s.beta[i])

            phi = jax.lax.fori_loop(0, k, undo, s.phi)

            def undo_psi(i, psi):
                return psi.at[i - 1].set(psi[i] - h)

            psi = jax.lax.fori_loop(1, k, undo_psi, s.psi)
            failed = trial.failed + 1
            reduce = jnp.where(failed > 3, jnp.minimum(0.5, jnp.sqrt(0.5 * rtol / erk)), 0.5)
            absh = jnp.maximum(reduce * s.absh, hmin)
            return s._replace(
                phi=phi,
                psi=psi,
                phase1=jnp.asarray(False),
                absh=absh,
                k=jnp.where(failed == 3, 1, knew),
            )

        retry = rejected & ~tolerance_failed & ~invalid
        s = jax.lax.cond(retry, restore, lambda s: s, s)
        return _Trial(
            s,
            jnp.where(retry, tdir * s.absh, h),
            jnp.where(retry, False, trial.done),
            trial.failed + rejected.astype(jnp.int32),
            ~rejected & ~invalid,
            tolerance_failed,
            invalid,
            p,
            delta,
            erk,
            erkm1,
            erkm2,
            knew,
            tnew,
        )

    trial = jax.lax.while_loop(
        lambda tr: ~tr.accepted & ~tr.tolerance_failed & ~tr.invalid, attempt, trial
    )

    def correct(tr):
        s, k, h = tr.state, tr.state.k, tr.h
        y = tr.prediction + h * s.g[k] * tr.delta
        yp = fun(tr.tnew, y)
        phi = s.phi.at[:, k].set(yp - s.phi[:, 0])
        phi = phi.at[:, k + 1].set(phi[:, k] - phi[:, k + 1])
        phi = jnp.where(jnp.arange(14) < k, phi + phi[:, k, None], phi)
        phase1 = s.phase1 & (tr.knew != k - 1) & (k != 12)
        erkp1 = s.absh * gstar[k] * weighted_norm(phi[:, k + 1])
        raise_order = ((k == 1) & (erkp1 < 0.5 * tr.erk)) | ((k > 1) & (k < 12) & (erkp1 < tr.erk))
        lower_order = (k > 1) & (tr.erkm1 <= jnp.minimum(tr.erk, erkp1))
        higher_k = jnp.where(lower_order, k - 1, jnp.where(raise_order, k + 1, k))
        higher_erk = jnp.where(lower_order, tr.erkm1, jnp.where(raise_order, erkp1, tr.erk))
        estimate_higher = k + 1 <= s.ns
        knext = jnp.where(
            phase1,
            k + 1,
            jnp.where(tr.knew == k - 1, k - 1, jnp.where(estimate_higher, higher_k, k)),
        )
        erk = jnp.where(tr.knew == k - 1, tr.erkm1, jnp.where(estimate_higher, higher_erk, tr.erk))
        reduce = (0.5 * rtol / erk) ** (1.0 / (knext + 1))
        absh = jnp.where(
            phase1 | (0.5 * rtol >= erk * 2.0 ** (knext + 1)),
            2 * s.absh,
            jnp.where(
                0.5 * rtol < erk, s.absh * jnp.maximum(0.5, jnp.minimum(0.9, reduce)), s.absh
            ),
        )
        return s._replace(
            t=tr.tnew,
            y=y,
            phi=phi,
            k=knext,
            klast=k,
            hlast=h,
            phase1=phase1,
            absh=jnp.where(tr.done, s.absh, absh),
            nfev=s.nfev + 1,
            done=tr.done,
        )

    out = jax.lax.cond(trial.accepted, correct, lambda tr: tr.state, trial)
    invalid = trial.invalid | ~jnp.all(jnp.isfinite(out.y))
    return out, trial.tolerance_failed, invalid


def _interpolate_point(time, tnew, ynew, klast, phi, psi):
    hi = time - tnew
    w = 1.0 / jnp.arange(1, 14, dtype=jnp.float64)
    g = jnp.zeros(13, dtype=jnp.float64).at[0].set(1.0)
    rho = g

    def update(j, carry):
        w, g, rho, term = carry
        gamma = (hi + term) / psi[j - 1]
        eta = hi / psi[j - 1]

        def update_w(i, w):
            return w.at[i].set(gamma * w[i] - eta * w[i + 1])

        w = jax.lax.fori_loop(0, klast + 1 - j, update_w, w)
        return w, g.at[j].set(w[0]), rho.at[j].set(gamma * rho[j - 1]), psi[j - 1]

    w, g, rho, term = jax.lax.fori_loop(1, klast + 1, update, (w, g, rho, jnp.asarray(0.0)))
    return ynew + hi * (phi[:, :13] @ g), phi[:, :13] @ rho


@partial(jax.jit, static_argnames=("return_derivative",))
def _ntrp113(times, tnew, ynew, klast, phi, psi, *, return_derivative=False):
    """Evaluate the literal native Adams polynomial, including extrapolation.

    Inputs are one accepted right endpoint's order, phi and psi history.
    Returns component-by-query values, and optionally their time derivative.

    Provenance
    ----------
    MATLAB source : installed R2025b private/ntrp113.m; @chebfun/odesol.m context.
    Chebfun commit: 7574c77
    """
    values, derivatives = jax.vmap(_interpolate_point, in_axes=(0, None, None, None, None, None))(
        jnp.atleast_1d(times), tnew, ynew, klast, phi, psi
    )
    return (values.T, derivatives.T) if return_derivative else values.T


def _truncate_at_event(previous, accepted, time, value):
    """Rebase terminal Adams history using ode113.m's pre-step phi/psi.

    Provenance: installed MATLAB R2025b ode113.m lines541-566. The event
    derivative comes from the accepted polynomial, not a new RHS evaluation.
    """
    order = int(accepted.klast)
    _, derivative = _ntrp113(time, accepted.t, accepted.y, accepted.klast,
                            accepted.phi, accepted.psi, return_derivative=True)
    psi = previous.psi
    beta = accepted.beta.at[0].set(1.0)
    step = time - previous.t
    temp = step
    for i in range(1, order):
        prior = psi[i - 1]
        psi = psi.at[i - 1].set(temp)
        temp = prior + step
        beta = beta.at[i].set(beta[i - 1] * psi[i - 1] / prior)
    psi = psi.at[order - 1].set(temp)
    phi = previous.phi
    phi = phi.at[:, 1:order].set(phi[:, 1:order] * beta[None, 1:order])
    shifted = jnp.concatenate((derivative, -phi[:, :order + 1]), axis=1)
    phi = phi.at[:, :order + 2].set(jnp.cumsum(shifted, axis=1))
    return accepted._replace(t=time, y=value, psi=psi, phi=phi, beta=beta,
                             done=jnp.asarray(True))


def _prepare_rhs(odefun):
    """Normalize an RHS once per public solve, shared across its restarts.

    A fresh wrapper for each independent solve preserves closure recapture.
    Source arithmetic is unchanged; only the JAX static callable identity is
    shared within constructODEsol's restart loop.
    """
    def rhs(t, y):
        return jnp.atleast_1d(jnp.asarray(odefun(t, y)))
    return rhs


def native_ode113(odefun, tspan, y0, options=None, *, max_steps=100000,
                  _prepared_rhs=None):
    """Solve a finite IVP using native variable-step Adams PECE orders1..12.

    Produces the one-output solver structure consumed by public ODESOL.
    Unsupported options reject explicitly; this initial backend does not
    implement mass/NonNegative/output callbacks. Native events use the Adams
    interpolant and terminal-history rebasing. max_steps is
    a Python resource cap, not a MATLAB algorithm parameter. The adaptive
    driver is eager; each accepted/rejected step and dense evaluation are JAX.

    Provenance
    ----------
    MATLAB source : installed R2025b ode113.m and private/odearguments.m;
    @chebfun/ode113.m and @chebfun/odesol.m integration context.
    Chebfun commit: 7574c77
    """
    import warnings

    if not jax.config.x64_enabled:
        raise ValueError("native ode113 port requires JAX x64")
    options = {} if options is None else dict(options)
    known = {"RelTol", "AbsTol", "InitialStep", "MaxStep", "MinStep", "NormControl", "Events"}

    def empty(v):
        return (
            v is None
            or (isinstance(v, (list, tuple, dict, str)) and not v)
            or getattr(v, "size", None) == 0
        )

    for key, value in options.items():
        if key not in known and not empty(value):
            raise NotImplementedError(f"native ode113 option {key} is not yet implemented")

    def opt(key, default):
        value = options.get(key)
        return default if empty(value) else value

    # odearguments converts tspan with tspan(:), in column-major order.
    span = jnp.asarray(tspan, dtype=jnp.float64).reshape(-1, order="F")
    if span.ndim != 1 or span.size < 2 or not bool(jnp.all(jnp.isfinite(span))):
        raise ValueError("native ode113 requires at least two finite times")
    direction = 1.0 if float(span[-1]) > float(span[0]) else -1.0
    if not bool(jnp.all(direction * jnp.diff(span) > 0)):
        raise ValueError("native ode113 times must be strictly monotone")
    # Source y0(:) accepts row, column and matrix-shaped initial data.
    y = jnp.asarray(y0).reshape(-1, order="F")
    if y.ndim != 1 or y.size == 0 or not bool(jnp.all(jnp.isfinite(y))):
        raise ValueError("native ode113 requires a finite initial state vector")
    y = y.astype(jnp.complex128 if jnp.iscomplexobj(y) else jnp.float64)

    rhs = _prepare_rhs(odefun) if _prepared_rhs is None else _prepared_rhs

    f0 = rhs(span[0], y)
    if f0.shape != y.shape or not bool(jnp.all(jnp.isfinite(f0))):
        raise ValueError("native ode113 RHS must return one finite derivative per component")
    y = y.astype(jnp.result_type(y, f0))
    f0 = f0.astype(y.dtype)
    rtol = jnp.asarray(opt("RelTol", 1e-3), dtype=jnp.float64)
    if rtol.ndim != 0 or not bool(jnp.isfinite(rtol) & (rtol > 0)):
        raise ValueError("RelTol must be a finite positive scalar")
    if float(rtol) < 100 * jnp.finfo(jnp.float64).eps:
        warnings.warn("RelTol increased to native100*eps floor", stacklevel=2)
        rtol = jnp.asarray(100 * jnp.finfo(jnp.float64).eps)
    atol = jnp.asarray(opt("AbsTol", 1e-6), dtype=jnp.float64)
    # odearguments checks length(atol)==neq before column-major atol(:).
    # Retain that source length guard rather than accepting any reshaped matrix.
    if atol.size != 1 and max(atol.shape) != y.size:
        raise ValueError("AbsTol must be scalar or one per component")
    atol = atol.reshape(-1, order="F")
    norm_control = opt("NormControl", "off")
    if isinstance(norm_control, str):
        if norm_control.lower() not in ("on", "off"):
            raise ValueError("NormControl must be on or off")
        norm_control = norm_control.lower() == "on"
    norm_control = bool(norm_control)
    if atol.ndim != 1 or atol.size not in (1, y.size) or (norm_control and atol.size != 1):
        raise ValueError("AbsTol must be scalar or one per component; NormControl requires scalar")
    if not bool(jnp.all(jnp.isfinite(atol) & (atol > 0))):
        raise ValueError("AbsTol must be finite and positive")
    threshold = atol[0] / rtol if norm_control else jnp.broadcast_to(atol, y.shape) / rtol
    t0, tfinal = span[0], span[-1]
    tlen = jnp.abs(tfinal - t0)
    safehmax = 16 * jnp.finfo(jnp.float64).eps * jnp.maximum(jnp.abs(t0), jnp.abs(tfinal))
    userhmax = jnp.asarray(opt("MaxStep", jnp.maximum(0.1 * tlen, safehmax)), dtype=jnp.float64)
    userhmin = jnp.asarray(opt("MinStep", 0.0), dtype=jnp.float64)
    if not bool(jnp.isfinite(userhmax) & (userhmax > 0)) or not bool(
        jnp.isfinite(userhmin) & (userhmin >= 0)
    ):
        raise ValueError("MaxStep must be positive and MinStep nonnegative")
    if not empty(options.get("MinStep")) and float(userhmin) == 0:
        raise ValueError("explicit MinStep must be positive")
    userhmin = jnp.minimum(userhmin, tlen)
    if not empty(options.get("MaxStep")):
        userhmax = jnp.minimum(userhmax, tlen)
    if empty(options.get("MaxStep")):
        userhmax = jnp.maximum(userhmax, userhmin)
    elif float(userhmax) < float(userhmin):
        raise ValueError("MaxStep must be at least MinStep")
    tiny = 16 * jnp.spacing(jnp.abs(t0))
    hmin, hmax = jnp.maximum(tiny, userhmin), jnp.maximum(tiny, userhmax)
    initial = opt("InitialStep", None)
    if initial is None and float(userhmax) == float(userhmin):
        initial = userhmin
    if initial is None:
        absh = jnp.minimum(hmax, jnp.abs(span[1] - span[0]))
        if norm_control:
            rh = (jnp.linalg.norm(f0) / jnp.maximum(jnp.linalg.norm(y), threshold)) / (
                0.25 * jnp.sqrt(rtol)
            )
        else:
            rh = jnp.max(jnp.abs(f0 / jnp.maximum(jnp.abs(y), threshold))) / (0.25 * jnp.sqrt(rtol))
        absh = jnp.where(absh * rh > 1, 1 / rh, absh)
        absh = jnp.maximum(absh, hmin)
    else:
        initial = jnp.abs(jnp.asarray(initial, dtype=jnp.float64))
        if initial.ndim != 0 or not bool(jnp.isfinite(initial) & (initial > 0)):
            raise ValueError("InitialStep must be finite and nonzero")
        absh = jnp.minimum(hmax, jnp.maximum(hmin, initial))
    real12 = jnp.zeros(12, dtype=jnp.float64)
    real13 = jnp.zeros(13, dtype=jnp.float64)
    one = jnp.asarray(1, jnp.int32)
    zero = jnp.asarray(0, jnp.int32)
    state = _State(
        t0,
        y,
        jnp.zeros((y.size, 14), dtype=y.dtype).at[:, 0].set(f0),
        real12,
        real12,
        real12,
        real13.at[0].set(1.0),
        real12,
        real12,
        real13.at[0].set(1.0).at[1].set(0.5),
        one,
        zero,
        jnp.asarray(0.0),
        zero,
        jnp.asarray(True),
        absh,
        one,
        zero,
        jnp.asarray(False),
    )
    from chebfunjax.utils.native_ode_events import _event_values, locate_events

    event = opt("Events", None)
    event_times, event_states, event_indices = [], [], []
    if event is not None:
        if not callable(event):
            raise ValueError("Events must be callable")
        event_value = _event_values(event, state.t, state.y)[0]
    history = [state]
    for _ in range(max_steps):
        previous = state
        state, tolerance_failed, invalid = _advance_step(
            rhs,
            state,
            tfinal,
            jnp.asarray(direction),
            threshold,
            rtol,
            userhmin,
            userhmax,
            norm_control=norm_control,
        )
        if bool(tolerance_failed):
            # Source warns and finalizes the accepted prefix, never appending
            # the rejected trial's predicted phi/psi history to the result.
            warnings.warn(
                "native ode113 tolerance cannot be met at minimum step; "
                "returning accepted partial solution",
                stacklevel=2,
            )
            break
        if bool(invalid):
            raise RuntimeError("native ode113 encountered nonfinite RHS/state")
        if event is not None:
            te, ye, ie, event_value, stopped = locate_events(
                event, event_value, previous.t, previous.y, state.t, state.y, t0,
                lambda query: _ntrp113(query, state.t, state.y, state.klast,
                                       state.phi, state.psi)[:, 0])
            event_times.extend(te)
            event_states.extend(ye)
            event_indices.extend(ie)
            if stopped:
                state = _truncate_at_event(previous, state, te[-1], ye[-1])
        history.append(state)
        if bool(state.done):
            break
    else:
        raise RuntimeError("native ode113 exceeded Python max_steps resource cap")
    mesh, values, orders, phis, psis = _assemble_history(history)

    @partial(jax.jit, static_argnames=("return_derivative",))
    def dense_kernel(times, *, return_derivative=False):
        queries = jnp.atleast_1d(jnp.asarray(times, dtype=jnp.float64))
        if mesh.size == 1:
            if return_derivative:
                raise ValueError("native DEVAL derivative requires an accepted interval")
            return jnp.broadcast_to(values[0, :, None], (values.shape[1], queries.size))
        indices = jnp.clip(
            jnp.searchsorted(direction * mesh, direction * queries, side="right"), 1, mesh.size - 1
        )
        ys, dys = jax.vmap(_interpolate_point)(
            queries, mesh[indices], values[indices], orders[indices], phis[indices], psis[indices]
        )
        # Native DEVAL returns stored state exactly at mesh nodes; its
        # derivative uses the interval to the right (left at final endpoint).
        exact_indices = jnp.clip(
            jnp.searchsorted(direction * mesh, direction * queries, side="left"), 0, mesh.size - 1
        )
        on_mesh = queries == mesh[exact_indices]
        ys = jnp.where(on_mesh[:, None], values[exact_indices], ys)
        return (ys.T, dys.T) if return_derivative else ys.T

    def dense(times, *, return_derivative=False):
        queries = jnp.atleast_1d(jnp.asarray(times, dtype=jnp.float64))
        if queries.ndim != 1 or not bool(jnp.all(jnp.isfinite(queries))):
            raise ValueError("native DEVAL requires a finite time vector")
        if not bool(
            jnp.all(
                (direction * (queries - mesh[0]) >= 0) & (direction * (queries - mesh[-1]) <= 0)
            )
        ):
            raise ValueError("native DEVAL query outside integration interval")
        return dense_kernel(queries, return_derivative=return_derivative)

    kmax = int(jnp.max(orders))
    # odefinalize trims history to maximum accepted order and leaves the
    # initial output slot zero, even though the internal initial phi=f0.
    exposed_phi = jnp.moveaxis(phis, 0, -1)[:, : kmax + 1, :].at[:, :, 0].set(0.0)
    exposed_psi = psis.T[:kmax, :].at[:, 0].set(0.0)
    result = {
        "solver": "ode113",
        "x": mesh,
        "y": values.T,
        "sol": dense,
        "extdata": {"options": dict(options), "odefun": odefun, "varargin": ()},
        "ie": jnp.asarray(event_indices, dtype=jnp.int32),
        "xe": jnp.stack(event_times) if event_times else jnp.empty(0),
        "idata": {
            "klastvec": orders,
            "phi3d": exposed_phi,
            "psi2d": exposed_psi,
            "idxNonNegative": jnp.empty(0, dtype=jnp.int32),
        },
        "stats": {
            "nsteps": len(history) - 1,
            "nfailed": int(state.nfailed),
            "nfevals": int(state.nfev),
            "tfinal": float(mesh[-1]),
        },
        "scope": "finite one-output native Adams; mass/nonnegative/output callbacks unported",
    }

    if event is not None:
        result["ye"] = (jnp.stack(event_states, axis=1) if event_states
                        else jnp.empty((y.size, 0), dtype=y.dtype))
    return result
