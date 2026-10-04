"""Compositional inversion helpers from MATLAB @chebfun/inv.m.

Provenance
----------
MATLAB source : @chebfun/inv.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp

_EPS = float(jnp.finfo(jnp.float64).eps)
_ALGORITHMS = ("roots", "newton", "bisection", "regulafalsi", "illinois", "brent")


def tol_union(A, B, tol=None):
    """Sorted union of real breakpoint vectors with MATLAB's tolerance merge.

    Exact duplicates are removed first. Every adjacent pair whose original
    gap is strictly below ``tol`` is replaced by its mean, and the right
    member of each such pair is deleted simultaneously. For chains of close
    points this deliberately has MATLAB's documented behavior: later pair
    means can themselves be deleted rather than recursively merged.

    Parameters
    ----------
    A, B : array-like
        Real vectors to union. Inputs are flattened, matching MATLAB's vector
        use in inverse-domain assembly.
    tol : float or None
        Absolute merge threshold. The default is
        ``100 * eps * max(norm(A, inf), norm(B, inf))``.

    Returns
    -------
    jax.Array
        Sorted unique/merged values as a one-dimensional float64 array.

    Provenance
    ----------
    MATLAB source : @chebfun/tolUnion.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford and
        The Chebfun Developers.

    JAX note
    --------
    This is an eager JAX operation: exact union and deletion produce a
    data-dependent output length, so it is not JIT-compatible.
    """
    a = jnp.asarray(A)
    b = jnp.asarray(B)
    if jnp.issubdtype(a.dtype, jnp.complexfloating) or jnp.issubdtype(
        b.dtype, jnp.complexfloating
    ):
        raise ValueError("tolUnion: inputs must be real vectors")
    a = jnp.asarray(a, dtype=jnp.float64).ravel()
    b = jnp.asarray(b, dtype=jnp.float64).ravel()
    if tol is None:
        norm_a = jnp.max(jnp.abs(a), initial=0.0)
        norm_b = jnp.max(jnp.abs(b), initial=0.0)
        tol = 100.0 * _EPS * jnp.maximum(norm_a, norm_b)
    tolerance = jnp.asarray(tol, dtype=jnp.float64)
    values = jnp.unique(jnp.concatenate((a, b)))
    if values.size < 2:
        return values

    close = jnp.diff(values) < tolerance
    close_indices = jnp.where(close)[0]
    merged = values.at[close_indices].set(
        0.5 * (values[close_indices] + values[close_indices + 1])
    )
    deleted = jnp.zeros(values.shape, dtype=jnp.bool_).at[
        close_indices + 1
    ].set(True)
    return jnp.compress(~deleted, merged)


def _onoff(value):
    if isinstance(value, str):
        if value.lower() not in ("on", "off"):
            raise ValueError("inv: expected 'on' or 'off'")
        return value.lower() == "on"
    return bool(value)


@eqx.filter_jit
def _brent(f, y, a, b, max_iterations: int = 512):
    """Vectorized source Brent loop from MATLAB ``fInverseBrent``.

    The iteration uses MATLAB's collective stopping tests, source trial order,
    product-based sign bracket updates, and returns the final trial ``s``.
    The 512-step cap is a Python safety adapter; reaching it while the source
    condition is still open raises instead of returning a partial iterate.

    Provenance
    ----------
    MATLAB source : @chebfun/inv.m, local function fInverseBrent
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford and
        The Chebfun Developers.

    JIT notes
    ---------
    JAX-traceable for a JAX-compatible callable ``f`` and static
    ``max_iterations``. Empty target arrays return an empty array directly as
    a Python shape adapter; MATLAB inverse construction does not sample an
    empty target vector.
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    if y.size == 0:
        return jnp.empty_like(y)

    a0 = jnp.asarray(a, dtype=jnp.float64)
    b0 = jnp.asarray(b, dtype=jnp.float64)
    fa0 = jnp.asarray(f(a0), dtype=jnp.float64) - y
    fb0 = jnp.asarray(f(b0), dtype=jnp.float64) - y
    a0 = jnp.broadcast_to(a0, y.shape)
    b0 = jnp.broadcast_to(b0, y.shape)
    fa0 = jnp.broadcast_to(fa0, y.shape)
    fb0 = jnp.broadcast_to(fb0, y.shape)

    # |f(a)| < |f(b)|: swap endpoints/residuals.
    swap0 = jnp.abs(fa0) < jnp.abs(fb0)
    a0, b0 = jnp.where(swap0, b0, a0), jnp.where(swap0, a0, b0)
    fa0, fb0 = jnp.where(swap0, fb0, fa0), jnp.where(swap0, fa0, fb0)

    c0, fc0 = a0, fa0
    mflag0 = jnp.ones(y.shape, dtype=jnp.bool_)
    s0, d0 = a0, c0
    fs0 = jnp.full(y.shape, jnp.inf, dtype=jnp.float64)
    i0 = jnp.asarray(0, dtype=jnp.int32)

    def source_continue(state):
        i, aa, bb, faa, fbb, cc, fcc, dd, mflag, ss, fs = state
        del faa, fbb, cc, fcc, dd, mflag, ss
        residual_open = jnp.max(jnp.abs(fs)) > _EPS
        bracket_scale = jnp.maximum(jnp.max(jnp.abs(bb)),
                                    jnp.max(jnp.abs(aa)))
        bracket_open = (jnp.max(jnp.abs(bb - aa))
                        > _EPS * bracket_scale)
        return (i < max_iterations) & residual_open & bracket_open

    def source_step(state):
        i, aa, bb, faa, fbb, cc, fcc, dd, mflag, _ss, _fs = state
        # Keep the arithmetic and trial selection order of fInverseBrent.
        s_iq = (
            aa * fbb * fcc / ((faa - fbb) * (faa - fcc))
            + bb * faa * fcc / ((fbb - faa) * (fbb - fcc))
            + cc * faa * fbb / ((fcc - faa) * (fcc - fbb))
        )
        s_sc = bb - fbb * (bb - aa) / (fbb - faa)
        s_bi = (aa + bb) / 2.0

        distinct = (faa != fcc) & (fbb != fcc)
        ss = jnp.where(distinct, s_iq, s_sc)
        boundary = (3.0 * aa + bb) / 4.0
        use_bisection = (
            ((boundary < bb) & ((ss < boundary) | (ss > bb)))
            | ((bb <= boundary) & ((ss < bb) | (ss > boundary)))
            | (mflag & (jnp.abs(ss - bb) >= jnp.abs(bb - cc) / 2.0))
            | (~mflag & (jnp.abs(ss - bb) >= jnp.abs(cc - dd) / 2.0))
            | (mflag & (jnp.abs(bb - cc) < _EPS))
            | (~mflag & (jnp.abs(cc - dd) < _EPS))
            | (jnp.abs(bb - aa) < _EPS)
        )
        ss = jnp.where(use_bisection, s_bi, ss)
        mflag = use_bisection

        fs = jnp.asarray(f(ss), dtype=jnp.float64) - y
        dd, cc, fcc = cc, bb, fbb

        # Literal MATLAB endpoint-product sign test, including its ordinary
        # floating-point overflow/underflow behavior.
        update_b = faa * fs <= 0.0
        bb = jnp.where(update_b, ss, bb)
        fbb = jnp.where(update_b, fs, fbb)
        aa = jnp.where(~update_b, ss, aa)
        faa = jnp.where(~update_b, fs, faa)

        swap = jnp.abs(faa) < jnp.abs(fbb)
        aa, bb = jnp.where(swap, bb, aa), jnp.where(swap, aa, bb)
        faa, fbb = jnp.where(swap, fbb, faa), jnp.where(swap, faa, fbb)
        return i + 1, aa, bb, faa, fbb, cc, fcc, dd, mflag, ss, fs

    state0 = (i0, a0, b0, fa0, fb0, c0, fc0, d0, mflag0, s0, fs0)
    state = jax.lax.while_loop(source_continue, source_step, state0)
    i, aa, bb, faa, fbb, cc, fcc, dd, mflag, ss, fs = state
    del aa, bb, faa, fbb, cc, fcc, dd, mflag

    residual_open = jnp.max(jnp.abs(fs)) > _EPS
    bracket_scale = jnp.maximum(jnp.max(jnp.abs(state[2])),
                                jnp.max(jnp.abs(state[1])))
    bracket_open = (jnp.max(jnp.abs(state[2] - state[1]))
                    > _EPS * bracket_scale)
    cap_exhausted = (i >= max_iterations) & residual_open & bracket_open
    return eqx.error_if(
        ss,
        cap_exhausted,
        "MATLAB-source Brent loop still unconverged at the explicit safety cap",
    )


@eqx.filter_jit
def _bisection(f, y, a, b, max_iterations: int = 2048):
    """Literal vectorized ``fInverseBisection`` iteration.

    The source uses an absolute EPS interval width and an EPS-wide residual
    dead band. ``max_iterations`` is an explicit Python safety cap; exhausting
    it while any interval remains open raises instead of returning a partial
    iterate.

    Provenance
    ----------
    MATLAB source : @chebfun/inv.m, local fInverseBisection
    Chebfun commit: 7574c77, lines 270-293
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    left0 = jnp.full_like(y, a)
    right0 = jnp.full_like(y, b)
    # MATLAB carries scalar endpoints; broadcast the initial midpoint to the
    # target shape so the JAX loop has invariant state shapes.
    c0 = jnp.full_like(y, (a + b) / 2)
    direction = jnp.sign(f(jnp.asarray(b)) - f(jnp.asarray(a)))
    i0 = jnp.asarray(0, dtype=jnp.int32)

    def condition(state):
        i, left, right, _c = state
        open_width = jnp.max(jnp.abs(right - left)) >= _EPS
        return (i < max_iterations) & open_width

    def step(state):
        i, left, right, c = state
        vals = direction * (f(c) - y)
        # Preserve source classification and assignment order:
        # I1=(vals<=-eps), I2=(vals>=eps), I3=the residual dead band.
        move_right = vals <= -_EPS
        move_left = vals >= _EPS
        dead_band = ~(move_right | move_left)
        left = (move_right * c + move_left * left + dead_band * c)
        right = (move_right * right + move_left * c + dead_band * c)
        c = (left + right) / 2
        return i + 1, left, right, c

    state = jax.lax.while_loop(condition, step,
                               (i0, left0, right0, c0))
    i, left, right, c = state
    del i
    exhausted = jnp.max(jnp.abs(right - left)) >= _EPS
    return eqx.error_if(c, exhausted,
                        "MATLAB-source bisection still open at safety cap")


def _false_position(f, y, a, b, illinois):
    aa, bb = jnp.full_like(y, a), jnp.full_like(y, b)
    fa, fb = f(aa) - y, f(bb) - y
    cc = bb - fb * (bb - aa) / (fb - fa)
    side = jnp.zeros_like(y, dtype=jnp.int32)
    xtol = _EPS * max(abs(a), abs(b), abs(b - a))
    for _ in range(512):
        fc = f(cc) - y
        change = (jnp.signbit(fa) != jnp.signbit(fc)) | (fc == 0)
        aa, fa = jnp.where(change, aa, cc), jnp.where(change, fa, fc)
        bb, fb = jnp.where(change, cc, bb), jnp.where(change, fc, fb)
        if illinois:
            fa = jnp.where(change & (side == 1), fa / 2, fa)
            fb = jnp.where(~change & (side == -1), fb / 2, fb)
        side = jnp.where(change, 1, -1)
        step = -fb * (bb - aa) / (fb - fa)
        nxt = bb + jnp.where(jnp.isfinite(step), step, 0)
        converged = float(jnp.max(jnp.abs(nxt - cc))) <= xtol
        cc = nxt
        if converged:
            break
    # A flat endpoint can make regula falsi stagnate before convergence.
    # Retain its converged values and safeguard unresolved ones with Brent.
    error = jnp.abs(f(cc) - y)
    scale = jnp.maximum(jnp.abs(y), max(abs(float(f(jnp.asarray(a)))),
                                     abs(float(f(jnp.asarray(b))))))
    if bool(jnp.any(error > 10 * _EPS * scale)):
        polished = _brent(f, y, a, b)
        cc = jnp.where(error > 10 * _EPS * scale, polished, cc)
    return cc


def _roots(f, y, tol):
    out = []
    endpoints = jnp.asarray(f.domain.breakpoints)
    values = f(endpoints)
    for target in y.ravel():
        rr = (f - target).roots()
        if rr.size != 1:
            error = jnp.abs(values - target)
            k = int(jnp.argmin(error))
            if float(error[k]) > 100 * tol * abs(float(values[k])):
                raise ValueError("CHEBFUN:CHEBFUN:inv:notmonotonic2: f must be monotonic")
            rr = endpoints[k:k + 1]
        out.append(rr[0])
    return jnp.asarray(out).reshape(y.shape)


def _inverse_domain(f):
    """Assemble inverse-domain extrema and one-sided breakpoint images.

    MATLAB first tolerance-unions the left/right values at interior
    breakpoints, then unions that result with the true min/max values. Keeping
    the two stages separate preserves source ordering and tolerance behavior.

    Provenance
    ----------
    MATLAB source : @chebfun/inv.m, lines 62-68; @chebfun/tolUnion.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford and
        The Chebfun Developers.
    """
    (_xmin, ymin), (_xmax, ymax) = f.minandmax()
    g_ends = jnp.asarray([ymin, ymax], dtype=jnp.float64)
    breaks = jnp.asarray(f.domain.breakpoints, dtype=jnp.float64)
    f_breaks = breaks[1:-1]
    if f_breaks.size:
        g_breaks_left = jnp.asarray(f(f_breaks, "left"), dtype=jnp.float64)
        g_breaks_right = jnp.asarray(f(f_breaks, "right"), dtype=jnp.float64)
    else:
        g_breaks_left = jnp.empty((0,), dtype=jnp.float64)
        g_breaks_right = jnp.empty((0,), dtype=jnp.float64)
    g_breaks = tol_union(g_breaks_left, g_breaks_right)
    return tuple(float(value) for value in tol_union(g_ends, g_breaks))


def _newton_or_roots(f, fp, y, a, b, tol, forward_length):
    """Choose MATLAB's dense Newton path or sparse roots fallback.

    Provenance
    ----------
    MATLAB source : @chebfun/inv.m, local fInverseNewton, lines 231-268
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford and
        The Chebfun Developers.
    """
    if y.size < forward_length:
        # fInverseNewton divides its tolerance by five before this branch.
        return _roots(f, y, tol / 5.0)
    return _newton(f, fp, y, a, b, tol)


@eqx.filter_jit
def _newton(f, fp, y, a, b, tol):
    """Literal MATLAB Newton inverse iteration for a dense target vector.

    Source begins at the left domain endpoint, carries each answer forward,
    uses ``tol/5``, applies raw Newton divisions, and returns the iterate after
    at most eleven updates. It has no interval guard or fallback solver.

    Provenance
    ----------
    MATLAB source : @chebfun/inv.m, local fInverseNewton
    Chebfun commit: 7574c77, lines 231-268

    Adapter note: Python preserves the target array shape because the public
    constructor callback contract differs from MATLAB's column-vector output.
    """
    y = jnp.asarray(y, dtype=jnp.float64)
    targets = y.reshape((-1,))
    tolerance = jnp.asarray(tol, dtype=y.dtype) / 5.0
    first = jnp.asarray(a, dtype=y.dtype)

    def solve_one(previous, target):
        residual0 = f(previous) - target
        count0 = jnp.asarray(0, dtype=jnp.int32)

        def condition(state):
            count, _point, residual = state
            return (jnp.abs(residual) > tolerance) & (count <= 10)

        def update(state):
            count, point, residual = state
            # Deliberately retain IEEE behavior for zero/nonfinite derivative,
            # as the MATLAB source performs this division without a guard.
            point = point - residual / fp(point)
            residual = f(point) - target
            return count + 1, point, residual

        _, point, _residual = jax.lax.while_loop(
            condition, update, (count0, previous, residual0))
        return point, point

    _, result = jax.lax.scan(solve_one, first, targets)
    return result.reshape(y.shape)


def _inverse(f, pref=None, *, algorithm="brent", eps=None,
             splitting=None, monocheck=False, rangecheck=False):
    from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
    from chebfunjax.chebpref import ChebfunPref

    pref = ChebfunPref(pref)
    algorithm = str(algorithm).lower()
    if algorithm not in _ALGORITHMS:
        raise ValueError("CHEBFUN:CHEBFUN:inv:badAlgo: unrecognized inverse algorithm")
    tol = _EPS if eps is None else float(eps)
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("inv: eps must be finite and positive")
    split = _onoff(pref.splitting if splitting is None else splitting)
    if f.isempty():
        return Chebfun.empty()
    if f.n_columns > 1:
        raise ValueError("CHEBFUN:CHEBFUN:inv:noquasi: inv requires a single column")
    if not f.isreal():
        raise ValueError("inv: function must be real and monotonic")
    if f.funs[0].tech.coeffs.ndim > 1:
        f = f.extract_columns(0)
    a, b = float(f.domain.a), float(f.domain.b)
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("inv: finite domain required")
    fp = f.diff() if _onoff(monocheck) or algorithm == "newton" else None
    if _onoff(monocheck):
        stationary = fp.roots()
        if stationary.size:
            distances = jnp.min(jnp.abs(stationary[:, None]
                                       - jnp.asarray(f.domain.breakpoints)), axis=1)
            if bool(jnp.any(distances > 100 * jnp.abs(f(stationary)) * tol)):
                raise ValueError("CHEBFUN:CHEBFUN:inv:doMonoCheck:notMonotonic: "
                                 "f must be monotonic")
            if not split:
                warnings.warn("F is monotonic, but its inverse has singular endpoints. "
                              "Enabling breakpoint detection is advised.", stacklevel=3)
    fa, fb = float(f(jnp.asarray(a))), float(f(jnp.asarray(b)))
    if fa == fb:
        raise ValueError("inv: function must be monotonic")
    # MATLAB uses minandmax (not just endpoint values) and performs two
    # separate tolerance unions: first one-sided break images, then extrema.
    domain = _inverse_domain(f)

    def values(target):
        y = jnp.asarray(target, dtype=jnp.float64)
        if algorithm == "roots":
            result = _roots(f, y, tol)
        elif algorithm == "newton":
            result = _newton_or_roots(f, fp, y, a, b, tol, len(f))
        elif algorithm == "bisection":
            result = _bisection(f, y, a, b)
        elif algorithm in ("regulafalsi", "illinois"):
            result = _false_position(f, y, a, b, algorithm == "illinois")
        else:
            result = _brent(f, y, a, b)
        return jnp.where(y == fa, a, jnp.where(y == fb, b, result))

    # MATLAB parseInputs sets these on a local preference copy: every inverse
    # disables the off-grid sample test and Newton uses resampling refinement.
    # Other algorithms retain the caller's refinementFunction preference.
    refinement = ("resampling" if algorithm == "newton"
                  else pref.refinementFunction)
    inverse = chebfun(values, domain=tuple(domain), eps=tol, splitting=split,
                     min_samples=len(f), max_length=pref.maxLength,
                     sample_test=False, refinement_function=refinement)
    if _onoff(rangecheck):
        (xmin, vmin), (xmax, vmax) = inverse.minandmax()
        x = Chebfun.identity(inverse.domain)
        inverse = (inverse + (xmax - x) * ((a - vmin) / (xmax - xmin))
                   + (x - xmin) * ((b - vmax) / (xmax - xmin)))
    return inverse
