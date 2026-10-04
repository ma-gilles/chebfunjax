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


def _bisection(f, y, a, b):
    left, right = jnp.full_like(y, a), jnp.full_like(y, b)
    increasing = float(f(jnp.asarray(a))) < float(f(jnp.asarray(b)))
    xtol = _EPS * max(abs(a), abs(b), abs(b - a))
    for _ in range(128):
        mid = left + (right - left) / 2
        value = f(mid)
        move_left = value < y if increasing else value > y
        exact = value == y
        left = jnp.where(move_left | exact, mid, left)
        right = jnp.where(~move_left | exact, mid, right)
        if float(jnp.max(right - left)) <= xtol:
            break
    return left + (right - left) / 2


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


def _newton(f, fp, y, a, b, tol):
    # MATLAB uses the previous sample's solution as the next initial guess.
    # A Brent safeguard handles derivative zeros and endpoint jumps.
    flat = y.ravel()
    previous = a if float(f(jnp.asarray(a))) < float(f(jnp.asarray(b))) else b
    out = []
    for target in flat:
        point = previous
        for _ in range(11):
            residual = float(f(jnp.asarray(point)) - target)
            if abs(residual) <= tol / 5:
                break
            derivative = float(fp(jnp.asarray(point)))
            if derivative == 0:
                break
            candidate = point - residual / derivative
            if not math.isfinite(candidate) or not a <= candidate <= b:
                break
            point = candidate
        if abs(float(f(jnp.asarray(point)) - target)) > tol:
            point = float(_brent(f, target, a, b))
        out.append(point)
        previous = point
    return jnp.asarray(out).reshape(y.shape)


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
    # Preserve both one-sided images: a jump in f becomes a constant piece
    # of its generalized inverse, and a kink becomes a breakpoint in g.
    boundaries = [fa, fb]
    for point in f.domain.breakpoints[1:-1]:
        boundaries.extend([float(f(point, "left")), float(f(point, "right"))])
    ordered = sorted(set(boundaries))
    union_tol = 100 * _EPS * max(abs(v) for v in ordered)
    domain = [ordered[0]]
    for value in ordered[1:]:
        if value - domain[-1] < union_tol:
            domain[-1] = (domain[-1] + value) / 2
        else:
            domain.append(value)

    def values(target):
        y = jnp.asarray(target, dtype=jnp.float64)
        if algorithm == "roots":
            result = _roots(f, y, tol)
        elif algorithm == "newton":
            result = (_newton(f, fp, y, a, b, tol) if y.size >= len(f)
                      else _roots(f, y, tol))
        elif algorithm == "bisection":
            result = _bisection(f, y, a, b)
        elif algorithm in ("regulafalsi", "illinois"):
            result = _false_position(f, y, a, b, algorithm == "illinois")
        else:
            result = _brent(f, y, a, b)
        return jnp.where(y == fa, a, jnp.where(y == fb, b, result))

    inverse = chebfun(values, domain=tuple(domain), eps=tol, splitting=split,
                     min_samples=len(f), max_length=pref.maxLength)
    if _onoff(rangecheck):
        (xmin, vmin), (xmax, vmax) = inverse.minandmax()
        x = Chebfun.identity(inverse.domain)
        inverse = (inverse + (xmax - x) * ((a - vmin) / (xmax - xmin))
                   + (x - xmin) * ((b - vmax) / (xmax - xmin)))
    return inverse
