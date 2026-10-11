"""Source convolution on finite piecewise Chebyshev representations.

Provenance
----------
Chebfun commit: 7574c77
@bndfun/conv.m, @chebfun/conv.m, @deltafun/conv.m.
Hale and Townsend's Legendre recurrence and cost-based quadrature switch.
"""

import math
import warnings

import jax
import jax.numpy as jnp

from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.utils.quadrature import chebpts_ab, legpts
from chebfunjax.utils.transforms import cheb2leg, leg2cheb

_EPS = 2.220446049250313e-16


def _trim(coeffs, multiplier=1):
    cutoff = int(
        jnp.max(
            jnp.where(
                jnp.abs(coeffs) > multiplier * _EPS * jnp.max(jnp.abs(coeffs)),
                jnp.arange(coeffs.size) + 1,
                0,
            )
        )
    )
    return coeffs[: max(cutoff, 1)]


@jax.jit
def _rec(alpha, beta, sign):
    size, n = alpha.size, beta.size
    sub = jnp.concatenate((jnp.ones(1), 1 / (2 * jnp.arange(1, size) + 1)))
    sup = -1 / (2 * jnp.arange(size) + 1)

    def apply_s(v):
        out = jnp.zeros_like(v).at[1:].set(sub[:-1] * v[:-1])
        out = out.at[:-1].add(sup[1:] * v[1:])
        return out.at[0].add(-sign * v[0])

    scl = 1 / (2 * jnp.arange(1, n + 1) - 1)
    scl = scl.at[1::2].multiply(-1)
    v = apply_s(alpha)
    bs = (scl * beta).at[0].set(0)
    gamma = (beta[0] * v).at[0].add(jnp.dot(v[:n], bs))
    if n == 1:
        return gamma
    new = apply_s(v) + sign * v
    old, v = v, new
    new = new.at[0].set(0)
    bs = (-bs * 3).at[1].set(0)
    gamma = (gamma + beta[1] * new).at[1].add(jnp.dot(new[:n], bs))

    def step(k, state):
        old, v, bs, gamma = state
        new = (2 * k - 3) * apply_s(v) + old
        new = jnp.where(jnp.arange(size) < k - 1, 0, new)
        bs = (-bs * ((k - 0.5) / (k - 1.5))).at[k - 1].set(0)
        gamma = (gamma + new * beta[k - 1]).at[k - 1].add(jnp.dot(new[:n], bs))
        return v, new, bs, gamma

    return jax.lax.fori_loop(3, n + 1, step, (old, v, bs, gamma))[3]


def _easy_conv(alpha, beta):
    alpha, beta = jnp.asarray(alpha), jnp.asarray(beta)
    if beta.size > alpha.size:
        alpha, beta = beta, alpha
    alpha = jnp.pad(alpha, (0, beta.size))
    left, right = _rec(alpha, beta, -1), _rec(-alpha, beta, 1)
    if beta.size == 1:
        return left, right
    return _trim(left), _trim(right)


@jax.jit
def _legendre_value(x, alpha):
    def step(k, state):
        old, cur = state
        degree = alpha.size - 1 - k
        new = (
            alpha[degree]
            + (2 * degree + 1) / (degree + 1) * x * cur
            - (degree + 1) / (degree + 2) * old
        )
        return cur, new

    old, cur = jax.lax.fori_loop(
        0,
        alpha.size - 1,
        step,
        (jnp.zeros_like(x, dtype=alpha.dtype), jnp.zeros_like(x, dtype=alpha.dtype)),
    )
    return alpha[0] + x * cur - 0.5 * old


def _map(x, a, b):
    return (x - a) / (b - a) - (b - x) / (b - a)


def _piece(coeffs, interval):
    from chebfunjax.chebfun1d.chebfun import _Piece

    return _Piece(Chebtech2.from_coeffs(coeffs), tuple(map(float, interval)))


def _restricted(piece, interval):
    p = piece.restrict(*map(float, interval))
    return p.with_tech(p.tech.simplify())


def _fix_maps(pieces):
    from chebfunjax.chebfun1d.chebfun import _Piece

    for k in range(1, len(pieces)):
        left, right = pieces[k - 1], pieces[k]
        b, c = left.interval[1], right.interval[0]
        if b != c:
            scale = max(*map(abs, left.interval), *map(abs, right.interval))
            if abs(b - c) >= 2 * _EPS * scale:
                raise ValueError("CHEBFUN:BNDFUN:conv:nonConsecutive")
            midpoint = (b + c) / 2
            pieces[k - 1] = _Piece(left.tech, (left.interval[0], midpoint))
            pieces[k] = _Piece(right.tech, (midpoint, right.interval[1]))
    return pieces


def convolve_pieces(f, g):
    """Hale--Townsend smooth-piece convolution with the native cost switch.

    Provenance
    ----------
    @bndfun/conv.m, Chebfun commit: 7574c77.
    """
    a, b = f.interval
    c, d = g.interval
    if b - a > d - c:
        return convolve_pieces(g, f)
    fc, gc = jnp.asarray(f.tech.coeffs), jnp.asarray(g.tech.coeffs)
    m, n = fc.size, gc.size
    patches = math.floor((d - c) / (b - a))
    x = chebpts_ab(n, b + c, a + d, kind=1)
    y = jnp.zeros(n, dtype=jnp.result_type(fc, gc))
    if patches > 1 and patches * ((m + n) * n) > (m + n) ** 3:
        left = convolve_pieces(f, g.restrict(c, c + (b - a)))[0]
        right = convolve_pieces(f, g.restrict(d - (b - a), d))[-1]
        t, w = legpts(math.ceil((m + n + 5) / 2), (a, b))
        values = g(x[:, None] - t[None, :]) @ (w * f(t))
        mid = _piece(_trim(Chebtech1.vals2coeffs(values), 10), (b + c, a + d))
        return _fix_maps([left, mid, right])
    fleg = cheb2leg(_trim(fc, 10))
    doms = [c + (b - a) * k for k in range(patches + 1)]
    for k in range(patches):
        dk = doms[k : k + 2]
        dl, dm, dr = a + dk[0], a + dk[1], b + dk[1]
        gk = _restricted(g, dk)
        hl, hr = _easy_conv(fleg, cheb2leg(gk.tech.coeffs))
        if k == 0:
            left = _piece(leg2cheb(hl), (dl, dm))
        else:
            mask = (dl <= x) & (x < dm)
            y = y + jnp.where(mask, _legendre_value(_map(x, dl, dm), hl), 0)
        if k < patches - 1:
            mask = (dm <= x) & (x < dr)
            y = y + jnp.where(mask, _legendre_value(_map(x, dm, dr), hr), 0)
    hs = max(abs(a), abs(b), abs(c), abs(d))
    spacing = math.ulp(hs)
    if abs((b - a) - (d - c)) < 10 * spacing:
        right = _piece(leg2cheb(hr), (d + a, d + b))
        pieces = [left, right]
    else:
        finish = a + c + patches * (b - a)
        gk = _restricted(g, (d - (b - a), d))
        hl, hr = _easy_conv(fleg, cheb2leg(gk.tech.coeffs))
        right = _piece(leg2cheb(hr), (d + a, d + b))
        remainder = d + a - finish
        df = (max(b - remainder, a), b)
        dg = (max(finish - b, c), min(d + a - b, d))
        if max(df[1] - df[0], dg[1] - dg[0]) > _EPS:
            mask = finish <= x
            y = jnp.where(mask, _legendre_value(_map(x, d - b + 2 * a, d + a), hl), y)
            fk, gk = _restricted(f, df), _restricted(g, dg)
            _, hr = _easy_conv(cheb2leg(fk.tech.coeffs), cheb2leg(gk.tech.coeffs))
            y = y + jnp.where(
                mask, _legendre_value(_map(x, finish, d + a), hr) * remainder / (b - a), 0
            )
        pieces = [left, _piece(Chebtech1.vals2coeffs(y), (b + c, a + d)), right]
    return _fix_maps([p.with_tech(p.tech * ((b - a) / 2)) for p in pieces])


def _as_chebfun(pieces):
    from chebfunjax.chebfun1d.chebfun import Chebfun

    return Chebfun(
        funs=pieces, domain=Domain(tuple([pieces[0].interval[0]] + [p.interval[1] for p in pieces]))
    )


def _myplus(f, g):
    from chebfunjax.chebfun1d.chebfun import tweak_domain

    f, g, _, _ = tweak_domain(f, g)
    a, b = g.domain.a, g.domain.b
    return f.define_interval((a, b), f.restrict((a, b)) + g)


def _scalar_storage(f):
    """Adapt a single smooth column to vector storage, retaining metadata."""
    from chebfunjax.chebfun1d.chebfun import Chebfun

    pieces = [
        p.with_tech(Chebfun._tech_with_coeffs(p.tech, p.tech.coeffs[:, 0]))
        if isinstance(p.tech, (Chebtech1, Chebtech2)) and p.tech.coeffs.ndim == 2
        else p
        for p in f.funs
    ]
    if all(p is q for p, q in zip(pieces, f.funs)):
        return f
    out = Chebfun(funs=pieces, domain=f.domain, deltas=f.deltas)
    if f._point_values is not None:
        object.__setattr__(out, "_point_values", jnp.asarray(f._point_values).reshape(-1))
    return Chebfun._as_transposed(out, f.is_transposed)


def convolution(f, g, *options):
    """Native flags, columns, orientation and piecewise convolution dispatch.

    Provenance
    ----------
    @chebfun/conv.m, Chebfun commit: 7574c77.
    """
    from chebfunjax.chebfun1d.chebfun import Chebfun, _delta_row, _Piece, chebfun
    from chebfunjax.chebfun1d.linalg import Quasimatrix

    trans = bool(getattr(f, "is_transposed", False))
    if trans != bool(getattr(g, "is_transposed", False)):
        raise ValueError("CHEBFUN:CHEBFUN:conv:transposed: CHEBFUN dimensions do not agree.")
    nf = len(f.cols) if isinstance(f, Quasimatrix) else (0 if f.isempty() else f.n_columns)
    ng = len(g.cols) if isinstance(g, Quasimatrix) else (0 if g.isempty() else g.n_columns)
    if nf > 1 or ng > 1:
        if nf != ng and nf != 1 and ng != 1:
            raise ValueError("CHEBFUN:CHEBFUN:conv:dimagree: CHEBFUN dimensions must agree.")
        fs = (
            f.cols
            if isinstance(f, Quasimatrix)
            else [f[k, :] if trans else f[:, k] for k in range(nf)]
        )
        gs = (
            g.cols
            if isinstance(g, Quasimatrix)
            else [g[k, :] if trans else g[:, k] for k in range(ng)]
        )
        out = [
            convolution(fs[k if nf > 1 else 0], gs[k if ng > 1 else 0], *options)
            for k in range(max(nf, ng))
        ]
        return Chebfun.vertcat(out) if trans else Chebfun.horzcat(out)
    if isinstance(f, Quasimatrix) and nf == 1:
        f = Chebfun._as_transposed(f.cols[0], trans)
    if isinstance(g, Quasimatrix) and ng == 1:
        g = Chebfun._as_transposed(g.cols[0], trans)
    if f.isempty() or g.isempty():
        return chebfun()
    same, old = False, False
    for option in options:
        flag = str(option).lower()
        if flag == "old":
            old = True
        elif flag == "same":
            same = True
        elif flag == "valid":
            raise ValueError("CHEBFUN:CHEBFUN:conv:validFlag: 'valid' is not yet supported.")
        elif flag != "full":
            raise ValueError(f"CHEBFUN:CHEBFUN:conv:badInput: Unknown input option {option}.")
    a, b, c, d = map(float, (f.domain.a, f.domain.b, g.domain.a, g.domain.b))
    if not all(map(math.isfinite, (a, b, c, d))):
        raise ValueError("CHEBFUN:CHEBFUN:conv:bounded: CONV only supports bounded domains.")
    f, g = _scalar_storage(f), _scalar_storage(g)
    if len(f.funs) + len(g.funs) > 50:
        warnings.warn(
            "CHEBFUN:CHEBFUN:conv:piecewise: Convolving many pieces can be very slow.", stacklevel=2
        )
    from chebfunjax.fun.singfun import Singfun

    if old or any(isinstance(p.tech, Singfun) for p in [*f.funs, *g.funs]):
        from chebfunjax.chebfun1d._convolution_quadrature import old_convolution

        h = old_convolution(f, g)
    else:
        # Preserve the upstream early swap, including its omission of flags.
        if b - a > d - c:
            return convolution(g, f)
        h = chebfun(0, domain=(a + c, b + d))
        if float(f.vscale) != 0 and float(g.vscale) != 0:
            for pf in f.funs:
                for pg in g.funs:
                    h = _myplus(h, _as_chebfun(convolve_pieces(pf, pg)))
        for rows, other in ((f.deltas, g), (g.deltas, f)):
            for row in rows:
                loc, mag, order = _delta_row(row)
                # Native outer conv visits individual FUN pairs. Its inner
                # @deltafun/conv differentiates the smooth FUN, not jumps
                # between FUNs. Retain only the smooth derivative pieces:
                # cvd1 confirms delta-prime * step has no jump delta upstream.
                derivative = other.diff(order) if order else other
                for p in derivative.funs:
                    shifted = _Piece(p.tech * mag, (p.interval[0] + loc, p.interval[1] + loc))
                    h = _myplus(h, _as_chebfun([shifted]))
        if f.deltas and g.deltas:
            accum = {}
            for fr in f.deltas:
                fl, fm, fo = _delta_row(fr)
                for gr in g.deltas:
                    gl, gm, go = _delta_row(gr)
                    key = (fl + gl, fo + go)
                    accum[key] = accum.get(key, 0) + fm * gm
            h = h._attach_deltas(
                h,
                tuple(
                    (loc, mag) if order == 0 else (loc, mag, order)
                    for (loc, order), mag in sorted(accum.items()) if mag != 0
                ),
            )
        values = jnp.asarray(h.point_values)
        for k, point in enumerate(h.domain.breakpoints[1:-1], 1):
            values = values.at[k].set((h.funs[k - 1](point) + h.funs[k](point)) / 2)
        h = h.set_point_values(values)
    if same:
        h = h.restrict((a, b))
    return Chebfun._as_transposed(h, trans).simplify()
