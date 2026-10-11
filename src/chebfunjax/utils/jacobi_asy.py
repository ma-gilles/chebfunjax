"""Jacobi asymptotic quadrature interior and endpoint kernels.

Provenance
----------
MATLAB source: jacpts.m, asy1/feval_asy1/asy2/asy2_higherterms.
Chebfun commit: 7574c77. Original authors: Nick Hale and Alex Townsend.
"""
from functools import partial

import jax
import jax.numpy as jnp

from .gamma_ratio import _stirling_correction


def _phi(n, a, b):
    j = jnp.arange(19, dtype=jnp.float64)
    vec = (.5+a+j)*(.5-a+j)/(j+1)/(2*n+a+b+j+2)
    p1 = jnp.concatenate((jnp.ones(1), jnp.cumprod(vec)))
    p1 = p1*jnp.where(jnp.arange(20)%4 >= 2, -1., 1.)
    p2 = jnp.eye(20)
    for ell in range(19):
        j = jnp.arange(19-ell, dtype=jnp.float64)
        vec = (.5+b+j)*(.5-b+j)/(j+1)/(2*n+a+b+j+ell+2)
        p2 = p2.at[ell+1:, ell].set(jnp.cumprod(vec))
    return p1[None, :]*p2


def _interior_constant(n, a, b):
    dsa = .5*a*a/n
    dsb = .5*b*b/n
    dsab = .25*(a+b)**2/n
    ds = dsa+dsb-dsab

    def cond(s):
        _, _, _, _, ds, total, previous = s
        return jnp.abs(ds/total)+previous > jnp.finfo(jnp.float64).eps/10

    def body(s):
        j, dsa, dsb, dsab, ds, total, _ = s
        previous = jnp.abs(ds/total)
        j = j+1
        tmp = -(j-1)/(j+1)/n
        dsa, dsb, dsab = tmp*dsa*a, tmp*dsb*b, .5*tmp*dsab*(a+b)
        ds = dsa+dsb-dsab
        return j, dsa, dsb, dsab, ds, total+ds, previous

    state = (1., dsa, dsb, dsab, ds, ds, ds)
    total = jax.lax.while_loop(cond, body, state)[5]
    p2 = jnp.exp(total)*jnp.sqrt(2*jnp.pi)*jnp.sqrt((n+a)*(n+b)/(2*n+a+b))/(2*n+a+b+1)
    c = p2*(_stirling_correction(jnp.float64(n+a))*_stirling_correction(jnp.float64(n+b))
            /_stirling_correction(jnp.float64(2*n+a+b)))*2/jnp.pi
    return c, c*(a+b+2*n)*(a+b+1+2*n)/(4*(a+n)*(b+n))


@partial(jax.jit, static_argnames=('n', 'a', 'b', 'final'))
def _interior_eval(n, a, b, t, mask, final=False):
    mm = jnp.arange(20, dtype=jnp.float64)[:, None]
    alpha = (.5*(2*n+a+b+1+mm))*t[None, :]-.5*(a+.5)*jnp.pi
    cosa, sina = jnp.cos(alpha), jnp.sin(alpha)
    if final:
        k = jnp.arange(1, t.size+1, dtype=jnp.float64)
        k = jnp.where(mask[0], k[::-1], k)
        ta = t.astype(jnp.float32).astype(jnp.float64)
        tb = t-ta
        hi, lo = n*ta, n*tb+(a+b+1)*.5*t
        pia, pib = 3.1415927410125732, -8.742278000372485e-08
        dh = (hi-(k-.25)*pia)+lo-.5*a*pia-(k-.25+.5*a)*pib

        def cond(s):
            j, _, _, _, _, _, done = s
            return (j <= 20) & ~done

        def body(s):
            j, total, sign, fact, power, square, _ = s
            dc = sign*power/fact
            done = jnp.max(jnp.where(mask, jnp.abs(dc), 0.)) < jnp.finfo(jnp.float64).eps/2000
            return j+1, total+dc, -sign, fact*(2*j+3)*(2*j+2), power*square, square, done

        tmp = jax.lax.while_loop(cond, body,
                                (0, jnp.zeros_like(t), 1., 1., dh, dh*dh, False))[1]
        tmp = tmp*jnp.where(jnp.arange(t.size)%2, -1., 1.)
        loc = jnp.argmax(jnp.where(mask, jnp.abs(tmp), -jnp.inf))
        tmp = jnp.sign(cosa[0, loc]*tmp[loc])*tmp
        cosa = cosa.at[0].set(tmp)
    cos2 = cosa*jnp.cos(t)+sina*jnp.sin(t)
    sin2 = sina*jnp.cos(t)-cosa*jnp.sin(t)
    sint = jnp.concatenate((jnp.ones((1, t.size)),
                           jnp.cumprod(jnp.broadcast_to(.5/jnp.sin(.5*t), (19, t.size)), axis=0)))
    cost = .5/jnp.cos(.5*t)
    phi, phi2 = _phi(n, a, b), _phi(n-1, a, b)
    even = jnp.arange(20)%2 == 0

    def cond(s):
        m, _, _, _, stop = s
        return (m < 20) & ~stop

    def body(s):
        m, total, total2, sc, _ = s
        ds1 = (jnp.where(even, phi[m], 0.)@sc)*cosa[m]
        ds2 = (jnp.where(~even, phi[m], 0.)@sc)*sina[m]
        ds12 = (jnp.where(even, phi2[m], 0.)@sc)*cos2[m]
        ds22 = (jnp.where(~even, phi2[m], 0.)@sc)*sin2[m]
        stop = (m > 10) & (jnp.max(jnp.where(mask, jnp.abs(ds1+ds2), 0.))
                          < jnp.finfo(jnp.float64).eps/100)
        total = jnp.where(stop, total, total+ds1+ds2)
        total2 = jnp.where(stop, total2, total2+ds12+ds22)
        sc = sc*jnp.where((jnp.arange(20) <= m)[:, None], cost, 1.)
        return m+1, total, total2, sc, stop

    _, total, total2, _, _ = jax.lax.while_loop(
        cond, body, (0, jnp.zeros_like(t), jnp.zeros_like(t), sint, False))
    c, c2 = _interior_constant(n, a, b)
    vals, vals2 = c*total, c2*total2
    ders = (n*(a-b-(2*n+a+b)*jnp.cos(t))*vals+2*(n+a)*(n+b)*vals2)/(2*n+a+b)/jnp.sin(t)
    denom = 1/jnp.real(jnp.sin(t/2)**(a+.5)*jnp.cos(t/2)**(b+.5))
    return vals*denom, ders*denom


def _interior_rule(n, a, b):
    k = (2*jnp.arange(n, 0, -1, dtype=jnp.float64)+a-.5)*jnp.pi/(2*n+a+b+1)
    tt = k+((.25-a*a)/jnp.tan(.5*k)-(.25-b*b)*jnp.tan(.5*k))/(2*n+a+b+1)**2
    right = tt[tt <= jnp.pi/2]
    mint = right[-10]
    count = max(int(jnp.argmax(right < mint)), 1)
    mask = jnp.arange(right.size) < count

    def solve(t, aa, bb, mask):
        dt = jnp.full_like(t, jnp.inf)
        j = 0
        while bool(jnp.max(jnp.where(mask, jnp.abs(dt), 0.)) > jnp.sqrt(jnp.finfo(jnp.float64).eps)/100) and j < 10:
            vals, ders = _interior_eval(n, aa, bb, t, mask)
            dt = vals/ders
            t = t+dt
            j += 1
        vals, ders = _interior_eval(n, aa, bb, t, mask, final=True)
        t = t+vals/ders
        return t, ders

    tr, dr = solve(right, a, b, mask)
    left = jnp.pi-tt[:n-right.size]
    mint = left[9]
    start = max(int(jnp.argmax(left > mint)), 0)
    mask = jnp.arange(left.size) >= start
    tl, dl = solve(left, b, a, mask)
    return (jnp.concatenate((-jnp.cos(tl), jnp.cos(tr))),
            jnp.concatenate((1/(dl*dl), 1/(dr*dr))),
            jnp.concatenate((jnp.sin(tl)/dl, jnp.sin(tr)/dr)))


def _boundary_coefficients(a, b, theta, n):
    from ..discretization.chebcolloc import ChebColloc2
    from ..domain import Domain

    c = max(float(jnp.max(theta)), 0.5)
    N = int(jnp.ceil(40 - n)) if n < 30 else 15 if c > jnp.pi / 2 - 0.5 else 10
    t = 0.5 * c * (jnp.sin(jnp.pi * jnp.arange(-(N - 1), N, 2) / (2 * (N - 1))) + 1)
    v = (-1.0) ** jnp.arange(N)
    v = v.at[0].multiply(0.5).at[-1].multiply(0.5)
    A = 0.25 - a * a
    B = 0.25 - b*b
    g = A * (1 / jnp.tan(t / 2) - 2 / t) - B * jnp.tan(t / 2)
    gp = A * (2 / t**2 - 0.5 / jnp.sin(t / 2) ** 2) - 0.5 * B / jnp.cos(t / 2) ** 2
    gpp = (
        A * (-4 / t**3 + 0.25 * jnp.sin(t) / jnp.sin(t / 2) ** 4)
        - 4 * B * jnp.sin(t / 2) ** 4 / jnp.sin(t) ** 3
    )
    g = g.at[0].set(0)
    gp = gp.at[0].set(-A / 6 - 0.5 * B)
    gpp = gpp.at[0].set(0)
    B0 = (0.25 * g / t).at[0].set(0.25 * (-A / 6 - 0.5 * B))
    B0p = (0.25 * (gp / t - g / t**2)).at[0].set(0)
    A1 = 0.125 * gp - (1 + 2 * a) / 2 * B0 - g * g / 32 - a * (A + 3 * B) / 24
    A1p = 0.125 * gpp - (1 + 2 * a) / 2 * B0p - gp * g / 16
    A1pt = (
        (A1p / t)
        .at[0]
        .set(-A / 720 - A * A / 576 - A * B / 96 - B * B / 64 - B / 48 + a * (A / 720 + B / 48))
    )
    f = -A * (
        1 / 12
        + t**2 / 240
        + t**4 / 6048
        + t**6 / 172800
        + t**8 / 5322240
        + 691 * t**10 / 118879488000
        + t**12 / 5748019200
        + 3617 * t**14 / 711374856192000
        + 43867 * t**16 / 300534953951232000
    )
    f = (
        jnp.where(t > 0.5, A * (1 / t**2 - 1 / (2 * jnp.sin(t / 2)) ** 2), f)
        - B / (2 * jnp.cos(t / 2)) ** 2
    )
    disc = ChebColloc2(N, Domain((-1.0, 1.0)))
    C = disc.cumsummat() * (0.5 * c)
    D = disc.diffmat() * (2 / c)
    tB1 = (-0.5 * A1p - (0.5 + a) * (C @ A1pt) + 0.5 * (C @ (f * A1))).at[0].set(0)
    B1 = (
        (tB1 / t)
        .at[0]
        .set(
            A / 720
            + A * A / 576
            + A * B / 96
            + B * B / 64
            + B / 48
            + a * (A * A / 576 + B * B / 64 + A * B / 96)
            - a * a * (A / 720 + B / 48)
        )
    )
    A2 = 0.5 * (D @ tB1) - (0.5 + a) * B1 - 0.5 * (C @ (f * tB1))
    A2 = A2 - A2[0]
    A2p = D @ A2
    A2p = A2p - A2p[0]

    def extrap(z):
        ww = (jnp.pi / 2 - t[1:]) * (-1.0) ** jnp.arange(N - 1)
        ww = ww.at[-1].multiply(0.5)
        return z.at[0].set(jnp.sum(ww * z[1:]) / jnp.sum(ww))

    A2pt = extrap(A2p / t)
    tB2 = -0.5 * A2p - (0.5 + a) * (C @ A2pt) + 0.5 * (C @ (f * A2))
    B2 = extrap(tB2 / t)
    A3 = 0.5 * (D @ tB2) - (0.5 + a) * B2 - 0.5 * (C @ (f * tB2))
    A3 = A3 - A3[0]
    return t, v, (tB1, A2, tB2, A3)


def _bessel_taylor(t, z, a):
    import math

    from .besselj import besselj

    kmax = min(int(jnp.ceil(jnp.abs(jnp.log(jnp.finfo(jnp.float64).eps)
                                    /jnp.log(jnp.max(jnp.abs(t)))))), 30)
    bj = [jnp.real(besselj(a+j, z)) for j in range(-kmax, kmax+1)]
    coefficients = [bj[kmax]]
    fact = 1.0
    for k in range(1, kmax+1):
        value = jnp.zeros_like(z)
        sign = 1
        for ell in range(k+1):
            value = value+sign*math.comb(k, ell)*bj[kmax+2*ell-k]
            sign = -sign
        fact = k*fact
        coefficients.append(value/2**k/fact)
    aa = jnp.stack(coefficients, axis=1)
    powers = jnp.stack([t**k for k in range(kmax+1)], axis=1)
    return jax.vmap(jnp.dot)(aa, powers)


@partial(jax.jit, static_argnames=('n', 'a', 'b'))
def _boundary_constants(n, a, b):
    rho = n+.5*(a+b+1)
    rho2 = rho-1
    ds = .5*a*a/n

    def cond(s):
        _, ds, total = s
        return jnp.abs(ds/total) > jnp.finfo(jnp.float64).eps/10

    def body(s):
        j, ds, total = s
        j = j+1
        ds = -(j-1)/(j+1)/n*(ds*a)
        return j, ds, total+ds

    total = jax.lax.while_loop(cond, body, (1., ds, ds))[2]
    p2 = jnp.exp(total)*jnp.sqrt((n+a)/n)*(n/rho)**a
    c = p2*(_stirling_correction(jnp.float64(n+a))
            /_stirling_correction(jnp.float64(n)))/jnp.sqrt(2.)
    c2 = c*n/(n+a)*(rho/rho2)**a
    return c, c2


def _boundary_rule(n, a, b, npts):
    import warnings

    from .bessel_roots_general import _bessel_roots_general
    from .besselj import besselj
    from .interpolation import bary

    # All calls from native asy use <=10 points per endpoint.
    jk = jnp.real(_bessel_roots_general(a, npts))
    rho = n+.5*(a+b+1)
    rho2 = n+.5*(a+b-1)
    phik = jk/rho
    t = phik+((a*a-.25)*(1-phik/jnp.tan(phik))/(2*phik)
              -.25*(a*a-b*b)*jnp.tan(.5*phik))/rho**2
    if bool(jnp.any(t > 1.1*jnp.pi/2)):
        warnings.warn('CHEBFUN:jacpts:asy2:theta: Theta > pi/2. Result may be inaccurate.',
                      stacklevel=2)
    grid, weights, coefficient_values = _boundary_coefficients(a, b, t, n)
    c, c2 = _boundary_constants(n, a, b)
    aa, bb = .25-a*a, .25-b*b

    def evaluate(t):
        ja = jnp.real(besselj(a, rho*t))
        jb = jnp.real(besselj(a+1, rho*t))
        jbb = jnp.real(besselj(a+1, rho2*t))
        # Native calls feval_asy2(...,1) on every iteration, not only last.
        jab = _bessel_taylor(-t, rho*t, a)
        gt = aa*(1/jnp.tan(t/2)-2/t)-bb*jnp.tan(t/2)
        gtdx = aa*(2/t**2-.5/jnp.sin(t/2)**2)-.5*bb/jnp.cos(t/2)**2
        tb0 = .25*gt
        a10 = a*(aa+3*bb)/24
        a1 = gtdx/8-(1+2*a)/8*gt/t-gt*gt/32-a10
        tb1, a2, tb2, a3 = (bary(t, value, grid, weights) for value in coefficient_values)
        vals = ja+jb*tb0/rho+ja*a1/rho**2+jb*tb1/rho**3+ja*a2/rho**4
        vals2 = jab+jbb*tb0/rho2+jab*a1/rho2**2+jbb*tb1/rho2**3+jab*a2/rho2**4
        vals = vals+jb*tb2/rho**5+ja*a3/rho**6
        vals2 = vals2+jbb*tb2/rho2**5+jab*a3/rho2**6
        valstmp = c*vals
        denom = jnp.sin(t/2)**(a+.5)*jnp.cos(t/2)**(b+.5)
        vals = jnp.sqrt(t)*valstmp/denom
        ders = (n*(a-b-(2*n+a+b)*jnp.cos(t))*valstmp
                +2*(n+a)*(n+b)*c2*vals2)/(2*n+a+b)
        ders = ders*(jnp.sqrt(t)/(denom*jnp.sin(t)))
        return vals, ders

    dt = jnp.full_like(t, jnp.inf)
    j = 0
    while bool(jnp.max(jnp.abs(dt)) > jnp.sqrt(jnp.finfo(jnp.float64).eps)/200) and j < 10:
        vals, ders = evaluate(t)
        dt = vals/ders
        t = t+dt
        j += 1
    vals, ders = evaluate(t)
    t = (t+vals/ders)[::-1]
    ders = ders[::-1]
    return jnp.cos(t), 1/(ders*ders), jnp.sin(t)/ders


def _jacobi_asy(n, a, b):
    if n <= 20:
        xr, wr, vr = _boundary_rule(n, a, b, (n+1)//2)
        xl, wl, vl = _boundary_rule(n, b, a, n//2)
        return (jnp.concatenate((-xl[::-1], xr)),
                jnp.concatenate((wl[::-1], wr)),
                jnp.concatenate((vl[::-1], vr)))
    x, w, v = _interior_rule(n, a, b)
    xr, wr, vr = _boundary_rule(n, a, b, 10)
    if a != b:
        xl, wl, vl = _boundary_rule(n, b, a, 10)
    else:
        xl, wl, vl = xr, wr, vr
    return (x.at[-10:].set(xr).at[:10].set(-xl[::-1]),
            w.at[-10:].set(wr).at[:10].set(wl[::-1]),
            v.at[-10:].set(vr).at[:10].set(vl[::-1]))
