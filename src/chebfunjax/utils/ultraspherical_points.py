"""Source ultraspherical quadrature algorithms.

Provenance
----------
MATLAB source : ultrapts.m
Chebfun commit: 7574c77
"""

import math
import warnings
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import gammaln

from .bessel_roots_general import _bessel_roots_general
from .gamma_ratio import _gamma_ratio


def _initial(n, lam, k):
    theta = (k - (1 - lam) * 0.5) / (n + lam) * jnp.pi
    cs = jnp.cos(theta) ** 2
    t = theta + lam * (1 - lam) / (2 * (n + lam) ** 2) * (
        1 - (6 + lam * (1 - lam) * (9 - 2 * cs)) / (12 * (n + lam) ** 2 * (1 - cs))
    ) / jnp.tan(theta)
    # A MATLAB IF on a vector is true only when every entry is true.
    if 0 < lam < 1:
        t = jnp.where(jnp.all(t < theta), theta, t)
    return t


def _poly(n, lam, x):
    def step(k, pair):
        p, p1 = pair
        q = 2 * (k - 1 + lam) / k * x * p - (k - 2 + 2 * lam) / k * p1
        return q, p

    p, p1 = lax.fori_loop(1, n + 1, step, (jnp.ones_like(x), jnp.zeros_like(x)))
    return p, (-n * x * p + (n + 2 * lam - 1) * p1) / (1 - x * x)


@partial(jax.jit, static_argnums=(0, 1))
def _rec(n, lam):
    m = (n + 1) // 2
    k = jnp.arange(1, m + 1, dtype=jnp.float64)
    x = jnp.cos(_initial(n, lam, k))
    if not 0 < lam < 1 and n > 21:
        bz = _bessel_roots_general(lam - 0.5, 10)
        nn = jnp.sqrt((n + lam) ** 2 + lam * (1 - lam) / 3)
        x = x.at[:10].set(
            jnp.cos(
                bz / nn - lam * (1 - lam) / 90 * (bz**3 + 2 * (lam * lam - lam - 0.75) * bz) / nn**5
            )
        )

    def condition(s):
        return (s[0] < 20) & (jnp.max(jnp.abs(s[2])) > jnp.finfo(jnp.float64).eps)

    def step(s):
        p, dp = _poly(n, lam, s[1])
        dx = p / dp
        return s[0] + 1, s[1] - dx, dx

    _, x, _ = lax.while_loop(condition, step, (0, x, jnp.full_like(x, jnp.inf)))
    _, dp = _poly(n, lam, x)
    c = (
        4 ** (1 - lam) * jnp.pi * _gamma_ratio(n + 1.0, 2 * lam - 1) / jnp.exp(2 * gammaln(lam + 1))
        if n >= 20
        else 4 ** (1 - lam)
        * jnp.pi
        * jnp.exp(gammaln(n + 2 * lam) - gammaln(n + 1.0) - 2 * gammaln(lam + 1))
    )
    w = c * lam * lam / ((1 - x * x) * dp**2)
    s = n % 2
    return (
        jnp.concatenate((-x[: m - s], x[::-1])),
        jnp.concatenate((w[: m - s], w[::-1])),
        jnp.concatenate((1 / dp[: m - s], 1 / dp[::-1])),
    )


def _interior_eval(n, lam, t):
    m = int(lam) - 1 if lam % 1 == 0 else 30
    j = jnp.arange(1, m + 1, dtype=jnp.float64)
    c = jnp.cumprod((j - lam) / (n + lam - j))
    d = lam * jnp.concatenate(
        (jnp.ones(1), jnp.cumprod((jnp.arange(1, m) + lam) / jnp.arange(2, m + 1)))
    )
    c = jnp.concatenate((jnp.ones(1), c * d))
    jj = jnp.arange(m + 1, dtype=jnp.float64)[:, None]
    alpha = (n + lam - jj) * t - 0.5 * jnp.pi * (jj + lam)
    twosin = jnp.broadcast_to(2 * jnp.sin(t), alpha.shape)
    denom = jnp.cumprod(twosin, axis=0) / twosin ** (1 - lam)
    vals = c @ (jnp.cos(alpha) / denom)
    ders = -c @ (
        ((n + lam - jj) * jnp.sin(alpha) + (jj + lam) * jnp.cos(alpha) / jnp.tan(t)) / denom
    )
    return vals, ders


@partial(jax.jit, static_argnums=(0, 1))
def _interior(n, lam):
    k = jnp.arange((n + 1) // 2, 0, -1, dtype=jnp.float64)
    t = _initial(n, lam, k)
    # Source nbdy is the whole half-grid for n<=21. Its first Newton
    # correction is always taken, then the empty interior residual stops
    # the loop; a final correction below is still performed.
    cut = 0 if n <= 21 else t.size - 10

    def cond(s):
        residual = jnp.max(jnp.abs(s[2][:cut])) if cut else jnp.float64(0)
        return (s[0] == 0) | (residual > jnp.sqrt(jnp.finfo(jnp.float64).eps) / 1000)

    def step(s):
        val, der = _interior_eval(n, lam, s[1])
        dt = val / der
        return s[0] + 1, s[1] - dt, dt

    _, t, _ = lax.while_loop(cond, step, (0, t, jnp.full_like(t, jnp.inf)))
    val, der = _interior_eval(n, lam, t)
    t = t - val / der
    c = 4 ** (-lam) * jnp.pi * _gamma_ratio(n + lam, lam) * _gamma_ratio(n + lam, 1 - lam)
    x, w, v = jnp.cos(t), c / der**2, jnp.sin(t) / der
    if n % 2:
        x = x.at[0].set(0)
        t = t.at[0].set(jnp.pi / 2)
        return (
            jnp.concatenate((-x[:0:-1], x)),
            jnp.concatenate((w[:0:-1], w)),
            -jnp.concatenate((v[:0:-1], v)),
            jnp.concatenate((jnp.pi - t[:0:-1], t)),
        )
    return (
        jnp.concatenate((-x[::-1], x)),
        jnp.concatenate((w[::-1], w)),
        jnp.concatenate((-v[::-1], v)),
        jnp.concatenate((jnp.pi - t[::-1], t)),
    )


def ultra_rule(n, lam, interval=None, method=None):
    """Return source nodes, quadrature weights, barycentric weights and angles.

    Python represents source row/column outputs as one-dimensional arrays.

    Provenance
    ----------
    MATLAB source : ultrapts.m
    Chebfun commit: 7574c77
    """
    from .quadrature import _ultrapts_gw, chebpts, legpts

    if lam <= -0.5:
        raise ValueError("CHEBFUN:ultrapts:sizeLAMBDA: LAMBDA must be greater than -1/2.")
    if lam >= 30:
        warnings.warn(
            "CHEBFUN:ultrapts:largeLAMBDA: LAMBDA >= 30. Results may not be accurate.", stacklevel=2
        )
    if isinstance(interval, str):
        if method is not None:
            raise ValueError("CHEBFUN:ultrapts:inputs: Method specified twice.")
        method, interval = interval, None
    explicit = method is not None
    method = "default" if method is None else method.lower()
    if method not in ("default", "rec", "asy", "gw"):
        raise ValueError("CHEBFUN:ultrapts:inputs: Unrecognised input string: " + method)
    if interval is not None:
        if len(interval) != 2:
            raise ValueError("CHEBFUN:ultrapts:inputs: Interval invalid.")
        # Validate concrete endpoints without introducing traced predicates.
        # Dynamic endpoints retain the established JAX mapping contract;
        # their validity must be checked by the caller before tracing.
        if not any(isinstance(x, jax.core.Tracer) for x in interval):
            if not all(math.isfinite(float(x)) for x in interval) or interval[0] >= interval[1]:
                raise ValueError("CHEBFUN:ultrapts:inputs: Interval invalid.")
    if int(n) != n or n < 0:
        raise ValueError("CHEBFUN:ultrapts:n: First input should be positive number.")
    n = int(n)
    if n == 0:
        e = jnp.empty(0, dtype=jnp.float64)
        return e, e, e, e
    if n <= 2:
        mass = jnp.sqrt(jnp.pi) * jnp.exp(gammaln(lam + 0.5) - gammaln(lam + 1))
        if n == 1:
            x = jnp.zeros(1)
            w = jnp.asarray([mass])
            v = jnp.ones(1)
            t = jnp.ones(1)
        else:
            x = jnp.array([-1.0, 1.0]) / jnp.sqrt(2 * (1 + lam))
            w = jnp.full(2, mass / 2)
            v = jnp.array([1.0, -1.0])
            t = jnp.arccos(x)
    elif lam == 0:
        x = chebpts(n, kind=1)
        w = jnp.full(n, jnp.pi / n)
        v = jnp.sin(jnp.pi * (jnp.arange(n) + 0.5) / n) * (-1.0) ** jnp.arange(n)
        t = jnp.arccos(x)
    elif lam == 1:
        x = chebpts(n + 2)[1:-1]
        w = jnp.pi / (n + 1) * (1 - x * x)
        v = (1 - x * x) * (-1.0) ** jnp.arange(n)
        t = jnp.arccos(x)
    elif lam == 0.5:
        if method == "rec":
            from .legendre_rec import _legpts_rec

            x, w, v = _legpts_rec(n)
            t = jnp.arccos(x)
        elif method == "gw":
            from .quadrature import _legpts_gw

            x, w = _legpts_gw(n)
            v = jnp.sqrt((1 - x * x) * w) * (-1.0) ** jnp.arange(n)
            v = v / jnp.max(jnp.abs(v))
            t = jnp.arccos(x)
        elif method == "asy":
            from .legendre_fast import _legpts_asy_with_theta

            x, w, v, t = _legpts_asy_with_theta(n)
        else:
            x, w, v, t = legpts(n, newtheta=True)
    else:
        threshold = (
            100
            if lam <= 3
            else 500
            if lam <= 8
            else 1000
            if lam <= 13
            else 2000
            if lam <= 20
            else 3000
        )
        selected = (
            "rec"
            if (not explicit and n < threshold) or method == "rec"
            else "gw"
            if method == "gw"
            else "asy"
        )
        if selected == "rec":
            x, w, v = _rec(n, lam)
            t = jnp.arccos(x)
        elif selected == "gw":
            x, w = _ultrapts_gw(n, lam)
            v = jnp.sqrt(1 - x * x) * jnp.sqrt(w)
            t = jnp.arccos(x)
        else:
            # n and lambda are static under the established quadrature JIT
            # contract. Evaluate the source's data-dependent construction
            # at trace time; the resulting rule is a compiled constant.
            with jax.ensure_compile_time_eval():
                x, w, v, t = _asy(n, lam)
        v = jnp.abs(v) * (-1.0) ** jnp.arange(n)
        v = v / jnp.max(jnp.abs(v))
    if interval is not None:
        a, b = interval
        x = (x + 1) / 2 * (b - a) + a
        w = (0.5 * (b - a)) ** (2 * lam) * w
        if lam == 0 and n > 2:
            # A traced mapped rule has a fixed complex angle dtype, allowing
            # nodes to cross +/-1 as endpoints vary without a Python branch.
            t = (
                jnp.arccos(x.astype(jnp.complex128))
                if isinstance(x, jax.core.Tracer) or bool(jnp.any(jnp.abs(x) > 1))
                else jnp.arccos(x)
            )
    return x, w, v, t


def _firstterms(a, theta, n):
    from ..discretization.chebcolloc import ChebColloc2
    from ..domain import Domain
    from .interpolation import bary

    c = max(float(jnp.max(theta)), 0.5)
    N = int(jnp.ceil(40 - n)) if n < 30 else 15 if c > jnp.pi / 2 - 0.5 else 10
    t = 0.5 * c * (jnp.sin(jnp.pi * jnp.arange(-(N - 1), N, 2) / (2 * (N - 1))) + 1)
    v = (-1.0) ** jnp.arange(N)
    v = v.at[0].multiply(0.5).at[-1].multiply(0.5)
    A = 0.25 - a * a
    B = A
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
    return (
        tuple(bary(theta, z, t, v) for z in (tB1, A2, tB2, A3)),
        A3[:10],
        C[:10, :10],
        D[:10, :10],
        f[:10],
    )


def _boundary_eval(n, lam, t, final=False):
    import math

    from .besselj import besselj as _besselj

    def besselj(order, z):
        # The boundary expansion evaluates real orders at positive arguments.
        return jnp.real(_besselj(order, z))

    a = lam - 0.5
    rho = n + lam
    r2 = rho - 1
    A = lam * (1 - lam)
    (tB1, A2, tB2, A3), Ak, C, D, f = _firstterms(a, t, n)
    Ja = besselj(a, rho * t)
    Jb = besselj(a + 1, rho * t)
    Jbb = besselj(a + 1, r2 * t)
    if final:
        # Source besselTaylor: exact binomial derivatives and 30 Taylor terms.
        bj = [besselj(a + j, rho * t) for j in range(-30, 31)]
        Jab = jnp.zeros_like(t)
        for k in range(31):
            coeff = sum(
                (-1.0) ** l * math.comb(k, l) * bj[30 + 2 * l - k] for l in range(k + 1)
            ) / float(2**k * math.factorial(k))
            Jab = Jab + coeff * (-t) ** k
    else:
        Jab = besselj(a, r2 * t)
    gt = 2 * A * (1 / jnp.tan(t) - 1 / t)
    gtdx = 0.5 * A * (4 / t**2 - 1 / jnp.sin(t / 2) ** 2 - 1 / jnp.cos(t / 2) ** 2)
    tB0 = 0.25 * gt
    A1 = gtdx / 8 - (1 + 2 * a) / 8 * gt / t - gt**2 / 32 - a * A / 6
    val = (
        Ja
        + Jb * tB0 / rho
        + Ja * A1 / rho**2
        + Jb * tB1 / rho**3
        + Ja * A2 / rho**4
        + Jb * tB2 / rho**5
        + Ja * A3 / rho**6
    )
    der = (
        Jab
        + Jbb * tB0 / r2
        + Jab * A1 / r2**2
        + Jbb * tB1 / r2**3
        + Jab * A2 / r2**4
        + Jbb * tB2 / r2**5
        + Jab * A3 / r2**6
    )
    denom = (jnp.sin(t) / 2) ** lam
    scale = jnp.sqrt(t) / denom
    C2 = n / (n + a) * (rho / r2) ** a

    def derivative(vv, dd):
        return (
            (-n * (2 * (n + lam) - 1) * jnp.cos(t) * vv + 2 * (n + a) ** 2 * C2 * dd)
            / (2 * (n + lam) - 1)
            * scale
            / jnp.sin(t)
        )

    ders = derivative(val, der)
    vals = scale * val
    if lam > 1 or lam < 0:
        ww = (jnp.pi / 2 - t[1:]) * (-1.0) ** jnp.arange(t.size - 1)
        ww = ww.at[-1].multiply(0.5)

        def extrap(z):
            return z.at[0].set(jnp.sum(ww * z[1:]) / jnp.sum(ww))

        for k in range(3, 61):
            Ap = D @ Ak
            Ap = Ap - Ap[0]
            Apt = extrap(Ap / t)
            tBk = -0.5 * Ap - (0.5 + a) * (C @ Apt) + 0.5 * (C @ (f * Ak))
            Bk = extrap(tBk / t)
            Ak1 = 0.5 * (D @ tBk) - (0.5 + a) * Bk - 0.5 * (C @ (f * tBk))
            Ak1 = Ak1 - Ak1[0]
            dv = Jb * tBk / rho ** (2 * k + 1) + Ja * Ak1 / rho ** (2 * k + 2)
            dd = Jbb * tBk / r2 ** (2 * k + 1) + Jab * Ak1 / r2 ** (2 * k + 2)
            delta = scale * dv
            deltad = derivative(dv, dd)
            vals = vals + delta
            ders = ders + deltad
            Ak = Ak1
            if not (
                jnp.max(jnp.abs(delta / vals)) > jnp.finfo(jnp.float64).eps
                and jnp.max(jnp.abs(deltad / ders)) > jnp.finfo(jnp.float64).eps
            ):
                break
    return vals, ders


def _asy(n, lam):
    x, w, v, t = _interior(n, lam)
    if n <= 21:
        return x, w, v, t
    if lam > 1 or lam < 0:
        bz = _bessel_roots_general(lam - 0.5, 10)
        nn = jnp.sqrt((n + lam) ** 2 + lam * (1 - lam) / 3)
        tb = bz / nn - lam * (1 - lam) / 90 * (bz**3 + 2 * (lam * lam - lam - 0.75) * bz) / nn**5
    else:
        tb = _initial(n, lam, jnp.arange(1, 11, dtype=jnp.float64))
    for _ in range(20):
        val, der = _boundary_eval(n, lam, tb)
        dt = val / der
        tb = tb + dt
        if jnp.max(jnp.abs(dt)) <= jnp.sqrt(jnp.finfo(jnp.float64).eps) / 1000:
            break
    val, der = _boundary_eval(n, lam, tb, True)
    tb = (tb + val / der)[::-1]
    der = der[::-1]
    c = 2 ** (2 * lam + 1) * (n + lam) ** (2 * lam - 1) * _gamma_ratio(n + 2 * lam, 1 - 2 * lam)
    c1 = (
        2 ** (2 * lam + 0.5)
        / jnp.sqrt(jnp.pi)
        * (n + lam) ** (lam - 0.5)
        * _gamma_ratio(n + 2 * lam, -lam)
    )
    xb = jnp.cos(tb)
    wb = c / der**2
    if bool(jnp.any(wb == 0)):
        warnings.warn(
            "CHEBFUN:ultrapts:largeNLAMDBA: Some WEIGHTS near the boundary region become zero due to overflow.",
            stacklevel=2,
        )
    vb = jnp.sin(tb) / (der / c1)
    return (
        x.at[-10:].set(xb).at[:10].set(-xb[::-1]),
        w.at[-10:].set(wb).at[:10].set(wb[::-1]),
        v.at[-10:].set(vb).at[:10].set(vb[::-1]),
        t.at[-10:].set(tb).at[:10].set(jnp.pi - tb[::-1]),
    )
