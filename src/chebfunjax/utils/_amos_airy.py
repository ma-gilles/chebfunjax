"""Source scalar arithmetic used by the small Hermite ASY path.

Provenance: Chebfun hermpts.m, commit
7574c77680d7e82b79626300bf255498271a72df; AMOS Airy/Bessel algorithms
by Donald E. Amos; Netlib fdlibm e_pow.c, e_exp.c, e_log.c,
e_rem_pio2.c and k_rem_pio2.c.

Fdlibm copyright (C) 1993, 2004 Sun Microsystems, Inc. All rights reserved.
Developed at SunSoft, a Sun Microsystems, Inc. business.
Permission to use, copy, modify, and distribute this software is freely
granted, provided that this notice is preserved.

These are private ordinary-domain source kernels, not complete replacements
for public transcendental APIs. Static mathematical coefficients are kept
in the repository. No captured operands, output tables or runtime fixtures.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from ._source_complex import _native_hypot, _native_moderate_sqrt, source_real_div
from ._source_phase import source_medium_phase_jit, source_order_phase, source_phase_jit
from ._source_real import add as _sadd
from ._source_real import div as _sdiv
from ._source_real import mul as _smul
from ._source_real import source_exp_jit, source_log_jit
from ._source_real import source_gamma_log as _gamma_log
from ._source_real import sub as _ssub

TOL = 2.220446049250313e-16
TINY = 2.2250738585072014e-308
ELIM = 2.303 * (1021 * 0.3010299956639812 - 3)
RL = 1.2 * min(0.3010299956639812 * 52, 18.0) + 3
ALIM = ELIM + max(-0.3010299956639812 * 52 * 2.303, -41.45)


def _complex(real, imag):
    return lax.complex(jnp.asarray(real, dtype=jnp.float64), jnp.asarray(imag, dtype=jnp.float64))


def _abs(z):
    return _native_hypot(z.real, z.imag)


def _mul(a, b):
    return _complex(a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real)


def _ordered_product(a, b):
    """Materialize source complex-product terms before adding them.

    The fixed four-step map is an arithmetic ordering boundary: CPU backend
    contraction otherwise changes separately rounded C++ source products.
    No input-dependent correction or precision change is performed.
    """
    left = jnp.stack((a.real, a.imag, a.real, a.imag))
    right = jnp.stack((b.real, b.imag, b.imag, b.real))
    products = lax.map(lambda pair: pair[0] * pair[1], jnp.stack((left, right), axis=-1))
    return _complex(products[0] - products[1], products[2] + products[3])


def _div(a, b):
    bm = 1 / _abs(b)
    (cc, cd) = (b.real * bm, b.imag * bm)
    return _complex((a.real * cc + a.imag * cd) * bm, (a.imag * cc - a.real * cd) * bm)


def _arg(z):
    (ar, ai) = (z.real, z.imag)
    theta = jnp.arctan(ai / ar)
    theta = jnp.where(
        (theta <= 0) & (ar < 0),
        theta + jnp.pi,
        jnp.where((theta > 0) & (ar < 0), theta - jnp.pi, theta),
    )
    theta = jnp.where(ar == 0, jnp.where(ai < 0, -jnp.pi / 2, jnp.pi / 2), theta)
    return jnp.where(ai == 0, jnp.where(ar < 0, jnp.pi, 0.0), theta)


def _sqrt(z):
    (ar, ai) = (z.real, z.imag)
    radius = jnp.sqrt(_abs(z))
    angle = 0.5 * _arg(z)
    value = _complex(radius * jnp.cos(angle), radius * jnp.sin(angle))
    d = radius * 0.7071067811865476
    value = jnp.where(ar == 0, _complex(d, jnp.where(ai < 0, -d, d)), value)
    value = jnp.where(
        ai == 0,
        jnp.where(ar < 0, _complex(0.0, jnp.sqrt(jnp.abs(ar))), _complex(jnp.sqrt(ar), 0.0)),
        value,
    )
    return jnp.where((ar == 0) & (ai == 0), _complex(0.0, 0.0), value)


def _log(z):
    return _complex(source_log_jit(_abs(z)), _arg(z))


def _exp(z):
    amplitude = source_exp_jit(z.real)
    (sine, cosine) = source_phase_jit(z.imag)
    value = _complex(_smul(amplitude, cosine), _smul(amplitude, sine))
    return jnp.where(z.imag == 0.0, _complex(amplitude, 0.0), value)


def _exp_phase(z, phase):
    """Source exponential with this invocation's freshly computed phase."""
    amplitude = source_exp_jit(z.real)
    (sine, cosine) = phase
    value = _complex(_smul(amplitude, cosine), _smul(amplitude, sine))
    return jnp.where(z.imag == 0.0, _complex(amplitude, 0.0), value)


def _exp_bounded(z):
    amplitude = source_exp_jit(z.real)
    (sine, cosine) = source_medium_phase_jit(z.imag)
    value = _complex(_smul(amplitude, cosine), _smul(amplitude, sine))
    return jnp.where(z.imag == 0.0, _complex(amplitude, 0.0), value)


@partial(jax.jit, static_argnames=("derivative",))
def _airy_small(z, derivative=False):
    """ZAIRY power-series branch for |z| <= 1, unscaled (KODE=1)."""
    z = jnp.asarray(z, dtype=jnp.complex128)
    az = _abs(z)
    fid = float(derivative)
    (c1, c2) = (0.3550280538878172, 0.2588194037928068)
    one = _complex(1.0, 0.0)
    z3 = _mul(_mul(z, z), z)
    az3 = az * (az * az)
    d1 = (2 + fid) * (3 + fid + fid)
    d2 = (3 - fid - fid) * (4 - fid)
    initial = (
        0,
        one,
        one,
        one,
        one,
        1.0,
        d1,
        d2,
        min(d1, d2),
        24 + 9 * fid,
        30 - 9 * fid,
        (az < TOL) | (az * az < TOL / az),
    )

    def condition(s):
        return (s[0] < 25) & ~s[-1]

    def step(s):
        (k, s1, s2, term1, term2, atrm, d1, d2, ad, ak, bk, _) = s
        (p1, p2) = (_mul(term1, z3), _mul(term2, z3))
        term1 = _complex(p1.real / d1, p1.imag / d1)
        term2 = _complex(p2.real / d2, p2.imag / d2)
        (s1, s2) = (s1 + term1, s2 + term2)
        atrm = atrm * az3 / ad
        (d1, d2) = (d1 + ak, d2 + bk)
        ad = jnp.minimum(d1, d2)
        return (k + 1, s1, s2, term1, term2, atrm, d1, d2, ad, ak + 18, bk + 18, atrm < TOL * ad)

    state = lax.while_loop(condition, step, initial)
    (s1, s2) = (state[1], state[2])
    if derivative:
        product = _mul(_mul(z, s1), z)
        ordinary = -c2 * s2 + jnp.where(az <= TOL, 0j, c1 / (1 + fid) * product)
        near = _complex(-c2, 0.0) + jnp.where(
            az <= jnp.sqrt(1000.0 * TINY),
            0j,
            c1 * _complex(0.5 * (z.real * z.real - z.imag * z.imag), z.real * z.imag),
        )
    else:
        ordinary = c1 * s1 - c2 * _mul(z, s2)
        near = _complex(c1, 0.0) - jnp.where(az <= 1000.0 * TINY, 0j, c2 * z)
    return jnp.where(az < TOL, near, ordinary)


@jax.jit
def _i_asymptotic(z, order, phases=None):
    """ZASYI specialized to N=1 and KODE=1; return (I, NZ).

    Provenance
    ----------
    Netlib AMOS zasyi.f, labels 10--70, 100 and 110. This private kernel
    retains the source branch precondition, Re(z)>=0 and |z|>=RL.
    """
    z = jnp.asarray(z, dtype=jnp.complex128)
    az = _abs(z)
    prefactor = jnp.sqrt(0.15915494309189535 / z)
    prefactor = _mul(prefactor, _exp(z) if phases is None else _exp_phase(z, phases[0]))
    fdn = (order + order) * (order + order)
    ez = z * 8.0
    aez = 8.0 * az
    sqk = fdn - 1.0
    atol = TOL / aez * jnp.abs(sqk)
    phase = _complex(
        -jnp.sin(order * jnp.pi), jnp.where(z.imag < 0, -1.0, 1.0) * jnp.cos(order * jnp.pi)
    )
    phase = jnp.where(z.imag == 0.0, _complex(0.0, 0.0), phase)
    one = _complex(1.0, 0.0)
    initial = (0, one, one, one, ez, 1.0, aez, 0.0, sqk, 1.0, False)

    def condition(s):
        return (s[0] < int(2 * RL) + 2) & ~s[-1]

    def step(s):
        (j, cs1, cs2, ck, dk, aa, bb, ak, sqk, sign, _) = s
        ck = _div(ck, dk) * sqk
        cs2 = cs2 + ck
        sign = -sign
        cs1 = cs1 + ck * sign
        dk = dk + ez
        aa = aa * jnp.abs(sqk) / bb
        bb = bb + aez
        ak = ak + 8.0
        sqk = sqk - ak
        return (j + 1, cs1, cs2, ck, dk, aa, bb, ak, sqk, sign, aa <= atol)

    state = lax.while_loop(condition, step, initial)
    exponential = _exp(-2.0 * z) if phases is None else _exp_phase(-2.0 * z, phases[1])
    correction = _mul(_mul(exponential, phase), state[2])
    s2 = state[1] + jnp.where(2 * z.real >= ELIM, 0j, correction)
    value = _ordered_product(s2, prefactor)
    status = jnp.where(jnp.abs(z.real) > ELIM, -1, jnp.where(state[-1], 0, -2))
    return (value, status)


@jax.jit
def _source_real_complex_divide(a, z):
    (br, bi) = (z.real, z.imag)
    zero = jnp.asarray(0.0, jnp.float64)

    def real_axis(_):
        return _complex(_sdiv(a, br), zero)

    def imaginary_axis(_):
        return _complex(zero, -_sdiv(a, bi))

    def general(_):

        def wide(_):
            r = _sdiv(bi, br)
            den = _sadd(br, _smul(bi, r))
            return _complex(_sdiv(a, den), _sdiv(-_smul(a, r), den))

        def narrow(_):
            r = _sdiv(br, bi)
            den = _sadd(bi, _smul(br, r))
            return _complex(_sdiv(_smul(a, r), den), _sdiv(-a, den))

        def equal(_):
            ar = jnp.abs(br)
            r = jnp.copysign(jnp.asarray(0.5), br)
            i = jnp.copysign(jnp.asarray(0.5), bi)
            return _complex(_sdiv(_smul(a, r), ar), _sdiv(-_smul(a, i), ar))

        return lax.cond(
            jnp.abs(br) > jnp.abs(bi),
            wide,
            lambda _: lax.cond(jnp.abs(br) == jnp.abs(bi), equal, narrow, None),
            None,
        )

    return lax.cond(
        bi == 0.0, real_axis, lambda _: lax.cond(br == 0.0, imaginary_axis, general, None), None
    )


_GAMMA_LOG_INTEGERS = (
    0.0,
    0.0,
    0.6931471805599453,
    1.791759469228055,
    3.1780538303479458,
    4.787491742782046,
    6.579251212010101,
    8.525161361065415,
    10.60460290274525,
    12.801827480081469,
    15.104412573075516,
    17.502307845873887,
    19.987214495661885,
    22.552163853123425,
    25.19122118273868,
    27.89927138384089,
    30.671860106080672,
    33.50507345013689,
    36.39544520803305,
    39.339884187199495,
    42.335616460753485,
    45.38013889847691,
    48.47118135183523,
    51.60667556776438,
    54.78472939811232,
    58.00360522298052,
    61.261701761002,
    64.55753862700634,
    67.88974313718154,
    71.25703896716801,
    74.65823634883016,
    78.0922235533153,
    81.55795945611504,
    85.05446701758152,
    88.58082754219768,
    92.1361756036871,
    95.7196945421432,
    99.33061245478743,
    102.96819861451381,
    106.63176026064346,
    110.32063971475739,
    114.0342117814617,
    117.77188139974507,
    121.53308151543864,
    125.3172711493569,
    129.12393363912722,
    132.95257503561632,
    136.80272263732635,
    140.67392364823425,
    144.5657439463449,
    148.47776695177302,
    152.40959258449735,
    156.3608363030788,
    160.3311282166309,
    164.32011226319517,
    168.32744544842765,
    172.3527971391628,
    176.39584840699735,
    180.45629141754378,
    184.53382886144948,
    188.6281734236716,
    192.7390472878449,
    196.86618167289,
    201.00931639928152,
    205.1681994826412,
    209.34258675253685,
    213.53224149456327,
    217.73693411395422,
    221.95644181913033,
    226.1905483237276,
    230.43904356577696,
    234.70172344281826,
    238.97838956183432,
    243.2688490029827,
    247.57291409618688,
    251.8904022097232,
    256.22113555000954,
    260.5649409718632,
    264.9216497985528,
    269.2910976510198,
    273.6731242856937,
    278.0675734403661,
    282.4742926876304,
    286.893133295427,
    291.3239500942703,
    295.76660135076065,
    300.22094864701415,
    304.6868567656687,
    309.1641935801469,
    313.65282994987905,
    318.1526396202093,
    322.66349912672615,
    327.1852877037752,
    331.7178871969285,
    336.26118197919845,
    340.815058870799,
    345.37940706226686,
    349.95411804077025,
    354.5390855194408,
    359.1342053695754,
)
_GAMMA_CF = (
    0.08333333333333333,
    -0.002777777777777778,
    0.0007936507936507937,
    -0.0005952380952380953,
    0.0008417508417508417,
    -0.0019175269175269176,
    0.00641025641025641,
    -0.029550653594771242,
    0.17964437236883057,
    -1.3924322169059011,
    13.402864044168393,
    -156.84828462600203,
    2193.1033333333335,
    -36108.77125372499,
    691472.268851313,
    -15238221.539407415,
    382900751.39141417,
    -10882266035.784391,
    347320283765.00226,
    -12369602142269.275,
    488788064793079.3,
    -2.1320333960919372e16,
)


@jax.jit
def _gamma_log_legacy(z):
    """AMOS DGAMLN positive-real branch, with IEEE binary64 constants."""
    z = jnp.asarray(z, dtype=jnp.float64)
    nz = z.astype(jnp.float32).astype(jnp.int32)
    zinc = jnp.where(z < 7.0, 7.0 - nz.astype(jnp.float64), 0.0)
    zdmy = z + zinc
    zp = 1.0 / zdmy
    cf = jnp.asarray(_GAMMA_CF, dtype=jnp.float64)
    first = cf[0] * zp

    def condition(s):
        return (s[0] < 22) & ~s[-1]

    def step(s):
        (k, power, total, _) = s
        power = power * (zp * zp)
        term = cf[k] * power
        done = jnp.abs(term) < first * TOL
        return (k + 1, power, jnp.where(done, total, total + term), done)

    state = lax.while_loop(condition, step, (1, zp, first, zp < TOL))
    product = lax.fori_loop(
        0, zinc.astype(jnp.int32), lambda i, p: p * (z + i.astype(jnp.float64)), 1.0
    )
    logarithm = jnp.log(zdmy)
    value = (
        zdmy * (logarithm - 1.0)
        - jnp.log(product)
        + 0.5 * (1.8378770664093456 - logarithm)
        + state[2]
    )
    table = jnp.asarray(_GAMMA_LOG_INTEGERS, dtype=jnp.float64)
    value = jnp.where((z <= 100.0) & (z == nz), table[jnp.clip(nz - 1, 0, 99)], value)
    return jnp.where(z > 0.0, value, jnp.nan)


@jax.jit
def _i_miller(z, order):
    """ZMLRI for Airy's orders (0<order<1), N=1, KODE=1.

    Provenance
    ----------
    Netlib AMOS zmlri.f. The ratio-search branch is retained even though
    the Airy continuation normally supplies |z|>2 to this kernel.
    """
    z = jnp.asarray(z, dtype=jnp.complex128)
    az = _abs(z)
    iaz = az.astype(jnp.float32).astype(jnp.int32)
    at = iaz.astype(jnp.float64) + 1.0
    raz = 1.0 / az
    direction = _complex(z.real * raz, -z.imag * raz)  # noqa: F841 - source step retained
    ck = _source_real_complex_divide(at, z)
    rz = _source_real_complex_divide(jnp.asarray(2.0), z)
    ack = _sdiv(_sadd(at, 1.0), az)
    rho = ack + jnp.sqrt(ack * ack - 1.0)
    rho2 = rho * rho
    tst = (rho2 + rho2) / ((rho2 - 1.0) * (rho - 1.0)) / TOL
    (zero, one) = (_complex(0.0, 0.0), _complex(1.0, 0.0))

    def search_condition(s):
        return (s[0] < 80) & ~s[-1]

    def search_step(s):
        (i, p1, p2, ck, ak, _) = s
        next_p = p1 - _mul(ck, p2)
        done = _abs(next_p) > tst * ak * ak
        return (i + 1, p2, next_p, ck + rz, jnp.where(done, ak, ak + 1.0), done)

    search = lax.while_loop(search_condition, search_step, (0, zero, one, ck, at, False))

    def ratio_search(_):
        initial_tst = jnp.sqrt(raz / TOL)

        def condition(s):
            return (s[0] < 80) & ~s[-1]

        def step(s):
            (k, p1, p2, ck, threshold, second, _) = s
            next_p = p1 - _mul(ck, p2)
            ck = ck + rz
            ap = _abs(next_p)
            reached = ap >= threshold
            ack = _abs(ck)
            flam = ack + jnp.sqrt(ack * ack - 1.0)
            fkap = ap / _abs(p2)
            rho = jnp.minimum(flam, fkap)
            threshold = jnp.where(
                reached & ~second, threshold * jnp.sqrt(rho / (rho * rho - 1.0)), threshold
            )
            return (k + 1, p2, next_p, ck, threshold, second | reached, second & reached)

        state = lax.while_loop(
            condition,
            step,
            (
                0,
                zero,
                one,
                _source_real_complex_divide(jnp.asarray(1.0), z),
                initial_tst,
                False,
                False,
            ),
        )
        return (state[0], state[-1])

    (ratio_k, ratio_ok) = lax.cond(
        iaz <= 0, ratio_search, lambda _: (jnp.asarray(0), jnp.asarray(True)), None
    )
    kk = jnp.maximum(search[0] + 1 + iaz, ratio_k + 1)
    fkk = kk.astype(jnp.float64)
    tfnf = order + order
    bk = source_exp_jit(
        _ssub(_ssub(_gamma_log(fkk + tfnf + 1.0), _gamma_log(fkk + 1.0)), _gamma_log(tfnf + 1.0))
    )

    def backward(_, state):
        (p1, p2, fkk, bk, total) = state
        factor = _sadd(fkk, order)
        (cr, ci) = (_smul(factor, rz.real), _smul(factor, rz.imag))
        nr = _ssub(_smul(cr, p2.real), _smul(ci, p2.imag))
        ni = _sadd(_smul(ci, p2.real), _smul(cr, p2.imag))
        next_p = _complex(_sadd(p1.real, nr), _sadd(p1.imag, ni))
        ak = _ssub(1.0, _sdiv(tfnf, _sadd(fkk, tfnf)))
        ack = _smul(bk, ak)
        weight = _sadd(ack, bk)
        total = _complex(
            _sadd(total.real, _smul(weight, p2.real)), _sadd(total.imag, _smul(weight, p2.imag))
        )
        return (p2, next_p, fkk - 1.0, ack, total)

    state = lax.fori_loop(
        0, kk, backward, (zero, _complex(1000.0 * TINY / TOL, 0.0), fkk, bk, zero)
    )
    value_unscaled = state[1]
    log_rz = _log(rz)
    exponent = _complex(
        _ssub(_sadd(_smul(-order, log_rz.real), z.real), _gamma_log(1.0 + order)),
        _sadd(_smul(-order, log_rz.imag), z.imag),
    )
    denominator = value_unscaled + state[4]
    scale = 1.0 / _abs(denominator)
    exp_value = _exp_bounded(exponent)
    normalization = _ordered_product(
        _complex(_smul(exp_value.real, scale), _smul(exp_value.imag, scale)),
        _complex(_smul(denominator.real, scale), -_smul(denominator.imag, scale)),
    )
    return (
        _ordered_product(value_unscaled, normalization),
        jnp.where(search[-1] & ratio_ok, 0, -2),
    )


@jax.jit
def _i_series(z, order):
    """ZSERI, N=1/KODE=1, ordinary-scale branch used by Airy continuation."""
    z = jnp.asarray(z, dtype=jnp.complex128)
    half = 0.5 * z
    cz = _mul(half, half)
    acz = _abs(cz)
    logarithm = _log(half)
    exponent = _complex(logarithm.real * order - _gamma_log(order + 1.0), logarithm.imag * order)
    coefficient = _exp_bounded(exponent)
    fnup = order + 1.0
    atol = TOL * acz / fnup
    one = _complex(1.0, 0.0)

    def condition(s):
        return ~s[-1]

    def step(s):
        (term, total, denominator, ak, aa, _) = s
        reciprocal = 1.0 / denominator
        term = _mul(term, cz) * reciprocal
        aa = aa * acz * reciprocal
        return (term, total + term, denominator + ak, ak + 2.0, aa, aa <= atol)

    state = lax.while_loop(condition, step, (one, one, fnup, fnup + 2.0, 2.0, acz < TOL * fnup))
    return (_mul(state[1], coefficient), jnp.asarray(0))


@jax.jit
def _k_bessel(z, order, phase=None):
    """ZBKNU for orders 1/3,2/3, N=1/KODE=1, ordinary exponent range.

    Provenance
    ----------
    Netlib AMOS zbknu.f labels 10--240, specialized to Airy's orders.
    The near-underflow scaled branch (labels 270 onward) is not yet ported;
    this private function requires Re(z) <= ALIM and |z| >= 2/3.
    """
    z = jnp.asarray(z, dtype=jnp.complex128)
    caz = _abs(z)
    rz = source_real_div(2.0, z)
    inu = (order + 0.5).astype(jnp.float32).astype(jnp.int32)
    dnu = order - inu.astype(jnp.float64)
    dnu2 = dnu * dnu

    def series(_):
        smu = _log(rz)
        fmu = smu * dnu
        sh = _complex(
            jnp.sinh(fmu.real) * jnp.cos(fmu.imag), jnp.cosh(fmu.real) * jnp.sin(fmu.imag)
        )
        ch = _complex(
            jnp.cosh(fmu.real) * jnp.cos(fmu.imag), jnp.sinh(fmu.real) * jnp.sin(fmu.imag)
        )
        fc = dnu * jnp.pi
        fc = fc / jnp.sin(fc)
        smu = sh / dnu
        t2 = jnp.exp(-_gamma_log(1.0 + dnu))
        t1 = 1.0 / (t2 * fc)
        g1 = (t1 - t2) / (dnu + dnu)
        g2 = (t1 + t2) * 0.5
        f = fc * (ch * g1 + smu * g2)
        efmu = _exp_bounded(fmu)
        p = 0.5 * efmu / t2
        q = _div(_complex(0.5, 0.0), efmu) / t1
        cz = 0.25 * _ordered_product(z, z)
        t1 = 0.25 * caz * caz
        one = _complex(1.0, 0.0)

        def condition(s):
            return s[-1] > TOL

        def step(s):
            (f, p, q, ck, s1, s2, ak, bk, a1) = s
            f = (f * ak + p + q) / bk
            p = p * (1.0 / (ak - dnu))
            q = q * (1.0 / (ak + dnu))
            rak = 1.0 / ak
            ck = _ordered_product(ck, cz) * rak
            s1 = _ordered_product(ck, f) + s1
            s2 = _ordered_product(ck, p - f * ak) + s2
            return (f, p, q, ck, s1, s2, ak + 1.0, bk + ak + ak + 1.0, a1 * t1 * rak)

        state = lax.while_loop(condition, step, (f, p, q, one, f, p, 1.0, 1.0 - dnu2, 1.0))
        value = jnp.where(inu == 0, state[4], _ordered_product(state[5], rz))
        return (value, jnp.asarray(0))

    def miller(_):
        coefficient = source_real_div(1.2533141373155003, _native_moderate_sqrt(z))
        amplitude = source_exp_jit(-z.real)
        (sine, cosine) = source_phase_jit(z.imag) if phase is None else phase
        coefficient = _ordered_product(
            coefficient, _complex(_smul(amplitude, cosine), -_smul(amplitude, sine))
        )
        cosine = jnp.abs(jnp.cos(jnp.pi * dnu))
        fhs = jnp.abs(0.25 - dnu2)
        binary_digits = min(max(52 * 0.3010299956639812 * 3.321928094, 12.0), 60.0)
        boundary = 0.6666666666666666 * binary_digits - 6.0
        angle = jnp.where(z.real == 0.0, 1.5707963267948966, jnp.abs(jnp.arctan(z.imag / z.real)))

        def forward_index(_):
            etest = cosine / (jnp.pi * caz * TOL)

            def condition(s):
                return (s[0] < 30) & ~s[-1]

            def step(s):
                (i, fk, fks, ck, p1, p2, fhs, _) = s
                ak = fhs / fks
                cb = ck / (fk + 1.0)
                pnext = cb * p2 - p1 * ak
                fks = fks + fk + fk + 2.0
                fhs = fhs + fk + fk
                fk = fk + 1.0
                return (i + 1, fk, fks, ck + 2.0, p2, pnext, fhs, etest < jnp.abs(pnext) * fk)

            state = lax.while_loop(
                condition, step, (0, 1.0, 2.0, caz + caz + 2.0, 0.0, 1.0, fhs, etest < 1.0)
            )
            fk = jnp.where(
                etest < 1.0, 1.0, state[1] + 1.909859317102744 * angle * jnp.sqrt(boundary / caz)
            )
            return (fk, jnp.where(state[-1], 0, -2))

        def estimated_index(_):
            a2 = jnp.sqrt(caz)
            ak = 1.8976999933151775 * cosine / (TOL * jnp.sqrt(a2))
            aa = 3.0 * angle / (1.0 + caz)
            bb = 14.7 * angle / (28.0 + caz)
            ak = (jnp.log(ak) + caz * jnp.cos(aa) / (1.0 + 0.008 * caz)) / jnp.cos(bb)
            return (0.12125 * ak * ak / caz + 1.5, jnp.asarray(0))

        (fk, status) = lax.cond(boundary <= caz, forward_index, estimated_index, None)
        k = fk.astype(jnp.float32).astype(jnp.int32)
        fk = k.astype(jnp.float64)
        zero = _complex(0.0, 0.0)
        seed = _complex(TOL, 0.0)

        def backward(_, s):
            (fk, fks, p1, p2, total) = s
            a1 = fks - fk
            ak = (fks + fk) / (a1 + fhs)
            rak = 2.0 / (fk + 1.0)
            cb = _complex((fk + z.real) * rak, z.imag * rak)
            next_p = (_ordered_product(p2, cb) - p1) * ak
            return (fk - 1.0, a1 - fk + 1.0, p2, next_p, total + next_p)

        state = lax.fori_loop(0, k, backward, (fk, fk * fk, zero, seed, seed))
        (p1, p2, total) = state[2:]
        reciprocal = 1.0 / _abs(total)
        s1 = _ordered_product(
            _ordered_product(coefficient, p2 * reciprocal),
            _complex(total.real * reciprocal, -total.imag * reciprocal),
        )
        reciprocal = 1.0 / _abs(p2)
        ratio = _ordered_product(
            p1 * reciprocal, _complex(p2.real * reciprocal, -p2.imag * reciprocal)
        )
        factor = _div(_complex(dnu + 0.5 - ratio.real, -ratio.imag), z)
        factor = _complex(factor.real + 1.0, factor.imag)
        s2 = _ordered_product(factor, s1)
        return (jnp.where(inu == 0, s1, s2), status)

    return lax.cond(caz <= 2.0, series, miller, None)


@jax.jit
def _airy_unscaled_dynamic(z, order, derivative):
    """Private ZAIRY translation without extreme-exponent scaling.

    Returns (value, status). Status -99 explicitly marks the unported
    extreme-exponent region; other nonzero statuses retain source meaning.
    This is a diagnostic kernel, not a replacement public Airy API.

    Provenance
    ----------
    Netlib AMOS ZAIRY/ZACAI, KODE=1, ID=0/1. Chebfun commit: 7574c77.
    """
    z = jnp.asarray(z, dtype=jnp.complex128)

    def large(_):
        root = _native_moderate_sqrt(z)
        product = _ordered_product(z, root)
        zeta = 0.6666666666666666 * product
        zr = jnp.where(z.real < 0.0, -jnp.abs(zeta.real), zeta.real)
        zr = jnp.where((z.imag == 0.0) & (z.real <= 0.0), 0.0, zr)
        zeta = _complex(zr, zeta.imag)
        direct = (zr >= 0.0) & (z.real > 0.0)
        k_argument = lax.cond(direct, lambda _: zeta, lambda _: -zeta, None)
        # Share only identical phase operands within this invocation. Keep
        # the original complex -2*z arithmetic used by ZASYI.
        phase_arguments = jnp.stack((k_argument.imag, (-2.0 * k_argument).imag))
        phases = lax.map(lambda argument: jnp.stack(source_phase_jit(argument)), phase_arguments)
        (k_value, k_status) = _k_bessel(k_argument, order, phase=phases[0])

        def continuation(_):
            reflected = -zeta
            az = _abs(zeta)
            (i_value, i_status) = lax.cond(
                (az <= 2.0) | (az * az * 0.25 <= order + 1.0),
                lambda _: _i_series(reflected, order),
                lambda _: lax.cond(
                    az >= RL,
                    lambda _: _i_asymptotic(reflected, order, phases=phases),
                    lambda _: _i_miller(reflected, order),
                    None,
                ),
                None,
            )
            sign = jnp.where(z.imag < 0.0, jnp.pi, -jnp.pi)
            (cspnr, cspni) = source_order_phase(order, jnp.where(z.imag < 0.0, 1.0, -1.0))
            value = _complex(
                _ssub(
                    _ssub(_smul(cspnr, k_value.real), _smul(cspni, k_value.imag)),
                    _smul(sign, i_value.imag),
                ),
                _sadd(
                    _sadd(_smul(cspnr, k_value.imag), _smul(cspni, k_value.real)),
                    _smul(sign, i_value.real),
                ),
            )
            return (value, jnp.where(i_status != 0, i_status, k_status))

        (bessel, status) = lax.cond(direct, lambda _: (k_value, k_status), continuation, None)
        scaled = bessel * 0.18377629847393068
        value = lax.cond(
            derivative,
            lambda _: -_ordered_product(z, scaled),
            lambda _: _ordered_product(root, scaled),
            None,
        )
        unsupported = jnp.abs(zr) >= ALIM
        range_limit = min(0.5 / TOL, 0.5 * 2147483647) ** 0.6666666666666666
        range_status = jnp.where(
            _abs(z) > range_limit, 4, jnp.where(_abs(z) > range_limit**0.5, 3, 0)
        )
        status = jnp.where(status == 0, range_status, status)
        invalid = unsupported | (range_status == 4)
        return (
            jnp.where(invalid, _complex(jnp.nan, jnp.nan), value),
            jnp.where(unsupported, -99, status),
        )

    return lax.cond(
        _abs(z) <= 1.0,
        lambda _: (
            lax.cond(
                derivative,
                lambda _: _airy_small(z, derivative=True),
                lambda _: _airy_small(z, derivative=False),
                None,
            ),
            jnp.asarray(0),
        ),
        large,
        None,
    )


def airy_unscaled_moderate(z, derivative=False):
    return _airy_unscaled_dynamic(
        jnp.asarray(z, jnp.complex128),
        jnp.asarray((1.0 + float(derivative)) / 3.0, jnp.float64),
        jnp.asarray(derivative, jnp.bool_),
    )
