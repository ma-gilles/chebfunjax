"""Ordered CPU binary64 Fourier arithmetic for length602.

Provenance: Chebfun7574c77680d7e82b79626300bf255498271a72df,
@chebtech2/coeffs2vals.m and @chebtech2/vals2coeffs.m call FFT/IFFT.
The MATLAB R2025b CPU execution uses7x2x43complex inverse stages and
2x7x43real forward stages. Source operation graphs were established from
neutral bounded entry/return observations and inspected arithmetic. All
array calculations use JAX. Phase constants use independently generated
integer octant folding and binary64 scalar angles; six radix7constants use
100decimal-digit trigonometric series. No native library, tables, operands
or reference outputs are read. Normalization divides by602 directly.

Moderate finite CPU controls qualify367220complex inverse outputwords,
275424nonredundant real FFT words,690real-stage and204inverse-stage words.
These counts do not establish full extreme-value/backend/performance parity.
"""

import math
from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp

from chebfunjax.utils._ordered_binary import binary

with localcontext() as ctx:
    ctx.prec = 100

    def atan_series(q):
        x = Decimal(1) / q
        total = term = x
        for k in range(1, 200):
            term *= -x * x
            total += term / (2 * k + 1)
        return total
    pi = 16 * atan_series(5) - 4 * atan_series(239)

    def trig(k, cosine):
        x = pi * k / 7
        total = term = Decimal(1) if cosine else x
        for j in range(1, 120):
            term *= -x * x / ((2 * j - 1) * (2 * j) if cosine else 2 * j * (2 * j + 1))
            total += term
        return float(total)
    (S1, S2, S3) = (trig(k, False) for k in (2, 4, 6))
    (C1, C2, C3) = (trig(k, True) for k in (2, 3, 1))

def add(a, b):
    return binary(a, b, 1)

def sub(a, b):
    return binary(a, b, 2)

def mul(a, b):
    return binary(a, b, 0)

def rotate(a):
    return jnp.stack((-a[..., 1], a[..., 0]), axis=-1)

def twiddle(z, w):
    return add(mul(rotate(z), w[..., 1, None]), mul(z, w[..., 0, None]))

@jax.jit
def radix2(x, w):
    z = twiddle(x[1], w)
    return jnp.stack((add(z, x[0]), sub(x[0], z)))

@jax.jit
def radix7(x, w):
    a = twiddle(x[2], w[1])
    b = twiddle(x[5], w[4])
    d25 = sub(a, b)
    s25 = add(b, a)
    a = twiddle(x[1], w[0])
    b = twiddle(x[6], w[5])
    d16 = sub(a, b)
    s16 = add(b, a)
    a = twiddle(x[3], w[2])
    b = twiddle(x[4], w[3])
    d34 = sub(a, b)
    s34 = add(b, a)
    y0 = add(add(add(s25, s34), s16), x[0])
    v2 = rotate(sub(sub(mul(d16, S2), mul(d25, S3)), mul(d34, S1)))
    a2 = add(sub(sub(x[0], mul(s16, C2)), mul(s25, C3)), mul(s34, C1))
    y2 = add(a2, v2)
    y5 = sub(a2, v2)
    v3 = rotate(add(sub(mul(d34, S2), mul(d25, S1)), mul(d16, S3)))
    a3 = add(sub(sub(x[0], mul(s16, C3)), mul(s34, C2)), mul(s25, C1))
    y3 = add(a3, v3)
    y4 = sub(a3, v3)
    v1 = rotate(add(add(mul(d25, S2), mul(d34, S3)), mul(d16, S1)))
    a1 = add(sub(sub(x[0], mul(s25, C2)), mul(s34, C3)), mul(s16, C1))
    y1 = add(a1, v1)
    y6 = sub(a1, v1)
    return jnp.stack((y0, y1, y2, y3, y4, y5, y6))

def phase_pair(k, n):
    q = 4 * (k % n)
    reflected = q > 2 * n
    if reflected:
        q = 4 * n - q
    quarter = q > n
    if quarter:
        q -= n
    diagonal = n - q < q
    if diagonal:
        q = n - q
    angle = float(q) * math.tau / float(4 * n)
    (c, s) = (math.cos(angle), math.sin(angle))
    if diagonal:
        (c, s) = (s, c)
    if quarter:
        (c, s) = (-s, c)
    if reflected:
        s = -s
    return (c, s)

def table43():
    return tuple((tuple((phase_pair(k * j, 43) for j in range(1, 22))) for k in range(1, 22)))

@jax.jit
def transform_pairs(x, twiddles):
    sums = binary(x[1:22], x[42:21:-1], 1)
    differences = binary(x[1:22], x[42:21:-1], 2)
    zero = jax.lax.fori_loop(0, 21, lambda j, acc: binary(acc, sums[j], 1), x[0])
    cosine = jnp.broadcast_to(x[0], (21, 2))
    sine = jnp.zeros((21, 2), dtype=x.dtype)

    def step(j, state):
        (c, s) = state
        wc = twiddles[:, j, 0, None]
        ws = twiddles[:, j, 1, None]
        return (binary(c, binary(sums[j], wc, 0), 1), binary(s, binary(differences[j], ws, 0), 1))
    (cosine, sine) = jax.lax.fori_loop(0, 21, step, (cosine, sine))
    positive = jnp.stack((binary(cosine[:, 0], sine[:, 1], 1), binary(cosine[:, 1], sine[:, 0], 2)), axis=1)
    negative = jnp.stack((binary(cosine[:, 0], sine[:, 1], 2), binary(cosine[:, 1], sine[:, 0], 1)), axis=1)
    return jnp.concatenate((zero[None, :], positive, negative[::-1]), axis=0)

TABLE = table43()

@jax.jit
def transform(x):
    return transform_pairs(x, jnp.asarray(TABLE, dtype=jnp.float64))

TWO = tuple((phase_pair(j, 86) for j in range(43)))

SEVEN = tuple((tuple((phase_pair(k * j, 602) for j in range(86))) for k in range(1, 7)))

@jax.jit
def inverse602_pairs(x):
    a = x.reshape(43, 14, 2).transpose(1, 0, 2)
    a = jax.vmap(lambda p: transform(p[:, ::-1])[:, ::-1])(a)
    a = a.reshape(2, 7, 43, 2).transpose(1, 0, 2, 3)
    w2 = jnp.asarray(TWO, jnp.float64)
    a = jax.vmap(lambda p: radix2(p, w2))(a).reshape(7, 86, 2)
    a = radix7(a, jnp.asarray(SEVEN, jnp.float64)).reshape(602, 2)
    return binary(a, jnp.asarray(602.0, jnp.float64), 3)

T7 = tuple((tuple((phase_pair(k * j, 301) for j in range(43))) for k in range(1, 7)))

TH = tuple((phase_pair(k, 602) for k in range(1, 151)))

def conjugate(x):
    return jnp.stack((x[..., 0], -x[..., 1]), axis=-1)

def forward_twiddle(x, w):
    return sub(mul(x, w[..., 0, None]), mul(rotate(x), w[..., 1, None]))

@jax.jit
def forward7(x, w):
    p3 = forward_twiddle(x[3], w[2])
    p4 = forward_twiddle(x[4], w[3])
    s34 = add(p4, p3)
    d43 = sub(p4, p3)
    p1 = forward_twiddle(x[1], w[0])
    p6 = forward_twiddle(x[6], w[5])
    s16 = add(p6, p1)
    d61 = sub(p6, p1)
    p2 = forward_twiddle(x[2], w[1])
    p5 = forward_twiddle(x[5], w[4])
    s25 = add(p5, p2)
    d52 = sub(p5, p2)
    y0 = add(add(add(s34, s25), s16), x[0])
    v4 = rotate(add(sub(mul(d43, S2), mul(d52, S1)), mul(d61, S3)))
    a4 = add(sub(sub(x[0], mul(s16, C3)), mul(s34, C2)), mul(s25, C1))
    y4 = sub(a4, v4)
    y3 = add(a4, v4)
    v5 = rotate(sub(sub(mul(d61, S2), mul(d52, S3)), mul(d43, S1)))
    a5 = add(sub(sub(x[0], mul(s16, C2)), mul(s25, C3)), mul(s34, C1))
    y5 = sub(a5, v5)
    y2 = add(a5, v5)
    v6 = rotate(add(add(mul(d52, S2), mul(d43, S3)), mul(d61, S1)))
    a1 = add(sub(sub(x[0], mul(s25, C2)), mul(s34, C3)), mul(s16, C1))
    y6 = sub(a1, v6)
    y1 = add(a1, v6)
    return jnp.stack((y0, y1, y2, y3, y4, y5, y6))

@jax.jit
def half_complex(left, right, w):
    a = conjugate(right)
    total = add(a, left)
    difference = sub(a, left)
    v = add(mul(rotate(difference), w[..., 0, None]), mul(difference, w[..., 1, None]))
    return (mul(add(total, v), 0.5), conjugate(mul(sub(total, v), 0.5)))

@jax.jit
def real602_pairs(x):
    z = jnp.stack((x[::2], x[1::2]), axis=-1)
    z = z.reshape(43, 7, 2).transpose(1, 0, 2)
    z = jax.vmap(transform)(z)
    z = forward7(z, jnp.asarray(T7, jnp.float64)).reshape(301, 2)
    (lo, hi) = half_complex(z[1:151], z[300:150:-1], jnp.asarray(TH, jnp.float64))
    zero = jnp.stack((add(z[0, 0], z[0, 1]), jnp.asarray(0.0, jnp.float64)))
    nyquist = jnp.stack((sub(z[0, 0], z[0, 1]), jnp.asarray(0.0, jnp.float64)))
    return jnp.concatenate((zero[None], lo, hi[::-1], nyquist[None]), axis=0)

@jax.jit
def complex_inverse602(values):
    "Return the length602complex inverse along axis0, preserving batch shape."
    shape = values.shape
    x = values.reshape(602, -1).T
    x = jnp.stack((jnp.real(x), jnp.imag(x)), axis=-1)
    y = jax.vmap(inverse602_pairs)(x).transpose(1, 0, 2)
    z = jax.lax.complex(y[..., 0], y[..., 1])
    return z.reshape(shape)


@jax.jit
def real_forward602_values(values):
    "Return real parts of302nonredundant forward frequencies along axis0."
    y = jax.vmap(real602_pairs)(values.reshape(602, -1).T)
    return y[..., 0].T.reshape((302,) + values.shape[1:])


@jax.jit
def real_even_inverse602_column(x):
    """Recombine Hermitian frequencies, then source radix7/43 inverse stages."""
    left = jnp.stack((x[1:151], jnp.zeros(150)), axis=-1)
    right = jnp.stack((x[300:150:-1], jnp.zeros(150)), axis=-1)
    a = conjugate(right)
    total = add(a, left)
    diff = sub(left, a)
    w = jnp.asarray(TH)
    v = sub(mul(rotate(diff), w[:, 0, None]), mul(diff, w[:, 1, None]))
    lo = add(total, v)
    hi = conjugate(sub(total, v))
    zero = jnp.stack((add(x[0], x[301]), sub(x[0], x[301])))
    y = jnp.concatenate((zero[None], lo, hi[::-1]), axis=0)
    y = y.reshape(43, 7, 2).transpose(1, 0, 2)
    y = jax.vmap(lambda p: transform(p[:, ::-1])[:, ::-1])(y)
    y = radix7(y, jnp.asarray(T7)).reshape(301, 2).reshape(602)
    return binary(y, jnp.asarray(602.0, jnp.float64), 3)[:302]


@jax.jit
def real_inverse602_values(values):
    """Real IFFT prefix for real-even602 inputs along axis0.

    Native CPU plan: hc2cbdftv_2_avx, radix7 inverse, generic43 inverse.
    The mirrored C2 transform supplies the required real-even inputs.
    """
    y = jax.vmap(real_even_inverse602_column)(values.reshape(602, -1).T)
    return y.T.reshape((302,) + values.shape[1:])
