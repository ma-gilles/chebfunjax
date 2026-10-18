"""Real-even length724 CPU transform through two Rader181 convolutions.

Chebfun7574c77 @chebtech2/vals2coeffs constructs a circular-even vector
and applies a normalized inverse FFT. The observed MATLAB R2025b plan for
length724 uses C2R, radix4 and Rader181 with convolution length180.
This algebraic C2R reduction uses the existing JAX length180 FFT, not a
word-for-word port of the native HC2C4 or length180 codelets.

Rader permutation/convolution normalization follows the public FFTW
algorithm (dft/rader.c, FFTW3.3.10). Constants depend only on transform
length, never data, tolerance, polynomial degree or retained fixtures.
"""

import jax
import jax.numpy as jnp

from chebfunjax.utils._fft602 import phase_pair
from chebfunjax.utils._ordered_binary import binary as op

P = 181
# 2,3,5 are distinct prime divisors of 180.
G = next(g for g in range(2, P) if all(pow(g, 180 // q, P) != 1 for q in (2, 3, 5)))
PERM = tuple(pow(G, j, P) for j in range(180))
OUT = tuple(pow(G, -j, P) for j in range(180))


def _multiply(a, b):
    r = op(op(a.real, b.real, 0), op(a.imag, b.imag, 0), 2)
    i = op(op(a.real, b.imag, 0), op(a.imag, b.real, 0), 1)
    return jax.lax.complex(r, i)


def _add(a, b):
    return jax.lax.complex(op(a.real, b.real, 1), op(a.imag, b.imag, 1))


def _subtract(a, b):
    return jax.lax.complex(op(a.real, b.real, 2), op(a.imag, b.imag, 2))


def _inverse181(x):
    """Unnormalized positive-sign prime DFT via a length180 convolution.

    If a[j]=x[g**j] and b[j]=exp(2pi*i*g**(-j)/181)/180,
    the unnormalized inverse FFT of FFT(a)*FFT(b) yields all non-DC
    outputs in g**(-j) order. Adding x[0] before the inverse adds it to
    every output, matching the source Rader normalization placement.
    """
    kernel = jnp.asarray(tuple(phase_pair(k, P) for k in OUT), jnp.float64)
    b = jax.lax.complex(
        op(kernel[:, 0], jnp.float64(180), 3), op(kernel[:, 1], jnp.float64(180), 3)
    )
    omega = jnp.fft.fft(b, axis=0).reshape((180,) + (1,) * (x.ndim - 1))
    a = jnp.fft.fft(x[jnp.asarray(PERM)], axis=0)
    z = _multiply(a, omega)
    z = z.at[0].set(_add(z[0], x[0]))
    y = jnp.conj(jnp.fft.fft(jnp.conj(z), axis=0))
    out = jnp.empty_like(x).at[0].set(_add(x[0], a[0]))
    return out.at[jnp.asarray(OUT)].set(y)


@jax.jit
def _c2r724(s):
    # Pack real output even/odd coordinates into one complex length362
    # inverse. For A=s[k], B=conj(s[362-k]), the packed spectrum is
    # (A+B)+i*exp(2pi*i*k/724)*(A-B). Omitting its usual factor1/2
    # permits the source final normalization by724. A radix2 split then
    # gives two length181 inverses; no convolution padding is used.
    a = s[:362]
    b = s[362:0:-1]
    plus = op(a, b, 1)
    minus = op(a, b, 2)
    tw = jnp.asarray(tuple(phase_pair(k, 724) for k in range(362)), jnp.float64)
    c = tw[:, 0, None]
    t = tw[:, 1, None]
    z = jax.lax.complex(op(plus, op(t, minus, 0), 2), op(c, minus, 0))
    u0 = _add(z[:181], z[181:])
    u1 = _subtract(z[:181], z[181:])
    tw2 = jnp.asarray(tuple(phase_pair(k, 362) for k in range(181)), jnp.float64)
    w = jax.lax.complex(tw2[:, 0, None], tw2[:, 1, None])
    u1 = _multiply(u1, w)
    y0 = _inverse181(u0)
    y1 = _inverse181(u1)
    y = jnp.stack((y0, y1), axis=1).reshape((362, s.shape[1]))
    raw = jnp.stack((y.real, y.imag), axis=1).reshape((724, s.shape[1]))
    return op(raw, jnp.float64(724), 3)


@jax.jit
def real_even_inverse724(values):
    """Normalized real IFFT of a circular-even length724 vector or matrix.

    Caller dispatches finite CPU float64 inputs. Fixed-width column chunks
    only bound workspace; no transform axis is padded or truncated.

    Provenance
    ----------
    MATLAB source : @chebtech2/vals2coeffs.m, real circular-even inverse FFT
    Chebfun commit: 7574c77
    Rader permutation/convolution algorithm: FFTW3.3.10 dft/rader.c;
    algebraic JAX implementation with existing length180 FFTs.
    """
    vector = values.ndim == 1
    half = values[:363].reshape((363, -1))
    width = half.shape[1]
    block = 256
    whole, tail = divmod(width, block)
    pieces = []
    if whole:
        chunks = half[:, : whole * block].reshape((363, whole, block))
        chunks = chunks.transpose((1, 0, 2))
        transformed = jax.lax.map(_c2r724, chunks)
        pieces.append(transformed.transpose((1, 0, 2)).reshape((724, whole * block)))
    if tail:
        pieces.append(_c2r724(half[:, whole * block :]))
    if not pieces:
        return jnp.empty_like(values)
    result = jnp.concatenate(pieces, axis=1) if len(pieces) > 1 else pieces[0]
    return result[:, 0] if vector else result
