"""One JAX construction engine for the source randnfun interfaces.

Normal draws are JAX draws, not MATLAB's normal transformation or NumPy's global
stream. Host entropy seeds the default stream once; subsequent calls advance it.
Construction is eager because the representation length depends on input metadata.

Provenance
----------
MATLAB source : randnfun.m, @chebtech/simplify.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
from __future__ import annotations

import math
import secrets
import threading

import jax
import jax.numpy as jnp

_DEFAULT_KEY = None
_STREAM_LOCK = threading.Lock()


def _next_key():
    global _DEFAULT_KEY
    with _STREAM_LOCK:
        if _DEFAULT_KEY is None:
            _DEFAULT_KEY = jax.random.key(secrets.randbits(32))
        _DEFAULT_KEY, draw = jax.random.split(_DEFAULT_KEY)
    return draw


def _normal_draw(key, rows, columns):
    # This shape/order is an explicit adapter for a column-major normal stream.
    # JAX normal values are NOT claimed bit-identical to MATLAB randn.
    return jax.random.normal(key, (columns, rows), dtype=jnp.float64).T


def _parse(args):
    lam = n = dom = None
    big = trig = cmplx = False
    for value in args:
        if isinstance(value, str):
            first = value[:1].lower()
            if first in ('n', 'b'):
                big = True
            elif first == 't':
                trig = True
            elif first == 'c':
                cmplx = True
            else:
                raise ValueError('CHEBFUN:randnfun: Unrecognized string input')
        else:
            a = jnp.asarray(value)
            if a.size != 1:
                dom = tuple(float(x) for x in a.ravel())
            elif lam is None or math.isnan(lam):
                lam = float(a.reshape(()))
            else:
                n = float(a.reshape(()))
    lam = 1.0 if lam is None or math.isnan(lam) else lam
    n = 1 if n is None or math.isnan(n) else n
    dom = (-1.0, 1.0) if dom is None or (dom and all(math.isnan(x) for x in dom)) else dom
    if not (lam > 0) or not (math.isfinite(lam) or lam == math.inf):
        raise ValueError('randnfun: wavelength must be positive')
    if not math.isfinite(n) or n < 0 or int(n) != n:
        raise ValueError('randnfun: column count must be a nonnegative integer')
    if len(dom) != 2 or not all(math.isfinite(x) for x in dom) or dom[0] >= dom[1]:
        raise ValueError('randnfun: expected two finite increasing domain endpoints')
    return lam, int(n), dom, big, trig, cmplx


def _periodic_coefficients(draws, length, big, cmplx):
    """Source reorder/symmetrization; draws shape is (2*n, 2*m+1)."""
    n = draws.shape[0] // 2
    modes = draws.shape[1]
    ii = jnp.concatenate((jnp.arange(modes - 1, -1, -2),
                          jnp.arange(1, modes, 2)))
    c = draws[:, ii].T
    c = (c[:, :n] + 1j * c[:, n:]) / jnp.sqrt(2.0)
    if not cmplx:
        c = (c + jnp.conj(c[::-1])) / jnp.sqrt(2.0)
    return c / jnp.sqrt(length if big else modes)


def _wrap(coefficients, dom, *, periodic, point_values=None):
    from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
    from chebfunjax.domain import Domain
    from chebfunjax.tech.chebtech import Chebtech2
    from chebfunjax.tech.trigtech import Trigtech

    if coefficients.size == 0:
        return Chebfun.empty()
    c = coefficients[:, 0] if coefficients.shape[1] == 1 else coefficients
    tech = Trigtech.from_coeffs(c) if periodic else Chebtech2.from_coeffs(c)
    f = Chebfun([_Piece(tech=tech, interval=dom)], Domain(dom))
    if point_values is None:
        if periodic:
            from chebfunjax.tech.trigtech import _trig_eval
            point_values = _trig_eval(coefficients, jnp.asarray([-1., 1.]),
                                      is_real=tech.is_real)
        else:
            signs = jnp.where(jnp.arange(coefficients.shape[0]) % 2, -1, 1)
            point_values = jnp.stack((jnp.sum(signs[:, None] * coefficients, axis=0),
                                      jnp.sum(coefficients, axis=0)))
    object.__setattr__(f, '_point_values', point_values)
    return f


def _from_nonperiodic_values(values, dom):
    """Numeric Chebfun endpoints, then source column-relative simplify.

    Provenance
    ----------
    MATLAB source : randnfun.m109-115; @chebfun/getValuesAtBreakpoints.m,
        @chebtech/{lval,rval,vscale,simplify}.m; @chebfun/simplify.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    from chebfunjax.tech.chebtech import _chop_columns
    from chebfunjax.utils.transforms import _coeffs2vals_jax, _vals2coeffs_jax

    coefficients = _vals2coeffs_jax(values)
    nold = coefficients.shape[0]
    signs = jnp.where(jnp.arange(nold) % 2, -1, 1)
    # Source numeric Chebfun endpoints are coefficient sums BEFORE simplify,
    # not the original samples or evaluations of the chopped representation.
    endpoints = jnp.stack((jnp.sum(signs[:, None] * coefficients, axis=0),
                           jnp.sum(coefficients, axis=0)))
    scale = jnp.max(jnp.abs(_coeffs2vals_jax(coefficients)), axis=0)
    tolerances = (1e-13 * scale) / scale
    # Preserve zero-column0/0: standardChop's zero-envelope branch returns1.
    padded_length = max(17, math.floor(nold*1.25+5+0.5))
    padded = jnp.pad(coefficients, ((0, padded_length-nold), (0, 0)))
    noisy = _vals2coeffs_jax(_coeffs2vals_jax(padded))
    cutoff = min(nold, _chop_columns(noisy, tolerances))
    return _wrap(coefficients[:cutoff], dom, periodic=False, point_values=endpoints)


def _construct(lam, n, dom, big, trig, cmplx, draw):
    from chebfunjax.domain import _linear_inverse_map
    from chebfunjax.tech.trigtech import Trigtech, _trig_eval
    from chebfunjax.utils.quadrature import chebpts_ab

    length = dom[1] - dom[0]
    if trig:
        m = math.floor(length / lam)
        c = _periodic_coefficients(draw(2*n, 2*m+1), length, big, cmplx)
        return _wrap(c, dom, periodic=True)
    dx = max(0.2, 2*lam/length)
    dom2 = (dom[0], dom[0] + (1+dx)*length)
    if lam == math.inf:
        # Literal source scalar branch ignores requested n here.
        c = draw(1, 1)
        if cmplx:
            c = (c + 1j*draw(1, 1)) / jnp.sqrt(2.0)
        if big:
            c = c / jnp.sqrt(dom2[1] - dom2[0])
        return _wrap(c, dom, periodic=False)
    m = math.floor(length / lam + 0.5)  # source positive round, not ties-to-even
    periodic_length = dom2[1] - dom2[0]
    m2 = math.floor(periodic_length / lam)
    c = _periodic_coefficients(draw(2*n, 2*m2+1), periodic_length, big, cmplx)
    if n == 0:
        return _wrap(c, dom, periodic=True)
    x = chebpts_ab(5*m+20, *dom)
    t = _linear_inverse_map(x, *dom2)
    # Direct source Horner kernel, avoiding concrete NumPy evaluation delegates.
    values = _trig_eval(c, t, is_real=Trigtech.from_coeffs(c).is_real)
    return _from_nonperiodic_values(values, dom)


def randnfun(*args, key=None, seed=None, domain=None, lam=None,
             big=False, trig=False, cmplx=False):
    """Source-shaped random Chebfun on [-1,1], with JAX key/seed adapters.

    Positional options follow MATLAB: wavelength, column count, domain vector,
    and big/trig/complex strings in any order (wavelength precedes count).
    Without an explicit key or seed, successive calls advance a private JAX
    stream seeded once from host entropy. Explicit key/seed calls are repeatable
    and do not advance that default stream. NumPy random.seed has no effect.
    Keys and seeds do not reproduce MATLAB normal draws. The narrow supported
    construction scope is positive wavelength, nonnegative integer column count,
    and two finite increasing domain endpoints. Host metadata/chopping is eager;
    the coefficient kernel and all new numerical arithmetic use JAX even with
    JIT disabled. Existing output-object consumers retain their own backends.

    Provenance
    ----------
    MATLAB source : randnfun.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    if key is not None and seed is not None:
        raise ValueError('randnfun: supply key or seed, not both')
    options = (() if lam is None else (lam,)) + args
    options += (() if domain is None else (domain,))
    options += tuple(flag for enabled, flag in ((big, 'big'), (trig, 'trig'),
                                                (cmplx, 'complex')) if enabled)
    parsed = _parse(options)
    local_key = jax.random.key(seed) if seed is not None else key

    def draw(rows, columns):
        nonlocal local_key
        if rows == 0 or columns == 0:
            return jnp.empty((rows, columns), dtype=jnp.float64)
        if local_key is None:
            draw_key = _next_key()
        else:
            local_key, draw_key = jax.random.split(local_key)
        return _normal_draw(draw_key, rows, columns)

    return _construct(*parsed, draw)
