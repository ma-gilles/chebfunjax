"""JAX fast Legendre transforms and Gauss-Legendre ASY helpers.

The module combines source implementations used by Chebfun's large-degree
Legendre conversion and inverse discrete Legendre transform. Algorithms are
ported from Chebfun commit 7574c77680d7e82b79626300bf255498271a72df:
``leg2cheb.m`` / ``leg2cheb_fast``, ``legpts.m`` local ``asy`` and
``bessel12atj0k``, ``besselroots.m``, and ``@chebfun/idlt.m`` local
``ndct_transpose`` / ``dst3_shifted_transpose``. The transforms use JAX arrays
and FFTs only.
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
Legendre transforms: Alex Townsend and Nick Hale; ASY quadrature: Ignace Bogaert.

The fast Leg2Cheb operation retains a static pivoted-Cholesky workspace bound
for JIT-compatible shapes. It returns a completion flag; callers must not use
an incomplete result as source-equivalent. The ASY quadrature helper supports
n >= 100. The IDLT Taylor correction accepts the ASY angle vector explicitly.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


@jax.jit(static_argnames=("trans", "normalize", "max_rank"))
def _leg2cheb_fast(
    c_leg: jax.Array,
    *,
    trans: bool = False,
    normalize: bool = False,
    max_rank: int = 128,
):
    """Apply a source fast Legendre-to-Chebyshev transform or its transpose.

    Args:
        c_leg: Coefficients shaped ``(N,)`` or ``(N, ncols)``.
        trans: Select MATLAB's source transpose branch.
        normalize: Apply MATLAB ``'norm'`` input scaling before transforming.
        max_rank: Static Cholesky workspace bound. Source needs no such bound;
            ``complete`` reports whether this helper reached its exact
            source residual threshold before filling the workspace.

    Returns:
        ``(c_cheb, complete)``. The output has the same rank as the input.
        A false completion flag means the result is truncated and must not be
        used as a qualified transform.

    Notes:
        * The caller chooses the source branch at N > 512. This helper does
          not implement the <=512 direct branch. That branch is written by
          MATLAB as ``L.'*idct1(c)``; its IDCT-I coefficient-recovery scaling
          is symmetric, so it is the ordinary matrix transpose of the forward
          direct operator. Keep the literal source operation when integrating.
        * The pivoted-Cholesky loop is JIT-compatible and data-dependent. The
          fixed workspace is an implementation constraint, surfaced explicitly
          through ``complete`` rather than silently changing the tolerance.

    Provenance:
        MATLAB source: leg2cheb.m, leg2cheb_fast, Chebfun commit 7574c77.
    """
    c_leg = jnp.asarray(c_leg)
    if c_leg.ndim not in (1, 2):
        raise ValueError("c_leg must be a vector or a 2-D column matrix")
    if not isinstance(trans, bool) or not isinstance(normalize, bool):
        raise ValueError("trans and normalize must be static booleans")
    if not isinstance(max_rank, int) or isinstance(max_rank, bool):
        raise ValueError("max_rank must be a static integer")
    vector_input = c_leg.ndim == 1
    if vector_input:
        c_leg = c_leg[:, None]
    n, ncols = c_leg.shape
    if n < 2:
        raise ValueError("fast helper requires at least two coefficient rows")
    if max_rank < 1:
        raise ValueError("max_rank must be positive")
    input_dtype = jnp.result_type(c_leg.dtype, jnp.float64)
    c_leg = c_leg.astype(input_dtype)
    if normalize:
        scale = jnp.sqrt(jnp.arange(n, dtype=jnp.float64) + 0.5)
        c_leg = c_leg * scale[:, None]
    real_dtype = jnp.real(c_leg).dtype
    vals = jnp.zeros((2 * n,), dtype=real_dtype)
    vals = vals.at[0].set(jnp.sqrt(jnp.pi))
    vals = vals.at[1].set(2.0 / jnp.sqrt(jnp.pi))

    # Literal scalar recurrence from MATLAB vals(i+1), vals(i+2), i=2:2:2*(N-1).
    def vals_body(i, v):
        v = v.at[i].set(v[i - 2] * (1.0 - 1.0 / i))
        v = v.at[i + 1].set(v[i - 1] * (1.0 - 1.0 / (i + 1.0)))
        return v

    vals = lax.fori_loop(1, n, lambda k, v: vals_body(2 * k, v), vals)

    # Pivoted Cholesky of the symmetric positive Hankel matrix, retaining
    # source first-maximum tie behavior (jnp.argmax returns the first index).
    tol = jnp.asarray(1e-14, dtype=real_dtype)
    rank_cap = min(n, max_rank)
    C = jnp.zeros((n, rank_cap), dtype=real_dtype)
    pivots = jnp.zeros((rank_cap,), dtype=real_dtype)
    d = vals[: 2 * n : 2]
    mx = jnp.max(d)
    idx = jnp.argmax(d)
    k = jnp.asarray(0, dtype=jnp.int32)

    def chol_cond(state):
        kk, _, _, diag, peak, _ = state
        return (peak > tol) & (kk < rank_cap)

    def chol_body(state):
        kk, cc, pp, diag, peak, pivot = state
        col = lax.dynamic_slice(vals, (pivot,), (n,))
        correction = cc @ (cc[pivot, :] * pp)
        col = col - correction
        cc = cc.at[:, kk].set(col)
        pp = pp.at[kk].set(1.0 / peak)
        diag = diag - col * col / peak
        peak = jnp.max(diag)
        pivot = jnp.argmax(diag)
        return kk + 1, cc, pp, diag, peak, pivot

    k, C, pivots, d, mx, idx = lax.while_loop(
        chol_cond, chol_body, (k, C, pivots, d, mx, idx)
    )
    complete = mx <= tol
    C = C * jnp.sqrt(pivots)[None, :]

    # T is the source upper-triangular Toeplitz row, with alternating
    # entries zeroed. FFT embedding follows the source 2N-point convolution.
    trow = vals[:n].at[1::2].set(0.0)
    zero = jnp.zeros((n,), dtype=real_dtype)
    if trans:
        symbol = jnp.fft.fft(jnp.concatenate((trow, zero)))
        c_leg = (2.0 / jnp.pi) * c_leg
        c_leg = c_leg.at[0, :].multiply(0.5)
    else:
        symbol = jnp.fft.fft(jnp.concatenate((trow[:1], zero, trow[:0:-1])))

    def apply_column(col):
        tmp = C * col[:, None]
        f1 = jnp.fft.fft(tmp, n=2 * n, axis=0)
        b = jnp.fft.ifft(f1 * symbol[:, None], axis=0)
        out = jnp.sum(b[:n, :] * C, axis=1)
        if trans:
            return out
        return (2.0 / jnp.pi) * out.at[0].multiply(0.5)

    out = jax.vmap(apply_column, in_axes=1, out_axes=1)(c_leg)
    if jnp.issubdtype(c_leg.dtype, jnp.complexfloating):
        out = out.astype(c_leg.dtype)
    else:
        out = jnp.real(out).astype(c_leg.dtype)
    return (out[:, 0] if vector_input else out), complete


_J0_ROOTS_20 = jnp.asarray(
    [
        2.4048255576957728,
        5.5200781102863106,
        8.6537279129110122,
        11.791534439014281,
        14.930917708487785,
        18.071063967910922,
        21.211636629879258,
        24.352471530749302,
        27.493479132040254,
        30.634606468431975,
        33.775820213573568,
        36.917098353664044,
        40.058425764628239,
        43.199791713176730,
        46.341188371661814,
        49.482609897397817,
        52.624051841114996,
        55.765510755019979,
        58.906983926080942,
        62.048469190227170,
    ],
    dtype=jnp.float64,
)

_J1_SQUARED_AT_J0_ROOTS_10 = jnp.asarray(
    [
        0.2695141239419169,
        0.1157801385822037,
        0.07368635113640822,
        0.05403757319811628,
        0.04266142901724309,
        0.03524210349099610,
        0.03002107010305467,
        0.02614739149530809,
        0.02315912182469139,
        0.02078382912226786,
    ],
    dtype=jnp.float64,
)


@partial(jax.jit, static_argnames=("m",))
def _besselroots0_jax(m: int) -> jax.Array:
    """First m J0 roots via the source McMahon expansion and exact table."""
    if m < 0:
        raise ValueError("m must be nonnegative")
    if m == 0:
        return jnp.empty((0,), dtype=jnp.float64)

    s = jnp.arange(1, m + 1, dtype=jnp.float64)
    mu = 0.0
    a1 = 1.0 / 8.0
    a3 = (7.0 * mu - 31.0) / 384.0
    a5 = 4.0 * (3779.0 + mu * (-982.0 + 83.0 * mu)) / 61440.0
    a7 = 6.0 * (-6277237.0 + mu * (1585743.0 + mu * (-153855.0 + 6949.0 * mu))) / 20643840.0
    a9 = 144.0 * (
        2092163573.0
        + mu * (-512062548.0 + mu * (48010494.0 + mu * (-2479316.0 + 70197.0 * mu)))
    ) / 11890851840.0
    a11 = 720.0 * (
        -8249725736393.0
        + mu * (
            1982611456181.0
            + mu * (
                -179289628602.0
                + mu * (8903961290.0 + mu * (-287149133.0 + 5592657.0 * mu))
            )
        )
    ) / 10463949619200.0
    a13 = 576.0 * (
        423748443625564327.0
        + mu
        * (
            -100847472093088506.0
            + mu
            * (
                8929489333108377.0
                + mu
                * (
                    -426353946885548.0
                    + mu
                    * (
                        13172003634537.0
                        + mu * (-291245357370.0 + mu * 4148944183.0)
                    )
                )
            )
        )
    ) / 13059009124761600.0

    b = 0.25 * (4.0 * s - 1.0) * jnp.pi
    coeffs = jnp.asarray(
        [a13, 0.0, a11, 0.0, a9, 0.0, a7, 0.0, a5, 0.0, a3, 0.0, a1, 0.0],
        dtype=jnp.float64,
    )
    roots = b - (mu - 1.0) * jnp.polyval(coeffs, 1.0 / b)
    n_table = min(m, 20)
    return roots.at[:n_table].set(_J0_ROOTS_20[:n_table])


@partial(jax.jit, static_argnames=("m",))
def _bessel12_at_j0_roots_jax(m: int) -> jax.Array:
    """Source expansion for J1(root(J0))**2, with first ten source values."""
    if m < 0:
        raise ValueError("m must be nonnegative")
    if m == 0:
        return jnp.empty((0,), dtype=jnp.float64)
    out = jnp.zeros((m,), dtype=jnp.float64)
    n_table = min(m, 10)
    out = out.at[:n_table].set(_J1_SQUARED_AT_J0_ROOTS_10[:n_table])
    if m <= 10:
        return out

    k = jnp.arange(11, m + 1, dtype=jnp.float64)
    ak = jnp.pi * (k - 0.25)
    z = (1.0 / ak) ** 2
    c1 = -171497088497.0 / 15206400.0
    c2 = 461797.0 / 1152.0
    c3 = -172913.0 / 8064.0
    c4 = 151.0 / 80.0
    c5 = -7.0 / 24.0
    # Literal source Horner grouping in ak2inv.
    poly = 2.0 + z**2 * (c5 + z * (c4 + z * (c3 + z * (c2 + z * c1))))
    return out.at[10:].set(poly / (jnp.pi * ak))


@partial(jax.jit, static_argnames=("n",))
def _legpts_asy_with_theta(n: int):
    """Return source ASY ``(x, w, v, theta)`` on [-1,1] for n>=100.

    Arrays use Python's one-dimensional convention and ascending x order.
    ``theta`` is the source ASY angle, aligned with x; it is computed directly
    from the Bessel-root expansion and is not reconstructed from ``acos(x)``.
    The source barycentric vector v has max absolute value 1 and alternating
    signs, matching the third output of legpts.m.

    This implements MATLAB's default/explicit ASY branch, selected by default
    for n>=100. It does not implement interval mapping, method dispatch, or
    trivial n<100 cases.

    Provenance:
        MATLAB legpts.m local ``asy`` and ``bessel12atj0k``, and besselroots.m.
        Chebfun commit 7574c77680d7e82b79626300bf255498271a72df.
    """
    if n < 100:
        raise ValueError("source ASY helper requires n >= 100")

    m = (n + 1) // 2
    jk = _besselroots0_jax(m)
    vn = 1.0 / (n + 0.5)
    a = jk * vn
    u = 1.0 / jnp.tan(a)
    ua = u * a
    u2 = u**2
    a2 = a**2

    f0 = a
    f1 = 0.125 * (ua - 1.0) / a
    if n < 10_000:
        a3 = a**3
        f2 = (6.0 * a2 * (1.0 + u2) + 25.0 - u * (31.0 * u2 + 33.0) * a3) / (384.0 * a3)
    else:
        f2 = 0.0
    if n < 1_000:
        u4 = u**4
        a3 = a**3
        a5 = a**5
        r30 = u * (2595.0 + 6350.0 * u2 + 3779.0 * u4) / 15360.0
        r31 = -(31.0 * u2 + 11.0) / 1024.0
        r32 = u / 512.0
        r33 = -25.0 / 3072.0
        r35 = -1073.0 / 5120.0
        f3 = r30 + r35 / a5 + (1.0 + u2) * (r31 / a + r32 / a2 + r33 / a3)
    else:
        f3 = 0.0
    theta_half = f0 + f1 * vn**2 + f2 * vn**4 + f3 * vn**6
    x_half = jnp.cos(theta_half)

    # Source ASY quadrature weight expansion W0..W3.
    w0 = 1.0
    w1 = 0.125 * (ua + a2 - 1.0) / a2
    if n < 10_000:
        a4 = a2**2
        u4 = u**4
        w2 = (
            81.0 - 31.0 * ua - 3.0 * (1.0 - 2.0 * u2) * a2 + 6.0 * u * a**3
            - (27.0 + 84.0 * u2 + 56.0 * u4) * a4
        ) / (384.0 * a4)
    else:
        w2 = 0.0
    if n < 1_000:
        u3 = u**3
        u4 = u**4
        u5 = u**5
        u6 = u3**2
        a3 = a**3
        a4 = a2**2
        a5 = a**5
        a6 = a3**2
        q30 = 187.0 / 96.0 * u4 + 295.0 / 256.0 * u2 + 151.0 / 160.0 * u6 + 153.0 / 1024.0
        q31 = -119.0 / 768.0 * u3 - 35.0 / 384.0 * u5 - 65.0 / 1024.0 * u
        q32 = 5.0 / 512.0 + 7.0 / 384.0 * u4 + 15.0 / 512.0 * u2
        q33 = u3 / 512.0 - 13.0 / 1536.0 * u
        q34 = -7.0 / 384.0 * u2 + 53.0 / 3072.0
        q35 = 3749.0 / 15360.0 * u
        q36 = -1125.0 / 1024.0
        w3 = q30 + q31 / a + q32 / a2 + q33 / a3 + q34 / a4 + q35 / a5 + q36 / a6
    else:
        w3 = 0.0
    j1sq = _bessel12_at_j0_roots_jax(m)
    w_half = 2.0 / (
        (j1sq / vn**2)
        * (a / jnp.sin(a))
        * (w0 + w1 * vn**2 + w2 * vn**4 + w3 * vn**6)
    )

    # Source ASY barycentric half-vector, if requested by the caller contract.
    v_half = jnp.sin(theta_half) / jnp.sqrt(2.0 / w_half)
    v_half = v_half / v_half[-1]

    if n % 2 == 0:
        x = jnp.concatenate((-x_half, x_half[::-1]))
        w = jnp.concatenate((w_half, w_half[::-1]))
        v = jnp.concatenate((v_half, v_half[::-1]))
        theta = jnp.concatenate((jnp.pi - theta_half, theta_half[::-1]))
    else:
        x = jnp.concatenate((-x_half[:-1], jnp.zeros((1,), dtype=jnp.float64), x_half[:-1][::-1]))
        w = jnp.concatenate((w_half, w_half[:-1][::-1]))
        v = jnp.concatenate((v_half, v_half[:-1][::-1]))
        theta = jnp.concatenate((jnp.pi - theta_half, theta_half[:-1][::-1]))

    v = jnp.abs(v) / jnp.max(jnp.abs(v))
    v = v * jnp.where(jnp.arange(n) % 2 == 0, 1.0, -1.0)
    return x, w, v, theta


def _dct2_source(u: jax.Array) -> jax.Array:
    """Chebfun's unnormalized DCT-II, using a JAX FFT for vector/matrix input."""
    u = jnp.asarray(u)
    vector = u.ndim == 1
    if vector:
        u = u[:, None]
    if u.ndim != 2:
        raise ValueError("DCT-II input must be a vector or matrix")
    n = u.shape[0]
    if n == 0:
        return u[:, 0] if vector else u

    def real_transform(x):
        ext = jnp.concatenate((x, x[::-1, :]), axis=0)
        spectrum = jnp.fft.fft(ext, axis=0)[:n, :]
        phase = jnp.exp(-1j * jnp.pi * jnp.arange(n) / (2.0 * n))[:, None]
        return 0.5 * jnp.real(spectrum * phase)

    if jnp.iscomplexobj(u):
        out = real_transform(jnp.real(u)) + 1j * real_transform(jnp.imag(u))
    else:
        out = real_transform(u)
    # @chebtech1/vals2coeffs zeros coefficients that symmetry proves are
    # exactly zero. Apply the test independently to each input column, as the
    # source's matrix path does after transforming real and imaginary parts.
    even_columns = jnp.all(u == u[::-1, :], axis=0)
    odd_columns = jnp.all(u == -u[::-1, :], axis=0)
    row_index = jnp.arange(n)
    zero_rows = ((row_index % 2 == 1)[:, None] & even_columns[None, :]) | (
        (row_index % 2 == 0)[:, None] & odd_columns[None, :]
    )
    out = jnp.where(zero_rows, jnp.zeros((), dtype=out.dtype), out)
    return out[:, 0] if vector else out


def _dst2_source(u: jax.Array) -> jax.Array:
    """Chebfun's DST-II, via the literal sign/reverse/DCT-II recipe."""
    u = jnp.asarray(u)
    vector = u.ndim == 1
    if vector:
        u = u[:, None]
    if u.ndim != 2:
        raise ValueError("DST-II input must be a vector or matrix")
    if u.shape[0] == 0:
        return u[:, 0] if vector else u
    signed = u.at[::2, :].multiply(-1)
    out = _dct2_source(signed[::-1, :])
    out = out.at[::2, :].multiply(-1)[::-1, :]
    return out[:, 0] if vector else out


def _dst3_shifted_transpose_source(u: jax.Array) -> jax.Array:
    """Source `dst3_shifted_transpose`: [0; DST-II(u)(1:end-1,:)]."""
    u = jnp.asarray(u)
    vector = u.ndim == 1
    matrix = u[:, None] if vector else u
    transformed = _dst2_source(matrix)
    out = jnp.concatenate((jnp.zeros_like(transformed[:1]), transformed[:-1]), axis=0)
    return out[:, 0] if vector else out


def _matrix_inf_norm_source(delta: jax.Array) -> jax.Array:
    """MATLAB `norm(delta, inf)`: largest absolute row sum."""
    delta = jnp.asarray(delta)
    if delta.ndim == 1:
        return jnp.max(jnp.abs(delta))
    return jnp.max(jnp.sum(jnp.abs(delta), axis=1))


def _idlt_ndct_transpose_source(
    weighted_values: jax.Array,
    legendre_theta: jax.Array,
    *,
    source_real_imag_quirk: bool = True,
) -> jax.Array:
    """Apply the MATLAB `ndct_transpose` Taylor expansion to weighted values.

    Inputs are ``weighted_values = w[:,None] * f`` and ``legendre_theta`` from
    :func:`_legpts_asy_with_theta`, aligned with the nodes. Return
    is the unscaled transform before MATLAB's subsequent `leg2cheb(...,'trans')`
    and row multiplier `(0:n-1)+.5`.

    The 18-term cap, eps stopping criterion, and the infinity norm of a matrix
    (maximum absolute row sum) match `@chebfun/idlt.m`. When source real/imag
    handling is enabled, purely imaginary input returns imag(c), as the
    source's `isreal(f)` / `isreal(1i*f)` post-check does.
    """
    f0 = jnp.asarray(weighted_values)
    vector = f0.ndim == 1
    f0 = f0[:, None] if vector else f0
    if f0.ndim != 2:
        raise ValueError("weighted_values must be a vector or matrix")
    n, m = f0.shape
    theta = jnp.asarray(legendre_theta, dtype=jnp.float64).reshape((n,))
    if n == 0:
        return f0[:, 0] if vector else f0

    # MATLAB reverses the GL rows and theta before comparing to first-kind
    # Chebyshev angles. Repair symmetric angle differences for endpoint
    # accuracy exactly as the source assignment does.
    t_leg = theta[::-1]
    f = f0[::-1, :]
    t_cheb = (jnp.arange(n, dtype=jnp.float64) + 0.5) * jnp.pi / n
    dt = t_leg - t_cheb
    # MATLAB's LHS index vector is descending while its RHS is ascending;
    # reverse the source slice to reproduce elementwise assignment order.
    dt = dt.at[n // 2 :].set(-dt[: (n + 1) // 2][::-1])
    dt = dt[:, None]

    nn = jnp.arange(n, dtype=jnp.float64)[:, None]
    c = _dct2_source(f)
    f_iter = f
    nn_power = jnp.ones_like(nn)
    factorial = 1.0
    active = jnp.asarray(True)
    eps = jnp.asarray(jnp.finfo(jnp.float64).eps)
    for ell in range(1, 18):
        f_iter = jnp.where(active, dt * f_iter, f_iter)
        nn_power = nn * nn_power
        factorial *= ell
        if ell % 2:
            transform = _dst3_shifted_transpose_source(f_iter)
        else:
            transform = _dct2_source(f_iter)
        pair = (ell + 1) // 2
        sign = -1.0 if pair % 2 else 1.0
        delta = (sign / factorial) * nn_power * transform
        # MATLAB adds the first term below eps, then stops. Mask later terms
        # instead of converting a traced norm to a Python bool; this keeps the
        # same cutoff decision under JIT with a fixed-size trace.
        c = c + jnp.where(active, delta, jnp.zeros_like(delta))
        # MATLAB norm(matrix, inf) is max row sum, not max element.
        active = active & (_matrix_inf_norm_source(delta) >= eps)

    if source_real_imag_quirk and jnp.iscomplexobj(f_iter):
        all_real = jnp.all(jnp.imag(f_iter) == 0)
        all_pure_imag = jnp.all(jnp.real(f_iter) == 0)
        if isinstance(f_iter, jax.core.Tracer):
            # A traced function cannot select a real output dtype from data.
            # Preserve the source's selected values while retaining complex
            # dtype, which is stable under JIT for arbitrary inputs.
            source_real = jnp.real(c).astype(c.dtype)
            source_imag = jnp.imag(c).astype(c.dtype)
            c = jnp.where(
                all_real,
                source_real,
                jnp.where(all_pure_imag, source_imag, c),
            )
        elif bool(all_real):
            c = jnp.real(c)
        elif bool(all_pure_imag):
            c = jnp.imag(c)
    return c[:, 0] if vector else c
