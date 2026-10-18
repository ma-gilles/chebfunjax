# Copyright (c) 1992-2025 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2000-2025 The University of California Berkeley. All
#                         rights reserved.
# Copyright (c) 2006-2025 The University of Colorado Denver.  All rights
#                         reserved.
#
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer listed
#   in this license in the documentation and/or other materials
#   provided with the distribution.
#
# - Neither the name of the copyright holders nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# The copyright holders provide no reassurances that the source code
# provided does not infringe any patent, copyright, or any other
# intellectual property rights of third parties.  The copyright holders
# disclaim any liability to any recipient for claims brought against
# recipient by any third party for infringement of that parties
# intellectual property rights.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Real QZ kernels derived from Netlib DLAG2/DHGEQZ (BSD notice retained above).
Primary sources: https://www.netlib.org/lapack/double/ DGGHRD, DHGEQZ,
DLAG2, DLASV2, DGGEV, DGGBAL, DTGEVC, DLALN2. Source algorithms are
expressed with JAX arrays and compiled kernels; no host eig provider.
"""

from functools import partial

import jax
import jax.numpy as j
from jax import lax

from . import _qz_gradual as gradual
from . import _qz_ieee as binary


@jax.jit
def lag2(A, B):
    """DLAG2 homogeneous real/complex conjugate shifts, including safe scaling.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    safe = j.finfo(A.dtype).tiny * 100
    rtmin = j.sqrt(safe)
    rtmax = 1 / rtmin
    anorm = j.maximum(j.max(j.sum(j.abs(A), axis=0)), safe)
    ascale = 1 / anorm
    a11, a12, a21, a22 = (A * ascale).reshape(-1)
    b11, b12, b22 = B[0, 0], B[0, 1], B[1, 1]
    bmin = rtmin * j.maximum(j.maximum(j.abs(b11), j.abs(b12)), j.maximum(j.abs(b22), rtmin))
    b11 = j.where(j.abs(b11) < bmin, j.copysign(bmin, b11), b11)
    b22 = j.where(j.abs(b22) < bmin, j.copysign(bmin, b22), b22)
    bnorm = j.maximum(j.maximum(j.abs(b11), j.abs(b12) + j.abs(b22)), safe)
    bsize = j.maximum(j.abs(b11), j.abs(b22))
    b11, b12, b22 = b11 / bsize, b12 / bsize, b22 / bsize
    bi11, bi22 = 1 / b11, 1 / b22
    s1, s2 = a11 * bi11, a22 * bi22
    ss = a21 * (bi11 * bi22)
    first = j.abs(s1) <= j.abs(s2)
    shift = j.where(first, s1, s2)
    as12 = a12 - shift * b12
    abi22 = j.where(first, (a22 - s1 * b22) * bi22 - ss * b12, -ss * b12)
    pp = 0.5 * j.where(first, abi22, (a11 - s2 * b11) * bi11 + abi22)
    qq = ss * as12

    def high(_):
        d = (rtmin * pp) ** 2 + qq * safe
        return d, j.sqrt(j.abs(d)) * rtmax

    def normal(_):
        def low(_):
            d = (rtmax * pp) ** 2 + qq / safe
            return d, j.sqrt(j.abs(d)) * rtmin

        return lax.cond(
            pp**2 + j.abs(qq) <= safe, low, lambda _: (pp**2 + qq, j.sqrt(j.abs(pp**2 + qq))), None
        )

    discr, r = lax.cond(j.abs(pp * rtmin) >= 1, high, normal, None)
    real = (discr >= 0) | (r == 0)
    wbig = shift + pp + j.copysign(r, pp)
    wsmall = shift + pp - j.copysign(r, pp)
    wdet = (a11 * a22 - a12 * a21) * (bi11 * bi22)
    wsmall = j.where(0.5 * j.abs(wbig) > j.maximum(j.abs(wsmall), safe), wdet / wbig, wsmall)
    wr1 = j.where(pp > abi22, j.minimum(wbig, wsmall), j.maximum(wbig, wsmall))
    wr2 = j.where(pp > abi22, j.maximum(wbig, wsmall), j.minimum(wbig, wsmall))
    wr1, wr2, wi = (
        j.where(real, wr1, shift + pp),
        j.where(real, wr2, shift + pp),
        j.where(real, 0.0, r),
    )
    c1 = bsize * (safe * j.maximum(1.0, ascale))
    c2, c3 = safe * j.maximum(1.0, bnorm), bsize * safe
    c4 = j.where((ascale <= 1) & (bsize <= 1), j.minimum(1.0, (ascale / safe) * bsize), 1.0)
    c5 = j.where((ascale <= 1) | (bsize <= 1), j.minimum(1.0, ascale * bsize), 1.0)

    def rescale(wr, im):
        wa = j.abs(wr) + j.abs(im)
        size = j.maximum(
            j.maximum(safe, c1),
            j.maximum(1.00001 * (wa * c2 + c3), j.minimum(c4, 0.5 * j.maximum(wa, c5))),
        )
        scale = j.where(
            size > 1,
            (j.maximum(ascale, bsize) / size) * j.minimum(ascale, bsize),
            (j.minimum(ascale, bsize) / size) * j.maximum(ascale, bsize),
        )
        return scale, wr / size, im / size

    scale1, wr1, wi = rescale(wr1, wi)
    scale2, wr2, _ = rescale(wr2, 0.0)
    return scale1, j.where(real, scale2, scale1), wr1, j.where(real, wr2, wr1), wi


def rotation(f, g):
    """Rotation.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    scale = j.maximum(j.abs(f), j.abs(g))
    scale = j.where(scale == 0, 1.0, scale)
    f, g = f / scale, g / scale
    r = j.hypot(f, g)
    r = j.copysign(j.where(r == 0, 1.0, r), f)
    return j.where(g == 0, 1.0, f / r), j.where(g == 0, 0.0, g / r)


def rows(X, p, q, c, s):
    """Rows.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    a, b = X[p], X[q]
    return X.at[p].set(gradual.muladd(a, c, b, s)).at[q].set(gradual.muladd(a, -s, b, c))


def cols(X, p, q, c, s):
    """Cols.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    return rows(X.T, p, q, c, s).T


def left(st, p, q, c, s):
    """Left.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    H, T, Q, Z = st
    return rows(H, p, q, c, s), rows(T, p, q, c, s), cols(Q, p, q, c, s), Z


def right(st, p, q, c, s):
    """Right.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    H, T, Q, Z = st
    return cols(H, p, q, c, s), cols(T, p, q, c, s), Q, cols(Z, p, q, c, s)


def reflector(x):
    """Reflector.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    scale = j.max(j.abs(x))
    y = x / j.where(scale > 0, scale, 1.0)
    norm = j.sqrt(j.sum(y * y))
    beta = -j.copysign(norm, y[0])
    tail = y[1:] / j.where(y[0] != beta, y[0] - beta, 1.0)
    tau = j.where(j.any(y[1:] != 0), (beta - y[0]) / j.where(beta != 0, beta, 1.0), 0.0)
    return j.concatenate((j.ones(1, dtype=x.dtype), tail)), tau


def house_left(st, k, v, tau):
    """House left.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    H, T, Q, Z = st
    ids = k + j.arange(3)

    def apply(X):
        block = X[ids, :]
        return X.at[ids, :].set(block - tau * j.outer(v, v @ block))

    return apply(H), apply(T), apply(Q.T).T, Z


def house_right(st, k, v, tau):
    """House right.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    H, T, Q, Z = st
    ids = k + j.arange(3)

    def apply(X):
        block = X[:, ids]
        return X.at[:, ids].set(block - tau * j.outer(block @ v, v))

    return apply(H), apply(T), Q, apply(Z)


def right_reflector(T, k):
    """DHGEQZ pivoted 2x2 solve constructing the right Householder.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    W = lax.dynamic_slice(T, (k + 1, k + 1), (2, 2))
    u = lax.dynamic_slice(T, (k + 1, k), (2, 1))[:, 0]
    flip = j.max(j.abs(W[0])) < j.max(j.abs(W[1]))
    W = j.where(flip, W[::-1], W)
    u = j.where(flip, u[::-1], u)
    cp = j.abs(W[0, 1]) > j.abs(W[0, 0])
    W = j.where(cp, W[:, ::-1], W)
    safe = j.finfo(T.dtype).tiny

    def solve(_):
        w11, w12, w21, w22 = W.reshape(-1)
        m = w21 / w11
        u1 = u[0]
        u2 = u[1] - m * u1
        w22 = w22 - m * w12

        def singular(_):
            return j.array([0.0, -w12 / w11, 1.0], dtype=T.dtype)

        def nonsingular(_):
            scale = j.minimum(
                1.0,
                j.minimum(
                    j.abs(w22) / j.maximum(j.abs(u2), safe), j.abs(w11) / j.maximum(j.abs(u1), safe)
                ),
            )
            x2 = scale * u2 / w22
            x1 = (scale * u1 - w12 * x2) / w11
            return j.array([scale, x1, x2])

        return lax.cond(j.abs(w22) < safe, singular, nonsingular, None)

    sol = lax.cond(
        j.max(j.abs(W)) < safe, lambda _: j.array([0.0, 1.0, 0.0], dtype=T.dtype), solve, None
    )
    scale, u1, u2 = sol
    u1, u2 = j.where(cp, u2, u1), j.where(cp, u1, u2)
    norm = j.hypot(scale, j.hypot(u1, u2))
    tau = 1 + scale / norm
    v = j.array([1.0, -u1 / (scale + norm), -u2 / (scale + norm)])
    return v, tau


@jax.jit
def double_sweep(st, lo, hi, ascale=None, bscale=None):
    """DHGEQZ Francis double shift, including singular pivot branches.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    H, T, Q, Z = st
    if ascale is None:
        ascale = 1 / j.maximum(j.linalg.norm(H), j.finfo(H.dtype).tiny)
    if bscale is None:
        bscale = 1 / j.maximum(j.linalg.norm(T), j.finfo(T.dtype).tiny)

    def ratios(k):
        return (
            (ascale * H[k, k]) / (bscale * T[k, k]),
            (ascale * H[k + 1, k]) / (bscale * T[k, k]),
            (ascale * H[k, k + 1]) / (bscale * T[k + 1, k + 1]),
            (ascale * H[k + 1, k + 1]) / (bscale * T[k + 1, k + 1]),
            T[k, k + 1] / T[k + 1, k + 1],
        )

    a, b, c, d, u = ratios(hi - 1)
    al, bl, cl, dl, ul = ratios(lo)
    el = (ascale * H[lo + 2, lo + 1]) / (bscale * T[lo + 1, lo + 1])
    x = j.array(
        [
            (a - al) * (d - al) - c * b + b * u * al + (cl - al * ul) * bl,
            ((dl - al) - bl * ul - (a - al) - (d - al) + b * u) * bl,
            el * bl,
        ]
    )
    firstv, firsttau = reflector(x)

    def step(k, st):
        H, T, Q, Z = st
        v, tau = lax.cond(
            k == lo,
            lambda _: (firstv, firsttau),
            lambda _: reflector(lax.dynamic_slice(H, (k, k - 1), (3, 1))[:, 0]),
            None,
        )
        H, T, Q, Z = house_left(st, k, v, tau)
        H = lax.cond(
            k > lo, lambda h: h.at[k + 1, k - 1].set(0.0).at[k + 2, k - 1].set(0.0), lambda h: h, H
        )
        v, tau = right_reflector(T, k)
        H, T, Q, Z = house_right((H, T, Q, Z), k, v, tau)
        return H, T.at[k + 1, k].set(0.0).at[k + 2, k].set(0.0), Q, Z

    st = lax.fori_loop(lo, hi - 1, step, st)
    H, T, Q, Z = st
    k = hi - 1
    c, s = rotation(H[k, k - 1], H[k + 1, k - 1])
    H, T, Q, Z = left(st, k, k + 1, c, s)
    H = H.at[k + 1, k - 1].set(0.0)
    c, s = rotation(T[k + 1, k + 1], T[k + 1, k])
    H, T, Q, Z = right((H, T, Q, Z), k + 1, k, c, s)
    return H, T.at[k + 1, k].set(0.0), Q, Z


@jax.jit
def lasv2(f, g, h):
    """DLASV2 signed singular values and rotations of triangular 2x2.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    swap = j.abs(h) > j.abs(f)
    ft, ht = j.where(swap, h, f), j.where(swap, f, h)
    fa, ha, ga = j.abs(ft), j.abs(ht), j.abs(g)
    pmax = j.where(ga > fa, 2, j.where(swap, 3, 1))

    def diagonal(_):
        return ha, fa, j.array(1.0), j.array(0.0), j.array(1.0), j.array(0.0)

    def nondiagonal(_):
        def large(_):
            ssmin = j.where(ha > 1, fa / (ga / ha), (fa / ga) * ha)
            return ssmin, ga, j.array(1.0), ht / g, ft / g, j.array(1.0)

        def normal(_):
            d = fa - ha
            ell = j.where(d == fa, 1.0, d / fa)
            m = g / ft
            t = 2 - ell
            mm = m * m
            s = j.sqrt(t * t + mm)
            rr = j.hypot(ell, m)
            a = 0.5 * (s + rr)
            tiny_t = j.where(
                ell == 0, j.copysign(2.0, ft) * j.copysign(1.0, g), g / j.copysign(d, ft) + m / t
            )
            t = j.where(mm == 0, tiny_t, (m / (s + t) + m / (rr + ell)) * (1 + a))
            ll = j.hypot(t, 2.0)
            crt, srt = 2 / ll, t / ll
            return ha / a, fa * a, (crt + srt * m) / a, (ht / ft) * srt / a, crt, srt

        return lax.cond((ga > fa) & (fa / ga < j.finfo(j.float64).eps / 2), large, normal, None)

    small, big, clt, slt, crt, srt = lax.cond(ga == 0, diagonal, nondiagonal, None)
    cl, sl = j.where(swap, srt, clt), j.where(swap, crt, slt)
    cr, sr = j.where(swap, slt, crt), j.where(swap, clt, srt)

    def sign(x):
        return j.copysign(1.0, x)

    tsign = j.where(
        pmax == 1,
        sign(cr) * sign(cl) * sign(f),
        j.where(pmax == 2, sign(sr) * sign(cl) * sign(g), sign(sr) * sign(sl) * sign(h)),
    )
    return j.copysign(small, tsign * sign(f) * sign(h)), j.copysign(big, tsign), sl, cl, sr, cr


@jax.jit
def standardize_pair(st, k):
    """Standardize pair.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    H, T, Q, Z = st
    small, big, sl, cl, sr, cr = lasv2(T[k, k], T[k, k + 1], T[k + 1, k + 1])
    flip = j.where(big < 0, -1.0, 1.0)
    cr, sr, big, small = cr * flip, sr * flip, big * flip, small * flip
    st = left(st, k, k + 1, cl, sl)
    H, T, Q, Z = right(st, k, k + 1, cr, sr)
    T = T.at[k, k].set(big).at[k, k + 1].set(0.0).at[k + 1, k].set(0.0).at[k + 1, k + 1].set(small)
    flip = j.where(small < 0, -1.0, 1.0)
    return (
        H.at[:, k + 1].multiply(flip),
        T.at[:, k + 1].multiply(flip),
        Q,
        Z.at[:, k + 1].multiply(flip),
    )


@jax.jit
def single_sweep(st, lo, hi, s1, wr):
    """Single sweep.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    H, T, _, _ = st
    c, s = rotation(s1 * H[lo, lo] - wr * T[lo, lo], s1 * H[lo + 1, lo])

    def step(k, carry):
        st, c, s = carry
        H, T, Q, Z = st
        c, s = lax.cond(
            k == lo, lambda _: (c, s), lambda _: rotation(H[k, k - 1], H[k + 1, k - 1]), None
        )
        previous = H[k, k - 1]
        H, T, Q, Z = left(st, k, k + 1, c, s)
        H = lax.cond(
            (k == lo) & (k > 0),
            lambda h: h.at[k, k - 1].set(previous).at[k + 1, k - 1].set(0.0),
            lambda h: h,
            H,
        )
        H = lax.cond(k > lo, lambda h: h.at[k + 1, k - 1].set(0.0), lambda h: h, H)
        c, s = rotation(T[k + 1, k + 1], T[k + 1, k])
        H, T, Q, Z = right((H, T, Q, Z), k + 1, k, c, s)
        return (H, T.at[k + 1, k].set(0.0), Q, Z), c, s

    return lax.fori_loop(lo, hi, step, (st, c, s))[0]


@jax.jit
def chase_zero(state, start, hi, leading):
    """Chase zero.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    def top(state):
        H, T, Q, Z = state
        c, s = rotation(H[start, start], H[start + 1, start])
        H, T, Q, Z = left(state, start, start + 1, c, s)
        H = H.at[start + 1, start].set(0)
        H = lax.cond(start > 0, lambda X: X.at[start + 1, start - 1].set(0), lambda X: X, H)
        return H, T, Q, Z

    def interior(state):
        def step(k, state):
            H, T, Q, Z = state
            c, s = rotation(T[k, k + 1], T[k + 1, k + 1])
            H, T, Q, Z = left(state, k, k + 1, c, s)
            T = T.at[k + 1, k + 1].set(0)
            c, s = rotation(H[k + 1, k], H[k + 1, k - 1])
            H, T, Q, Z = right((H, T, Q, Z), k, k - 1, c, s)
            return H.at[k + 1, k - 1].set(0), T, Q, Z

        H, T, Q, Z = lax.fori_loop(start, hi, step, state)
        c, s = rotation(H[hi, hi], H[hi, hi - 1])
        H, T, Q, Z = right((H, T, Q, Z), hi, hi - 1, c, s)
        return H.at[hi, hi - 1].set(0), T, Q, Z

    return lax.cond(leading, top, interior, state)


@jax.jit
def inspect(H, T, hi, atol, btol, ascale):
    """Inspect.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    n = H.shape[0]
    indices = j.arange(n)
    sub = j.concatenate([j.zeros(1), j.abs(j.diag(H, -1))])
    split = (indices <= hi) & (sub <= atol)
    lo = j.max(j.where(split, indices, 0))
    small = j.where(indices == hi, j.abs(j.diag(T)) <= btol, j.abs(j.diag(T)) < btol) | (
        j.diag(T) == 0
    )
    zeros = (indices >= lo) & (indices <= hi) & small
    z = j.max(j.where(zeros, indices, -1))
    temp = j.abs(H[z, z - 1])
    temp2 = j.abs(H[z, z])
    scale = j.maximum(temp, temp2)
    scale = j.where((scale < 1) & (scale > 0), scale, 1.0)
    temp, temp2 = temp / scale, temp2 / scale
    two = (
        (z >= 0)
        & (z < hi)
        & (z > lo)
        & (temp * (ascale * j.abs(H[z + 1, z])) <= temp2 * (ascale * atol))
    )
    return lo, z, two


def schur(A, B, max_sweeps=None):
    """Real generalized Schur iteration with homogeneous eigenvalue pairs.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    A, B = j.asarray(A, dtype=j.float64), j.asarray(B, dtype=j.float64)
    n = A.shape[0]
    small = float(j.sqrt(j.finfo(j.float64).tiny) / j.finfo(j.float64).eps)
    large = 1 / small

    def scale_input(X):
        # Integer magnitude ordering includes subnormals under CPU FTZ.
        bits = lax.bitcast_convert_type(X, j.uint64) & j.uint64(0x7FFFFFFFFFFFFFFF)
        magnitude = float(lax.bitcast_convert_type(j.max(bits), j.float64))
        target = min(max(magnitude, small), large) if magnitude else 1.0
        scale = magnitude / target if magnitude else 1.0
        return binary.divide(X, scale), scale

    As, sa = scale_input(A)
    Bs, sb = scale_input(B)
    Ab, Bb, rp, cp, active_lo, active_hi = balance_permute(As, Bs)
    H, T, Qp, Zp = reduce_balanced(
        Ab,
        Bb,
        active_lo,
        active_hi,
        bool(gradual._minimum_exponent(Bb[active_lo : active_hi + 1, active_lo:]) < -1022 + 53),
    )
    Q = j.zeros_like(Qp).at[rp].set(Qp)
    Z = j.zeros_like(Zp).at[cp].set(Zp)
    st = H, T, Q, Z
    eps = j.finfo(j.float64).eps
    tiny = j.finfo(j.float64).tiny
    atol = j.maximum(
        tiny, eps * j.linalg.norm(H[active_lo : active_hi + 1, active_lo : active_hi + 1])
    )
    btol = j.maximum(
        tiny, eps * j.linalg.norm(T[active_lo : active_hi + 1, active_lo : active_hi + 1])
    )
    anorm = stable_norm(H[active_lo : active_hi + 1, active_lo : active_hi + 1])
    bnorm = stable_norm(T[active_lo : active_hi + 1, active_lo : active_hi + 1])
    ascale = 1 / j.maximum(tiny, anorm)
    bscale = 1 / j.maximum(tiny, bnorm)
    atol = j.maximum(tiny, eps * anorm)
    btol = j.maximum(tiny, eps * bnorm)
    hi = n - 1
    iteration = total = 0
    exceptional = 0.0
    budget = 30 * n if max_sweeps is None else max_sweeps
    pairs = []
    while hi >= 0:
        H, T, Q, Z = st
        lo, z, two = map(int, inspect(H, T, hi, atol, btol, ascale))
        if lo > 0:
            H = H.at[lo, lo - 1].set(0.0)
            st = H, T, Q, Z
        if lo == hi:
            sign = j.where(j.signbit(T[hi, hi]), -1.0, 1.0)
            st = (
                H.at[:, hi].set(binary.multiply(H[:, hi], sign)),
                T.at[:, hi].set(binary.multiply(T[:, hi], sign)),
                Q,
                Z.at[:, hi].multiply(sign),
            )
            hi -= 1
            iteration = 0
            exceptional = 0.0
            continue
        if total >= budget:
            raise RuntimeError(f"real QZ nonconvergence active={hi + 1} sweeps={total}")
        if z >= 0:
            st = chase_zero((H, T.at[z, z].set(0.0), Q, Z), z, hi, (z == lo) or bool(two))
            total += 1
            continue
        iteration += 1
        total += 1
        s1, s2, wr, wr2, wi = lag2(
            H[hi - 1 : hi + 1, hi - 1 : hi + 1], T[hi - 1 : hi + 1, hi - 1 : hi + 1]
        )
        if iteration % 10 == 0:
            increment = j.where(
                (budget * tiny) * j.abs(H[hi - 1, hi]) < j.abs(T[hi - 1, hi - 1]),
                H[hi - 1, hi] / T[hi - 1, hi - 1],
                1 / (tiny * budget),
            )
            exceptional += float(increment)
            s1 = j.array(1.0)
            wr = j.array(exceptional)
            wi = j.array(0.0)
        if float(wi) == 0:
            bound = j.minimum(ascale, 1.0) * (0.5 / tiny)
            shift_scale = j.minimum(1.0, bound / s1)
            bound = j.minimum(bscale, 1.0) * (0.5 / tiny)
            shift_scale = j.minimum(shift_scale, bound / j.maximum(j.abs(wr), tiny))
            s1, wr = s1 * shift_scale, wr * shift_scale
            start = single_start(H, T, lo, hi, s1, wr, ascale, atol)
            st = single_sweep(st, start, hi, s1, wr)
        elif hi == lo + 1:
            st = standardize_pair(st, lo)
            H, T, Q, Z = st
            s1, s2, wr, wr2, wi = lag2(H[lo : hi + 1, lo : hi + 1], T[lo : hi + 1, lo : hi + 1])
            if float(wi) != 0:
                pairs.append((lo, s1, s2, wr, wr2, wi))
                hi -= 2
                iteration = 0
                exceptional = 0.0
        else:
            st = double_sweep(st, lo, hi, ascale, bscale)
    H, T, Q, Z = st
    alpha = j.diag(H).astype(j.complex128)
    beta = j.diag(T)
    for k, s1, s2, wr, wr2, wi in pairs:
        ap, bp = pair_homogeneous(H[k : k + 2, k : k + 2], T[k : k + 2, k : k + 2], s1, wr, wi)
        alpha = alpha.at[k : k + 2].set(ap)
        beta = beta.at[k : k + 2].set(bp)
    restored = lax.complex(binary.multiply(j.real(alpha), sa), binary.multiply(j.imag(alpha), sa))
    return dict(
        S=H,
        T=T,
        Q=Q,
        Z=Z,
        alpha=restored,
        beta=binary.multiply(beta, sb),
        alpha_scaled=alpha,
        beta_scaled=beta,
        pairs=[x[0] for x in pairs],
        sweeps=total,
        a_scale=sa,
        b_scale=sb,
    )


@jax.jit
def pair_homogeneous(A, B, s1, wr, wi):
    """DHGEQZ QZVAL normalization for a standardized complex 2x2 block.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    a11, a12, a21, a22 = A.reshape(-1)
    b11, b22 = B[0, 0], B[1, 1]
    c11r = s1 * a11 - wr * b11
    c11i = -wi * b11
    c12 = s1 * a12
    c21 = s1 * a21
    c22r = s1 * a22 - wr * b22
    c22i = -wi * b22
    safe = j.finfo(A.dtype).tiny

    def first(_):
        norm = j.hypot(c12, j.hypot(c11r, c11i))
        return c12 / norm, -c11r / norm, -c11i / norm

    def second(_):
        cz = j.hypot(c22r, c22i)

        def usual(_):
            tr, ti = c22r / cz, c22i / cz
            norm = j.hypot(cz, c21)
            return cz / norm, -c21 * tr / norm, c21 * ti / norm

        return lax.cond(
            cz <= safe, lambda _: (j.array(0.0), j.array(1.0), j.array(0.0)), usual, None
        )

    cz, szr, szi = lax.cond(
        j.abs(c11r) + j.abs(c11i) + j.abs(c12) > j.abs(c21) + j.abs(c22r) + j.abs(c22i),
        first,
        second,
        None,
    )

    def from_b(_):
        return cz * b11, szr * b22, -szi * b22

    def from_a(_):
        a1r = cz * a11 + szr * a12
        a1i = szi * a12
        a2r = cz * a21 + szr * a22
        a2i = szi * a22
        cq = j.hypot(a1r, a1i)

        def usual(_):
            tr, ti = a1r / cq, a1i / cq
            return cq, tr * a2r + ti * a2i, ti * a2r - tr * a2i

        return lax.cond(
            cq <= safe, lambda _: (j.array(0.0), j.array(1.0), j.array(0.0)), usual, None
        )

    cq, sqr, sqi = lax.cond(
        s1 * j.sum(j.abs(A)) > (j.abs(wr) + j.abs(wi)) * (j.abs(b11) + j.abs(b22)),
        from_b,
        from_a,
        None,
    )
    norm = j.hypot(cq, j.hypot(sqr, sqi))
    cq, sqr, sqi = cq / norm, sqr / norm, sqi / norm
    tr, ti = sqr * szr - sqi * szi, sqr * szi + sqi * szr
    b1 = j.hypot(cq * cz * b11 + tr * b22, ti * b22)
    b2 = j.hypot(cq * cz * b22 + tr * b11, -ti * b11)
    return j.array(
        [lax.complex((wr * b1) / s1, (wi * b1) / s1), lax.complex((wr * b2) / s1, -(wi * b2) / s1)]
    ), j.array([b1, b2])


def abs1(z):
    """Abs1.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    return j.abs(j.real(z)) + j.abs(j.imag(z))


def safe_div(a, b):
    # Smith division after denominator normalization; avoids squaring b.
    """Safe div.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    scale = j.maximum(j.abs(j.real(b)), j.abs(j.imag(b)))
    br, bi = j.real(b) / scale, j.imag(b) / scale
    ar, ai = j.real(a) / scale, j.imag(a) / scale
    den = br * br + bi * bi
    return lax.complex((ar * br + ai * bi) / den, (ai * br - ar * bi) / den)


@jax.jit
def solve2(C, rhs, dmin):
    """DLALN2 complete-pivot block solve with small-pivot perturbation.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    small = j.maximum(dmin, 2 * j.finfo(j.float64).tiny)
    big = 1 / (2 * j.finfo(j.float64).tiny)
    # Fortran column-major tie ordering.
    pivot = j.argmax(abs1(C.T).reshape(-1))
    row, col = pivot % 2, pivot // 2
    original_rhs = rhs
    C = C[j.array([row, 1 - row])][:, j.array([col, 1 - col])]
    rhs = rhs[j.array([row, 1 - row])]
    cmax = j.max(abs1(C))
    bnorm = j.max(abs1(rhs))

    def tiny(_):
        scale = j.where((small < 1) & (bnorm > 1) & (bnorm > big * small), 1 / bnorm, 1.0)
        return (scale / small) * original_rhs, scale

    def regular(_):
        inverse = safe_div(j.array(1 + 0j), C[0, 0])
        mult = C[1, 0] * inverse
        u12 = C[0, 1] * inverse
        pivot2 = C[1, 1] - mult * C[0, 1]
        pnorm = abs1(pivot2)
        pivot2 = j.where(pnorm < small, small + 0j, pivot2)
        y2 = rhs[1] - mult * rhs[0]
        bound = j.maximum(abs1(rhs[0]) * (pnorm * abs1(inverse)), abs1(y2))
        scale = j.where((bound > 1) & (pnorm < 1) & (bound >= big * pnorm), 1 / bound, 1.0)
        x2 = safe_div(scale * y2, pivot2)
        x1 = inverse * (scale * rhs[0]) - u12 * x2
        return j.array([x1, x2]), scale

    x, scale = lax.cond(cmax < small, tiny, regular, None)
    x = j.where(cmax < small, x, x[j.array([col, 1 - col])])
    xmax = j.max(abs1(x))
    extra = j.where((xmax > 1) & (cmax > 1) & (xmax > big / cmax), cmax / big, 1.0)
    return x * extra, scale * extra


def balance_permute(A, B):
    """DGGBAL JOB=P simultaneous sparsity permutations; no rank estimate.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    from ._qz_ieee import nonzero

    n = len(A)
    lo = 0
    hi = n - 1
    rp = j.arange(n)
    cp = j.arange(n)

    def exchange(A, B, rp, cp, i, k, m):
        ri = j.arange(n).at[i].set(m).at[m].set(i)
        ci = j.arange(n).at[k].set(m).at[m].set(k)
        return A[ri][:, ci], B[ri][:, ci], rp[ri], cp[ci]

    while hi > 0:
        found = False
        nz = nonzero(A) | nonzero(B)
        for i in range(hi, -1, -1):
            columns = j.where(nz[i, : hi + 1], size=hi + 1, fill_value=-1)[0].tolist()
            columns = [x for x in columns if x >= 0]
            if len(columns) <= 1:
                k = columns[0] if columns else hi
                A, B, rp, cp = exchange(A, B, rp, cp, i, k, hi)
                hi -= 1
                found = True
                break
        if not found:
            break
    while lo < hi:
        found = False
        nz = nonzero(A) | nonzero(B)
        for k in range(lo, hi + 1):
            rows = j.where(nz[lo : hi + 1, k], size=hi - lo + 1, fill_value=-1)[0].tolist()
            rows = [x + lo for x in rows if x >= 0]
            if len(rows) <= 1:
                i = rows[0] if rows else hi
                A, B, rp, cp = exchange(A, B, rp, cp, i, k, lo)
                lo += 1
                found = True
                break
        if not found:
            break
    return A, B, rp, cp, lo, hi


@partial(jax.jit, static_argnames=("lo", "hi", "safe_qr"))
def reduce_balanced(A, B, lo, hi, safe_qr=False):
    """Reduce balanced.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    n = len(A)
    H = A
    T = B
    Q = j.eye(n)
    Z = j.eye(n)
    if hi > lo:
        q, t = (
            gradual.qr(B[lo : hi + 1, lo:])
            if safe_qr
            else j.linalg.qr(B[lo : hi + 1, lo:], mode="complete")
        )
        T = T.at[lo : hi + 1, lo:].set(t)
        H = H.at[lo : hi + 1, lo:].set(gradual.matmul(q.T, H[lo : hi + 1, lo:]))
        Q = Q.at[lo : hi + 1, lo : hi + 1].set(q)
    st = H, T, Q, Z

    def column(k, st):
        def eliminate(t, st):
            rr = hi - t
            H, T, Q, Z = st
            c, s = rotation(H[rr - 1, k], H[rr, k])
            H, T, Q, Z = left(st, rr - 1, rr, c, s)
            H = H.at[rr, k].set(0.0)
            c, s = rotation(T[rr, rr], T[rr, rr - 1])
            H, T, Q, Z = right((H, T, Q, Z), rr, rr - 1, c, s)
            return H, T.at[rr, rr - 1].set(0.0), Q, Z

        return lax.fori_loop(0, hi - k - 1, eliminate, st)

    return lax.fori_loop(lo, max(lo, hi - 1), column, st)


@jax.jit
def vectors_kernel(H, T, Z, alpha, beta, pairs):
    """Vectors kernel.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    if len(H) <= 1:
        return j.eye(len(H), dtype=j.complex128)
    n = len(H)
    hn = j.linalg.norm(H, ord=1)
    tn = j.linalg.norm(T, ord=1)
    eps = j.finfo(j.float64).eps
    safe = j.finfo(j.float64).tiny
    eye = j.eye(n, dtype=j.complex128)

    def eigenvector(index, V):
        def conjugate(_):
            return j.conj(V[:, index - 1])

        def compute(_):
            a, b = alpha[index], beta[index]
            scale = j.maximum(j.maximum(abs1(a), abs1(b)), safe)
            a, b = a / scale, b / scale
            C = b * H - a * T
            dmin = j.maximum(j.maximum(eps * abs1(b) * hn, eps * abs1(a) * tn), safe)

            def pair(_):
                end = index + 1
                lower = C[end, index]
                diagonal = C[end, end]
                v = lax.cond(
                    abs1(lower) >= abs1(diagonal),
                    lambda _: j.array([-safe_div(diagonal, lower), 1 + 0j]),
                    lambda _: j.array([1 + 0j, -safe_div(C[index, index], C[index, end])]),
                    None,
                )
                return j.zeros(n, dtype=j.complex128).at[index].set(v[0]).at[end].set(v[1])

            x = lax.cond(pairs[index], pair, lambda _: eye[:, index], None)

            def back(carry):
                k, x = carry
                paired = (k > 0) & pairs[k - 1]

                def two(_):
                    start = k - 1
                    block = lax.dynamic_slice(C, (start, start), (2, 2))
                    rhs = -(lax.dynamic_slice(C, (start, 0), (2, n)) @ x)
                    xx, scale = solve2(block, rhs, dmin)
                    return start - 1, (x * scale).at[start].set(xx[0]).at[k].set(xx[1])

                def one(_):
                    rhs = -(C[k, :] @ x)
                    pivot = C[k, k]
                    pivot = j.where(abs1(pivot) < dmin, dmin + 0j, pivot)
                    return k - 1, x.at[k].set(safe_div(rhs, pivot))

                k, x = lax.cond(paired, two, one, None)
                return k, x / j.maximum(1.0, j.max(abs1(x)))

            _, x = lax.while_loop(lambda carry: carry[0] >= 0, back, (index - 1, x))
            v = j.where((a == 0) & (b == 0), eye[:, index], Z @ x)
            return v / j.maximum(j.linalg.norm(v), safe)

        v = lax.cond((index > 0) & pairs[index - 1], conjugate, compute, None)
        return V.at[:, index].set(v)

    return lax.fori_loop(0, n, eigenvector, j.zeros((n, n), dtype=j.complex128))


def right_vectors_compiled(out):
    """Right vectors compiled.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    pairs = j.zeros(len(out["S"]), dtype=j.bool_)
    if out["pairs"]:
        pairs = pairs.at[j.array(out["pairs"])].set(True)
    return vectors_kernel(
        out["S"], out["T"], out["Z"], out["alpha_scaled"], out["beta_scaled"], pairs
    )


@jax.jit
def stable_norm(X):
    """Stable norm.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    scale = j.max(j.abs(X))
    return scale * j.sqrt(j.sum((X / j.where(scale > 0, scale, 1.0)) ** 2))


@jax.jit
def single_start(H, T, lo, hi, s1, wr, ascale, atol):
    """Single start.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    def check(t, start):
        k = hi - 1 - t
        temp = j.abs(s1 * H[k, k - 1])
        temp2 = j.abs(s1 * H[k, k] - wr * T[k, k])
        scale = j.maximum(temp, temp2)
        scale = j.where((scale < 1) & (scale > 0), scale, 1.0)
        temp, temp2 = temp / scale, temp2 / scale
        small = j.abs(ascale * H[k + 1, k]) * temp <= (ascale * atol) * temp2
        return j.where((start == lo) & small, k, start)

    return lax.fori_loop(0, hi - lo - 1, check, lo)
