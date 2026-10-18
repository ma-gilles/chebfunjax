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

"""DSTEQR/DLAEV2-style Hermitian tridiagonal eigensolver.

Primary algorithms: Netlib LAPACK double/dsteqr.f, double/dlaev2.f.
QR direction is represented by reversing the active block, then doing QL.
Used by operators._eig_hermitian for the dense eig boundary.
"""

import jax
import jax.numpy as j
from jax import lax

from chebfunjax.operators import _qz_ieee as binary

EPS = j.finfo(j.float64).eps / 2
TINY = j.finfo(j.float64).tiny
SSFMIN = TINY**0.5 / EPS**2
SSFMAX = (1 / TINY) ** 0.5 / 3
MASK = j.uint64(0x7FFFFFFFFFFFFFFF)
SIGN = j.uint64(0x8000000000000000)


def words(x):
    """Words.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    return lax.bitcast_convert_type(x, j.uint64)


def sqrt_abs(x):
    """Sqrt abs.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    _, m, e = binary.parts(x)
    return binary.compose(j.uint64(0), j.sqrt(m * j.where(e % 2, 2.0, 1.0)), e // 2)


def exact_hermitian(A):
    """Exact hermitian.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    def canon(x):
        w = words(x)
        return j.where((w & MASK) == 0, j.uint64(0), w)

    return j.all(canon(j.real(A)) == canon(j.real(A.T))) & j.all(
        canon(j.imag(A)) == canon(-j.imag(A.T))
    )


@jax.jit
def eig2(a, b, c):
    """Eig2.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    sm, df, tb = a + c, a - c, 2 * b
    rt = j.hypot(df, tb)
    big, small = j.where(j.abs(a) > j.abs(c), a, c), j.where(j.abs(a) > j.abs(c), c, a)
    r1 = j.where(sm < 0, 0.5 * (sm - rt), 0.5 * (sm + rt))
    r2 = j.where(sm == 0, -0.5 * rt, (big / r1) * small - (b / r1) * b)
    cs = j.where(df >= 0, df + rt, df - rt)
    ct = -tb / j.where(cs != 0, cs, 1.0)
    tn = -cs / j.where(tb != 0, tb, 1.0)
    sn1 = j.where(
        j.abs(cs) > j.abs(tb),
        1 / j.sqrt(1 + ct * ct),
        j.where(tb == 0, 0.0, tn / j.sqrt(1 + tn * tn)),
    )
    cs1 = j.where(
        j.abs(cs) > j.abs(tb),
        ct / j.sqrt(1 + ct * ct),
        j.where(tb == 0, 1.0, 1 / j.sqrt(1 + tn * tn)),
    )
    switch = j.where(sm < 0, -1, 1) == j.where(df >= 0, 1, -1)
    return r1, r2, j.where(switch, -sn1, cs1), j.where(switch, cs1, sn1)


@jax.jit
def sweep(d, e, Z, l, m):
    """Sweep.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    g = (d[l + 1] - d[l]) / (2 * e[l])
    g = d[m] - d[l] + e[l] / (g + j.copysign(j.hypot(g, 1.0), g))

    def body(k, state):
        d, e, Z, g, s, c, p = state
        i = m - 1 - k
        f = s * e[i]
        b = c * e[i]
        r = j.copysign(j.hypot(g, f), g)
        c = j.where(r == 0, 1.0, g / r)
        s = j.where(r == 0, 0.0, f / r)
        e = lax.cond(i != m - 1, lambda x: x.at[i + 1].set(r), lambda x: x, e)
        g = d[i + 1] - p
        r = (d[i] - g) * s + 2 * c * b
        p = s * r
        d = d.at[i + 1].set(g + p)
        g = c * r - b
        u, v = Z[:, i], Z[:, i + 1]
        Z = Z.at[:, i].set(c * u - s * v).at[:, i + 1].set(s * u + c * v)
        return d, e, Z, g, s, c, p

    d, e, Z, g, _, _, p = lax.fori_loop(
        0, m - l, body, (d, e, Z, g, j.array(1.0), j.array(1.0), j.array(0.0))
    )
    return d.at[l].add(-p), e.at[l].set(g), Z


def eigh_tridiagonal(d, e, Z=None, maxiter=None):
    """Eigh tridiagonal.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    d, e = j.asarray(d, j.float64), j.asarray(e, j.float64)
    n = len(d)
    if e.shape != (max(0, n - 1),):
        raise ValueError("tridiagonal dimensions")
    Z = j.eye(n) if Z is None else j.asarray(Z)
    total = 0
    limit = 30 * n if maxiter is None else maxiter
    start = 0
    while start < n:
        end = start
        while end < n - 1:
            threshold = binary.multiply(
                binary.multiply(sqrt_abs(d[end]), sqrt_abs(d[end + 1])), EPS
            )
            if bool(
                (~binary.nonzero(e[end]))
                | (
                    j.isfinite(e[end])
                    & j.isfinite(threshold)
                    & ((words(e[end]) & MASK) <= (words(threshold) & MASK))
                )
            ):
                e = e.at[end].set(0.0)
                break
            end += 1
        if end == start:
            start += 1
            continue
        anorm = lax.bitcast_convert_type(
            j.maximum(j.max(words(d[start : end + 1]) & MASK), j.max(words(e[start:end]) & MASK)),
            j.float64,
        )
        target = (
            SSFMAX
            if bool(anorm > SSFMAX)
            else SSFMIN
            if bool((words(anorm) & MASK) < (words(j.array(SSFMIN)) & MASK))
            else anorm
        )
        scaled = bool(
            (anorm > SSFMAX)
            | (j.isfinite(anorm) & ((words(anorm) & MASK) < (words(j.array(SSFMIN)) & MASK)))
        )
        ds, es = d[start : end + 1], e[start:end]
        if scaled:
            ds = binary.multiply(binary.divide(ds, anorm), target)
            es = binary.multiply(binary.divide(es, anorm), target)
        zs = Z[:, start : end + 1]
        reverse = bool(j.abs(ds[-1]) < j.abs(ds[0]))
        if reverse:
            ds, es, zs = ds[::-1], es[::-1], zs[:, ::-1]
        l = 0
        k = len(ds)
        while l < k:
            m = l
            while m < k - 1 and not bool(
                es[m] ** 2 <= EPS**2 * j.abs(ds[m]) * j.abs(ds[m + 1]) + TINY
            ):
                m += 1
            if m < k - 1:
                es = es.at[m].set(0.0)
            if m == l:
                l += 1
                continue
            if m == l + 1:
                a, b, c, s = eig2(ds[l], es[l], ds[l + 1])
                u, v = zs[:, l], zs[:, l + 1]
                ds = ds.at[l].set(a).at[l + 1].set(b)
                es = es.at[l].set(0.0)
                zs = zs.at[:, l].set(c * u + s * v).at[:, l + 1].set(-s * u + c * v)
                l += 2
                continue
            if total >= limit:
                raise RuntimeError("DSTEQR iteration did not converge")
            total += 1
            ds, es, zs = sweep(ds, es, zs, l, m)
        if reverse:
            ds, zs = ds[::-1], zs[:, ::-1]
        if scaled:
            ds = binary.multiply(binary.divide(ds, target), anorm)
        d = d.at[start : end + 1].set(ds)
        Z = Z.at[:, start : end + 1].set(zs)
        start = end + 1

    return sort_words(d, Z)


@jax.jit
def sort_words(d, Z):
    """Sort words.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    w = lax.bitcast_convert_type(d, j.uint64)
    n = len(w)

    def outer(i, state):
        w, Z = state

        def inner(k, best):
            pw, qw = w[best], w[k]
            pk = j.where((pw & SIGN) != 0, ~pw, pw ^ SIGN)
            qk = j.where((qw & SIGN) != 0, ~qw, qw ^ SIGN)
            less = (
                ((pw & MASK) <= j.uint64(0x7FF0000000000000))
                & ((qw & MASK) <= j.uint64(0x7FF0000000000000))
                & (qk < pk)
                & (((pw & MASK) != 0) | ((qw & MASK) != 0))
            )
            return j.where(less, k, best)

        best = lax.fori_loop(i + 1, n, inner, i)
        wi, wb = w[i], w[best]
        zi, zb = Z[:, i], Z[:, best]
        return w.at[i].set(wb).at[best].set(wi), Z.at[:, i].set(zb).at[:, best].set(zi)

    w, Z = lax.fori_loop(0, n - 1, outer, (w, Z))
    return lax.bitcast_convert_type(w, j.float64), Z
