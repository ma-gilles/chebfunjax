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

"""Generalized Hermitian eigensolver for the dense eig boundary.

LAPACK DPOTF2/ZPOTF2 Cholesky and DSYTD2/ZHETD2 Householder reduction
principles, with explicit congruence transformations and gradual arithmetic.
DSTEQR/DLAEV2 tridiagonal eigensolver in _eig_tridiagonal.py.
The Netlib BSD notice is retained above.
"""

import jax
import jax.numpy as j
from jax import lax

from chebfunjax.operators import _qz_gradual as gradual
from chebfunjax.operators import _qz_ieee as binary

from ._eig_tridiagonal import MASK, eigh_tridiagonal, exact_hermitian, sqrt_abs, words


@jax.jit
def cholesky_kernel(B):
    """Cholesky kernel.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    n = len(B)
    indices = j.arange(n)

    def column(k, state):
        L, info = state
        dot = gradual.matmul(L[k, :][None, :], j.conj(L[k, :]))[0]
        p = binary.add(j.real(B[k, k]), -j.real(dot))
        good = ((words(p) & MASK) != 0) & ((words(p) >> j.uint64(63)) == 0) & j.isfinite(p)
        info = j.where((info == 0) & ~good, k + 1, info)
        diag = sqrt_abs(j.where(good, p, 1.0))
        cross = gradual.matmul(L, j.conj(L[k, :]))
        col = gradual.divide(gradual.add(B[:, k], -cross), diag)
        col = j.where(indices > k, col, j.where(indices == k, diag, L[:, k]))
        return L.at[:, k].set(col), info

    return lax.fori_loop(0, n, column, (j.zeros_like(B), j.array(0)))


def cholesky(B):
    """Cholesky.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    L, info = cholesky_kernel(B)
    return L if int(info) == 0 else None


@jax.jit
def triangular_solve(L, R):
    """Triangular solve.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    def row(k, X):
        dot = gradual.matmul(L[k, :][None, :], X)[0]
        return X.at[k].set(gradual.divide(gradual.add(R[k], -dot), L[k, k]))

    return lax.fori_loop(0, len(L), row, j.zeros_like(R))


def lower_storage(C):
    """Lower storage.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    lower = j.tril(C, -1)
    out = gradual.add(lower, j.conj(lower.T))
    return out.at[j.diag_indices(len(C))].set(j.real(j.diag(C)))


def reflector(x):
    """Reflector.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    bits = j.maximum(words(j.real(x)) & MASK, words(j.imag(x)) & MASK)
    scale = lax.bitcast_convert_type(j.max(bits), j.float64)
    y = gradual.divide(x, j.where(binary.nonzero(scale), scale, 1.0))
    alpha = y[0]
    tail = y.at[0].set(0.0)
    inactive = ~j.any(
        binary.nonzero(j.real(tail)) | binary.nonzero(j.imag(tail))
    ) & ~binary.nonzero(j.imag(alpha))
    if bool(inactive):
        return j.zeros_like(x).at[0].set(1.0), j.array(0.0, dtype=x.dtype), j.real(x[0])
    norm = j.sqrt(j.sum(j.real(tail) ** 2 + j.imag(tail) ** 2))
    beta = -j.copysign(j.hypot(j.real(alpha), j.hypot(j.imag(alpha), norm)), j.real(alpha))
    tau = gradual.divide(gradual.add(beta, -alpha), beta)
    v = gradual.divide(tail, gradual.add(alpha, -beta)).at[0].set(1.0)
    return v, tau, binary.multiply(beta, scale)


@jax.jit
def reduce_hermitian(C):
    """Reduce hermitian.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    n = len(C)
    indices = j.arange(n)

    def column(k, state):
        C, Q = state
        x = j.where(indices > k, C[:, k], 0.0)
        bits = j.maximum(words(j.real(x)) & MASK, words(j.imag(x)) & MASK)
        scale = lax.bitcast_convert_type(j.max(bits), j.float64)
        y = gradual.divide(x, j.where(binary.nonzero(scale), scale, 1.0))
        alpha = y[k + 1]
        tail = y.at[k + 1].set(0.0)
        norm = j.sqrt(j.sum(j.real(tail) ** 2 + j.imag(tail) ** 2))
        beta = -j.copysign(j.hypot(j.real(alpha), j.hypot(j.imag(alpha), norm)), j.real(alpha))
        inactive = ~j.any(
            binary.nonzero(j.real(tail)) | binary.nonzero(j.imag(tail))
        ) & ~binary.nonzero(j.imag(alpha))
        tau = j.where(inactive, 0.0, gradual.divide(gradual.add(beta, -alpha), beta))
        denominator = j.where(inactive, 1.0, gradual.add(alpha, -beta))
        v = gradual.divide(tail, denominator).at[k + 1].set(1.0)
        beta = j.where(inactive, j.real(x[k + 1]), binary.multiply(beta, scale))

        def apply(state):
            C, Q = state
            w = gradual.multiply(
                tau,
                gradual.matmul(j.where((indices[:, None] > k) & (indices[None, :] > k), C, 0.0), v),
            )
            w = j.where(indices > k, w, 0.0)
            dot = gradual.matmul(j.conj(w)[None, :], v)[0]
            correction = gradual.multiply(-0.5, gradual.multiply(tau, dot))
            w = gradual.add(w, gradual.multiply(correction, v))
            update = gradual.add(
                gradual.multiply(v[:, None], j.conj(w)[None, :]),
                gradual.multiply(w[:, None], j.conj(v)[None, :]),
            )
            C = lower_storage(
                j.where((indices[:, None] > k) & (indices[None, :] > k), gradual.add(C, -update), C)
            )
            col = j.where(indices > k, 0.0, C[:, k]).at[k + 1].set(beta)
            C = C.at[:, k].set(col).at[k, :].set(j.conj(col))
            qv = gradual.matmul(j.where(indices[None, :] > k, Q, 0.0), v)
            Q = j.where(
                indices[None, :] > k,
                gradual.add(
                    Q, -gradual.multiply(gradual.multiply(qv[:, None], tau), j.conj(v)[None, :])
                ),
                Q,
            )
            return C, Q

        return lax.cond(inactive, lambda state: state, apply, (C, Q))

    C, Q = lax.fori_loop(0, n - 1, column, (C, j.eye(n, dtype=C.dtype)))
    return j.real(j.diag(C)), j.real(j.diag(C, -1)), Q


@jax.jit
def congruence(A, L):
    """Congruence.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    n = len(A)
    indices = j.arange(n)

    def column(k, C):
        akk = binary.divide(j.real(C[k, k]), binary.multiply(j.real(L[k, k]), j.real(L[k, k])))
        C = C.at[k, k].set(akk)

        def tail_update(C):
            b = j.where(indices > k, L[:, k], 0.0)
            a = j.where(indices > k, gradual.divide(C[:, k], j.real(L[k, k])), 0.0)
            correction = gradual.multiply(binary.multiply(-0.5, akk), b)
            a = gradual.add(a, correction)
            update = gradual.add(
                gradual.multiply(a[:, None], j.conj(b)[None, :]),
                gradual.multiply(b[:, None], j.conj(a)[None, :]),
            )
            C = lower_storage(
                j.where((indices[:, None] > k) & (indices[None, :] > k), gradual.add(C, -update), C)
            )
            a = gradual.add(a, correction)

            def solve_row(i, x):
                dot = gradual.matmul(L[i, :][None, :], x)[0]
                value = gradual.divide(gradual.add(a[i], -dot), L[i, i])
                return x.at[i].set(value)

            x = lax.fori_loop(k + 1, n, solve_row, j.zeros(n, dtype=A.dtype))
            col = j.where(indices > k, x, C[:, k])
            return C.at[:, k].set(col).at[k, :].set(j.conj(col))

        return lax.cond(k < n - 1, tail_update, lambda C: C, C)

    return lax.fori_loop(0, n, column, A)


def real_scale(X, factor):
    """Real scale.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    if j.issubdtype(X.dtype, j.complexfloating):
        return lax.complex(binary.multiply(j.real(X), factor), binary.multiply(j.imag(X), factor))
    return binary.multiply(X, factor)


def real_divide(X, factor):
    """Real divide.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    if j.issubdtype(X.dtype, j.complexfloating):
        return lax.complex(binary.divide(j.real(X), factor), binary.divide(j.imag(X), factor))
    return binary.divide(X, factor)


def generalized_hermitian(A, B):
    """Generalized hermitian.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    dtype = j.result_type(A, B, j.float64)
    A, B = j.asarray(A, dtype), j.asarray(B, dtype)
    if not bool(exact_hermitian(A) & exact_hermitian(B)):
        return None
    L = cholesky(B)
    if L is None:
        return None
    C = congruence(A, L)
    # DSYEV/ZHEEV quick return precedes norm-based scaling.
    if len(C) == 1:
        vectors = triangular_solve(j.conj(L.T), j.eye(1, dtype=C.dtype))
        return j.real(j.diag(C)), j.real(vectors)
    # DLANSY/ZLANHE maximum modulus, including gradual subnormal transport.
    if j.issubdtype(C.dtype, j.complexfloating):
        components = j.maximum(words(j.real(C)) & MASK, words(j.imag(C)) & MASK)
        local_scale = lax.bitcast_convert_type(components, j.float64)
        divisor = j.where(j.isfinite(local_scale) & binary.nonzero(local_scale), local_scale, 1.0)
        modulus = binary.multiply(
            j.hypot(binary.divide(j.real(C), divisor), binary.divide(j.imag(C), divisor)), divisor
        )
    else:
        modulus = lax.bitcast_convert_type(words(C) & MASK, j.float64)
    scale = lax.bitcast_convert_type(j.max(words(modulus) & MASK), j.float64)
    sml = j.finfo(j.float64).tiny / j.finfo(j.float64).eps
    rmin, rmax = sml**0.5, (1 / sml) ** 0.5
    target = j.where(scale > rmax, rmax, j.where(scale < rmin, rmin, scale))
    if bool(binary.nonzero(scale) & ((scale < rmin) | (scale > rmax))):
        C = real_scale(real_divide(C, scale), target)
    else:
        target = j.array(1.0)
        scale = j.array(1.0)
    d, e, Q = reduce_hermitian(C)
    vals, V = eigh_tridiagonal(d, e, Q)
    vals = binary.multiply(binary.divide(vals, target), scale)
    # Solve L^H X=V by reversal; this preserves B-orthonormalization.
    X = triangular_solve(j.conj(L.T)[::-1, ::-1], V[::-1])[::-1]
    return vals, X
