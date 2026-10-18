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

"""JAX arithmetic for gradual-underflow-sensitive linear transformations.
The slow branch is selected by IEEE exponent range, never spectral rank.
"""

import jax
import jax.numpy as j
from jax import lax

from . import _qz_ieee as binary


def _minimum_exponent(X):
    def real_min(x):
        _, m, e = binary.parts(x)
        return j.min(j.where(m != 0, e, 4096))

    if j.issubdtype(X.dtype, j.complexfloating):
        return j.minimum(real_min(j.real(X)), real_min(j.imag(X)))
    return real_min(X)


def multiply(a, b):
    """Multiply.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    if j.issubdtype(j.result_type(a, b), j.complexfloating):
        ar, ai, br, bi = j.real(a), j.imag(a), j.real(b), j.imag(b)
        return lax.complex(
            binary.add(binary.multiply(ar, br), -binary.multiply(ai, bi)),
            binary.add(binary.multiply(ar, bi), binary.multiply(ai, br)),
        )
    return binary.multiply(a, b)


def add(a, b):
    """Add.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    if j.issubdtype(j.result_type(a, b), j.complexfloating):
        return lax.complex(binary.add(j.real(a), j.real(b)), binary.add(j.imag(a), j.imag(b)))
    return binary.add(a, b)


@jax.jit
def matmul(A, B):
    """Matmul.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    vector = B.ndim == 1
    if vector:
        B = B[:, None]
    # Products this small can enter subnormal range after at most 53 bits
    # of cancellation; the bound follows binary64 precision and emin.
    risky = _minimum_exponent(A) + _minimum_exponent(B) < -1022 + 53

    def software(_):
        def term(k, total):
            return add(total, multiply(A[:, k, None], B[k, None, :]))

        return lax.fori_loop(
            0, A.shape[1], term, j.zeros((A.shape[0], B.shape[1]), dtype=j.result_type(A, B))
        )

    out = lax.cond(risky, software, lambda _: A @ B, None)
    return out[:, 0] if vector else out


@jax.jit
def muladd(a, c, b, s):
    """Muladd.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    risky = (
        j.minimum(
            _minimum_exponent(a) + _minimum_exponent(j.asarray(c)),
            _minimum_exponent(b) + _minimum_exponent(j.asarray(s)),
        )
        < -1022 + 53
    )
    return lax.cond(
        risky, lambda _: add(multiply(a, c), multiply(b, s)), lambda _: a * c + b * s, None
    )


def divide(a, b):
    """Divide.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    if not j.issubdtype(j.result_type(a, b), j.complexfloating):
        return binary.divide(a, b)
    br, bi = j.real(b), j.imag(b)
    bits = j.maximum(
        lax.bitcast_convert_type(br, j.uint64) & j.uint64(0x7FFFFFFFFFFFFFFF),
        lax.bitcast_convert_type(bi, j.uint64) & j.uint64(0x7FFFFFFFFFFFFFFF),
    )
    scale = lax.bitcast_convert_type(bits, j.float64)
    ar, ai = binary.divide(j.real(a), scale), binary.divide(j.imag(a), scale)
    br, bi = binary.divide(br, scale), binary.divide(bi, scale)
    denominator = br * br + bi * bi
    return lax.complex(
        binary.divide(binary.add(binary.multiply(ar, br), binary.multiply(ai, bi)), denominator),
        binary.divide(binary.add(binary.multiply(ai, br), -binary.multiply(ar, bi)), denominator),
    )


@jax.jit
def qr(B):
    """DLARFG/ZLARFG Householder QR with binary64 gradual transport.

    Used only for exponent-sensitive blocks; normal blocks retain JAX QR.
    Normalizing before reflector formation follows LAPACK's safe scaling,
    while software multiply/add preserves subnormal trailing updates.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    m, n = B.shape
    indices = j.arange(m)

    def column(k, st):
        T, Q = st
        x = j.where(indices >= k, T[:, k], 0.0)
        bits = lax.bitcast_convert_type(j.real(x), j.uint64) & j.uint64(0x7FFFFFFFFFFFFFFF)
        if j.issubdtype(B.dtype, j.complexfloating):
            bits = j.maximum(
                bits, lax.bitcast_convert_type(j.imag(x), j.uint64) & j.uint64(0x7FFFFFFFFFFFFFFF)
            )
        scale = lax.bitcast_convert_type(j.max(bits), j.float64)
        y = divide(x, j.where(binary.nonzero(scale), scale, 1.0))
        alpha = y[k]
        tail = j.where(indices > k, y, 0.0)
        norm = j.sqrt(j.sum(j.real(tail) ** 2 + j.imag(tail) ** 2))
        beta = -j.copysign(j.hypot(j.real(alpha), j.hypot(j.imag(alpha), norm)), j.real(alpha))
        inactive = (~j.any(binary.nonzero(j.real(tail)) | binary.nonzero(j.imag(tail)))) & (
            ~binary.nonzero(j.imag(alpha))
        )

        def reflect(_):
            tau = divide(add(beta, -alpha), beta)
            v = divide(tail, add(alpha, -beta)).at[k].set(1.0)
            vt = matmul(j.conj(v)[None, :], T)[0]
            Tnew = add(T, -multiply(multiply(v[:, None], j.conj(tau)), vt[None, :]))
            qv = matmul(Q, v)
            Qnew = add(Q, -multiply(multiply(qv[:, None], tau), j.conj(v)[None, :]))
            Tnew = Tnew.at[k, k].set(binary.multiply(beta, scale))
            Tnew = Tnew.at[:, k].set(j.where(indices > k, 0.0, Tnew[:, k]))
            return Tnew, Qnew

        return lax.cond(inactive, lambda _: (T, Q), reflect, None)

    T, Q = lax.fori_loop(0, min(m, n), column, (B, j.eye(m, dtype=B.dtype)))
    return Q, T
