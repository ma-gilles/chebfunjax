"""JAX complex QZ for the dense Chebfun generalized eigenvalue boundary.

Provenance
----------
Chebfun commit: 7574c77
MATLAB @linop/eigs.m:getEigenvalues invokes dense eig(A,B) before source
constraint deflation, target selection, mode filtering and adaptive refinement.
This module implements that numerical boundary using JAX primitives only.

Algorithm references: Netlib LAPACK ZGGHRD, ZHGEQZ, ZLARTG and ZTGEVC,
https://www.netlib.org/lapack/complex16/ . The unitary reduction, ordered QZ
splits/BTOL deflation, complex shifts and scaled triangular solve are expressed
independently in JAX. The following BSD notice accompanies this derived work.
This is a generalized Schur solver, not a singular Kronecker decomposition.
"""
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

from functools import partial

import jax
import jax.numpy as j
from jax import lax

from . import _qz_gradual as gradual
from . import _qz_ieee as binary


def _magnitude(z):
    return j.hypot(j.real(z), j.imag(z))


def _rotation(f, g):
    # Normalize real components separately, as ZLARTG does, so complex
    # division cannot square a very large or very small denominator.
    fs = j.maximum(j.abs(j.real(f)), j.abs(j.imag(f)))
    gs = j.maximum(j.abs(j.real(g)), j.abs(j.imag(g)))
    scale = j.maximum(fs, gs)
    scale = j.where(scale > 0, scale, 1.0)
    fd = j.where(fs > 0, fs, 1.0)
    gd = j.where(gs > 0, gs, 1.0)
    fn = lax.complex(j.real(f) / fd, j.imag(f) / fd)
    gn = lax.complex(j.real(g) / gd, j.imag(g) / gd)
    af = (fs / scale) * _magnitude(fn)
    ag = (gs / scale) * _magnitude(gn)
    norm = j.hypot(af, ag)
    norm = j.where(norm > 0, norm, 1.0)
    phase = fn / j.where(_magnitude(fn) > 0, _magnitude(fn), 1.0)
    phase = j.where(fs > 0, phase, 1 + 0j)
    c = j.where((fs == 0) & (gs == 0), 1.0, af / norm)
    s = phase * j.conj(gn) * ((gs / scale) / norm)
    return c, s


def _rows(X, p, q, c, s):
    a, b = X[p, :], X[q, :]
    return (
        X.at[p, :].set(gradual.muladd(a, c, b, s)).at[q, :].set(gradual.muladd(a, -j.conj(s), b, c))
    )


def _cols(X, p, q, c, s):
    a, b = X[:, p], X[:, q]
    return (
        X.at[:, p].set(gradual.muladd(a, c, b, s)).at[:, q].set(gradual.muladd(a, -j.conj(s), b, c))
    )


def _left(state, p, q, c, s):
    H, T, Q, Z = state
    return _rows(H, p, q, c, s), _rows(T, p, q, c, s), _cols(Q, p, q, c, j.conj(s)), Z


def _right(state, p, q, c, s):
    H, T, Q, Z = state
    return _cols(H, p, q, c, s), _cols(T, p, q, c, s), Q, _cols(Z, p, q, c, s)


@jax.jit
def _reduce(A, B):
    n = A.shape[0]
    Q, T = j.linalg.qr(B)
    H = Q.conj().T @ A
    Z = j.eye(n, dtype=A.dtype)

    def column(k, state):
        def eliminate(t, state):
            r = n - 1 - t
            H, T, Q, Z = state
            c, s = _rotation(H[r - 1, k], H[r, k])
            state = _left(state, r - 1, r, c, s)
            H, T, Q, Z = state
            H = H.at[r, k].set(0)
            state = (H, T, Q, Z)
            c, s = _rotation(T[r, r], T[r, r - 1])
            H, T, Q, Z = _right(state, r, r - 1, c, s)
            return H, T.at[r, r - 1].set(0), Q, Z

        return lax.fori_loop(0, n - k - 2, eliminate, state)

    return lax.fori_loop(0, max(n - 2, 0), column, (H, T, Q, Z))


@jax.jit
def _sweep(state, lo, hi, shift, ascale, bscale):
    H, T, _, _ = state
    c, s = _rotation(ascale * H[lo, lo] - shift * (bscale * T[lo, lo]), ascale * H[lo + 1, lo])

    def step(k, carry):
        state, c, s = carry

        def next_rot(_):
            H, T, Q, Z = state
            c, s = _rotation(H[k, k - 1], H[k + 1, k - 1])
            return c, s

        c, s = lax.cond(k == lo, lambda _: (c, s), next_rot, operand=None)
        previous = state[0][k, k - 1]
        H, T, Q, Z = _left(state, k, k + 1, c, s)
        H = lax.cond(
            (k == lo) & (k > 0),
            lambda h: h.at[k, k - 1].set(previous).at[k + 1, k - 1].set(0.0),
            lambda h: h,
            H,
        )
        H = lax.cond(k > lo, lambda X: X.at[k + 1, k - 1].set(0), lambda X: X, H)
        c, s = _rotation(T[k + 1, k + 1], T[k + 1, k])
        H, T, Q, Z = _right((H, T, Q, Z), k + 1, k, c, s)
        return (H, T.at[k + 1, k].set(0), Q, Z), c, s

    return lax.fori_loop(lo, hi, step, (state, c, s))[0]


@jax.jit
def _chase_zero(state, start, hi, leading):
    def top(state):
        H, T, Q, Z = state
        c, s = _rotation(H[start, start], H[start + 1, start])
        H, T, Q, Z = _left(state, start, start + 1, c, s)
        H = H.at[start + 1, start].set(0)
        H = lax.cond(start > 0, lambda X: X.at[start + 1, start - 1].set(0), lambda X: X, H)
        return H, T, Q, Z

    def interior(state):
        def step(k, state):
            H, T, Q, Z = state
            c, s = _rotation(T[k, k + 1], T[k + 1, k + 1])
            H, T, Q, Z = _left(state, k, k + 1, c, s)
            T = T.at[k + 1, k + 1].set(0)
            c, s = _rotation(H[k + 1, k], H[k + 1, k - 1])
            H, T, Q, Z = _right((H, T, Q, Z), k, k - 1, c, s)
            return H.at[k + 1, k - 1].set(0), T, Q, Z

        H, T, Q, Z = lax.fori_loop(start, hi, step, state)
        c, s = _rotation(H[hi, hi], H[hi, hi - 1])
        H, T, Q, Z = _right((H, T, Q, Z), hi, hi - 1, c, s)
        return H.at[hi, hi - 1].set(0), T, Q, Z

    return lax.cond(leading, top, interior, state)


@jax.jit
def _inspect(H, T, hi, atol, btol, ascale):
    n = H.shape[0]
    indices = j.arange(n)
    sub = j.concatenate([j.zeros(1), _abs1(j.diag(H, -1))])
    split = (indices <= hi) & (sub <= atol)
    lo = j.max(j.where(split, indices, 0))
    small = j.where(indices == hi, j.abs(j.diag(T)) <= btol, j.abs(j.diag(T)) < btol) | (
        j.diag(T) == 0
    )
    zeros = (indices >= lo) & (indices <= hi) & small
    z = j.max(j.where(zeros, indices, -1))
    temp = _abs1(H[z, z - 1])
    temp2 = _abs1(H[z, z])
    scale = j.maximum(temp, temp2)
    scale = j.where((scale < 1) & (scale > 0), scale, 1.0)
    temp, temp2 = temp / scale, temp2 / scale
    two = (
        (z >= 0)
        & (z < hi)
        & (z > lo)
        & (temp * (ascale * _abs1(H[z + 1, z])) <= temp2 * (ascale * atol))
    )
    return lo, z, two


@jax.jit
def _shift_value(H, T, hi, iteration, exceptional, ascale, bscale):
    H = H * ascale
    T = T * bscale
    a = H[hi - 1, hi - 1] / T[hi - 1, hi - 1]
    b = H[hi - 1, hi] / T[hi, hi]
    c = H[hi, hi - 1] / T[hi - 1, hi - 1]
    d = H[hi, hi] / T[hi, hi]
    u = T[hi - 1, hi] / T[hi, hi]
    v = d - u * c
    t = (a + v) / 2
    disc = j.sqrt(t * t + b * c - a * d)
    wilkinson = j.where(j.real(j.conj(t - v) * disc) <= 0, t + disc, t - disc)
    exceptional = exceptional + j.conj(H[hi - 1, hi] / T[hi - 1, hi - 1])
    return j.where(iteration % 10 == 0, exceptional, wilkinson), exceptional


def _schur(A, B, max_sweeps=None, mass_deflation="native"):
    if mass_deflation not in ("native", "exact"):
        raise ValueError("unknown deflation policy")
    A, B = j.asarray(A), j.asarray(B)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape != B.shape:
        raise ValueError("equal square matrices required")
    if not bool(j.all(j.isfinite(A)) & j.all(j.isfinite(B))):
        raise ValueError("finite matrices required")
    A = A.astype(j.complex128)
    B = B.astype(j.complex128)
    n = len(A)
    if n == 0:
        raise NotImplementedError("empty pencil")

    As, sa = _scale_input(A)
    Bs, sb = _scale_input(B)
    Ab, Bb, rp, cp, active_lo, active_hi = _balance_permute(As, Bs)
    H, T, Qp, Zp = _reduce_balanced(
        Ab,
        Bb,
        active_lo,
        active_hi,
        bool(gradual._minimum_exponent(Bb[active_lo : active_hi + 1, active_lo:]) < -1022 + 53),
    )
    Q = j.zeros_like(Qp).at[rp].set(Qp)
    Z = j.zeros_like(Zp).at[cp].set(Zp)
    state = H, T, Q, Z
    anorm = _stable_norm(H[active_lo : active_hi + 1, active_lo : active_hi + 1])
    bnorm = _stable_norm(T[active_lo : active_hi + 1, active_lo : active_hi + 1])
    atol = j.maximum(j.finfo(j.float64).tiny, j.finfo(j.float64).eps * anorm)
    btol = (
        j.maximum(j.finfo(j.float64).tiny, j.finfo(j.float64).eps * bnorm)
        if mass_deflation == "native"
        else 0.0
    )
    ascale = 1 / j.maximum(j.finfo(j.float64).tiny, anorm)
    bscale = 1 / j.maximum(j.finfo(j.float64).tiny, bnorm)
    hi = n - 1
    it = 0
    total = 0
    exceptional = 0j
    budget = 30 * n if max_sweeps is None else max_sweeps
    while hi >= 0:
        H, T, Q, Z = state
        lo, z, two = map(int, _inspect(H, T, hi, atol, btol, ascale))
        if lo > 0:
            H = H.at[lo, lo - 1].set(0)
            state = H, T, Q, Z
        if lo == hi:
            mag = _magnitude(T[hi, hi])
            if float(mag) > j.finfo(j.float64).tiny:
                phase = lax.complex(j.real(T[hi, hi]) / mag, -j.imag(T[hi, hi]) / mag)
                H = H.at[:, hi].set(_complex_multiply(H[:, hi], phase))
                T = T.at[:, hi].set(_complex_multiply(T[:, hi], phase))
                Z = Z.at[:, hi].set(_complex_multiply(Z[:, hi], phase))
                T = T.at[hi, hi].set(mag)
            else:
                T = T.at[hi, hi].set(0.0)
            state = H, T, Q, Z
            hi -= 1
            it = 0
            exceptional = 0j
            continue
        if total >= budget:
            raise RuntimeError(f"QZ nonconvergence: active={hi + 1}, sweeps={total}")
        if z >= 0:
            state = (H, T.at[z, z].set(0), Q, Z)
            state = _chase_zero(state, z, hi, (z == lo) or bool(two))
            total += 1
            continue
        it += 1
        total += 1
        shift, ex = _shift_value(H, T, hi, it, exceptional, ascale, bscale)
        if it % 10 == 0:
            exceptional = ex
        if not bool(j.isfinite(shift)):
            raise NotImplementedError("unsafe shift range; no mass-zero substitution")
        start = _single_start(H, T, lo, hi, shift, ascale, bscale, atol)
        state = _sweep(state, start, hi, shift, ascale, bscale)
    H, T, Q, Z = state
    alpha = _real_scale(j.diag(H), sa)
    beta = _real_scale(j.diag(T), sb)
    return dict(
        S=H,
        T=T,
        Q=Q,
        Z=Z,
        alpha=alpha,
        beta=beta,
        a_scale=sa,
        b_scale=sb,
        sweeps=total,
        mass_deflation=mass_deflation,
        indeterminate=(alpha == 0) & (beta == 0),
    )


@jax.jit
def _right_vectors(H, T, Z):
    """Scaled triangular back substitution, following TGEVC's safe-pivot idea.

    Per-step normalization avoids growing intermediate components. A repeated
    or defective eigenvalue can yield dependent vectors, as it mathematically
    must; no full eigenbasis assertion is made.
    """
    n = len(H)
    eps = j.finfo(j.float64).eps
    tiny = j.finfo(j.float64).tiny
    hn = j.linalg.norm(H, ord=1)
    tn = j.linalg.norm(T, ord=1)

    def column(k, X):
        a, b = H[k, k], T[k, k]
        indeterminate = (a == 0) & (b == 0)
        pair = j.maximum(j.abs(a), j.abs(b))
        pair = j.where(indeterminate, 1.0, pair)
        a = a / pair
        b = b / pair
        M = b * H - a * T
        dmin = j.maximum(tiny, eps * (j.abs(b) * hn + j.abs(a) * tn))
        x = j.zeros(n, dtype=H.dtype).at[k].set(1)

        def back(t, x):
            r = k - 1 - t
            d = M[r, r]
            phase = j.where(j.abs(d) > 0, d / j.where(j.abs(d) > 0, j.abs(d), 1), 1 + 0j)
            d = j.where(j.abs(d) < dmin, phase * dmin, d)
            rhs = -j.dot(M[r, :], x)
            x = x.at[r].set(rhs / d)
            return x / j.maximum(1.0, j.max(j.abs(x)))

        x = lax.fori_loop(0, k, back, x)
        x = Z @ x
        x = x / j.linalg.norm(x)
        # ZTGEVC returns a unit placeholder for an indeterminate pair.
        x = j.where(indeterminate, j.eye(n, dtype=H.dtype)[:, k], x)
        return X.at[:, k].set(x)

    return lax.fori_loop(0, n, column, j.zeros_like(H))


def _eigenpairs(A, B, **kwargs):
    result = _schur(A, B, **kwargs)
    result["vectors"] = _right_vectors(result["S"], result["T"], result["Z"])
    if not bool(j.all(j.isfinite(result["vectors"]))):
        raise RuntimeError("nonfinite triangular eigenvectors")
    return result


def generalized_eig(A, B):
    """Return eigenvalues and normalized right vectors of ``A x = lambda B x``.

    Homogeneous pairs are retained internally. Exact zero beta is converted to
    infinity only at this compatibility boundary. A zero alpha/beta pair
    yields NaN for the caller to filter, with a unit vector placeholder.
    Unsupported exponent range or QZ nonconvergence raises explicitly.

    Provenance
    ----------
    Chebfun commit: 7574c77
    MATLAB @linop/eigs.m:getEigenvalues dense eig(A,B) call; LAPACK
    ZGGHRD/ZHGEQZ/ZTGEVC algorithms, BSD notice at module top.
    """
    A, B = j.asarray(A), j.asarray(B)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or B.shape != A.shape:
        raise ValueError("equal square matrices required")
    if A.shape[0] == 0:
        return j.empty((0,), dtype=j.complex128), j.empty((0, 0), dtype=j.complex128)
    result = _eigenpairs(A, B)
    alpha, beta = result["alpha"], result["beta"]
    ar, ai = j.real(alpha), j.imag(alpha)
    br = j.real(beta)
    zero = ~binary.nonzero(br)
    indeterminate = zero & ~binary.nonzero(ar) & ~binary.nonzero(ai)
    rr = binary.divide(ar, br)
    ii = binary.divide(ai, br)
    rr = j.where(zero, j.where(indeterminate, j.nan, j.where(binary.nonzero(ar), rr, 0.0)), rr)
    # Native MATLAB preserves a zero imaginary component for 0:0,
    # even when other homogeneous alpha values are genuinely complex.
    ii = j.where(zero, j.where(binary.nonzero(ai), ii, 0.0), ii)
    values = lax.complex(rr, ii)
    return values, result["vectors"]


def _balance_permute(A, B):
    """DGGBAL JOB=P simultaneous sparsity permutations; no rank estimate."""
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
        nz = nonzero(j.real(A)) | nonzero(j.imag(A)) | nonzero(j.real(B)) | nonzero(j.imag(B))
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
        nz = nonzero(j.real(A)) | nonzero(j.imag(A)) | nonzero(j.real(B)) | nonzero(j.imag(B))
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
def _reduce_balanced(A, B, lo, hi, safe_qr=False):
    n = len(A)
    H = A
    T = B
    Q = j.eye(n, dtype=A.dtype)
    Z = j.eye(n, dtype=A.dtype)
    if hi > lo:
        q, t = (
            gradual.qr(B[lo : hi + 1, lo:])
            if safe_qr
            else j.linalg.qr(B[lo : hi + 1, lo:], mode="complete")
        )
        T = T.at[lo : hi + 1, lo:].set(t)
        H = H.at[lo : hi + 1, lo:].set(gradual.matmul(q.conj().T, H[lo : hi + 1, lo:]))
        Q = Q.at[lo : hi + 1, lo : hi + 1].set(q)
    st = H, T, Q, Z

    def column(k, st):
        def eliminate(t, st):
            rr = hi - t
            H, T, Q, Z = st
            c, s = _rotation(H[rr - 1, k], H[rr, k])
            H, T, Q, Z = _left(st, rr - 1, rr, c, s)
            H = H.at[rr, k].set(0.0)
            c, s = _rotation(T[rr, rr], T[rr, rr - 1])
            H, T, Q, Z = _right((H, T, Q, Z), rr, rr - 1, c, s)
            return H, T.at[rr, rr - 1].set(0.0), Q, Z

        return lax.fori_loop(0, hi - k - 1, eliminate, st)

    return lax.fori_loop(lo, max(lo, hi - 1), column, st)


@jax.jit
def _real_scale(X, s):
    return lax.complex(binary.multiply(j.real(X), s), binary.multiply(j.imag(X), s))


@jax.jit
def _complex_multiply(X, s):
    # Identity phase is common for isolated real-valued complex entries;
    # avoid flushing unchanged subnormal components in that branch.
    def general(_):
        ar = binary.multiply(j.real(X), j.real(s))
        ai = binary.multiply(j.imag(X), j.imag(s))
        br = binary.multiply(j.real(X), j.imag(s))
        bi = binary.multiply(j.imag(X), j.real(s))
        return lax.complex(binary.add(ar, -ai), binary.add(br, bi))

    return lax.cond(j.imag(s) == 0, lambda _: _real_scale(X, j.real(s)), general, None)


def _scale_input(X):
    component = j.maximum(
        lax.bitcast_convert_type(j.real(X), j.uint64) & j.uint64(0x7FFFFFFFFFFFFFFF),
        lax.bitcast_convert_type(j.imag(X), j.uint64) & j.uint64(0x7FFFFFFFFFFFFFFF),
    )
    largest = float(lax.bitcast_convert_type(j.max(component), j.float64))
    if largest == 0:
        return X, 1.0
    scaled = lax.complex(binary.divide(j.real(X), largest), binary.divide(j.imag(X), largest))
    factor = float(j.max(_magnitude(scaled)))
    small = float(j.sqrt(j.finfo(j.float64).tiny) / j.finfo(j.float64).eps)
    large = 1 / small
    if largest > large / factor:
        scale = (largest / large) * factor
    elif largest < small / factor:
        scale = (largest / small) * factor
    else:
        scale = 1.0
    return lax.complex(binary.divide(j.real(X), scale), binary.divide(j.imag(X), scale)), scale


@jax.jit
def _stable_norm(X):
    magnitude = _magnitude(X)
    scale = j.max(magnitude)
    return scale * j.sqrt(j.sum((magnitude / j.where(scale > 0, scale, 1.0)) ** 2))


def _abs1(z):
    return j.abs(j.real(z)) + j.abs(j.imag(z))


@jax.jit
def _single_start(H, T, lo, hi, shift, ascale, bscale, atol):
    def check(t, start):
        k = hi - 1 - t
        temp = _abs1(ascale * H[k, k] - shift * (bscale * T[k, k]))
        temp2 = ascale * _abs1(H[k + 1, k])
        scale = j.maximum(temp, temp2)
        scale = j.where((scale < 1) & (scale > 0), scale, 1.0)
        temp, temp2 = temp / scale, temp2 / scale
        small = _abs1(H[k, k - 1]) * temp2 <= temp * atol
        return j.where((start == lo) & small, k, start)

    return lax.fori_loop(0, hi - lo - 1, check, lo)
