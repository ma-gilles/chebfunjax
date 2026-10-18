"""JAX dense generalized eigenvalue boundary for Chebfun 7574c77.

Native MATLAB @linop/eigs.m:getEigenvalues invokes dense eig(A,B).
Real and complex generalized Schur paths retain homogeneous finite/infinite
and indeterminate pairs; source callers own filtering and adaptivity.
This module does not implement a singular Kronecker decomposition.
Algorithm provenance and Netlib BSD notices accompany the implementation
modules. No host eigensolver fallback is used.
"""

import jax.numpy as j
from jax import lax

from . import _qz_ieee as binary
from . import _qz_real as real


def generalized_eig(A, B):
    """Generalized eig.

    Provenance
    ----------
    MATLAB source: @linop/eigs.m, dense eig(A, B) boundary at 7574c77.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    Supporting JAX implementation of the LAPACK algorithms described in
    this module and operators._generalized_qz; no host numerical provider.
    """
    A, B = j.asarray(A), j.asarray(B)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape != B.shape:
        raise ValueError("expected equal square matrices")
    if not bool(j.all(j.isfinite(A)) & j.all(j.isfinite(B))):
        raise ValueError("array must not contain infs or NaNs")
    if len(A) == 0:
        return j.empty((0,), dtype=j.complex128), j.empty((0, 0), dtype=j.float64)
    from ._eig_hermitian import generalized_hermitian

    definite = generalized_hermitian(A, B)
    if definite is not None:
        return definite
    if j.issubdtype(j.result_type(A, B), j.complexfloating):
        from ._qz_complex import generalized_eig as complex_eig

        return complex_eig(A, B)
    out = real.schur(A, B)
    a, b = out["alpha"], out["beta"]
    ar, ai = j.real(a), j.imag(a)
    rr = binary.divide(ar, b)
    ii = binary.divide(ai, b)
    zero = ~binary.nonzero(b)
    indeterminate = zero & ~binary.nonzero(ar) & ~binary.nonzero(ai)
    rr = j.where(zero, j.where(indeterminate, j.nan, j.where(binary.nonzero(ar), rr, 0.0)), rr)
    # Native MATLAB preserves a zero imaginary component for 0:0,
    # even when other homogeneous alpha values are genuinely complex.
    ii = j.where(zero, j.where(binary.nonzero(ai), ii, 0.0), ii)
    values = lax.complex(rr, ii)
    vectors = real.right_vectors_compiled(out)
    if not bool(j.any(binary.nonzero(ai))):
        vectors = j.real(vectors)
    return values, vectors
