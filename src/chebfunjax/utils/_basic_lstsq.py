"""Basic least-squares solutions for dense rectangular double matrices.

Provenance
----------
MATLAB source : complex/ConformalL.m, the rectangular ``A\\U`` operation
Chebfun commit: 7574c77 (example invokes MATLAB's built-in backslash)
The backend is not Chebfun source. The pivoted-QR model is documented in
Moler, Numerical Computing with MATLAB, section 5.7:
https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/moler/leastsquares.pdf
The dense rectangular contract is documented at:
https://www.mathworks.com/help/matlab/ref/double.mldivide.html
Rank decisions are qualified against retained R2025b Update 2 CPU controls.
This is not a general square or sparse backslash implementation.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
from jax.scipy.linalg import qr, solve_triangular


def _basic_qr_rank(r: jnp.ndarray, rows: int, columns: int):
    """Return the leading rank and observed dense-backslash threshold.

    The threshold uses eps * abs(R11), not MATLAB eps(abs(R11)), and
    therefore is zero for a zero matrix. Equality excludes a pivot.
    """
    diagonal = jnp.abs(jnp.diag(r))
    threshold = max(rows, columns) * jnp.finfo(diagonal.dtype).eps * diagonal[0]
    count = diagonal.size
    rank = int(jnp.min(jnp.where(diagonal > threshold, count, jnp.arange(count))))
    return rank, threshold


def _basic_lstsq(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Solve a finite, nonempty dense rectangular system by basic pivoted QR.

    Numerical arrays use real or complex binary64. A vector RHS returns a
    vector; a matrix RHS returns a matrix. Variables outside the leading
    numerical-rank pivot block are exactly zero. The solution need not have
    minimum norm. Pivot choices can vary between numerical backends.

    This eager adapter extracts a scalar rank before solving the literal
    leading triangular block; it is not JIT-safe. Square/sparse solver
    dispatch and differentiation through rank or pivot changes are outside
    its contract.

    Provenance
    ----------
    MATLAB source : complex/ConformalL.m (``A\\U``)
    Chebfun commit: 7574c77
    Dense rectangular basic-QR semantics: MathWorks mldivide documentation
    and Moler section 5.7, linked in the module documentation. The rank rule
    is qualified with R2025b controls, not inferred from a fitted rank.
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    if a.ndim != 2:
        raise ValueError("basic least squares requires a matrix")
    rows, columns = a.shape
    if rows == columns or min(rows, columns) == 0:
        raise ValueError("basic least squares requires a nonempty rectangular matrix")
    if b.ndim not in (1, 2) or b.shape[0] != rows:
        raise ValueError("RHS must be a vector or matrix with the same row count")
    a = a.astype(jnp.complex128 if jnp.iscomplexobj(a) else jnp.float64)
    b = b.astype(jnp.result_type(a.dtype, b.dtype, jnp.float64))
    vector_rhs = b.ndim == 1
    if vector_rhs:
        b = b[:, None]
    q, r, pivots = qr(a, mode="economic", pivoting=True)
    rank, threshold = _basic_qr_rank(r, rows, columns)
    if rank < min(rows, columns):
        warnings.warn(
            f"Rank deficient, rank = {rank}, tol = {float(threshold):.6e}.",
            RuntimeWarning,
            stacklevel=2,
        )
    solution = jnp.zeros((columns, b.shape[1]), dtype=jnp.result_type(a, b))
    if rank:
        projected = q.conj().T @ b
        leading = solve_triangular(r[:rank, :rank], projected[:rank], lower=False)
        solution = solution.at[pivots[:rank], :].set(leading)
    return solution[:, 0] if vector_rhs else solution
