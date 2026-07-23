# uses-numpy: 2D PDE solver assembles Kronecker systems with numpy (not JIT-safe)
"""2D differential operator for PDEs on rectangles.

:class:`Chebop2` solves linear PDEs of the form::

    L[u](x, y) = f(x, y),   (x, y) ∈ [xa, xb] × [ya, yb]

with Dirichlet boundary conditions on some or all four edges.

The method assembles the PDE as a sum of Kronecker products of 1D collocation
differentiation matrices (Townsend & Olver 2015), imposes boundary conditions
by row replacement in the full Kronecker system, and solves via
``numpy.linalg.solve``.  For rank-2 operators (Laplacian, Helmholtz) the
Bartels-Stewart algorithm is available as an optional fast-path.

Typical usage::

    import jax.numpy as jnp
    from chebfunjax.operators.chebop2 import Chebop2

    # Poisson equation: u_xx + u_yy = f, zero Dirichlet BCs
    N = Chebop2(lambda u: u.diff(2, 0) + u.diff(0, 2))
    N.bc = 0.0
    u = N.solve(lambda x, y: -2.0 * (1 - x**2) - 2.0 * (1 - y**2))
    # exact solution: u = (1 - x^2) * (1 - y^2)

Translated from MATLAB Chebfun class ``@chebop2`` (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.

References
----------
A. Townsend and S. Olver, "The automatic solution of partial differential
equations using a global spectral method", J. Comput. Phys., 299 (2015),
pp. 106-123.
"""

from __future__ import annotations

import warnings
from typing import Callable

import jax.numpy as jnp
import numpy as np

from chebfunjax.utils.quadrature import chebpts

# ---------------------------------------------------------------------------
# Machine epsilon
# ---------------------------------------------------------------------------
_EPS = float(jnp.finfo(jnp.float64).eps)


# ===========================================================================
# 1D spectral matrices and helpers
# ===========================================================================


def _diffmat_cheb2_np(n: int, order: int, domain: tuple[float, float]) -> np.ndarray:
    """Chebyshev-collocation differentiation matrix of given order.

    Returns the n×n matrix D such that ``D @ u_vals`` gives the values of
    the ``order``-th derivative at the same n Chebyshev-2 collocation points.

    Provenance
    ----------
    MATLAB source : @chebcolloc2/diffmat.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.utils.diffmat import diffmat
    return np.array(diffmat(n, order, domain=domain), dtype=np.float64)


def _cheb2_pts_np(n: int, domain: tuple[float, float]) -> np.ndarray:
    """Physical Chebyshev-2 collocation points on *domain* (ascending order)."""
    a, b = domain
    t = np.array(chebpts(n, kind=2), dtype=np.float64)
    return 0.5 * (b - a) * t + 0.5 * (a + b)


# ===========================================================================
# Bartels-Stewart solver  (AXB^T + CXD^T = E)
# ===========================================================================


def bartels_stewart(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    E: np.ndarray,
) -> np.ndarray:
    """Solve the generalized Sylvester equation ``A X B^T + C X D^T = E``.

    Uses the Bartels-Stewart algorithm via the QZ decomposition of (A, C) and
    (D, B).

    Parameters
    ----------
    A, C : np.ndarray, shape (m, m)
        Coefficient matrices in the y-direction.
    B, D : np.ndarray, shape (n, n)
        Coefficient matrices in the x-direction.
    E : np.ndarray, shape (m, n)
        Right-hand side matrix.

    Returns
    -------
    X : np.ndarray, shape (m, n)
        Solution to A X B^T + C X D^T = E.

    Notes
    -----
    This solver is NOT used by default in :class:`Chebop2` (which uses the
    full Kronecker approach for correctness).  It is provided as a public
    utility for users who need the Bartels-Stewart solver directly.

    Provenance
    ----------
    MATLAB source : @chebop2/bartelsStewart.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: J. D. Gardiner, A. J. Laub, J. J. Amato, & C. B. Moler,
        "Solution of the Sylvester matrix equation AXB^T + CXD^T = E",
        ACM TOMS, 18(2), 223-231 (1992).

    See Also
    --------
    Chebop2.solve
    """
    import scipy.linalg

    if np.linalg.norm(E) < 10 * _EPS:
        return np.zeros_like(E)

    m = A.shape[0]
    n = B.shape[0]

    # QZ decomposition of (A, C): A = Q1 P Z1^T, C = Q1 S Z1^T (real orthogonal
    # Q1, Z1).  P and S are quasi-triangular; do NOT force them upper triangular
    # -- the column recursion below solves full m x m systems with them, so
    # zeroing their 2x2-block subdiagonals corrupts the transformed pencil.
    P, S, Q1, Z1 = scipy.linalg.qz(A, C, output="real")

    # QZ decomposition of (D, B): D = Q2 T Z2^T, B = Q2 R Z2^T.  T is (quasi-)
    # upper triangular and R is upper triangular, which is what enables the
    # column-by-column back-substitution over the right pencil.
    T, R, Q2, Z2 = scipy.linalg.qz(D, B, output="real")

    # With Y = Z1^T X Z2 the equation reduces to P Y R^T + S Y T^T = F where the
    # transformed RHS is F = Q1^T E Q2 (the solution is recovered as
    # X = Z1 Y Z2^T below).
    F = Q1.T @ E @ Q2

    # Backward substitution: build solution Y column by column
    Y = np.zeros((m, n), dtype=np.float64)
    PY = np.zeros((m, n), dtype=np.float64)
    SY = np.zeros((m, n), dtype=np.float64)

    k = n - 1
    while k >= 1:
        t_off = T[k, k - 1]
        t_diag = max(abs(T[k, k]), abs(T[k - 1, k - 1]), 1.0)
        if abs(t_off) < _EPS * t_diag:
            rhs = F[:, k].copy()
            for jj in range(k + 1, n):
                rhs -= R[k, jj] * PY[:, jj] + T[k, jj] * SY[:, jj]
            Mkk = R[k, k] * P + T[k, k] * S
            Y[:, k] = np.linalg.solve(Mkk, rhs)
            PY[:, k] = P @ Y[:, k]
            SY[:, k] = S @ Y[:, k]
            k -= 1
        else:
            rhs1 = F[:, k - 1].copy()
            rhs2 = F[:, k].copy()
            for jj in range(k + 1, n):
                Pyj = PY[:, jj]
                Syj = SY[:, jj]
                rhs1 -= R[k - 1, jj] * Pyj + T[k - 1, jj] * Syj
                rhs2 -= R[k, jj] * Pyj + T[k, jj] * Syj

            M11 = R[k - 1, k - 1] * P + T[k - 1, k - 1] * S
            M12 = R[k - 1, k] * P + T[k - 1, k] * S
            M21 = R[k, k - 1] * P + T[k, k - 1] * S
            M22 = R[k, k] * P + T[k, k] * S

            SM = np.zeros((2 * m, 2 * m), dtype=np.float64)
            SM[:m, :m] = M11
            SM[:m, m:] = M12
            SM[m:, :m] = M21
            SM[m:, m:] = M22

            sol = np.linalg.solve(SM, np.concatenate([rhs1, rhs2]))
            Y[:, k - 1] = sol[:m]
            Y[:, k] = sol[m:]
            PY[:, k] = P @ Y[:, k]
            PY[:, k - 1] = P @ Y[:, k - 1]
            SY[:, k] = S @ Y[:, k]
            SY[:, k - 1] = S @ Y[:, k - 1]
            k -= 2

    if k == 0:
        rhs = F[:, 0].copy()
        for jj in range(1, n):
            rhs -= R[0, jj] * PY[:, jj] + T[0, jj] * SY[:, jj]
        M00 = R[0, 0] * P + T[0, 0] * S
        Y[:, 0] = np.linalg.solve(M00, rhs)

    X = Z1 @ Y @ Z2.T
    return X


# ===========================================================================
# Coefficient-space (ultraspherical) discretization and solve
#
# This is MATLAB Chebfun's actual @chebop2 method: represent the PDO as a
# separable-rank expansion, discretize each 1D term with banded ultraspherical
# operators (conversion S, differentiation D, multiplication M), eliminate the
# boundary DOFs (constructBC + zeroDOF + canonicalBC), solve the resulting
# generalized Sylvester equation, then re-impose the boundary rows.  It returns
# the solution's Chebyshev coefficient matrix directly, reaching MATLAB's
# ~eps accuracy (the value-space Kronecker solve floors at ~1e-12).
#
# Provenance
# ----------
# MATLAB source : @chebop2/discretize.m, @chebop2/denseSolve.m,
#     @chebop2/constructBC.m, @ultraS/*
# Chebfun commit: 7574c77
# Original authors: Copyright 2017 by The University of Oxford
#     and The Chebfun Developers.
# ===========================================================================


def _ultra_diffmat(n: int, k: int) -> np.ndarray:
    """Unscaled ultraspherical differentiation matrix C^{(0)} -> C^{(k)}."""
    from chebfunjax.discretization.ultras import diffmat as _dm
    return np.asarray(_dm(n, k), dtype=np.float64)


def _ultra_convertmat(n: int, k1: int, k2: int) -> np.ndarray:
    """Ultraspherical conversion matrix C^{(k1)} -> C^{(k2+1)} (identity if k2<k1)."""
    from chebfunjax.discretization.ultras import convertmat as _cm
    return np.asarray(_cm(n, k1, k2), dtype=np.float64)


def _ultra_multmat(n: int, a: np.ndarray, lam: int) -> np.ndarray:
    """Ultraspherical multiplication matrix by ``a`` (Cheb-T coeffs) in C^{(lam)}."""
    from chebfunjax.discretization.ultras import multmat as _mm
    return np.asarray(_mm(n, jnp.asarray(a, dtype=jnp.float64), lam), dtype=np.float64)


def _cheb_coeffs_1d(fn, n: int, dom: tuple[float, float]) -> np.ndarray:
    """Chebyshev-T coefficients (length n) of a 1D function on ``dom``.

    Provenance
    ----------
    MATLAB source : chebfun constructor (bcArg.coeffs)
    Chebfun commit: 7574c77
    """
    from chebfunjax.utils.transforms import vals2coeffs
    a, b = dom
    t = np.array(chebpts(n, kind=2), dtype=np.float64)
    pts = 0.5 * (b - a) * t + 0.5 * (a + b)
    vals = np.asarray(fn(jnp.asarray(pts, dtype=jnp.float64)), dtype=np.complex128)
    if np.max(np.abs(vals.imag)) < 1e-13 * max(np.max(np.abs(vals)), 1.0):
        vals = vals.real
    c = np.array(vals2coeffs(jnp.asarray(vals)), dtype=vals.dtype)
    return c


def _cheb_coeffs_2d(f, m: int, n: int,
                    dom: tuple[float, float, float, float]) -> np.ndarray:
    """2D Chebyshev-T coefficient matrix (rows=y, cols=x) of ``f`` on ``dom``.

    ``C[i, j]`` is the coefficient of ``T_i(y) T_j(x)``.

    Provenance
    ----------
    MATLAB source : chebfun2/chebcoeffs2
    Chebfun commit: 7574c77
    """
    from chebfunjax.utils.transforms import vals2coeffs
    xa, xb, ya, yb = dom
    tx = np.array(chebpts(n, kind=2), dtype=np.float64)
    ty = np.array(chebpts(m, kind=2), dtype=np.float64)
    xpts = 0.5 * (xb - xa) * tx + 0.5 * (xa + xb)
    ypts = 0.5 * (yb - ya) * ty + 0.5 * (ya + yb)
    XX, YY = np.meshgrid(xpts, ypts)  # (m, n)
    V = np.asarray(
        f(jnp.asarray(XX, dtype=jnp.float64), jnp.asarray(YY, dtype=jnp.float64)),
        dtype=np.complex128,
    )
    if np.max(np.abs(V.imag)) < 1e-13 * max(np.max(np.abs(V)), 1.0):
        V = V.real
    dt = V.dtype
    C = np.empty((m, n), dtype=dt)
    for j in range(n):
        C[:, j] = np.array(vals2coeffs(jnp.asarray(V[:, j])), dtype=dt)
    for i in range(m):
        C[i, :] = np.array(vals2coeffs(jnp.asarray(C[i, :])), dtype=dt)
    return C


def _unconstrained_matrix_equation(ode_col, n: int, order: int,
                                   dom: tuple[float, float]) -> np.ndarray:
    """Build one 1D ultraspherical ODE operator, size n x n, in C^{(order)}.

    ``ode_col`` is the sequence of derivative coefficients ``[c_0, c_1, ...]``:
    each ``c_k`` multiplies ``D^k``.  A ``c_k`` may be a scalar (constant
    coefficient) or a length-p Chebyshev-T coefficient vector (variable
    coefficient), handled via a multiplication matrix.

    Provenance
    ----------
    MATLAB source : @chebop2/discretize.m (unconstrainedMatrixEquation)
    Chebfun commit: 7574c77
    """
    a, b = dom
    B = None
    for kk in range(len(ode_col)):
        c = ode_col[kk]
        if c is None:
            continue
        S = _ultra_convertmat(n, kk, order - 1)       # C^{(kk)} -> C^{(order)}
        D = ((2.0 / (b - a)) ** kk) * _ultra_diffmat(n, kk)
        if np.ndim(c) == 0:
            if c == 0:
                continue
            A = c * (S @ D)
        else:
            cvec = np.asarray(c)
            if np.max(np.abs(cvec)) == 0:
                continue
            M = _ultra_multmat(n, cvec.real if np.iscomplexobj(cvec)
                               and np.max(np.abs(cvec.imag)) < 1e-14 else cvec, kk)
            A = S @ M @ D
        B = A if B is None else B + A
    if B is None:
        B = np.zeros((n, n), dtype=np.float64)
    return B


def _cheb_values(k: int, n: int, x: float) -> np.ndarray:
    """Values of ``T_j^{(k)}(x)`` for j=0..n-1 at ``x`` in {-1, 1}.

    Provenance
    ----------
    MATLAB source : @chebop2/constructBC.m (chebValues)
    Chebfun commit: 7574c77
    """
    if k == 0:
        return x ** np.arange(n, dtype=np.float64)
    ll, kk = np.meshgrid(np.arange(n, dtype=np.float64),
                         np.arange(k, dtype=np.float64))
    factor = np.prod((ll ** 2 - kk ** 2) / (2.0 * kk + 1.0), axis=0)
    return (x ** np.arange(1, n + 1, dtype=np.float64)) * factor


def _canonical_bc(B: np.ndarray, G: np.ndarray):
    """Reduce boundary rows to canonical (unit upper-triangular) form.

    Returns ``(B, G, P)`` where the leading ``nbc x nbc`` block of ``B @ P``
    is the identity after LU + scaling, so that ``B X = G`` can be used to
    eliminate degrees of freedom.  ``P`` is a permutation matrix.

    Provenance
    ----------
    MATLAB source : @chebop2/discretize.m (canonicalBC, nonsingularPermute)
    Chebfun commit: 7574c77
    """
    import scipy.linalg
    if B.size == 0:
        return B, G, None
    nbc, dim = B.shape
    # nonsingularPermute: find leading nbc columns forming a nonsingular block.
    k = 0
    while np.linalg.matrix_rank(B[:, k:k + nbc]) < nbc:
        k += 1
        if nbc + k > dim:
            raise RuntimeError(
                "Chebop2: boundary conditions are linearly dependent.")
    perm = list(range(k, nbc + k)) + list(range(0, k)) + list(range(nbc + k, dim))
    P = np.eye(dim, dtype=np.float64)[:, perm]
    B = B @ P
    PL, U = scipy.linalg.lu(B, permute_l=True)  # PL @ U = B
    B = U
    G = np.linalg.solve(PL, G)
    d = np.diag(B).copy()
    Dinv = np.diag(1.0 / d)
    B = Dinv @ B
    G = Dinv @ G
    return B, G, P


def _zero_dof(C1: np.ndarray, C2: np.ndarray, E: np.ndarray,
              B: np.ndarray, G: np.ndarray):
    """Eliminate boundary degrees of freedom from a matrix-equation term.

    Provenance
    ----------
    MATLAB source : @chebop2/discretize.m (zeroDOF)
    Chebfun commit: 7574c77
    """
    C1 = C1.copy()
    E = E.copy()
    for ii in range(B.shape[0]):
        for kk in range(C1.shape[0]):
            if abs(C1[kk, ii]) > 10.0 * _EPS:
                c = C1[kk, ii]
                C1[kk, :] = C1[kk, :] - c * B[ii, :]
                E[kk, :] = E[kk, :] - c * (G[ii, :] @ C2.T)
    return C1, E


def _impose_boundary_conditions(X, bb, gg, Px, Py, m, n):
    """Re-impose the eliminated boundary rows onto the interior solution.

    Provenance
    ----------
    MATLAB source : @chebop2/denseSolve.m (imposeBoundaryConditions)
    Chebfun commit: 7574c77
    """
    # bb entries are (bcn, ncond) bcrow matrices; gg entries (een, ncond).
    L, R, U, D = bb
    lg, rg, ug, dg = gg

    def hcat(parts):
        parts = [p for p in parts if p is not None]
        return np.hstack(parts) if parts else None

    Uc = U.shape[1] if U is not None else 0
    Dc = D.shape[1] if D is not None else 0
    cs = Uc + Dc
    Lc = L.shape[1] if L is not None else 0
    Rc = R.shape[1] if R is not None else 0
    rs = Lc + Rc

    By = hcat([U, D])    # (m, cs)
    Gy = hcat([ug, dg])  # (n, cs)
    if By is not None:
        By = Py.T @ By
        rows = X.shape[1]
        X12 = np.linalg.solve(
            By[:cs, :].T,
            Gy[rs:rs + rows, :].T - By[cs:cs + X.shape[0], :].T @ X,
        )
        X = np.vstack([X12, X])

    Bx = hcat([L, R])
    Gx = hcat([lg, rg])
    if Bx is not None:
        Bx = Px.T @ Bx
        rows = X.shape[0]
        X2 = np.linalg.solve(
            Bx[:rs, :].T,
            Gx[:rows, :].T - Bx[rs:rs + X.shape[1], :].T @ X.T,
        ).T
        X = np.hstack([X2, X])

    if X.shape[0] < m:
        X = np.vstack([X, np.zeros((m - X.shape[0], X.shape[1]), dtype=X.dtype)])
    if X.shape[1] < n:
        X = np.hstack([X, np.zeros((X.shape[0], n - X.shape[1]), dtype=X.dtype)])
    if Px is not None:
        X = X @ Px.T
    if Py is not None:
        X = Py @ X
    return X


def _reduced_solve(CC, rhs, rk):
    """Solve the reduced matrix equation ``sum_j CC[j][0] X CC[j][1].' = rhs``.

    Rank 1 is a pair of triangular-free solves; rank 2 (real) uses the
    Bartels-Stewart generalized Sylvester solver; higher rank (or complex
    rank 2) falls back to the dense Kronecker solve.

    Provenance
    ----------
    MATLAB source : @chebop2/denseSolve.m
    Chebfun commit: 7574c77
    """
    complex_sys = np.iscomplexobj(rhs) or any(
        np.iscomplexobj(c) for pair in CC for c in pair)

    if rk == 1:
        A, B = CC[0][0], CC[0][1]
        Y = np.linalg.solve(A, rhs)
        return np.linalg.solve(B, Y.T).T

    if rk == 2 and not complex_sys:
        return bartels_stewart(CC[0][0], CC[0][1], CC[1][0], CC[1][1], rhs)

    # Rank >= 2: dense Kronecker solve (also the complex rank-2 path).
    p = CC[0][0].shape[0]
    q = CC[0][1].shape[0]
    sz = p * q
    dt = np.complex128 if complex_sys else np.float64
    K = np.zeros((sz, sz), dtype=dt)
    for jj in range(rk):
        K = K + np.kron(CC[jj][1], CC[jj][0])
    b = rhs.ravel("F")
    x = np.linalg.solve(K, b)
    return x.reshape(p, q, order="F")


def _is_resolved_coeffs(C: np.ndarray, tol: float) -> bool:
    """Check that a Chebyshev-coefficient solution matrix has a decayed tail.

    Provenance
    ----------
    MATLAB source : @chebop2/solvepde.m (resolution check)
    Chebfun commit: 7574c77
    """
    if C.size == 0:
        return True
    m, n = C.shape
    scale = max(float(np.max(np.abs(C))), 1e-300)
    thresh = tol * scale
    tail_rows = np.max(np.abs(C[max(0, m - 3):, :])) if m >= 1 else 0.0
    tail_cols = np.max(np.abs(C[:, max(0, n - 3):])) if n >= 1 else 0.0
    return bool(tail_rows < thresh and tail_cols < thresh)


# ===========================================================================
# Boundary-condition proxies for general (Neumann/Robin) constraints
# ===========================================================================


class _BCZeroProxy:
    """Zero-valued 1D solution proxy used to extract a BC's forcing term.

    Behaves as an additive/multiplicative zero so that evaluating a BC lambda
    ``@(t, u) c*u + ... + f(t)`` with ``u`` set to this proxy returns ``f(t)``.

    Provenance
    ----------
    MATLAB source : @chebop2/constructBC.m (bcArg(x, 0*x))
    Chebfun commit: 7574c77
    """

    def diff(self, *args, **kwargs):
        return self

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other

    def __sub__(self, other):
        return -other

    def __rsub__(self, other):
        return other

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __neg__(self):
        return self

    def __truediv__(self, other):
        return self


class _BCProbeProxy:
    """1D solution proxy that records derivative-order coefficients of a BC.

    ``diff(k)`` shifts the recorded order; scalar multiplication scales the
    coefficients; additive terms that are pure functions of the coordinate
    (the forcing) are discarded, isolating the linear operator on ``u``.

    Provenance
    ----------
    MATLAB source : @chebop2/recoverCoeffs.m
    Chebfun commit: 7574c77
    """

    def __init__(self, terms: dict | None = None) -> None:
        self._terms: dict = terms if terms is not None else {0: 1.0}

    def diff(self, k: int = 1):
        return _BCProbeProxy({order + k: c for order, c in self._terms.items()})

    def __add__(self, other):
        if isinstance(other, _BCProbeProxy):
            new = dict(self._terms)
            for order, c in other._terms.items():
                new[order] = new.get(order, 0.0) + c
            return _BCProbeProxy(new)
        # Pure function/scalar forcing -- discard (handled separately).
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, _BCProbeProxy):
            return self.__add__(other.__neg__())
        return self

    def __rsub__(self, other):
        return self.__neg__()

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return _BCProbeProxy({o: c * other for o, c in self._terms.items()})
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return _BCProbeProxy({o: -c for o, c in self._terms.items()})

    def __truediv__(self, other):
        if isinstance(other, (int, float, complex)):
            return self.__mul__(1.0 / other)
        return NotImplemented


# ===========================================================================
# Main Chebop2 class
# ===========================================================================


class Chebop2:
    """Linear 2D differential operator for PDEs on rectangles.

    :class:`Chebop2` solves linear constant-coefficient PDEs of the form::

        Σ_{j,k} a_{jk} * ∂^j/∂y^j ∂^k/∂x^k u  =  f(x, y)

    with user-specified Dirichlet boundary conditions on the four edges.

    The solve algorithm (Townsend & Olver 2015):

    1. Represent the operator as a coefficient matrix ``A`` where
       ``A[j, k]`` is the coefficient of ``∂^j/∂y^j ∂^k/∂x^k``.
    2. Compute the SVD of ``A`` to get a rank-r decomposition.
    3. Discretize each 1D piece using Chebyshev-collocation diffmats.
    4. Form the full n²×n² Kronecker matrix and impose BCs by row replacement.
    5. Solve with ``numpy.linalg.solve``.

    Parameters
    ----------
    op : callable or None
        The differential operator.  Must be a lambda accepting a
        :class:`_Chebop2Proxy` and returning a :class:`_Chebop2Proxy`.
        Example: ``lambda u: u.diff(2, 0) + u.diff(0, 2)``  (Laplacian).
    domain : tuple of 4 floats, default ``(-1, 1, -1, 1)``
        ``(xa, xb, ya, yb)`` — physical rectangle.

    Attributes
    ----------
    op, domain, lbc, rbc, ubc, dbc, bc

    BC specifications
    -----------------
    Each BC attribute accepts:

    * ``scalar c``          — constant Dirichlet ``u|_edge = c``
    * ``callable f(t)``     — non-constant Dirichlet along the edge;
      ``t`` is a JAX array of physical coordinates parallel to the edge.
    * ``None``              — no BC on that edge.

    The shorthand ``N.bc = c`` sets all four BCs simultaneously.

    Examples
    --------
    **Poisson with zero Dirichlet BCs on [-1,1]²:**

    >>> from chebfunjax.operators.chebop2 import Chebop2
    >>> import jax.numpy as jnp
    >>> N = Chebop2(lambda u: u.diff(2, 0) + u.diff(0, 2))
    >>> N.bc = 0.0
    >>> f = lambda x, y: -2.0 * (1.0 - x**2) - 2.0 * (1.0 - y**2)
    >>> u = N.solve(f, n=20)
    >>> # exact: u(x, y) = (1 - x^2)(1 - y^2)

    **Helmholtz equation:**

    >>> k = 2.0
    >>> N = Chebop2(lambda u: u.diff(2, 0) + u.diff(0, 2) + k**2 * u)
    >>> N.bc = 0.0

    Notes
    -----
    * Only constant-coefficient PDEs are currently supported.
    * The operator lambda must use ``u.diff(yorder, xorder)`` calls and
      scalar arithmetic.
    * Only Dirichlet BCs are supported.
    * The adaptive solver doubles the grid until the last 8 rows/columns of
      the coefficient matrix decay below the tolerance.

    Provenance
    ----------
    MATLAB source : @chebop2/chebop2.m, @chebop2/solvepde.m,
        @chebop2/denseSolve.m, @chebop2/discretize.m,
        @chebop2/bartelsStewart.m, @chebop2/constructBC.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    Algorithm: A. Townsend and S. Olver, "The automatic solution of partial
        differential equations using a global spectral method",
        J. Comput. Phys., 299 (2015), pp. 106-123.

    See Also
    --------
    Chebop, Linop, bartels_stewart
    """

    def __init__(
        self,
        op: Callable | None = None,
        domain: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0),
    ) -> None:
        if len(domain) != 4:
            raise ValueError(
                f"Chebop2: domain must be a 4-tuple (xa, xb, ya, yb), "
                f"got length {len(domain)}."
            )
        xa, xb, ya, yb = domain
        if xb <= xa or yb <= ya:
            raise ValueError(
                f"Chebop2: domain must have xa < xb and ya < yb, "
                f"got ({xa}, {xb}, {ya}, {yb})."
            )
        self.domain: tuple[float, float, float, float] = (
            float(xa), float(xb), float(ya), float(yb)
        )
        self.op: Callable | None = op
        self._lbc = None
        self._rbc = None
        self._ubc = None
        self._dbc = None
        self._coeffs: np.ndarray | None = None
        self._xorder: int = 0
        self._yorder: int = 0
        if op is not None:
            self._extract_coeffs()

    # ------------------------------------------------------------------
    # BC properties
    # ------------------------------------------------------------------

    @property
    def lbc(self):
        """Left boundary condition (x = xa).  Scalar, callable, or None."""
        return self._lbc

    @lbc.setter
    def lbc(self, val):
        self._lbc = val

    @property
    def rbc(self):
        """Right boundary condition (x = xb).  Scalar, callable, or None."""
        return self._rbc

    @rbc.setter
    def rbc(self, val):
        self._rbc = val

    @property
    def ubc(self):
        """Upper boundary condition (y = yb).  Scalar, callable, or None."""
        return self._ubc

    @ubc.setter
    def ubc(self, val):
        self._ubc = val

    @property
    def dbc(self):
        """Lower boundary condition (y = ya).  Scalar, callable, or None."""
        return self._dbc

    @dbc.setter
    def dbc(self, val):
        self._dbc = val

    @property
    def coeffs(self):
        """Constant-coefficient PDO matrix (MATLAB ``@chebop2`` layout).

        Returns an ``(xorder+1, yorder+1)`` array ``C`` where ``C[k, j]`` is
        the coefficient of ``∂^k/∂x^k ∂^j/∂y^j``.  This is the transpose of
        the internal ``_coeffs`` matrix (whose rows index the y-order), and
        matches the ordering of MATLAB ``N.coeffs`` for constant-coefficient
        operators.

        Provenance
        ----------
        MATLAB source : @chebop2/chebop2.m (coeffs property)
        Chebfun commit: 7574c77
        """
        if self._coeffs is None:
            if self.op is None:
                return np.zeros((1, 1), dtype=np.float64)
            self._extract_coeffs()
        return np.array(self._coeffs.T, dtype=np.float64)

    @property
    def xorder(self) -> int:
        """Highest x-derivative order appearing in the operator."""
        if self._coeffs is None and self.op is not None:
            self._extract_coeffs()
        return self._xorder

    @property
    def yorder(self) -> int:
        """Highest y-derivative order appearing in the operator."""
        if self._coeffs is None and self.op is not None:
            self._extract_coeffs()
        return self._yorder

    @property
    def bc(self):
        """Read lbc (write sets all four BCs simultaneously)."""
        return self._lbc

    @bc.setter
    def bc(self, val):
        """Set all four boundary conditions to the same value."""
        self._lbc = val
        self._rbc = val
        self._ubc = val
        self._dbc = val

    # ------------------------------------------------------------------
    # Coefficient extraction from op lambda
    # ------------------------------------------------------------------

    def _extract_coeffs(self) -> None:
        """Extract the constant-coefficient matrix from ``self.op``.

        Sets ``self._coeffs`` (shape (yorder+1, xorder+1)),
        ``self._xorder``, and ``self._yorder``.

        Provenance
        ----------
        MATLAB source : @chebop2/chebop2.m (constructor)
        Chebfun commit: 7574c77
        """
        proxy = _Chebop2Proxy()
        result = self.op(proxy)
        if not isinstance(result, _Chebop2Proxy):
            raise TypeError(
                "Chebop2: operator must return a _Chebop2Proxy term.  "
                "Make sure the lambda uses only u.diff(dy, dx) and scalar "
                "arithmetic (e.g., lambda u: u.diff(2,0) + u.diff(0,2))."
            )
        A = result._coeffs_matrix()
        tol = 10.0 * _EPS
        A[np.abs(A) < tol] = 0.0
        self._coeffs = A
        nonzero_rows = np.where(np.any(A != 0, axis=1))[0]
        nonzero_cols = np.where(np.any(A != 0, axis=0))[0]
        self._yorder = int(nonzero_rows[-1]) if len(nonzero_rows) > 0 else 0
        self._xorder = int(nonzero_cols[-1]) if len(nonzero_cols) > 0 else 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        f=0.0,
        n: int | None = None,
        n_min: int = 9,
        n_max: int = 257,
        tol: float = 1e-10,
    ):
        """Solve the PDE ``L[u] = f`` with the attached boundary conditions.

        Parameters
        ----------
        f : scalar or callable, default 0.0
            Right-hand side.  If scalar, treated as a constant function.
            If callable, must accept two 2D JAX arrays (X, Y) from meshgrid
            (shape (m, n)) and return a 2D array of the same shape.
        n : int or None
            Fixed grid size (same in both x and y).  If ``None``, uses
            adaptive doubling until convergence or ``n_max``.
        n_min : int, default 9
            Minimum grid size for adaptive loop.
        n_max : int, default 257
            Maximum grid size for adaptive loop.
        tol : float, default 1e-10
            Coefficient tail tolerance for convergence check.

        Returns
        -------
        u : SeparableApprox
            Solution as a low-rank 2D function.  Evaluate via ``u(x, y)``.

        Raises
        ------
        RuntimeError
            If ``N.op`` is ``None``.
        RuntimeWarning
            If the adaptive loop reaches ``n_max`` without convergence.

        Provenance
        ----------
        MATLAB source : @chebop2/solvepde.m, @chebop2/denseSolve.m
        Chebfun commit: 7574c77
        """
        if self.op is None:
            raise RuntimeError(
                "Chebop2.solve: operator is not set. "
                "Assign N.op = lambda u: ... before solving."
            )
        if self._coeffs is None:
            self._extract_coeffs()

        # Prefer the coefficient-space (ultraspherical) solver -- MATLAB's
        # actual method -- which reaches ~eps accuracy.  Fall back to the
        # value-space Kronecker collocation solver if it cannot handle the
        # problem (raises), preserving previous behaviour.
        use_coeff = self._can_use_coeff_solve()

        if n is not None:
            if use_coeff:
                try:
                    C = self._coeff_solve(f, n, n)
                    return self._wrap_coeffs(C)
                except Exception:
                    pass
            X = self._dense_solve(f, n, n)
            return self._wrap_solution(X)

        # Adaptive loop: double the grid until resolved.  The coefficient-space
        # path resolves to ~eps (like MATLAB solvepde), so it uses a tighter
        # relative tail tolerance than the value-space convergence check.
        coeff_tol = min(tol, 1e-13)
        sz = n_min
        for _ in range(20):
            if use_coeff:
                try:
                    C = self._coeff_solve(f, sz, sz)
                    if _is_resolved_coeffs(C, coeff_tol):
                        return self._wrap_coeffs(C)
                    old_sz = sz
                    sz = _next_grid(sz)
                    if sz >= n_max:
                        warnings.warn(
                            f"Chebop2.solve: adaptive loop reached n_max={n_max} "
                            f"without convergence (tol={tol}). Returning best "
                            f"available solution.",
                            stacklevel=2,
                        )
                        return self._wrap_coeffs(self._coeff_solve(f, old_sz, old_sz))
                    continue
                except Exception:
                    use_coeff = False  # fall back for the rest of the loop
            X = self._dense_solve(f, sz, sz)
            if _is_resolved_vals(X, tol):
                return self._wrap_solution(X)
            old_sz = sz
            sz = _next_grid(sz)
            if sz >= n_max:
                warnings.warn(
                    f"Chebop2.solve: adaptive loop reached n_max={n_max} without "
                    f"convergence (tol={tol}). Returning best available solution.",
                    stacklevel=2,
                )
                return self._wrap_solution(self._dense_solve(f, old_sz, old_sz))
        return self._wrap_solution(self._dense_solve(f, sz, sz))

    def __truediv__(self, f):
        """``N \\ f`` — solve N[u] = f."""
        return self.solve(f)

    def __add__(self, other: "Chebop2") -> "Chebop2":
        """Sum of two constant-coefficient operators on the same domain.

        The coefficient matrices are added (with zero-padding to a common
        shape) and a new :class:`Chebop2` with the combined operator is
        returned.  Boundary conditions are not carried over.

        Provenance
        ----------
        MATLAB source : @chebop2/plus.m
        Chebfun commit: 7574c77
        """
        if not isinstance(other, Chebop2):
            return NotImplemented
        if self.domain != other.domain:
            raise ValueError(
                "Chebop2.__add__: operators must share the same domain, "
                f"got {self.domain} and {other.domain}."
            )
        if self._coeffs is None:
            self._extract_coeffs()
        if other._coeffs is None:
            other._extract_coeffs()
        A, B = self._coeffs, other._coeffs
        rows = max(A.shape[0], B.shape[0])
        cols = max(A.shape[1], B.shape[1])
        S = np.zeros((rows, cols), dtype=np.float64)
        S[: A.shape[0], : A.shape[1]] += A
        S[: B.shape[0], : B.shape[1]] += B

        new = Chebop2(domain=self.domain)
        new._coeffs = S
        new.op = _op_from_coeffs(S)
        nz_r = np.where(np.any(S != 0, axis=1))[0]
        nz_c = np.where(np.any(S != 0, axis=0))[0]
        new._yorder = int(nz_r[-1]) if len(nz_r) > 0 else 0
        new._xorder = int(nz_c[-1]) if len(nz_c) > 0 else 0
        return new

    def __sub__(self, other: "Chebop2") -> "Chebop2":
        """Difference of two constant-coefficient operators (``N1 - N2``)."""
        if not isinstance(other, Chebop2):
            return NotImplemented
        if other._coeffs is None:
            other._extract_coeffs()
        neg = Chebop2(domain=other.domain)
        neg._coeffs = -np.array(other._coeffs, dtype=np.float64)
        neg.op = _op_from_coeffs(neg._coeffs)
        neg._xorder = other._xorder
        neg._yorder = other._yorder
        return self.__add__(neg)

    # ------------------------------------------------------------------
    # Fixed-size solve (value space, full Kronecker)
    # ------------------------------------------------------------------

    def _dense_solve(self, f, m: int, n: int) -> np.ndarray:
        """Solve at fixed grid size m (y-direction) × n (x-direction).

        Returns the m×n matrix of function values ``U[i, j] = u(y_i, x_j)``
        at Chebyshev-2 collocation points.

        Algorithm
        ---------
        1. Build 1D collocation diffmats for y (size m) and x (size n).
        2. Form the full mn×mn Kronecker matrix ``K = Σ_r RIGHT_r ⊗ LEFT_r``.
        3. Sample BC values at the boundary collocation points.
        4. Replace boundary rows in K with identity rows and set RHS to BC values.
        5. Solve ``K @ vec(U) = vec(F)`` via ``numpy.linalg.solve``.

        The collocation points are the ascending Chebyshev-2 nodes, so:
        - ``y_pts[0] = ya``, ``y_pts[-1] = yb``
        - ``x_pts[0] = xa``, ``x_pts[-1] = xb``

        Provenance
        ----------
        MATLAB source : @chebop2/denseSolve.m, @chebop2/discretize.m
        Chebfun commit: 7574c77
        """
        xa, xb, ya, yb = self.domain
        A_op = self._coeffs      # shape (yorder+1, xorder+1)
        xorder = self._xorder
        yorder = self._yorder

        # ----------------------------------------------------------------
        # 1. SVD decomposition of A_op to get rank-r representation
        #    A_op = Σ_r sigma_r * v_r v_r^T  (in terms of ODE coeff vectors)
        #    A_op.T has shape (xorder+1, yorder+1)
        #    SVD: U_svd (xorder+1, r), svals (r,), V_svd.T (r, yorder+1)
        # ----------------------------------------------------------------
        U_svd, svals, Vt_svd = np.linalg.svd(A_op.T, full_matrices=False)
        V_svd = Vt_svd.T  # (yorder+1, rk)
        tol_svd = max(_EPS, 1e-14)
        rk = max(1, int(np.sum(np.abs(svals) / max(svals[0], 1e-300) > tol_svd)))
        U_svd = U_svd[:, :rk]
        svals = svals[:rk]
        V_svd = V_svd[:, :rk]

        # ----------------------------------------------------------------
        # 2. Build 1D operator matrices
        #    LEFT_r: m×m y-direction ODE operator
        #    RIGHT_r: n×n x-direction ODE operator
        # ----------------------------------------------------------------
        CC_left = []
        CC_right = []
        for r in range(rk):
            # y-direction: Ly_r = Σ_k V_svd[k, r] * D_y^k
            Ly = np.zeros((m, m), dtype=np.float64)
            for k in range(yorder + 1):
                c = V_svd[k, r]
                if abs(c) > _EPS:
                    Ly += c * _diffmat_cheb2_np(m, k, (ya, yb))

            # x-direction: Rx_r = Σ_k U_svd[k, r] * D_x^k
            Rx = np.zeros((n, n), dtype=np.float64)
            for k in range(xorder + 1):
                c = U_svd[k, r]
                if abs(c) > _EPS:
                    Rx += c * _diffmat_cheb2_np(n, k, (xa, xb))

            sv_sqrt = np.sqrt(abs(svals[r]))
            sign_sv = np.sign(svals[r]) if svals[r] != 0 else 1.0
            CC_left.append(sv_sqrt * sign_sv * Ly)
            CC_right.append(sv_sqrt * Rx)

        # ----------------------------------------------------------------
        # 3. Build the RHS value matrix F
        # ----------------------------------------------------------------
        x_pts = _cheb2_pts_np(n, (xa, xb))  # ascending: xa..xb
        y_pts = _cheb2_pts_np(m, (ya, yb))  # ascending: ya..yb
        xx, yy = np.meshgrid(x_pts, y_pts)  # shape (m, n), yy[i,j]=y_i, xx[i,j]=x_j

        if callable(f):
            F_vals = np.array(
                f(jnp.asarray(xx, dtype=jnp.float64),
                  jnp.asarray(yy, dtype=jnp.float64)),
                dtype=np.float64,
            )
        else:
            F_vals = np.full((m, n), float(f), dtype=np.float64)

        # ----------------------------------------------------------------
        # 4. Build boundary value vectors
        #    BC values at the boundary collocation points
        # ----------------------------------------------------------------
        # y-BCs: dbc at y=ya (pts[0]), ubc at y=yb (pts[-1])
        # x-BCs: lbc at x=xa (pts[0]), rbc at x=xb (pts[-1])

        bc_dbc_vals = _eval_bc_on_pts(self._dbc, x_pts) if self._dbc is not None else None
        bc_ubc_vals = _eval_bc_on_pts(self._ubc, x_pts) if self._ubc is not None else None
        bc_lbc_vals = _eval_bc_on_pts(self._lbc, y_pts) if self._lbc is not None else None
        bc_rbc_vals = _eval_bc_on_pts(self._rbc, y_pts) if self._rbc is not None else None

        # ----------------------------------------------------------------
        # 5. Assemble full Kronecker matrix K = Σ_r RIGHT_r ⊗ LEFT_r
        #    The vectorization is column-major: vec(U)[i + j*m] = U[i, j]
        #    D_y^2 U + U D_x^2^T <=> (I_n ⊗ D_y^2 + D_x^2 ⊗ I_m) vec(U)
        #    But for general sum: Σ_r LEFT_r U RIGHT_r^T
        #                         <=> Σ_r (RIGHT_r ⊗ LEFT_r) vec(U)
        # ----------------------------------------------------------------
        sz = m * n
        K = np.zeros((sz, sz), dtype=np.float64)
        for r in range(rk):
            K += np.kron(CC_right[r], CC_left[r])

        rhs = F_vals.ravel("F").copy()  # column-major vectorization

        # ----------------------------------------------------------------
        # 6. Impose boundary conditions by row replacement
        # ----------------------------------------------------------------
        # Cheb-2 pts ordering: pts[0]=domain_start, pts[-1]=domain_end
        # U[i, j] = u(y_pts[i], x_pts[j])
        # In column-major vec: U[i, j] -> index i + j*m

        # y=ya BCs (dbc): U[0, j] = dbc(x_pts[j]) for all j
        if bc_dbc_vals is not None:
            for j in range(n):
                ri = 0 + j * m
                K[ri, :] = 0.0
                K[ri, ri] = 1.0
                rhs[ri] = bc_dbc_vals[j]

        # y=yb BCs (ubc): U[m-1, j] = ubc(x_pts[j]) for all j
        if bc_ubc_vals is not None:
            for j in range(n):
                ri = (m - 1) + j * m
                K[ri, :] = 0.0
                K[ri, ri] = 1.0
                rhs[ri] = bc_ubc_vals[j]

        # x=xa BCs (lbc): U[i, 0] = lbc(y_pts[i]) for all i
        if bc_lbc_vals is not None:
            for i in range(m):
                ri = i + 0 * m
                K[ri, :] = 0.0
                K[ri, ri] = 1.0
                rhs[ri] = bc_lbc_vals[i]

        # x=xb BCs (rbc): U[i, n-1] = rbc(y_pts[i]) for all i
        if bc_rbc_vals is not None:
            for i in range(m):
                ri = i + (n - 1) * m
                K[ri, :] = 0.0
                K[ri, ri] = 1.0
                rhs[ri] = bc_rbc_vals[i]

        # ----------------------------------------------------------------
        # 7. Solve K @ vec(U) = vec(F)
        # ----------------------------------------------------------------
        U_vec = np.linalg.solve(K, rhs)
        U = U_vec.reshape(m, n, order="F")

        return U

    # ------------------------------------------------------------------
    # Coefficient-space (ultraspherical) solve
    # ------------------------------------------------------------------

    def _can_use_coeff_solve(self) -> bool:
        """Whether the coefficient-space solver can handle this problem.

        The ultraspherical path handles constant- and variable-coefficient
        (real or complex) PDOs with Dirichlet, Neumann, or Robin boundary
        conditions.  It requires that the operator have at least one
        derivative in each direction that has a boundary condition (so the
        BC-elimination bookkeeping is well posed).  Anything it cannot form
        raises inside :meth:`_coeff_solve` and triggers the value-space
        fallback, so this predicate only needs to avoid the obvious misfits.
        """
        if self._coeffs is None:
            return False
        A = self._coeffs
        if A.shape[0] < 1 or A.shape[1] < 1:
            return False
        # Need a positive order in a direction before we can impose BCs there.
        has_x_bc = (self._lbc is not None) or (self._rbc is not None)
        has_y_bc = (self._dbc is not None) or (self._ubc is not None)
        if has_x_bc and self._xorder < 1:
            return False
        if has_y_bc and self._yorder < 1:
            return False
        return True

    def _construct_bc(self, bc_spec, bcpos: int, een: int, bcn: int,
                      dom: tuple[float, float], scl: tuple[float, float],
                      order: int):
        """Discretize a single boundary condition (MATLAB ``constructBC``).

        Returns ``(bcrow, bcvalue)`` where ``bcrow`` (length ``bcn``) is the
        boundary functional acting on the Chebyshev coefficients in the
        direction perpendicular to the edge, and ``bcvalue`` (length ``een``)
        holds the Chebyshev coefficients of the (nonhomogeneous) data along
        the edge.

        Handles Dirichlet (scalar or one-argument callable), general
        Neumann/Robin conditions ``lambda t, u: c0*u + c1*u' + ... + f(t)``
        (two-argument callable), and multiple conditions on one edge expressed
        as a list ``lambda t, u: [u - g0(t), diff(u) - g1(t)]``.  The returned
        arrays are 2D with one column per condition.

        Provenance
        ----------
        MATLAB source : @chebop2/constructBC.m
        Chebfun commit: 7574c77
        """
        import inspect

        from chebfunjax.utils.transforms import vals2coeffs

        # Determine the callable arity (Dirichlet=1 arg, Neumann/Robin=2 args).
        nargs = None
        if callable(bc_spec):
            try:
                nargs = len(inspect.signature(bc_spec).parameters)
            except (ValueError, TypeError):
                nargs = 1

        # --- Dirichlet: scalar or one-argument callable ---
        if isinstance(bc_spec, (int, float, complex)) or (callable(bc_spec)
                                                          and nargs == 1):
            bcrow = _cheb_values(0, bcn, float(bcpos)).reshape(bcn, 1)
            if isinstance(bc_spec, (int, float, complex)):
                dt = np.complex128 if isinstance(bc_spec, complex) else np.float64
                bcvalue = np.zeros((een, 1), dtype=dt)
                bcvalue[0, 0] = bc_spec
            else:
                data = _cheb_coeffs_1d(bc_spec, een, dom)
                bcvalue = np.zeros((een, 1), dtype=data.dtype)
                L = min(een, len(data))
                bcvalue[:L, 0] = data[:L]
            return bcrow, bcvalue

        # --- General / multi-condition: two-argument callable ---
        if callable(bc_spec) and nargs == 2:
            a, b = dom
            t = np.array(chebpts(een, kind=2), dtype=np.float64)
            tpts = 0.5 * (b - a) * t + 0.5 * (a + b)
            dx = abs(2.0 / (scl[1] - scl[0]))

            # 1) Forcing f(t): evaluate with u == 0 (each list entry a column).
            f_res = bc_spec(jnp.asarray(tpts, dtype=jnp.float64), _BCZeroProxy())
            f_list = list(f_res) if isinstance(f_res, (list, tuple)) else [f_res]

            # 2) Constants c_k: probe with u carrying derivative structure.
            p_res = bc_spec(jnp.asarray(tpts, dtype=jnp.float64),
                            _BCProbeProxy({0: 1.0}))
            p_list = list(p_res) if isinstance(p_res, (list, tuple)) else [p_res]
            if len(p_list) != len(f_list):
                raise TypeError("Chebop2: inconsistent boundary condition list.")

            ncond = len(p_list)
            bcvalue_cols = []
            bcrow_cols = []
            for jj in range(ncond):
                # Forcing column (bcvalue = -f).
                fj = f_list[jj]
                if isinstance(fj, _BCZeroProxy):
                    bcvalue_cols.append(np.zeros(een, dtype=np.float64))
                else:
                    fv = np.asarray(fj, dtype=np.complex128)
                    if np.max(np.abs(fv.imag)) < 1e-13 * max(np.max(np.abs(fv)), 1.0):
                        fv = fv.real
                    fc = np.array(vals2coeffs(jnp.asarray(fv)), dtype=fv.dtype)
                    col = np.zeros(een, dtype=fc.dtype)
                    L = min(een, len(fc))
                    col[:L] = -fc[:L]
                    bcvalue_cols.append(col)

                # Operator row column.
                pj = p_list[jj]
                if not isinstance(pj, _BCProbeProxy):
                    raise TypeError("Chebop2: unsupported boundary condition form.")
                terms = pj._terms
                any_c = any(isinstance(c, complex) for c in terms.values())
                row = np.zeros(bcn, dtype=np.complex128 if any_c else np.float64)
                for k, c in terms.items():
                    if c == 0:
                        continue
                    row = row + c * (dx ** k) * _cheb_values(k, bcn, float(bcpos))
                bcrow_cols.append(row)

            dt = (np.complex128
                  if any(np.iscomplexobj(c) for c in bcvalue_cols + bcrow_cols)
                  else np.float64)
            bcvalue = np.array(bcvalue_cols, dtype=dt).T   # (een, ncond)
            bcrow = np.array(bcrow_cols, dtype=dt).T       # (bcn, ncond)
            return bcrow, bcvalue

        raise TypeError("Chebop2: unrecognised boundary condition syntax.")

    def _coeff_solve(self, f, m: int, n: int) -> np.ndarray:
        """Coefficient-space (ultraspherical) solve at fixed size m x n.

        Returns the ``m x n`` Chebyshev-T coefficient matrix ``X`` of the
        solution, ``X[i, j]`` being the coefficient of ``T_i(y) T_j(x)``.

        This mirrors MATLAB ``@chebop2/discretize`` + ``denseSolve``: separable
        rank expansion of the PDO, banded ultraspherical 1D operators, boundary
        DOF elimination, generalized Sylvester (or Kronecker) solve, then
        re-imposition of the boundary rows.

        Provenance
        ----------
        MATLAB source : @chebop2/discretize.m, @chebop2/denseSolve.m
        Chebfun commit: 7574c77
        """
        xa, xb, ya, yb = self.domain
        A = self._coeffs                      # (yorder+1, xorder+1)
        xorder = self._xorder
        yorder = self._yorder

        # ---- separable rank expansion via SVD of A.' (rows=x, cols=y) ----
        U_svd, svals, Vt_svd = np.linalg.svd(A.T, full_matrices=False)
        V_svd = Vt_svd.T
        tol = 10.0 * _EPS
        rk = max(1, int(np.sum(svals > tol * max(svals[0], 1e-300))))
        U_svd = U_svd[:, :rk]     # rows index x-order
        svals = svals[:rk]
        V_svd = V_svd[:, :rk]     # rows index y-order

        # ---- 1D ultraspherical operators for each rank term ----
        CC = []
        for jj in range(rk):
            RIGHT = _unconstrained_matrix_equation(U_svd[:, jj], n, xorder, (xa, xb))
            LEFT = _unconstrained_matrix_equation(V_svd[:, jj], m, yorder, (ya, yb))
            sv = np.sqrt(svals[jj])
            CC.append([sv * LEFT, sv * RIGHT])

        # ---- boundary conditions ----
        bcLeft = bcRight = bcUp = bcDown = None
        leftVal = rightVal = upVal = downVal = None
        if self._lbc is not None:
            bcLeft, leftVal = self._construct_bc(
                self._lbc, -1, m, n, (ya, yb), (xa, xb), xorder)
        if self._rbc is not None:
            bcRight, rightVal = self._construct_bc(
                self._rbc, 1, m, n, (ya, yb), (xa, xb), xorder)
        if self._ubc is not None:
            bcUp, upVal = self._construct_bc(
                self._ubc, 1, n, m, (xa, xb), (ya, yb), yorder)
        if self._dbc is not None:
            bcDown, downVal = self._construct_bc(
                self._dbc, -1, n, m, (xa, xb), (ya, yb), yorder)

        def _stack(rows, vals):
            # Each bcrow is (bcn, ncond); stack their transposes so every
            # condition becomes one row of B (MATLAB [bcUp.'; bcDown.']).
            R = [r.T for r in rows if r is not None]
            Vv = [v.T for v in vals if v is not None]
            if not R:
                return np.zeros((0, 0)), np.zeros((0, 0))
            return np.vstack(R), np.vstack(Vv)

        By, Gy = _stack([bcUp, bcDown], [upVal, downVal])
        Bx, Gx = _stack([bcLeft, bcRight], [leftVal, rightVal])
        if By.size:
            By, Gy, Py = _canonical_bc(By, Gy)
        else:
            Py = None
        if Bx.size:
            Bx, Gx, Px = _canonical_bc(Bx, Gx)
            Bx = Bx.T
            Gx = Gx.T
        else:
            Px = None

        # ---- right-hand side coefficient matrix, mapped to C^{(order)} ----
        if callable(f):
            F = _cheb_coeffs_2d(f, m, n, self.domain)
        elif f == 0:
            F = np.zeros((m, n), dtype=np.float64)
        else:
            F = np.zeros((m, n), dtype=np.float64)
            F[0, 0] = float(f)
        lmap = _ultra_convertmat(m, 0, yorder - 1)
        rmap = _ultra_convertmat(n, 0, xorder - 1)
        E = lmap @ F @ rmap.T

        # Promote to complex if any operator/BC/RHS piece is complex.
        pieces = [c for pair in CC for c in pair] + [E]
        for arr in (By, Gy, Bx, Gx):
            if arr is not None and getattr(arr, "size", 0):
                pieces.append(arr)
        if any(np.iscomplexobj(p) for p in pieces):
            CC = [[np.asarray(c, dtype=np.complex128) for c in pair] for pair in CC]
            E = np.asarray(E, dtype=np.complex128)

        # ---- eliminate boundary DOFs (zeroDOF) ----
        for jj in range(rk):
            if By.size:
                C, E = _zero_dof(CC[jj][0], CC[jj][1], E, By, Gy)
                CC[jj][0] = C
            if Bx.size:
                C, E = _zero_dof(CC[jj][1], CC[jj][0], E.T, Bx.T, Gx.T)
                CC[jj][1] = C
                E = E.T

        # ---- remove degrees of freedom (truncation) ----
        mo = max(xorder, yorder)
        nn = n - mo
        mm = m - mo
        df1 = max(0, xorder - yorder)
        df2 = max(0, yorder - xorder)
        for jj in range(rk):
            CC[jj][0] = CC[jj][0][:mm, yorder:m - df1]
            CC[jj][1] = CC[jj][1][:nn, xorder:n - df2]
        rhs = E[:mm, :nn]

        # ---- solve the reduced (generalized Sylvester / Kronecker) system ----
        X = _reduced_solve(CC, rhs, rk)

        # ---- re-impose the boundary rows ----
        bb = [bcLeft, bcRight, bcUp, bcDown]
        gg = [leftVal, rightVal, upVal, downVal]
        X = _impose_boundary_conditions(X, bb, gg, Px, Py, m, n)
        return X

    # ------------------------------------------------------------------
    # Wrap solution as SeparableApprox
    # ------------------------------------------------------------------

    def _wrap_solution(self, U_vals: np.ndarray):
        """Wrap the value matrix U_vals as a SeparableApprox.

        Parameters
        ----------
        U_vals : np.ndarray, shape (m, n)
            Function values at a Chebyshev-2 tensor grid (ascending ordering).

        Returns
        -------
        SeparableApprox
            Low-rank representation of the solution.

        Provenance
        ----------
        MATLAB source : @chebop2/solvepde.m
        Chebfun commit: 7574c77
        """
        from chebfunjax.utils.transforms import vals2coeffs

        m, n = U_vals.shape

        # Convert value matrix to Chebyshev coefficient matrix
        C = np.zeros((m, n), dtype=np.float64)
        for j in range(n):
            col = jnp.asarray(U_vals[:, j], dtype=jnp.float64)
            C[:, j] = np.array(vals2coeffs(col), dtype=np.float64)
        for i in range(m):
            row = jnp.asarray(C[i, :], dtype=jnp.float64)
            C[i, :] = np.array(vals2coeffs(row), dtype=np.float64)

        return self._wrap_coeffs(C)

    def _wrap_coeffs(self, C: np.ndarray):
        """Wrap a Chebyshev-coefficient matrix as a SeparableApprox.

        ``C[i, j]`` is the coefficient of ``T_i(y) T_j(x)``.  A low-rank SVD
        of ``C`` gives the column (functions of y) and row (functions of x)
        Chebtech2 pieces.  A tiny imaginary part (from a complex-coefficient
        solve) is dropped when negligible.

        Provenance
        ----------
        MATLAB source : @chebop2/solvepde.m (chebfun2 from coeffs)
        Chebfun commit: 7574c77
        """
        from chebfunjax.chebfun2d.separable_approx import SeparableApprox
        from chebfunjax.tech.chebtech import Chebtech2

        xa, xb, ya, yb = self.domain
        C = np.asarray(C)
        if np.iscomplexobj(C) and np.max(np.abs(C.imag)) < np.sqrt(_EPS):
            C = C.real
        dt = jnp.complex128 if np.iscomplexobj(C) else jnp.float64

        U_svd, s, Vt_svd = np.linalg.svd(C, full_matrices=False)
        tol_sa = _EPS * 10
        rk = max(1, int(np.sum(s / max(s[0], 1e-300) > tol_sa)))
        U_svd = U_svd[:, :rk]
        s = s[:rk]
        Vt_svd = Vt_svd[:rk, :]

        cols_list = []
        rows_list = []
        for r in range(rk):
            cols_list.append(Chebtech2(jnp.asarray(U_svd[:, r], dtype=dt)))
            rows_list.append(Chebtech2(jnp.asarray(Vt_svd[r, :], dtype=dt)))

        pivots = jnp.asarray(s, dtype=jnp.float64)

        return SeparableApprox(
            cols=cols_list,
            rows=rows_list,
            pivots=pivots,
            domain=(xa, xb, ya, yb),
        )

    def __repr__(self) -> str:
        xa, xb, ya, yb = self.domain
        n_bcs = sum(
            1 for v in [self._lbc, self._rbc, self._ubc, self._dbc]
            if v is not None
        )
        return (
            f"Chebop2(domain=({xa}, {xb}, {ya}, {yb}), "
            f"op={'set' if self.op is not None else 'None'}, "
            f"n_bcs={n_bcs})"
        )


# ===========================================================================
# _Chebop2Proxy — symbolic proxy for extracting PDE operator coefficients
# ===========================================================================


class _Chebop2Proxy:
    """Symbolic 2D function proxy for extracting PDE operator coefficients.

    When the user writes ``lambda u: u.diff(2, 0) + u.diff(0, 2)``, ``u``
    is replaced with an instance of this class.  Calling ``u.diff(j, k)``
    records a term with derivative order ``(j, k)`` in the operator.
    Arithmetic accumulates coefficients.

    Internal state: dict mapping ``(yorder, xorder) -> scalar coefficient``.

    Provenance
    ----------
    Inspired by MATLAB's ``adchebfun2`` automatic differentiation.
    Chebfun commit: 7574c77
    """

    def __init__(self, terms: dict | None = None) -> None:
        self._terms: dict[tuple[int, int], float] = (
            terms if terms is not None else {(0, 0): 1.0}
        )

    def diff(self, yorder: int = 0, xorder: int = 0) -> "_Chebop2Proxy":
        """Return the (yorder, xorder) partial derivative of this proxy."""
        new_terms: dict[tuple[int, int], float] = {}
        for (j, k), c in self._terms.items():
            key = (j + yorder, k + xorder)
            new_terms[key] = new_terms.get(key, 0.0) + c
        return _Chebop2Proxy(new_terms)

    def __add__(self, other) -> "_Chebop2Proxy":
        if isinstance(other, (int, float)):
            new_terms = dict(self._terms)
            new_terms[(0, 0)] = new_terms.get((0, 0), 0.0) + float(other)
            return _Chebop2Proxy(new_terms)
        if isinstance(other, _Chebop2Proxy):
            new_terms = dict(self._terms)
            for key, c in other._terms.items():
                new_terms[key] = new_terms.get(key, 0.0) + c
            return _Chebop2Proxy(new_terms)
        return NotImplemented

    def __radd__(self, other) -> "_Chebop2Proxy":
        return self.__add__(other)

    def __sub__(self, other) -> "_Chebop2Proxy":
        if isinstance(other, (int, float)):
            return self.__add__(-float(other))
        if isinstance(other, _Chebop2Proxy):
            return self.__add__(other.__neg__())
        return NotImplemented

    def __rsub__(self, other) -> "_Chebop2Proxy":
        return self.__neg__().__add__(other)

    def __mul__(self, other) -> "_Chebop2Proxy":
        if isinstance(other, (int, float)):
            c = float(other)
            return _Chebop2Proxy({key: val * c for key, val in self._terms.items()})
        return NotImplemented

    def __rmul__(self, other) -> "_Chebop2Proxy":
        return self.__mul__(other)

    def __neg__(self) -> "_Chebop2Proxy":
        return _Chebop2Proxy({key: -val for key, val in self._terms.items()})

    def __truediv__(self, other) -> "_Chebop2Proxy":
        if isinstance(other, (int, float)):
            return self.__mul__(1.0 / float(other))
        return NotImplemented

    def _coeffs_matrix(self) -> np.ndarray:
        """Build coefficient matrix A[j, k] = coeff of d^j/dy^j d^k/dx^k."""
        if not self._terms:
            return np.zeros((1, 1), dtype=np.float64)
        max_j = max(j for j, k in self._terms)
        max_k = max(k for j, k in self._terms)
        A = np.zeros((max_j + 1, max_k + 1), dtype=np.float64)
        for (j, k), c in self._terms.items():
            A[j, k] = c
        return A

    def __repr__(self) -> str:
        return f"_Chebop2Proxy(terms={self._terms})"


# ===========================================================================
# Module-level private helpers
# ===========================================================================


def _eval_bc_on_pts(bc_spec, pts: np.ndarray) -> np.ndarray:
    """Evaluate a BC specification at physical collocation points.

    Parameters
    ----------
    bc_spec : scalar or callable
        Boundary condition.  If scalar, constant value.  If callable,
        evaluated at ``pts`` (a JAX array).
    pts : np.ndarray, shape (n,)
        Physical collocation points along the edge.

    Returns
    -------
    np.ndarray, shape (n,)

    Provenance
    ----------
    MATLAB source : @chebop2/constructBC.m
    Chebfun commit: 7574c77
    """
    if isinstance(bc_spec, (int, float)):
        return np.full(len(pts), float(bc_spec), dtype=np.float64)
    if callable(bc_spec):
        return np.array(
            bc_spec(jnp.asarray(pts, dtype=jnp.float64)),
            dtype=np.float64,
        )
    return np.asarray(bc_spec, dtype=np.float64).ravel()


def _op_from_coeffs(S: np.ndarray) -> Callable:
    """Build an operator lambda that reproduces a coefficient matrix ``S``.

    ``S[j, k]`` is the coefficient of ``∂^j/∂y^j ∂^k/∂x^k`` (the internal
    ``_coeffs`` layout).  The returned callable maps a :class:`_Chebop2Proxy`
    to the corresponding proxy term, so that :class:`Chebop2` objects produced
    by arithmetic remain solvable.
    """

    def op(u: "_Chebop2Proxy") -> "_Chebop2Proxy":
        term: _Chebop2Proxy | None = None
        for j in range(S.shape[0]):
            for k in range(S.shape[1]):
                c = float(S[j, k])
                if c != 0.0:
                    piece = c * u.diff(j, k)
                    term = piece if term is None else term + piece
        return term if term is not None else 0.0 * u.diff(0, 0)

    return op


# ---------------------------------------------------------------------------
# Free operator helpers for building operator lambdas (MATLAB-style syntax)
# ---------------------------------------------------------------------------


def diffx(u: "_Chebop2Proxy", n: int = 1) -> "_Chebop2Proxy":
    """``n``-th partial derivative in x (MATLAB ``diffx(u, n)``)."""
    return u.diff(0, n)


def diffy(u: "_Chebop2Proxy", n: int = 1) -> "_Chebop2Proxy":
    """``n``-th partial derivative in y (MATLAB ``diffy(u, n)``)."""
    return u.diff(n, 0)


def laplacian(u: "_Chebop2Proxy") -> "_Chebop2Proxy":
    """2D Laplacian ``u_xx + u_yy`` (MATLAB ``laplacian(u)``)."""
    return u.diff(0, 2) + u.diff(2, 0)


def lap(u: "_Chebop2Proxy") -> "_Chebop2Proxy":
    """Alias for :func:`laplacian` (MATLAB ``lap(u)``)."""
    return laplacian(u)


def gradient(u: "_Chebop2Proxy") -> tuple["_Chebop2Proxy", "_Chebop2Proxy"]:
    """2D gradient ``(u_x, u_y)`` (MATLAB ``gradient(u)``)."""
    return (u.diff(0, 1), u.diff(1, 0))


def grad(u: "_Chebop2Proxy") -> tuple["_Chebop2Proxy", "_Chebop2Proxy"]:
    """Alias for :func:`gradient` (MATLAB ``grad(u)``)."""
    return gradient(u)


def divergence(vec: tuple["_Chebop2Proxy", "_Chebop2Proxy"]) -> "_Chebop2Proxy":
    """2D divergence ``F1_x + F2_y`` of a vector field (MATLAB ``divergence``)."""
    f1, f2 = vec
    return f1.diff(0, 1) + f2.diff(1, 0)


def div(vec: tuple["_Chebop2Proxy", "_Chebop2Proxy"]) -> "_Chebop2Proxy":
    """Alias for :func:`divergence` (MATLAB ``div(F)``)."""
    return divergence(vec)


def _next_grid(n: int) -> int:
    """Return the next adaptive grid size (double the interior)."""
    if n <= 1:
        return 3
    p = int(np.floor(np.log2(n - 1)))
    return 2 ** (p + 1) + 1


def _is_resolved_vals(U: np.ndarray, tol: float) -> bool:
    """Check if the solution value matrix is resolved by checking Cheb coeffs."""
    from chebfunjax.utils.transforms import vals2coeffs

    if U.size == 0:
        return True
    m, n = U.shape
    # Check a sample of columns and rows for coefficient decay
    n_check = min(4, n)
    n_check_r = min(4, m)

    scale = max(np.max(np.abs(U)), 1e-300)
    tail_abs = tol * scale

    # Check columns (y-direction)
    for j in range(0, n, max(1, n // n_check)):
        col = jnp.asarray(U[:, j], dtype=jnp.float64)
        c = np.array(vals2coeffs(col), dtype=np.float64)
        if np.max(np.abs(c[max(0, m - 4):])) > tail_abs * 20:
            return False

    # Check rows (x-direction)
    for i in range(0, m, max(1, m // n_check_r)):
        row = jnp.asarray(U[i, :], dtype=jnp.float64)
        c = np.array(vals2coeffs(row), dtype=np.float64)
        if np.max(np.abs(c[max(0, n - 4):])) > tail_abs * 20:
            return False

    return True
