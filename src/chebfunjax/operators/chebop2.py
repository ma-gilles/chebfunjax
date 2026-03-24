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

    # QZ decomposition of (A, C): A = Q1 P Z1^H, C = Q1 S Z1^H
    P, S, Q1, Z1 = scipy.linalg.qz(A, C, output="real")
    P = np.triu(P)
    S = np.triu(S)

    # QZ decomposition of (D, B): D = Q2 T Z2^H, B = Q2 R Z2^H
    T, R, Q2, Z2 = scipy.linalg.qz(D, B, output="real")

    # Transform the RHS: F = Q1 E Q2^T
    F = Q1 @ E @ Q2.T

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

        if n is not None:
            X = self._dense_solve(f, n, n)
            return self._wrap_solution(X)

        # Adaptive loop: double the grid until resolved
        sz = n_min
        X = None
        for _ in range(20):
            X = self._dense_solve(f, sz, sz)
            if _is_resolved_vals(X, tol):
                break
            old_sz = sz
            sz = _next_grid(sz)
            if sz >= n_max:
                warnings.warn(
                    f"Chebop2.solve: adaptive loop reached n_max={n_max} without "
                    f"convergence (tol={tol}). Returning best available solution.",
                    stacklevel=2,
                )
                X = self._dense_solve(f, old_sz, old_sz)
                break
        return self._wrap_solution(X)

    def __truediv__(self, f):
        """``N \\ f`` — solve N[u] = f."""
        return self.solve(f)

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
        from chebfunjax.chebfun2d.separable_approx import SeparableApprox
        from chebfunjax.utils.transforms import vals2coeffs

        xa, xb, ya, yb = self.domain
        m, n = U_vals.shape

        # Convert value matrix to Chebyshev coefficient matrix
        C = np.zeros((m, n), dtype=np.float64)
        for j in range(n):
            col = jnp.asarray(U_vals[:, j], dtype=jnp.float64)
            C[:, j] = np.array(vals2coeffs(col), dtype=np.float64)
        for i in range(m):
            row = jnp.asarray(C[i, :], dtype=jnp.float64)
            C[i, :] = np.array(vals2coeffs(row), dtype=np.float64)

        # Low-rank SVD compression of the coefficient matrix
        U_svd, s, Vt_svd = np.linalg.svd(C, full_matrices=False)
        tol_sa = _EPS * 10
        rk = max(1, int(np.sum(s / max(s[0], 1e-300) > tol_sa)))
        U_svd = U_svd[:, :rk]
        s = s[:rk]
        Vt_svd = Vt_svd[:rk, :]

        from chebfunjax.tech.chebtech import Chebtech2

        cols_list = []
        rows_list = []
        for r in range(rk):
            col_c = jnp.asarray(U_svd[:, r], dtype=jnp.float64)
            row_c = jnp.asarray(Vt_svd[r, :], dtype=jnp.float64)
            cols_list.append(Chebtech2(col_c))
            rows_list.append(Chebtech2(row_c))

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
