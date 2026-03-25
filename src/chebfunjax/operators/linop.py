"""Linear operator with boundary conditions for spectral ODE/BVP solving.

Provides :class:`Linop`, a high-level wrapper around an :class:`OperatorBlock`
(or sum/composition of blocks) together with a list of
:class:`FunctionalBlock` boundary conditions.  The two key operations are:

- ``solve(f, n)``    — solve the BVP ``L*u = f`` with the attached BCs
- ``eigs(n, k)``     — compute the *k* most-resolved eigenvalues of *L*

The discretization strategy follows MATLAB Chebfun's ``@chebcolloc2``:

1. Assemble the ``n x n`` operator matrix.
2. Replace the last ``n_bc`` rows (one per BC) with the barycentric-row
   vectors from the :class:`FunctionalBlock` evaluations.
3. Replace the corresponding RHS entries with the BC values.
4. Solve via ``jnp.linalg.solve``.
5. Interpret the ``n`` solution values as function values at the ``n``
   Chebyshev-2 points and wrap them in a :class:`~chebfunjax.chebfun1d.Chebfun`.

For ``eigs``, the constrained rows are similarly imposed on the *left* of the
eigenproblem: the BC rows replace the last ``n_bc`` rows of the operator matrix
and the corresponding rows of the RHS identity are zeroed out.  The eigenvalues
coming from the BC rows are artificially large and are discarded.

Translated from MATLAB Chebfun classes ``@linop`` and ``@linopConstraint``
(commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import jax.numpy as jnp

from chebfunjax.domain import Domain
from chebfunjax.operators.blocks import (
    ChebColloc2Disc,
    FunctionalBlock,
    OperatorBlock,
)
from chebfunjax.utils.quadrature import chebpts

# ---------------------------------------------------------------------------
# Lazy import to avoid circular dependency with chebfun1d
# ---------------------------------------------------------------------------

def _chebfun_from_values(values, domain: tuple[float, float]):
    """Wrap collocation values as a Chebfun (lazy import to avoid cycles)."""
    from chebfunjax.chebfun1d.chebfun import Chebfun
    dom = Domain(domain)
    return Chebfun.from_values(jnp.asarray(values, dtype=jnp.float64), dom)


def _chebfun_call(f, x):
    """Evaluate a Chebfun or scalar at physical points x."""
    if hasattr(f, "__call__") and not isinstance(f, (int, float)):
        return f(jnp.asarray(x, dtype=jnp.float64))
    return jnp.full_like(x, float(f), dtype=jnp.float64)


# ===========================================================================
# Linop
# ===========================================================================


class Linop:
    """Linear differential operator with attached boundary conditions.

    A :class:`Linop` packages:

    - an :class:`~chebfunjax.operators.blocks.OperatorBlock` *L* (the
      differential operator, e.g. ``D(order=2) + I()``),
    - a list of :class:`~chebfunjax.operators.blocks.FunctionalBlock`
      boundary conditions *bcs* (e.g. ``eval_at(a)`` and ``eval_at(b)``),
    - the physical domain ``(a, b)``.

    The primary entry points are :meth:`solve` (``L\f`` equivalent) and
    :meth:`eigs`.

    Parameters
    ----------
    L : OperatorBlock
        The linear differential operator.
    bcs : list of FunctionalBlock
        Boundary-condition functionals.  Their ``matrix(disc)`` method returns
        the ``n``-length row that replaces the corresponding row of the
        operator matrix.
    domain : (float, float), default (-1, 1)
        Physical domain ``[a, b]``.
    bc_values : list of float, optional
        Right-hand-side values for the boundary conditions (default: all 0).

    Examples
    --------
    Solve u'' = -1, u(-1) = u(1) = 0 (exact solution u = (1-x²)/2):

    >>> from chebfunjax.operators.blocks import D, I, eval_at
    >>> from chebfunjax.operators.linop import Linop
    >>> L = Linop(D(order=2), [eval_at(-1.0), eval_at(1.0)],
    ...           domain=(-1.0, 1.0), bc_values=[0.0, 0.0])
    >>> u = L.solve(lambda x: -jnp.ones_like(x))
    >>> import jax.numpy as jnp
    >>> abs(float(u(0.0)) - 0.5) < 1e-12
    True

    Notes
    -----
    Construction is NOT JIT-safe (operator assembly uses Python control flow).
    The ``solve`` method calls ``jnp.linalg.solve`` which is JIT-safe given
    fixed *n*.

    Boundary conditions are imposed by replacing the *last* ``n_bc`` rows of
    the operator matrix.  This follows the standard Chebfun convention (the
    operator rows at the outermost Chebyshev points, which coincide with the
    physical boundary, are the natural rows to replace).

    Provenance
    ----------
    MATLAB source : @linop/linop.m, @linop/linsolve.m, @linop/mldivide.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.

    See Also
    --------
    Chebop, OperatorBlock, FunctionalBlock
    """

    def __init__(
        self,
        L: OperatorBlock,
        bcs: Sequence[FunctionalBlock] | None = None,
        domain: tuple[float, float] = (-1.0, 1.0),
        bc_values: Sequence[float] | None = None,
    ) -> None:
        if not isinstance(L, OperatorBlock):
            raise TypeError(
                f"Linop: L must be an OperatorBlock, got {type(L).__name__}."
            )
        self.L = L
        self.bcs: list[FunctionalBlock] = list(bcs) if bcs is not None else []
        self.domain = tuple(float(v) for v in domain)
        if bc_values is not None:
            self.bc_values: list[float] = [float(v) for v in bc_values]
        else:
            self.bc_values = [0.0] * len(self.bcs)
        if len(self.bc_values) != len(self.bcs):
            raise ValueError(
                f"Linop: length of bc_values ({len(self.bc_values)}) must "
                f"equal number of BCs ({len(self.bcs)})."
            )

    # ------------------------------------------------------------------
    # Core: assemble and solve
    # ------------------------------------------------------------------

    def _assemble(self, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Assemble the square n×n system matrix and rhs placeholder.

        Returns
        -------
        A : jnp.ndarray, shape (n, n)
            Operator matrix with BC rows substituted in.
        bc_rows_and_vals : (rows, vals) stored for rhs construction.
        """
        disc = ChebColloc2Disc(n, self.domain)
        # Full operator matrix (n x n)
        A = self.L.matrix(disc)

        n_bc = len(self.bcs)
        if n_bc > n:
            raise ValueError(
                f"Linop._assemble: more BCs ({n_bc}) than discretization "
                f"points ({n}). Increase n."
            )

        # Replace the last n_bc rows with BC rows
        for i, bc in enumerate(self.bcs):
            row = bc.matrix(disc)         # shape (n,)
            row_idx = n - n_bc + i
            A = A.at[row_idx, :].set(row)

        return A

    def solve(
        self,
        f,
        n: int | None = None,
        n_min: int = 8,
        n_max: int = 4096,
        tol: float = 1e-10,
    ):
        """Solve the BVP ``L*u = f`` with the attached boundary conditions.

        Discretizes at increasing sizes until the Chebyshev coefficients of
        the solution decay below ``tol`` relative to their maximum.

        Parameters
        ----------
        f : callable or scalar
            Right-hand side.  If callable, called with the ``n`` Chebyshev-2
            collocation points on ``self.domain`` to produce the RHS vector.
            If scalar, treated as a constant function.
        n : int or None
            Fixed discretization size.  If given, no adaptive loop is used.
        n_min : int, default 8
            Minimum discretization size for the adaptive loop.
        n_max : int, default 2048
            Maximum discretization size for the adaptive loop.
        tol : float, default 1e-10
            Coefficient decay tolerance for convergence check.

        Returns
        -------
        u : Chebfun
            Solution on ``self.domain``.

        Raises
        ------
        RuntimeError
            If the adaptive loop reaches ``n_max`` without convergence.

        Notes
        -----
        Adaptive loop is NOT JIT-safe (Python-level while loop with
        data-dependent termination).  The inner linear solve is JAX.

        Provenance
        ----------
        MATLAB source : @linop/linsolve.m, @linop/mldivide.m
        Chebfun commit: 7574c77
        """
        if n is not None:
            # Fixed-size solve
            u_vals = self._solve_at(n, f)
            return _chebfun_from_values(u_vals, self.domain)

        # Adaptive loop: powers of 2 + 1 (Chebyshev-2 convention)
        sizes = [n_min]
        cur = n_min
        while cur < n_max:
            cur = min(cur * 2, n_max)
            sizes.append(cur)

        for sz in sizes:
            u_vals = self._solve_at(sz, f)
            # Check convergence via coefficient decay
            from chebfunjax.utils.transforms import vals2coeffs
            coeffs = vals2coeffs(u_vals)
            if self._is_happy(coeffs, tol):
                return _chebfun_from_values(u_vals, self.domain)

        warnings.warn(
            f"Linop.solve: adaptive loop reached n_max={n_max} without "
            f"convergence (tol={tol}). Returning best available solution.",
            stacklevel=2,
        )
        return _chebfun_from_values(u_vals, self.domain)

    def _solve_at(self, n: int, f) -> jnp.ndarray:
        """Solve at fixed discretization size n, return values at Cheb-2 pts."""
        ChebColloc2Disc(n, self.domain)
        # Compute physical Chebyshev-2 points on [a, b]
        a, b = self.domain
        t_ref = chebpts(n, kind=2)  # reference [-1, 1]
        x_pts = 0.5 * (b - a) * t_ref + 0.5 * (a + b)

        # Build RHS
        if callable(f):
            rhs = jnp.asarray(f(x_pts), dtype=jnp.float64)
        else:
            rhs = jnp.full(n, float(f), dtype=jnp.float64)

        # Assemble the square system
        A = self._assemble(n)

        # Replace last n_bc entries of rhs with BC values
        n_bc = len(self.bcs)
        for i, val in enumerate(self.bc_values):
            row_idx = n - n_bc + i
            rhs = rhs.at[row_idx].set(float(val))

        return jnp.linalg.solve(A, rhs)

    @staticmethod
    def _is_happy(coeffs: jnp.ndarray, tol: float) -> bool:
        """Check if Chebyshev coefficient tail has decayed below tol."""
        from chebfunjax.utils.misc import standard_chop
        n = coeffs.shape[0]
        if n < 4:
            return False
        cutoff = standard_chop(coeffs, tol=tol)
        # Happy if the cutoff is significantly smaller than n
        return int(cutoff) < n - 1

    # ------------------------------------------------------------------
    # Eigenvalue problem
    # ------------------------------------------------------------------

    def eigs(
        self,
        n: int | None = None,
        k: int = 6,
        n_default: int = 64,
        sigma: float | str | None = None,
    ) -> jnp.ndarray:
        """Compute eigenvalues of the constrained operator.

        Discretizes *L* at size *n*, imposes BC rows as in :meth:`solve`,
        then calls ``jnp.linalg.eig`` on the resulting ``n x n`` generalized
        eigenproblem::

            A * v = lambda * B * v

        where *A* is the BC-constrained operator and *B* is the identity with
        the same BC rows zeroed out (so the BC-constrained rows do not
        contribute finite eigenvalues).

        Returns the *k* eigenvalues that appear most resolved (smallest
        magnitude first, so the lowest-frequency modes come first for
        differential operators).

        Parameters
        ----------
        n : int or None
            Discretization size.  Defaults to ``n_default``.
        k : int, default 6
            Number of eigenvalues to return.
        n_default : int, default 64
            Discretization size used when ``n`` is ``None``.
        sigma : float or str or None
            Target: a scalar means "nearest to sigma", ``None`` means
            "smallest magnitude".  Strings ``'LM'``, ``'SM'``, ``'LR'``,
            ``'SR'`` are also accepted.

        Returns
        -------
        lam : jnp.ndarray, shape (k,)
            The *k* selected eigenvalues (real part sorted ascending).

        Notes
        -----
        Uses dense ``jnp.linalg.eig`` (not sparse ARPACK) — suitable for
        moderate *n* (≤ 1000).  Spurious eigenvalues from the BC rows are
        removed by deflation: the ``n_bc`` largest-magnitude eigenvalues
        are discarded before selecting the *k* target eigenvalues.

        Provenance
        ----------
        MATLAB source : @linop/eigs.m (``getEigenvalues`` helper)
        Chebfun commit: 7574c77
        Original authors: Copyright 2017 by The University of Oxford
            and The Chebfun Developers.

        See Also
        --------
        solve
        """
        sz = n if n is not None else n_default
        disc = ChebColloc2Disc(sz, self.domain)

        # Build operator matrix
        A = self.L.matrix(disc)

        n_bc = len(self.bcs)

        # Build identity (rhs of generalized eigenproblem)
        B = jnp.eye(sz, dtype=jnp.float64)

        # Replace last n_bc rows of A with BC rows; zero corresponding rows in B
        for i, bc in enumerate(self.bcs):
            row = bc.matrix(disc)          # shape (sz,)
            row_idx = sz - n_bc + i
            A = A.at[row_idx, :].set(row)
            B = B.at[row_idx, :].set(0.0)  # deflate BC rows

        # Solve generalized eigenproblem A v = lambda B v using dense eig.
        # Convert to standard form by solving B \ A where possible, but B
        # is singular (BC rows are zero) so we use scipy.linalg.eig instead.
        import numpy as np
        import scipy.linalg

        A_np = np.array(A)
        B_np = np.array(B)

        lam_all, _ = scipy.linalg.eig(A_np, B_np)

        # Deflate: the n_bc largest-magnitude eigenvalues (from BC rows) are
        # spurious (infinite or very large). Force them to inf so they're
        # excluded.
        lam_complex = jnp.array(lam_all, dtype=jnp.complex128)
        jnp.isfinite(lam_complex)

        # Sort by distance to sigma (or smallest magnitude if sigma is None)
        lam_np = np.array(lam_all)
        if sigma is None or sigma == "SM":
            key = np.abs(lam_np)
        elif sigma == "LM" or sigma is jnp.inf:
            key = -np.abs(lam_np)
        elif sigma == "LR":
            key = -np.real(lam_np)
        elif sigma == "SR":
            key = np.real(lam_np)
        elif sigma == "LI":
            key = -np.imag(lam_np)
        elif sigma == "SI":
            key = np.imag(lam_np)
        elif isinstance(sigma, (int, float)):
            key = np.abs(lam_np - sigma)
        else:
            raise ValueError(
                f"Linop.eigs: unrecognised sigma={sigma!r}. Use None, a "
                f"scalar, 'SM', 'LM', 'LR', 'SR', 'LI', or 'SI'."
            )

        # Set key=inf for non-finite eigenvalues so they're last
        finite_np = np.isfinite(lam_np)
        key[~finite_np] = np.inf

        # Deflate the n_bc largest-magnitude eigenvalues (spurious BC modes)
        # by setting them to inf in the key array.
        if n_bc > 0:
            mag_order = np.argsort(np.abs(lam_np))[::-1]
            deflate_idx = mag_order[:n_bc]
            key[deflate_idx] = np.inf

        sorted_idx = np.argsort(key, kind="stable")
        selected = sorted_idx[:k]

        lam_sel = lam_np[selected]

        # Return real parts if imaginary parts are negligible
        if np.max(np.abs(np.imag(lam_sel))) < 1e-8 * np.max(np.abs(np.real(lam_sel)) + 1e-300):
            return jnp.array(np.real(lam_sel), dtype=jnp.float64)
        return jnp.array(lam_sel, dtype=jnp.complex128)

    # ------------------------------------------------------------------
    # Operator: L \ f  (mldivide)
    # ------------------------------------------------------------------

    def __matmul__(self, f):
        """``L @ f`` — apply operator to a Chebfun (returns Chebfun)."""
        raise NotImplementedError(
            "Linop.__matmul__: operator application not yet implemented. "
            "Use Linop.solve(f) to solve L*u = f."
        )

    def __truediv__(self, f):
        """``L \\ f`` equivalent: solve L*u = f."""
        return self.solve(f)

    def __repr__(self) -> str:
        return (
            f"Linop(order={self.L.order}, domain={self.domain}, "
            f"n_bcs={len(self.bcs)})"
        )
