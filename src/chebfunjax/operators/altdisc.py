"""Alternative discretizations for linops: ultraS, chebcolloc1.

MATLAB's opDiscretization hierarchy lets ``prefs.discretization`` pick
``@ultraS`` (the Olver-Townsend sparse ultraspherical spectral method)
or ``@chebcolloc1`` (collocation at first-kind points) instead of the
default ``@chebcolloc2``.  chebfunjax's operator blocks are closures
rather than symbolic terms, so this module rebuilds each block in the
requested space from its chebcolloc2 matrix:

* differential blocks are recovered as
  ``L[u] = c_0(x) u + c_1(x) u' + ... + c_m(x) u^(m)``
  by probing the chebcolloc2 matrix on scaled monomials (forward
  substitution), and the ultraS matrix is then assembled as
  ``sum_k S_{k->d} M_k[c_k] D_k`` with the sparse ultraspherical
  conversion/differentiation/multiplication operators;
* any other block (integral operators, compositions that are not a pure
  differential form) is transferred exactly: a linear operator that maps
  degree-``<n`` polynomials to degree-``<n`` polynomials is fully
  determined by its chebcolloc2 matrix, so ``R21 @ M2 @ B12`` is its
  chebcolloc1 matrix and ``C @ M2 @ V`` its Chebyshev-coefficient
  matrix, with ``B12/R21`` barycentric resampling between the grids and
  ``V/C`` the coeffs<->values maps.

Block systems, scalar variables, functional rows, piecewise domains and
generalized (``A v = lam B v``) pencils are all supported; the row space
of equation ``i`` is ``C^(d_i)`` (ultraS) or the first-kind grid
(chebcolloc1), reduced by ``d_i`` rows per interval to make room for the
side conditions, exactly as MATLAB's ``reduce`` does.

Provenance
----------
MATLAB source : @ultraS/ultraS.m, @chebcolloc1/chebcolloc1.m,
    @opDiscretization/matrix.m, @linop/eigs.m, @linop/linsolve.m
    (prefs.discretization dispatch)
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

from math import factorial

import jax.numpy as jnp
import numpy as np  # uses-numpy: dense/sparse assembly on concrete data

from chebfunjax.operators.blocks import (
    ChebColloc2Disc,
    D,
    FunctionalBlock,
    OperatorBlock,
)
from chebfunjax.operators.chebop2 import (
    _ultra_convertmat,
    _ultra_diffmat,
    _ultra_multmat,
)
from chebfunjax.utils.quadrature import chebpts
from chebfunjax.utils.transforms import vals2coeffs

_EPS = float(np.finfo(np.float64).eps)


def extract_coefficients(block, dom, order: int, n: int = 64):
    """Recover the derivative-coefficient functions of a scalar block.

    Returns a list ``[c_0, ..., c_order]`` of length-``n`` value vectors
    on the chebcolloc2 grid of ``dom`` such that the block acts as
    ``sum_k diag(c_k) D^k`` (forward substitution on monomial probes of
    the block's chebcolloc2 matrix).

    Provenance
    ----------
    MATLAB source : the differential form recovered by
        @linop/linop.m block coefficients; probing strategy as in the
        chebop linearizer.
    Chebfun commit: 7574c77
    """
    a, b = float(dom[0]), float(dom[1])
    disc = ChebColloc2Disc(n, (a, b))
    M = np.asarray(block.matrix(disc), dtype=complex)
    if np.max(np.abs(M.imag)) < 1e-14 * max(np.max(np.abs(M)), 1.0):
        M = M.real
    t = np.asarray(chebpts(n, kind=2))
    h = 0.5 * (b - a)
    y = t  # reference variable
    Lp = [M @ (y ** k / factorial(k)) for k in range(order + 1)]
    coeffs = []
    for k in range(order + 1):
        acc = Lp[k].copy()
        for j in range(k):
            acc = acc - coeffs[j] * (y ** (k - j)) / (h ** j
                                                      * factorial(k - j))
        coeffs.append(acc * (h ** k))
    return coeffs, t


def _cheb1_pts_weights(n: int):
    """First-kind points and barycentric weights on [-1, 1]."""
    th = (2 * np.arange(n)[::-1] + 1) * np.pi / (2 * n)
    x = np.cos(th)
    w = (-1.0) ** np.arange(n) * np.sin(th)
    return x, w


def _cheb2_pts_weights(n: int):
    """Second-kind points and barycentric weights on [-1, 1]."""
    x = np.asarray(chebpts(n, kind=2))
    w = (-1.0) ** np.arange(n)
    w[0] *= 0.5
    w[-1] *= 0.5
    return x, w


def _bary_diffmat(x, w, k: int = 1):
    """Barycentric differentiation matrix of order ``k`` on nodes ``x``
    (Welfert's recursion; Berrut & Trefethen SIAM Rev. 46, 2004)."""
    X = x[:, None] - x[None, :]
    np.fill_diagonal(X, 1.0)
    W = w[None, :] / w[:, None]
    D_ = W / X
    np.fill_diagonal(D_, 0.0)
    np.fill_diagonal(D_, -np.sum(D_, axis=1))
    Dm = D_
    for m in range(2, k + 1):
        diag_prev = np.diag(Dm)[:, None]
        Dm = (m / X) * (W * diag_prev - Dm)
        np.fill_diagonal(Dm, 0.0)
        np.fill_diagonal(Dm, -np.sum(Dm, axis=1))
    return Dm


def _bary_eval_row(x, w, t: float):
    """Barycentric evaluation row at ``t`` for nodes/weights (x, w)."""
    d = t - x
    hit = np.abs(d) < 1e-14
    if np.any(hit):
        row = np.zeros(len(x))
        row[np.argmax(hit)] = 1.0
        return row
    num = w / d
    return num / np.sum(num)


class _RefMats:
    """Reference-interval transfer matrices, computed once per size."""

    def __init__(self, n: int):
        self.n = n
        self.x1, self.w1 = _cheb1_pts_weights(n)
        self.x2, self.w2 = _cheb2_pts_weights(n)
        # cheb1 values -> cheb2 values (interpolation) and back.
        self.B12 = np.vstack([_bary_eval_row(self.x1, self.w1, t)
                              for t in self.x2])
        self.R21 = np.vstack([_bary_eval_row(self.x2, self.w2, t)
                              for t in self.x1])
        # Chebyshev-T coeffs -> cheb2 values, and its inverse.
        eye = np.eye(n)
        self.C = np.column_stack([
            np.asarray(vals2coeffs(jnp.asarray(eye[:, j])))
            for j in range(n)])
        self.V = np.linalg.inv(self.C)

    def proj1(self, red: int):
        """Barycentric down-projection: n cheb1 values -> n-red."""
        if red <= 0:
            return np.eye(self.n)
        xr, _ = _cheb1_pts_weights(self.n - red)
        return np.vstack([_bary_eval_row(self.x1, self.w1, t)
                          for t in xr])


def _repair_endpoints(cv: np.ndarray, ref: "_RefMats") -> np.ndarray:
    """Replace a coefficient sample's endpoint values by barycentric
    extrapolation from the interior nodes.

    Coefficient functions probed through a block's chebcolloc2 matrix
    carry the piecewise chebfun's stored breakpoint pointValues (e.g.
    the average across a jump) at the interval endpoints, which are not
    the one-sided limits of the smooth restriction; interior samples
    define it exactly, so extrapolate from them (removing a node x_r
    from a barycentric node set scales the remaining weights by
    (x_j - x_r))."""
    n = cv.shape[0]
    if n < 4:
        return cv
    x_in = ref.x2[1:-1]
    w_in = ref.w2[1:-1] * (x_in * x_in - 1.0)
    out = cv.copy()
    for pos, t in ((0, -1.0), (n - 1, 1.0)):
        num = w_in / (t - x_in)
        out[pos] = np.dot(num, cv[1:-1]) / np.sum(num)
    return out


def _multmat_c(n: int, c_t: np.ndarray, k: int) -> np.ndarray:
    """Complex-safe ultraspherical multiplication matrix (multmat is
    linear in the coefficient vector, so split real/imag parts)."""
    if np.iscomplexobj(c_t):
        return (_ultra_multmat(n, c_t.real, k)
                + 1j * _ultra_multmat(n, c_t.imag, k))
    return _ultra_multmat(n, c_t, k)


def _ultras_op_block(blk, dom_k, d_row: int, ref: _RefMats):
    """UltraS matrix of one operator block on one interval, rows in
    C^(d_row).  True Olver-Townsend assembly from the recovered
    differential form when it validates; exact coefficient-space
    transfer of the chebcolloc2 matrix otherwise."""
    n = ref.n
    a, b = float(dom_k[0]), float(dom_k[1])
    coeff_vals, M2 = _differential_form(blk, (a, b), ref)
    if coeff_vals is not None:
        A = np.zeros((n, n), dtype=complex if any(
            np.iscomplexobj(c) for c in coeff_vals) else float)
        for k, cv in enumerate(coeff_vals):
            if np.max(np.abs(cv)) < 1e2 * _EPS * max(
                    1.0, float(max(np.max(np.abs(c))
                                   for c in coeff_vals))):
                continue
            c_t = np.asarray(vals2coeffs(jnp.asarray(cv)))
            S = _ultra_convertmat(n, k, d_row - 1)
            Dk = ((2.0 / (b - a)) ** k) * _ultra_diffmat(n, k)
            Mk = _multmat_c(n, c_t, k)
            A = A + S @ Mk @ Dk
        return A
    # Not a pure differential form (e.g. an integral operator):
    # exact transfer through coefficient space.
    S0 = _ultra_convertmat(n, 0, d_row - 1)
    return S0 @ ref.C @ M2 @ ref.V


def _differential_form(blk, dom_k, ref: _RefMats):
    """Recover (and endpoint-repair) the differential coefficients of a
    block on one interval; returns ``(coeffs, M2)`` with ``coeffs``
    ``None`` when the block is not a pure differential form.  The
    validation compares only the interior collocation rows, where the
    chebcolloc2 matrix is unaffected by breakpoint pointValues."""
    n = ref.n
    a, b = float(dom_k[0]), float(dom_k[1])
    m = int(blk.order)
    disc = ChebColloc2Disc(n, (a, b))
    M2 = np.asarray(blk.matrix(disc), dtype=complex)
    if np.max(np.abs(M2.imag)) < 1e-14 * max(np.max(np.abs(M2)), 1.0):
        M2 = M2.real
    coeff_vals, _t = extract_coefficients(blk, (a, b), m, n=n)
    M2r = np.zeros_like(M2)
    for k, cv in enumerate(coeff_vals):
        Dk2 = (np.asarray(D((a, b), k).matrix(disc)) if k
               else np.eye(n))
        M2r = M2r + cv[:, None] * Dk2
    scl = max(np.max(np.abs(M2)), 1.0)
    if np.max(np.abs(M2r[1:-1] - M2[1:-1])) < 1e-7 * scl:
        return [_repair_endpoints(np.asarray(cv), ref)
                for cv in coeff_vals], M2
    return None, M2


def _colloc1_op_block(blk, dom_k, ref: _RefMats):
    """chebcolloc1 matrix of one operator block on one interval.

    Differential blocks are assembled natively (repaired coefficient
    samples on the first-kind grid times Welfert barycentric
    differentiation matrices -- the first-kind grid contains no interval
    endpoints, so breakpoint pointValues never enter); anything else is
    transferred exactly from the chebcolloc2 matrix."""
    n = ref.n
    a, b = float(dom_k[0]), float(dom_k[1])
    coeff_vals, M2 = _differential_form(blk, (a, b), ref)
    if coeff_vals is not None:
        scale = 2.0 / (b - a)
        A = np.zeros((n, n), dtype=complex if any(
            np.iscomplexobj(c) for c in coeff_vals) else float)
        for k, cv in enumerate(coeff_vals):
            if np.max(np.abs(cv)) == 0:
                continue
            tech_c = np.asarray(vals2coeffs(jnp.asarray(cv)))
            # Evaluate the coefficient interpolant on the cheb1 grid.
            c1 = np.polynomial.chebyshev.chebval(ref.x1, tech_c)
            Dk = ((scale ** k) * _bary_diffmat(ref.x1, ref.w1, k)
                  if k else np.eye(n))
            A = A + c1[:, None] * Dk
        return A
    return ref.R21 @ M2 @ ref.B12


def _fun_row_transfer(seg, ref: _RefMats, discretization: str):
    """Map one interval's segment of a colloc2 functional row into the
    requested trial space (cheb1 values or T coefficients)."""
    if discretization == "ultraS":
        return seg @ ref.V
    return seg @ ref.B12


class SystemDisc:
    """A block linop discretized under ultraS or chebcolloc1.

    Attributes
    ----------
    A : ndarray
        Square matrix ``[side-condition rows; projected operator rows]``.
    con_vals : list of float
        The side-condition values (continuity rows first).

    Provenance
    ----------
    MATLAB source : @ultraS/ultraS.m, @chebcolloc1/chebcolloc1.m via
        @opDiscretization/matrix.m and @chebDiscretization/reduce.m
    Chebfun commit: 7574c77
    """

    def __init__(self, L, n: int, discretization: str, dom=None,
                 row_order_min=None):
        if discretization not in ("ultraS", "chebcolloc1"):
            raise ValueError(
                f"Unknown discretization {discretization!r}; expected "
                "'ultraS' or 'chebcolloc1'.")
        self.disc = discretization
        self.n = n = int(n)
        dom = tuple(float(v) for v in (L.domain if dom is None else dom))
        self.dom = dom
        self.K = K = len(dom) - 1
        L2 = L.derive_continuity(dom) if not L.continuity else L
        self.L = L2
        self.isfun = isfun = L2.is_fun_variable()
        self.funcrow = [L2._row_is_functional(i) for i in range(L2.nrows)]
        self.ref = _RefMats(n)

        # Differential order (output space) of each operator row.  For a
        # generalized pencil the row space must fit both operators, so a
        # per-row floor may be supplied (MATLAB uses the max diffOrder of
        # the pencil when reducing).
        self.row_order = []
        for i in range(L2.nrows):
            orders = [blk.order for blk in L2.A.blocks[i]
                      if isinstance(blk, OperatorBlock)]
            d = max(orders) if orders else 0
            if row_order_min is not None:
                d = max(d, int(row_order_min[i]))
            self.row_order.append(d)

        # Row reduction: d_i rows per interval per operator row; the
        # side conditions must exactly fill the removed rows.
        con_rows = L2.continuity + L2.constraint
        ncon = len(con_rows)
        nfrow = sum(1 for f in self.funcrow if f)
        cols = sum(K * n if isfun[j] else 1 for j in range(L2.ncols))
        rows = ncon + nfrow + sum(K * (n - self.row_order[i])
                                  for i in range(L2.nrows)
                                  if not self.funcrow[i])
        if rows != cols:
            raise ValueError(
                "CHEBFUN:LINOP:linsolve:notSquare -- the operator does "
                f"not have the correct number of side conditions "
                f"(matrix is {rows}x{cols}).")

        self.col_off = []
        off = 0
        for j in range(L2.ncols):
            self.col_off.append(off)
            off += K * n if isfun[j] else 1
        self.ncols_total = off

        disc2 = ChebColloc2Disc([n] * K, dom)
        self._disc2 = disc2
        self.con_rows = np.vstack(
            [self._functional_row(row) for row, _v in con_rows]
        ) if con_rows else np.zeros((0, off))
        self.con_vals = [v for _row, v in con_rows]
        self.A = np.vstack([self.con_rows] + [
            self._equation_rows(L2.A.blocks[i], i)
            for i in range(L2.nrows)])

    # -- assembly ------------------------------------------------------

    def _functional_row(self, row) -> np.ndarray:
        """One side-condition / functional row over all variables."""
        n, K, ref = self.n, self.K, self.ref
        parts = []
        for j, blk in enumerate(row):
            if isinstance(blk, FunctionalBlock):
                rv = np.asarray(blk.matrix(self._disc2), dtype=float)
                parts.append(np.concatenate(
                    [_fun_row_transfer(rv[k * n:(k + 1) * n], ref,
                                       self.disc) for k in range(K)]))
            elif isinstance(blk, (int, float, complex)):
                parts.append(np.asarray([blk], dtype=complex)
                             if isinstance(blk, complex)
                             else np.asarray([float(blk)]))
            else:
                raise TypeError(
                    "SystemDisc: functional rows may only contain "
                    "FunctionalBlocks and scalars.")
        return np.concatenate(parts)[None, :]

    def _fun_column(self, blk, red: int) -> np.ndarray:
        """A Chebfun (or constant) block: scalar variable -> function."""
        n, ref = self.n, self.ref
        cols = []
        for k in range(self.K):
            a, b = self.dom[k], self.dom[k + 1]
            if self.disc == "ultraS":
                pts = ref.x2 * 0.5 * (b - a) + 0.5 * (a + b)
                fv = (np.asarray(blk(jnp.asarray(pts))).ravel()
                      if callable(blk) else np.full(n, float(blk)))
                fc = ref.C @ fv
                d = red
                S0 = _ultra_convertmat(n, 0, d - 1)
                cols.append((S0 @ fc)[: n - red])
            else:
                pts = ref.x1 * 0.5 * (b - a) + 0.5 * (a + b)
                fv = (np.asarray(blk(jnp.asarray(pts))).ravel()
                      if callable(blk) else np.full(n, float(blk)))
                cols.append(ref.proj1(red) @ fv)
        return np.concatenate(cols)[:, None]

    def _equation_rows(self, row, i: int) -> np.ndarray:
        """All discrete rows of block-row ``i`` (projected)."""
        if self.funcrow[i]:
            return self._functional_row(row)
        n, K, ref = self.n, self.K, self.ref
        red = self.row_order[i]
        d_row = max(red, 0)
        parts = []
        for j, blk in enumerate(row):
            if isinstance(blk, OperatorBlock):
                mats = []
                for k in range(K):
                    dk = (self.dom[k], self.dom[k + 1])
                    if self.disc == "ultraS":
                        M = _ultras_op_block(blk, dk, d_row, ref)
                        mats.append(M[: n - red])
                    else:
                        M = _colloc1_op_block(blk, dk, ref)
                        mats.append(ref.proj1(red) @ M)
                blkdiag = np.zeros(
                    (sum(m.shape[0] for m in mats), K * n),
                    dtype=complex if any(np.iscomplexobj(m)
                                         for m in mats) else float)
                r0 = 0
                for k, m in enumerate(mats):
                    blkdiag[r0:r0 + m.shape[0], k * n:(k + 1) * n] = m
                    r0 += m.shape[0]
                parts.append(blkdiag)
            elif isinstance(blk, FunctionalBlock):
                raise TypeError(
                    "SystemDisc: a FunctionalBlock cannot occupy an "
                    "operator row.")
            else:
                # Chebfun or scalar coefficient of a scalar variable.
                parts.append(self._fun_column(blk, red))
        return np.concatenate(
            [p.astype(complex) if any(np.iscomplexobj(q) for q in parts)
             else p for p in parts], axis=1)

    # -- right-hand side and recovery ----------------------------------

    def rhs(self, entries) -> np.ndarray:
        """Stack [side-condition values; projected f rows]."""
        n, ref = self.n, self.ref
        segs = [np.asarray([complex(v) if isinstance(v, complex)
                            else float(v) for v in self.con_vals])]
        ei = 0
        for i in range(self.L.nrows):
            entry = entries[ei] if ei < len(entries) else 0.0
            ei += 1
            if self.funcrow[i]:
                segs.append(np.asarray([float(entry)]))
                continue
            red = self.row_order[i]
            for k in range(self.K):
                a, b = self.dom[k], self.dom[k + 1]
                if self.disc == "ultraS":
                    pts = ref.x2 * 0.5 * (b - a) + 0.5 * (a + b)
                    fv = (np.asarray(entry(jnp.asarray(pts))).ravel()
                          if callable(entry)
                          else np.full(n, float(entry)))
                    S0 = _ultra_convertmat(n, 0, red - 1)
                    segs.append((S0 @ (ref.C @ fv))[: n - red])
                else:
                    pts = ref.x1 * 0.5 * (b - a) + 0.5 * (a + b)
                    fv = (np.asarray(entry(jnp.asarray(pts))).ravel()
                          if callable(entry)
                          else np.full(n, float(entry)))
                    segs.append(ref.proj1(red) @ fv)
        return np.concatenate(segs)

    def mass(self, B=None) -> np.ndarray:
        """The pencil right side: zeros on the side-condition rows, the
        discretized ``B`` (identity when ``None``) below, sharing this
        discretization's row spaces and reduction."""
        n, K = self.n, self.K
        ncon = self.con_rows.shape[0]
        rows = [np.zeros((ncon, self.ncols_total))]
        for i in range(self.L.nrows):
            if self.funcrow[i]:
                rows.append(np.zeros((1, self.ncols_total)))
                continue
            red = self.row_order[i]
            if B is None:
                block_row = [None] * self.L.ncols
            else:
                block_row = list(B.A.blocks[i])
            parts = []
            for j in range(self.L.ncols):
                width = K * n if self.isfun[j] else 1
                blk = None if B is None else block_row[j]
                if B is None:
                    if j != i or not self.isfun[j]:
                        parts.append(np.zeros((K * (n - red), width)))
                        continue
                    from chebfunjax.operators.blocks import I as _I
                    blk = _I(self.dom)
                if isinstance(blk, OperatorBlock):
                    mats = []
                    for k in range(K):
                        dk = (self.dom[k], self.dom[k + 1])
                        if self.disc == "ultraS":
                            M = _ultras_op_block(blk, dk,
                                                 max(red, 0), self.ref)
                            mats.append(M[: n - red])
                        else:
                            M = _colloc1_op_block(blk, dk, self.ref)
                            mats.append(self.ref.proj1(red) @ M)
                    bd = np.zeros(
                        (sum(m.shape[0] for m in mats), K * n),
                        dtype=complex if any(np.iscomplexobj(m)
                                             for m in mats) else float)
                    r0 = 0
                    for k, m in enumerate(mats):
                        bd[r0:r0 + m.shape[0], k * n:(k + 1) * n] = m
                        r0 += m.shape[0]
                    parts.append(bd)
                elif blk is None or (isinstance(blk, (int, float))
                                     and blk == 0):
                    parts.append(np.zeros((K * (n - red), width)))
                else:
                    parts.append(self._fun_column(blk, red))
            rows.append(np.concatenate(
                [p.astype(complex) for p in parts]
                if any(np.iscomplexobj(p) for p in parts) else parts,
                axis=1))
        return np.vstack([r.astype(complex) for r in rows]
                         if any(np.iscomplexobj(r) for r in rows)
                         else rows)

    def trial_vector(self, entries) -> np.ndarray:
        """Sample per-variable functions/scalars into the trial space
        (Chebyshev-T coefficients per interval for ultraS, first-kind
        values per interval for chebcolloc1)."""
        ref = self.ref
        segs = []
        for j in range(self.L.ncols):
            entry = entries[j]
            if not self.isfun[j]:
                segs.append(np.asarray([float(entry)]))
                continue
            for k in range(self.K):
                a, b = self.dom[k], self.dom[k + 1]
                if self.disc == "ultraS":
                    pts = ref.x2 * 0.5 * (b - a) + 0.5 * (a + b)
                    fv = np.asarray(entry(jnp.asarray(pts))).ravel()
                    segs.append(ref.C @ fv)
                else:
                    pts = ref.x1 * 0.5 * (b - a) + 0.5 * (a + b)
                    segs.append(np.asarray(entry(jnp.asarray(pts)))
                                .ravel())
        return np.concatenate(segs)

    def recover(self, v):
        """Map a solution vector back to per-variable Chebfuns/scalars."""
        from chebfunjax.operators.blocklinop import (
            _chebfun_from_pieces,
            _piece_from_values,
        )
        from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
        n, K = self.n, self.K
        v = np.asarray(v)
        out = []
        for j in range(self.L.ncols):
            off = self.col_off[j]
            if not self.isfun[j]:
                val = v[off]
                out.append(complex(val) if np.iscomplexobj(v)
                           else float(np.real(val)))
                continue
            pieces = []
            for k in range(K):
                seg = jnp.asarray(v[off + k * n: off + (k + 1) * n])
                if self.disc == "ultraS":
                    tech = Chebtech2.from_coeffs(seg).simplify()
                else:
                    t1 = Chebtech1.from_values(seg)
                    tech = Chebtech2.from_coeffs(
                        jnp.asarray(t1.coeffs)).simplify()
                pieces.append(_piece_from_values(
                    jnp.asarray(tech.values), self.dom[k],
                    self.dom[k + 1]))
            out.append(_chebfun_from_pieces(pieces, self.dom))
        return out


def system_matrices(L, n: int, discretization: str,
                    dom=None, row_order_min=None) -> SystemDisc:
    """Discretize a BlockLinop under ultraS or chebcolloc1.

    Provenance
    ----------
    MATLAB source : @opDiscretization/matrix.m with
        prefs.discretization = @ultraS | @chebcolloc1
    Chebfun commit: 7574c77
    """
    return SystemDisc(L, n, discretization, dom=dom,
                      row_order_min=row_order_min)
