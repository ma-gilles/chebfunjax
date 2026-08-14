"""Alternative 1-D discretizations for scalar linops: ultraS, chebcolloc1.

MATLAB's opDiscretization hierarchy lets ``prefs.discretization`` pick
``@ultraS`` (the Olver-Townsend sparse ultraspherical spectral method)
or ``@chebcolloc1`` (collocation at first-kind points) instead of the
default ``@chebcolloc2``.  chebfunjax's operator blocks are closures
rather than symbolic terms, so this module recovers the differential
form

    L[u] = c_0(x) u + c_1(x) u' + ... + c_m(x) u^(m)

from any scalar :class:`~chebfunjax.operators.blocks.OperatorBlock` by
probing its chebcolloc2 matrix on scaled monomials (the same forward
substitution the chebop linearizer uses), then assembles the requested
discretization from the recovered coefficients:

* ``ultraS``   — coefficient space: ``sum_k S_{k->m} M_k[c_k] D_k``
  with the sparse ultraspherical conversion/differentiation/
  multiplication operators (shared with the chebop2 solver);
* ``chebcolloc1`` — barycentric differentiation matrices on the
  first-kind grid, boundary rows by barycentric extrapolation.

Provenance
----------
MATLAB source : @ultraS/ultraS.m, @chebcolloc1/chebcolloc1.m,
    @linop/eigs.m, @linop/linsolve.m (prefs.discretization dispatch)
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

from math import factorial

import jax.numpy as jnp
import numpy as np  # uses-numpy: dense/sparse assembly on concrete data

from chebfunjax.operators.blocks import ChebColloc2Disc
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


def _bary_diffmat(x, w, k: int = 1):
    """Barycentric differentiation matrix of order ``k`` on nodes ``x``
    (Welfert's recursion; Berrut & Trefethen SIAM Rev. 46, 2004)."""
    X = x[:, None] - x[None, :]
    np.fill_diagonal(X, 1.0)
    W = w[None, :] / w[:, None]
    D = W / X
    np.fill_diagonal(D, 0.0)
    np.fill_diagonal(D, -np.sum(D, axis=1))
    Dm = D
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


def scalar_matrices(L, n: int, discretization: str):
    """Discretize a scalar BlockLinop as (A, B_mass, recover).

    ``A`` is the square matrix ``[constraint rows; projected operator]``
    and ``B_mass`` the matching mass matrix (zero constraint rows above
    the projected identity), so that ``A v = lam * B_mass v`` is the
    discrete eigenproblem and ``A v = [bc values; P f]`` the linear
    solve.  ``recover(v)`` maps a solution vector back to a Chebfun.

    Provenance
    ----------
    MATLAB source : @ultraS/matrix.m, @chebcolloc1/... via
        @opDiscretization/matrix.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.chebfun1d.chebfun import Chebfun
    from chebfunjax.domain import Domain
    from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

    if L.nrows != 1 or L.ncols != 1:
        raise NotImplementedError(
            "ultraS/chebcolloc1 discretizations support scalar (single "
            "block) linops; block systems use chebcolloc2.")
    dom = tuple(float(v) for v in L.domain)
    if len(dom) != 2:
        raise NotImplementedError(
            "ultraS/chebcolloc1 discretizations support a single "
            "interval.")
    a, b = dom
    block = L.A.blocks[0][0]
    m = int(block.order)
    coeff_vals, t2 = extract_coefficients(block, dom, m, n=max(n, 33))
    ncon = len(L.constraint)

    if discretization == "ultraS":
        # Coefficient space: rows = [BC rows; first n-ncon rows of
        # sum_k S_k M_k[c_k] D_k]; mass = S_{0->m}.
        A_op = np.zeros((n, n))
        for k, cv in enumerate(coeff_vals):
            if np.max(np.abs(cv)) < 1e2 * _EPS * max(
                    1.0, float(np.max(np.abs(np.concatenate(
                        [np.atleast_1d(np.max(np.abs(c)))
                         for c in coeff_vals]))))):
                continue
            c_t = np.asarray(vals2coeffs(jnp.asarray(cv)))
            S = _ultra_convertmat(n, k, m - 1)
            Dk = ((2.0 / (b - a)) ** k) * _ultra_diffmat(n, k)
            Mk = _ultra_multmat(n, c_t, k)
            A_op = A_op + S @ Mk @ Dk
        S0m = _ultra_convertmat(n, 0, m - 1)

        # Constraint rows in coefficient space: a functional row on the
        # colloc2 grid maps to coefficients via row @ coeffs2vals, i.e.
        # row_c = row_v @ V where V is the coeffs->values matrix.
        disc2 = ChebColloc2Disc(n, dom)
        eye = np.eye(n)
        V = np.column_stack([
            np.asarray(Chebtech2.from_coeffs(
                jnp.asarray(eye[:, j])).values)
            for j in range(n)])
        con_rows = []
        for row_list, _val in L.constraint:
            rv = np.asarray(row_list[0].matrix(disc2), dtype=float)
            con_rows.append(rv @ V)
        Abig = np.vstack(con_rows + [A_op[: n - ncon]])
        Bmass = np.vstack([np.zeros((ncon, n)), S0m[: n - ncon]])

        def recover(v):
            tech = Chebtech2.from_coeffs(jnp.asarray(v)).simplify()
            return Chebfun.from_values(
                jnp.asarray(tech.values), Domain(dom))

        def rhs_vec(fvals_fn):
            # f expanded in C^(m-1): values at colloc2 -> T coeffs -> S.
            fv = fvals_fn(t2 * 0.5 * (b - a) + 0.5 * (a + b))
            fc = np.asarray(vals2coeffs(jnp.asarray(np.asarray(fv))))
            fc = np.resize(fc, n)
            return (S0m @ fc)[: n - ncon]

        return Abig, Bmass, recover, rhs_vec

    if discretization == "chebcolloc1":
        x1, w1 = _cheb1_pts_weights(n)
        pts = x1 * 0.5 * (b - a) + 0.5 * (a + b)
        scale = 2.0 / (b - a)
        # Interpolate the recovered coefficients onto the cheb1 grid.
        c_on_1 = []
        for cv in coeff_vals:
            tech = Chebtech2.from_coeffs(
                np.asarray(vals2coeffs(jnp.asarray(cv))))
            c_on_1.append(np.asarray(tech(jnp.asarray(x1))))
        A_op = np.zeros((n, n))
        for k, cv in enumerate(c_on_1):
            if np.max(np.abs(cv)) == 0:
                continue
            Dk = (scale ** k) * _bary_diffmat(x1, w1, k) if k else np.eye(n)
            A_op = A_op + cv[:, None] * Dk
        # Rectangularization: project rows onto n - ncon first-kind
        # points via barycentric resampling.
        xr, _ = _cheb1_pts_weights(n - ncon)
        P = np.vstack([_bary_eval_row(x1, w1, t) for t in xr])
        con_rows = []
        for row_list, _val in L.constraint:
            # Functional rows: rebuild on the cheb1 grid via barycentric
            # evaluation of the same functional's action.  Endpoint
            # evals and derivative evals dominate; probe the functional
            # on the colloc2 grid and transfer through interpolation.
            disc2 = ChebColloc2Disc(n, dom)
            rv = np.asarray(row_list[0].matrix(disc2), dtype=float)
            # row on cheb2 values -> row on cheb1 values: rv @ (cheb2
            # values of the cheb1 cardinal functions) = rv @ T where
            # T[i, j] = ell_j^{(1)}(x2_i).
            x2 = np.asarray(chebpts(n, kind=2))
            T = np.vstack([_bary_eval_row(x1, w1, t) for t in x2])
            con_rows.append(rv @ T)
        Abig = np.vstack(con_rows + [P @ A_op])
        Bmass = np.vstack([np.zeros((ncon, n)), P])

        def recover(v):
            tech1 = Chebtech1.from_values(jnp.asarray(v))
            tech2 = Chebtech2.from_coeffs(
                jnp.asarray(tech1.coeffs)).simplify()
            return Chebfun.from_values(
                jnp.asarray(tech2.values), Domain(dom))

        def rhs_vec(fvals_fn):
            fv = np.asarray(fvals_fn(pts))
            return P @ fv

        return Abig, Bmass, recover, rhs_vec

    raise ValueError(
        f"Unknown discretization {discretization!r}; expected "
        "'ultraS' or 'chebcolloc1'.")
