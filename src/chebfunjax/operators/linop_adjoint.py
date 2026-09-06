"""Adjoint of a block linop with side constraints, and singular values
(MATLAB ``@linop/linopAdjoint.m`` and ``@linop/svds.m``; Fable 5).

Provenance
----------
MATLAB source : @linop/linopAdjoint.m, @linop/svds.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and The
    Chebfun Developers.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np  # uses-numpy: host-side dense linear algebra

from chebfunjax.operators.adjoint import _compmat, _rref
from chebfunjax.operators.blocklinop import BlockLinop
from chebfunjax.operators.blocks import (
    D,
    FunctionalBlock,
    OperatorBlock,
    eval_at,
    mult,
    zero_functional,
    zeros_op,
)


def _is_chebfun(b) -> bool:
    return hasattr(b, "funs") and hasattr(b, "domain")


def _block_coeffs(B, dom):
    """Coefficient Chebfuns ``[a_m, ..., a_0]`` (descending order, MATLAB
    ``toCoeff``) of a block; scalars and Chebfuns are multiplication
    operators."""
    from chebfunjax.chebfun1d.chebfun import chebfun
    a, b = float(dom[0]), float(dom[-1])
    if _is_chebfun(B):
        return [B]
    if isinstance(B, (int, float, complex)):
        return [chebfun(lambda x: 0.0 * x + B, domain=(a, b))]
    coeffs = list(B.coeff_list())
    out = []
    for c in coeffs:
        if _is_chebfun(c):
            out.append(c)
        else:
            out.append(chebfun(lambda x, _c=c: 0.0 * x + _c, domain=(a, b)))
    return out


def _block_order(B) -> int:
    if isinstance(B, OperatorBlock):
        return int(B.order)
    return 0


def adjoint_formal(L: BlockLinop):
    """Formal adjoint ``L*`` (integration by parts, no boundary
    conditions) as a :class:`BlockLinop`, and the callable
    ``op(x, v1, ..., vn)`` applying it to Chebfuns.

    Provenance
    ----------
    MATLAB source : @linop/linopAdjoint.m (adjointFormal)
    Chebfun commit: 7574c77
    """
    dom = tuple(float(v) for v in L.domain)
    a, b = dom[0], dom[-1]
    nrows, ncols = L.nrows, L.ncols
    blocks = L.A.blocks
    star_blocks = [[None] * nrows for _ in range(ncols)]
    adj_lists = [[None] * nrows for _ in range(ncols)]
    for ii in range(ncols):
        for jj in range(nrows):
            B = blocks[jj][ii]
            coeffs = _block_coeffs(B, dom)     # [a_dor, ..., a_0]
            dor = len(coeffs) - 1
            adj = [0.0 * coeffs[0] for _ in range(dor + 1)]   # descending
            for k in range(dor + 1):
                for ell in range(k + 1):
                    term = coeffs[dor - k].diff(k - ell).conj() * float(
                        (-1) ** k * math.comb(k, ell))
                    adj[dor - ell] = adj[dor - ell] + term
            blk = None
            for k in range(dor + 1):
                t = mult(adj[dor - k], (a, b)) * D((a, b), k)
                blk = t if blk is None else blk + t
            star_blocks[ii][jj] = blk if blk is not None else zeros_op((a, b))
            adj_lists[ii][jj] = adj

    Lstar = BlockLinop(star_blocks, domain=dom)

    def op(x, *vs):
        vs = list(vs)
        outs = []
        for ii in range(ncols):
            acc = None
            for jj in range(nrows):
                adj = adj_lists[ii][jj]
                dor = len(adj) - 1
                v = vs[jj]
                for k in range(dor + 1):
                    ck = adj[dor - k]
                    if float(ck.norm()) == 0.0:
                        continue
                    t = ck * (v.diff(k) if k > 0 else v)
                    acc = t if acc is None else acc + t
            outs.append(acc if acc is not None else 0.0 * vs[0])
        return outs[0] if ncols == 1 else outs
    return Lstar, op


def _end_value_rows(dom, ndeg, nder):
    """Values of ``T_0 .. T_{ndeg-1}`` and their first ``nder-1``
    derivatives at both endpoints: (Ul, Ur) of shape (nder, ndeg)."""
    from chebfunjax.chebfun1d.chebfun import chebpoly
    a, b = float(dom[0]), float(dom[-1])
    Ul = np.zeros((nder, ndeg))
    Ur = np.zeros((nder, ndeg))
    polys = [chebpoly(p, (a, b)) for p in range(ndeg)]
    for p, T in enumerate(polys):
        g = T
        for k in range(nder):
            Ul[k, p] = float(np.asarray(g(jnp.asarray(a))))
            Ur[k, p] = float(np.asarray(g(jnp.asarray(b))))
            g = g.diff()
    return Ul, Ur, polys


def _apply_row(row, funcs):
    """Apply a constraint row (list of functionals / scalars) to a list of
    Chebfuns."""
    tot = 0.0
    for r, f in zip(row, funcs):
        if isinstance(r, FunctionalBlock):
            tot = tot + complex(np.asarray(r.apply(f)))
        elif isinstance(r, (int, float, complex)) and r != 0:
            tot = tot + r * complex(np.asarray(f(jnp.asarray(float(row[0].domain[0])))))
    return tot


def _bc_row_functional(coef_row, dor_list, dom, star_type):
    """Functional (list per variable) for one adjoint boundary row of
    ``Bstar`` (columns: left endpoint derivative stack, then right)."""
    a, b = float(dom[0]), float(dom[-1])
    nin = len(dor_list)
    half = sum(dor_list)
    funcs = []
    col = 0
    for j in range(nin):
        f = None
        for k in range(dor_list[j]):
            cl = coef_row[col + k]
            cr = coef_row[half + col + k]
            for c, x0 in ((cl, a), (cr, b)):
                if c == 0:
                    continue
                t = eval_at(x0, (a, b)) * D((a, b), k) if k > 0 \
                    else eval_at(x0, (a, b))
                t = t if c == 1 else float(np.real(c)) * t
                f = t if f is None else f + t
        col += dor_list[j]
        funcs.append(f if f is not None else zero_functional((a, b)))
    return funcs


def _bc_row_callable(coef_row, dor_list, dom, star_type):
    """Callable ``v (or v1, v2, ...) -> Chebfun expression`` for a
    left/right adjoint condition (MATLAB bcHandles); for mixed rows the
    callable takes ``(x, v...)`` and returns a number."""
    a, b = float(dom[0]), float(dom[-1])
    nin = len(dor_list)
    half = sum(dor_list)

    def g(*vs):
        if star_type == 2:
            vs = vs[1:]
        out = None
        col = 0
        for j in range(nin):
            v = vs[j]
            for k in range(dor_list[j]):
                for c, x0, side in ((coef_row[col + k], a, 0),
                                    (coef_row[half + col + k], b, 1)):
                    if c == 0:
                        continue
                    t = v.diff(k) if k > 0 else v
                    if star_type == 2:
                        t = t(jnp.asarray(x0))
                    t = t if c == 1 else float(np.real(c)) * t
                    out = t if out is None else out + t
            col += dor_list[j]
        return out
    return g


def linop_adjoint(L: BlockLinop, bc_type: str = "bvp"):
    """Adjoint of ``L`` with its side conditions (MATLAB
    ``linopAdjoint(L, bcType)``): returns ``(Lstar, op, bcOpL, bcOpR,
    bcOpM)`` -- the adjoint linop carrying the adjoint constraints, the
    callable ``op(x, v...)`` for its action, and the left / right /
    mixed adjoint boundary-condition handles (``None`` when absent;
    ``bcOpM == 'periodic'`` for periodic problems).

    Provenance
    ----------
    MATLAB source : @linop/linopAdjoint.m
    Chebfun commit: 7574c77
    """
    if bc_type not in ("bvp", "periodic"):
        raise ValueError("CHEBFUN:LINOP:linopAdjoint:bcType: bcType must be "
                         "'bvp' or 'periodic'.")
    dom = tuple(float(v) for v in L.domain)
    a, b = dom[0], dom[-1]
    Lstar, op = adjoint_formal(L)
    nout, nin = L.nrows, L.ncols
    blocks = L.A.blocks
    dor = [max(_block_order(blocks[i][j]) for i in range(nout))
           for j in range(nin)]
    if bc_type == "periodic":
        Lstar.constraint = []
        Lstar.periodic = True
        return Lstar, op, None, None, "periodic"
    cons = list(L.constraint)
    nbcs = len(cons)
    sdor = sum(dor)
    nadj = 2 * sdor - nbcs
    if nadj <= 0 and nbcs > 0 and nadj < 0:
        raise ValueError("CHEBFUN:LINOP:adjoint:boundaryconditions: too "
                         "many boundary conditions.")
    # fU: constraints applied to the Chebyshev basis of each variable;
    # endVals: endpoint derivative values of that basis.
    fU = np.zeros((nbcs, 2 * sdor), dtype=complex)
    endL_blocks, endR_blocks = [], []
    col = 0
    for ii in range(nin):
        ndeg = 2 * dor[ii]
        Ul, Ur, polys = _end_value_rows(dom, ndeg, max(dor[ii], 1))
        Ul = Ul[:dor[ii], :]
        Ur = Ur[:dor[ii], :]
        for p, T in enumerate(polys):
            funcs = [T if j == ii else 0.0 * T for j in range(nin)]
            for r, (row, _val) in enumerate(cons):
                fU[r, col + p] = _apply_row(row, funcs)
        endL_blocks.append(Ul)
        endR_blocks.append(Ur)
        col += ndeg

    def _blkdiag(bl):
        m = sum(x.shape[0] for x in bl)
        n = sum(x.shape[1] for x in bl)
        out = np.zeros((m, n))
        r = c = 0
        for x in bl:
            out[r:r + x.shape[0], c:c + x.shape[1]] = x
            r += x.shape[0]
            c += x.shape[1]
        return out
    endVals = np.vstack([_blkdiag(endL_blocks), _blkdiag(endR_blocks)])
    if nbcs > 0:
        B = np.linalg.solve(endVals.T, fU.T).T
        if np.linalg.matrix_rank(B) != nbcs:
            raise ValueError("CHEBFUN:LINOP:adjoint:boundaryconditions: "
                             "Boundary conditions of L are not linearly "
                             "independent.")
        B = _rref(np.real(B))
        B[np.abs(B - 1) < 10 * np.finfo(float).eps] = 1
        B[np.abs(B) < 10 * np.finfo(float).eps] = 0
        q, _ = np.linalg.qr(B.T, mode="complete")
        nulB = _rref(q[:, nbcs:].T)
    else:
        nulB = np.eye(2 * sdor)
    compML = [[_compmat(a, list(reversed(_block_coeffs(blocks[i][j], dom))))
               for j in range(nin)] for i in range(nout)]
    compMR = [[_compmat(b, list(reversed(_block_coeffs(blocks[i][j], dom))))
               for j in range(nin)] for i in range(nout)]

    def _cell2mat(cells):
        return np.block([[np.atleast_2d(c) for c in row] for row in cells])
    ML = _cell2mat(compML)
    MR = _cell2mat(compMR)
    compM = np.zeros((ML.shape[0] + MR.shape[0], ML.shape[1] + MR.shape[1]))
    compM[:ML.shape[0], :ML.shape[1]] = -ML
    compM[ML.shape[0]:, ML.shape[1]:] = MR
    Bstar = _rref(nulB @ compM)
    Bstar[np.abs(Bstar - 1) < 10 * np.finfo(float).eps] = 1
    Bstar[np.abs(Bstar) < 10 * np.finfo(float).eps] = 0
    nadjbcs = Bstar.shape[0]
    star_types = np.full(nadjbcs, 2)
    for i in range(nadjbcs):
        if np.max(np.abs(Bstar[i, :sdor])) == 0:
            star_types[i] = 1
        elif np.max(np.abs(Bstar[i, sdor:])) == 0:
            star_types[i] = 0
    order = np.argsort(star_types, kind="stable")
    Bstar, star_types = Bstar[order], star_types[order]
    Lrows, Rrows, Mrows = [], [], []
    Lstar.constraint = []
    for i in range(nadjbcs):
        row = Bstar[i]
        if np.max(np.abs(row)) == 0:
            continue
        Lstar = Lstar.add_constraint(
            _bc_row_functional(row, dor, dom, star_types[i]), 0.0)
        g = _bc_row_callable(row, dor, dom, star_types[i])
        (Lrows if star_types[i] == 0 else Rrows if star_types[i] == 1
         else Mrows).append(g)

    def _bundle(rows, mixed=False):
        if not rows:
            return None
        if len(rows) == 1:
            return rows[0]
        return lambda *vs: [g(*vs) for g in rows]
    return Lstar, op, _bundle(Lrows), _bundle(Rrows), _bundle(Mrows)


def svds(L: BlockLinop, k: int = 6, bc_type: str = "bvp", n: int = 129,
         bvp_tol: float = 5e-13):
    """Largest ``k`` singular values (and singular functions) of a block
    linop with side constraints (MATLAB ``svds(L, k, bcType, pref)``):
    the eigenvalues of the super-operator ``[0 L*; L 0]`` nearest zero,
    inverted and sorted.  Returns ``(U, S, V)`` with ``U``/``V`` lists
    of Chebfuns (one per singular triplet, normalised) and ``S`` the
    diagonal matrix of singular values.

    Provenance
    ----------
    MATLAB source : @linop/svds.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.chebfun1d.chebfun import chebfun
    dom = tuple(float(v) for v in L.domain)
    a, b = dom[0], dom[-1]
    Lstar, _op, _l, _r, _m = linop_adjoint(L, bc_type)
    m, nn = L.nrows, L.ncols
    nm = nn + m
    z = chebfun(lambda x: 0.0 * x, domain=(a, b))
    blocks = [[mult(z, (a, b)) for _ in range(nm)] for _ in range(nm)]
    for ii in range(m):
        for jj in range(nn):
            blocks[nn + ii][jj] = L.A.blocks[ii][jj]
    for ii in range(nn):
        for jj in range(m):
            blocks[ii][nn + jj] = Lstar.A.blocks[ii][jj]
    superA = BlockLinop(blocks, domain=dom)
    zf = zero_functional((a, b))
    for row, val in L.constraint:
        superA = superA.add_constraint(list(row) + [zf] * m, 0.0)
    for row, val in Lstar.constraint:
        superA = superA.add_constraint([zf] * nn + list(row), 0.0)
    dor = max(max(_block_order(L.A.blocks[i][j]) for j in range(nn))
              for i in range(m))
    nc = len(L.constraint)
    nulA = dor * nn - nc
    nsvals = 2 + 2 * k - abs(nulA)
    lam, vecs = superA.eigs(k=nsvals, sigma=0.0, n=n, rayleigh=True)
    lam = np.asarray(lam)
    if np.max(np.abs(np.imag(lam))) > 0 and np.max(np.abs(np.imag(lam))) \
            > 1e-8 * np.max(np.abs(lam)):
        raise ValueError("CHEBFUN:LINOP:svds:real: Computed singular values "
                         "are not strictly real.")
    Dv = np.array(np.real(lam), dtype=float)
    Dv[np.abs(Dv) < bvp_tol * np.max(np.abs(Dv))] = 0.0
    with np.errstate(divide="ignore"):
        inv = np.where(Dv != 0, 1.0 / np.where(Dv == 0, 1.0, Dv), np.inf)
    idx = np.argsort(-inv, kind="stable")
    inv = inv[idx]
    vecs = [vecs[i] for i in idx]
    sel = list(range(k - 1, -1, -1))
    svals = 1.0 / inv[sel]
    U, V = [], []
    for i in sel:
        if hasattr(vecs[i], "blocks"):
            comps = [b for row in vecs[i].blocks for b in row]
        else:
            comps = list(vecs[i])
        v = comps[:nn]
        u = comps[nn:]
        nv = math.sqrt(sum(float(np.real(np.asarray(c.norm(2)))) ** 2
                           for c in v))
        nu = math.sqrt(sum(float(np.real(np.asarray(c.norm(2)))) ** 2
                           for c in u))
        nv = 1.0 if nv < bvp_tol else nv
        nu = 1.0 if nu < bvp_tol else nu
        V.append([c * (1.0 / nv) for c in v] if nn > 1 else v[0] * (1.0 / nv))
        U.append([c * (1.0 / nu) for c in u] if m > 1 else u[0] * (1.0 / nu))
    return U, np.diag(svals), V
