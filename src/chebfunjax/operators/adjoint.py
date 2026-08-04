"""Adjoints of linear chebops (scalar case).

Port of MATLAB Chebfun's ``@chebop/adjoint.m`` and
``@linop/linopAdjoint.m`` (adjointFormal, adjointBCs, compmat) for
scalar differential operators with endpoint boundary conditions.

The formal adjoint of ``L u = sum_k a_k u^(k)`` is

    L* v = sum_l [ sum_{k>=l} (-1)^k C(k,l) conj(a_k)^{(k-l)} ] v^(l),

and the adjoint boundary conditions are the null space of the original
conditions pushed through the boundary bilinear form (complementarity
matrix from integration by parts).

Provenance
----------
MATLAB source : @chebop/adjoint.m, @linop/linopAdjoint.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

import math

# uses-numpy: small dense linear algebra (rref, QR null space, and the
# boundary complementarity matrix) on matrices of size 2*diffOrder; these
# are host-side setup, not JIT-compiled array code.
import jax.numpy as jnp
import numpy as np

__all__ = ["adjoint"]


def _rref(A, tol=1e-10):
    """Reduced row echelon form (MATLAB rref) for small dense arrays."""
    A = np.array(A, dtype=float)
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        piv = r + int(np.argmax(np.abs(A[r:, c])))
        if abs(A[piv, c]) <= tol:
            continue
        A[[r, piv]] = A[[piv, r]]
        A[r] = A[r] / A[r, c]
        for i in range(rows):
            if i != r:
                A[i] = A[i] - A[i, c] * A[r]
        r += 1
    return A


def _extract_coeff_chebfuns(N, dor):
    """Coefficient chebfuns a_0..a_dor of the linear operator N.op.

    Monomial probes L[x^k/k!] = sum_{j<=k} a_j x^{k-j}/(k-j)!,
    forward-substituted with chebfun arithmetic.
    """
    import inspect

    from chebfunjax.chebfun1d.chebfun import Chebfun
    from chebfunjax.domain import Domain

    a, b = (float(N.domain[0]), float(N.domain[-1]))
    dom = Domain((a, b))
    xf = Chebfun.identity(dom)
    nargs = len(inspect.signature(N.op).parameters)

    def apply_op(u):
        return N.op(xf, u) if nargs > 1 else N.op(u)

    op0 = apply_op(0.0 * xf)
    outs = []
    for k in range(dor + 1):
        mono = xf**k * (1.0 / math.factorial(k)) if k > 0 \
            else 1.0 + 0.0 * xf
        outs.append(apply_op(mono))

    coeffs = []
    for k in range(dor + 1):
        ck = outs[k] - op0
        for j in range(k):
            ck = ck - coeffs[j] * (xf ** (k - j)
                                   * (1.0 / math.factorial(k - j)))
        coeffs.append(ck.simplify())
    return coeffs, xf, dom


def _compmat(x, coeffs):
    """Complementarity matrix at endpoint x (MATLAB compmat).

    ``coeffs[j]`` is the coefficient chebfun of u^(j); n = len - 1.
    """
    n = len(coeffs) - 1
    if n == 0:
        return np.zeros((1, 1))
    A = np.zeros((n, n))
    for kk in range(n, 0, -1):
        cf = coeffs[kk]
        for ii in range(kk):
            for jj in range(kk - ii):
                d = kk - ii - jj - 1
                val = float(np.asarray(
                    cf.diff(d)(jnp.asarray(float(x)))))
                A[ii, jj] += ((-1) ** (kk - ii - 1)
                              * math.comb(kk - ii - 1, jj) * val)
    return A


def _bc_functional_rows(N, dor, dom):
    """Express N's boundary conditions as rows over the endpoint basis
    [u(a), .., u^(dor-1)(a), u(b), .., u^(dor-1)(b)] (matrix B).

    MATLAB probes the constraint functionals on the first 2*dor
    Chebyshev polynomials and solves against their endpoint data.
    """
    from chebfunjax.chebfun1d.chebfun import Chebfun
    from chebfunjax.utils.polynomials import chebpoly

    a, b = float(dom.a), float(dom.b)
    nb = 2 * dor
    U = []
    for j in range(nb):
        cf = Chebfun.from_coeffs(jnp.asarray(chebpoly(j)))
        U.append(cf.new_domain((a, b)))

    def _bc_vals(spec, x0, u):
        """Homogeneous action of a BC spec on chebfun u (list of floats)."""
        if spec is None:
            return []
        if isinstance(spec, (int, float)):
            return [float(np.asarray(u(jnp.asarray(x0))))]
        if isinstance(spec, (list, tuple)):
            return [float(np.asarray(u.diff(i)(jnp.asarray(x0))))
                    for i in range(len(spec))]
        out = spec(u)
        if not isinstance(out, (list, tuple)):
            out = [out]
        vals = []
        for o in out:
            if isinstance(o, (int, float)):
                vals.append(float(o))
            else:
                vals.append(float(np.asarray(o(jnp.asarray(x0)))))
        # subtract affine part (value on the zero function)
        z = 0.0 * u
        out0 = spec(z)
        if not isinstance(out0, (list, tuple)):
            out0 = [out0]
        for i, o in enumerate(out0):
            v0 = (float(o) if isinstance(o, (int, float))
                  else float(np.asarray(o(jnp.asarray(x0)))))
            vals[i] -= v0
        return vals

    fU = []       # nbcs x nb
    for j, u in enumerate(U):
        col = (_bc_vals(N._lbc_raw, a, u)
               + _bc_vals(N._rbc_raw, b, u))
        fU.append(col)
    fU = np.asarray(fU, dtype=float).T          # (nbcs, nb)
    if fU.size == 0:
        fU = np.zeros((0, nb))

    endvals = np.zeros((nb, nb))
    for j, u in enumerate(U):
        for d in range(dor):
            endvals[d, j] = float(np.asarray(u.diff(d)(jnp.asarray(a))))
            endvals[dor + d, j] = float(np.asarray(
                u.diff(d)(jnp.asarray(b))))

    B = np.linalg.solve(endvals.T, fU.T).T       # (nbcs, 2*dor)
    if np.linalg.matrix_rank(B) != B.shape[0]:
        raise ValueError(
            "adjoint: boundary conditions of L are not linearly "
            "independent.")
    B = _rref(B)
    B[np.abs(B - 1) < 1e-10] = 1
    B[np.abs(B) < 1e-10] = 0
    return B


def adjoint(N):
    """Adjoint of a scalar linear chebop with endpoint BCs.

    Returns a new :class:`~chebfunjax.operators.chebop.Chebop` with the
    formal-adjoint operator and the adjoint boundary conditions; the
    display metadata mirrors MATLAB's (including its ``a11_k``
    coefficient labels for non-constant coefficients).
    """
    from chebfunjax.operators.chebop import Chebop

    dor = N._op_order()
    a, b = (float(N.domain[0]), float(N.domain[-1]))
    if dor == 0:
        return N

    coeffs, xf, dom = _extract_coeff_chebfuns(N, dor)

    # Formal adjoint coefficients.
    adj = [0.0 * xf for _ in range(dor + 1)]
    for k in range(dor + 1):
        for line in range(k + 1):
            term = coeffs[k].diff(k - line) * float(
                (-1) ** k * math.comb(k, line))
            adj[line] = adj[line] + term
    adj = [c.simplify() for c in adj]

    def _is_const(c, val):
        try:
            return (len(c) == 1
                    and abs(float(np.asarray(c(jnp.asarray(a)))) - val)
                    < 1e-12)
        except Exception:
            return False

    def _is_zero(c):
        try:
            return float(c.norm()) < 1e-12
        except Exception:
            return False

    # Adjoint operator handle (closure over coefficient chebfuns) and
    # MATLAB-style op display string.
    terms = []
    for k in range(dor, -1, -1):
        ck = adj[k]
        if _is_zero(ck):
            continue
        vstr = ("v" if k == 0
                else ("diff(v)" if k == 1 else f"diff(v,{k})"))
        if _is_const(ck, 1.0):
            terms.append("+" + vstr)
        elif _is_const(ck, -1.0):
            terms.append("-" + vstr)
        else:
            terms.append(f"+a11_{k}.*{vstr}")
    op_str = "".join(terms)
    if op_str.startswith("+"):
        op_str = op_str[1:]

    adj_local = list(adj)

    def op(x, v):
        out = None
        for k in range(dor + 1):
            if _is_zero(adj_local[k]):
                continue
            term = adj_local[k] * v.diff(k) if k > 0 \
                else adj_local[k] * v
            out = term if out is None else out + term
        return out

    # Adjoint boundary conditions.
    B = _bc_functional_rows(N, dor, dom)
    nbcs = B.shape[0]
    nadj = 2 * dor - nbcs

    q, _ = np.linalg.qr(B.T, mode="complete")
    nulB = _rref(q[:, nbcs:].T) if nadj > 0 else np.zeros((0, 2 * dor))

    compM = np.zeros((2 * dor, 2 * dor))
    compM[:dor, :dor] = -_compmat(a, coeffs)
    compM[dor:, dor:] = _compmat(b, coeffs)
    Bstar = _rref(nulB @ compM)
    Bstar[np.abs(Bstar - 1) < 1e-10] = 1
    Bstar[np.abs(Bstar) < 1e-10] = 0

    star_types = np.full(Bstar.shape[0], 2)
    for i in range(Bstar.shape[0]):
        if np.max(np.abs(Bstar[i, :dor])) == 0:
            star_types[i] = 1
        elif np.max(np.abs(Bstar[i, dor:])) == 0:
            star_types[i] = 0
    order = np.argsort(star_types, kind="stable")
    Bstar, star_types = Bstar[order], star_types[order]

    def _row_expr(row_half):
        """Build (callable v -> chebfun expr, display string) for one
        endpoint half-row of Bstar."""
        def make(coefs):
            def g(v, _c=tuple(coefs)):
                out = None
                for k, ck in enumerate(_c):
                    if ck == 0:
                        continue
                    t = v.diff(k) if k > 0 else v
                    t = t if ck == 1 else float(ck) * t
                    out = t if out is None else out + t
                return out
            return g
        names = []
        for k, ck in enumerate(row_half):
            if ck == 0:
                continue
            vstr = ("v" if k == 0
                    else ("diff(v)" if k == 1 else f"diff(v,{k})"))
            names.append(vstr if ck == 1 else f"{ck:g}*{vstr}")
        return make(row_half), "+".join(names)

    lbc_rows, rbc_rows = [], []
    lbc_strs, rbc_strs = [], []
    for i in range(Bstar.shape[0]):
        if star_types[i] == 0:
            g, s = _row_expr(Bstar[i, :dor])
            lbc_rows.append(g)
            lbc_strs.append(s)
        elif star_types[i] == 1:
            g, s = _row_expr(Bstar[i, dor:])
            rbc_rows.append(g)
            rbc_strs.append(s)
        else:
            raise NotImplementedError(
                "adjoint: mixed (coupled-endpoint) adjoint boundary "
                "conditions are not supported yet.")

    Ns = Chebop(domain=(a, b))
    Ns.op = op
    if lbc_rows:
        Ns.lbc = (lbc_rows[0] if len(lbc_rows) == 1
                  else (lambda v, _r=tuple(lbc_rows):
                        [g(v) for g in _r]))
    if rbc_rows:
        Ns.rbc = (rbc_rows[0] if len(rbc_rows) == 1
                  else (lambda v, _r=tuple(rbc_rows):
                        [g(v) for g in _r]))
    # Display metadata (consumed by Chebop.__repr__ when present).
    Ns._disp_op_str = op_str
    Ns._disp_var = "v"
    Ns._disp_lbc = lbc_strs
    Ns._disp_rbc = rbc_strs
    Ns._adj_coeffs = adj
    return Ns
