"""Chebop BVP solves and eigenproblems under ultraS / chebcolloc1.

MATLAB solves a chebop under ``prefs.discretization = @ultraS`` or
``@chebcolloc1`` by discretizing the linearized operator of each Newton
step in the requested space.  chebfunjax's chebop records its operator
as a closure, so this module recovers the Frechet derivative at the
current iterate in *differential form* by probing the operator with
scaled monomial perturbations (central differences), assembles typed
:class:`~chebfunjax.operators.blocks.OperatorBlock` rows from the
recovered coefficient chebfuns, linearizes the boundary conditions into
``eval_at * D^k`` functionals the same way, and solves each Newton step
through :meth:`BlockLinop.linsolve` with the requested backend.

Provenance
----------
MATLAB source : @chebop/solvebvp.m, @chebop/linearize.m with
    prefs.discretization = @ultraS | @chebcolloc1
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

import inspect
from math import factorial

import jax.numpy as jnp
import numpy as np  # uses-numpy: concrete Newton bookkeeping

_EPS_FD = 1e-6
_MAXK = 4


def _identity(dom):
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(lambda t: t, domain=dom)


def _zero_fun(dom):
    from chebfunjax.chebfun1d.chebfun import chebfun
    return chebfun(lambda t: 0.0 * t, domain=dom)


def _mono(dom, p: int):
    """The scaled reference monomial ``y^p / p!`` with
    ``y = (2x - (a+b)) / (b-a)``."""
    from chebfunjax.chebfun1d.chebfun import chebfun
    a, b = float(dom[0]), float(dom[-1])
    return chebfun(
        lambda t, _p=p: (((2.0 * t - (a + b)) / (b - a)) ** _p)
        / factorial(_p), domain=dom)


def _op_arity(fn) -> int:
    try:
        return len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return 1


def _apply_op(N, U):
    """Evaluate the chebop operator on chebfun arguments."""
    x = _identity(N.domain)
    nargs = _op_arity(N.op)
    out = N.op(x, *U) if nargs > len(U) else N.op(*U)
    if isinstance(out, (list, tuple)):
        return list(out)
    return [out]


def _vscale(f) -> float:
    # Sampled scale estimate.  Never use abs(f).max() here: abs() runs
    # an adaptive construction with rootfinding, which explodes on the
    # finite-difference-noise coefficients this module produces.
    xs = jnp.linspace(float(f.domain.breakpoints[0]),
                      float(f.domain.breakpoints[-1]), 65)
    return float(jnp.max(jnp.abs(jnp.asarray(f(xs)))))


def _res_norm(R, dom) -> float:
    a, b = float(dom[0]), float(dom[-1])
    xs = jnp.linspace(a + 1e-9 * (b - a), b - 1e-9 * (b - a), 101)
    return max(float(jnp.max(jnp.abs(jnp.asarray(r(xs))))) for r in R)


def _frechet_blocks(N, U, f_list, dom, maxk=None):
    """Linearize the operator at ``U``: typed block rows plus the
    residual chebfuns ``R_i = F_i(U) - f_i``.  ``maxk`` limits the
    per-variable probe order (pass the sniffed orders on Jacobian
    refreshes to avoid re-probing with unnecessary monomials)."""
    from chebfunjax.operators.blocks import D as _D
    from chebfunjax.operators.blocks import mult

    m = len(U)
    base = _apply_op(N, U)
    n_eq = len(base)
    R = []
    for i in range(n_eq):
        fi = f_list[i] if i < len(f_list) else 0.0
        R.append(base[i] - fi)

    h2 = 2.0 / (float(dom[-1]) - float(dom[0]))
    blocks = [[None] * m for _ in range(n_eq)]
    var_orders = [0] * m
    if maxk is None:
        maxk = [_MAXK] * m
    for j in range(m):
        # Probe with monomials (central differences: the recovered
        # coefficients are divided by h2^p, which amplifies forward-
        # difference noise past the order-sniffing threshold on wide
        # domains), then forward-substitute the coefficients.
        G = []
        for p in range(maxk[j] + 1):
            v = _mono(dom, p)
            Up = list(U)
            Um = list(U)
            Up[j] = U[j] + _EPS_FD * v
            Um[j] = U[j] - _EPS_FD * v
            Fp = _apply_op(N, Up)
            Fm = _apply_op(N, Um)
            G.append([(Fp[i] - Fm[i]) * (0.5 / _EPS_FD)
                      for i in range(n_eq)])
        monos = [_mono(dom, p) for p in range(maxk[j] + 1)]
        for i in range(n_eq):
            coeffs = []
            for p in range(maxk[j] + 1):
                acc = G[p][i]
                for k in range(p):
                    acc = acc - coeffs[k] * (h2 ** k) * monos[p - k]
                coeffs.append(acc * (1.0 / h2 ** p))
            scale = max([_vscale(c) for c in coeffs] + [1e-30])
            order = 0
            for k in range(maxk[j], -1, -1):
                if _vscale(coeffs[k]) > 1e-5 * scale:
                    order = k
                    break
            blk = None
            for k in range(order + 1):
                if _vscale(coeffs[k]) <= 1e-8 * scale and k < order:
                    continue
                term = (mult(coeffs[k]) if k == 0
                        else mult(coeffs[k]) * (_D(dom) ** k))
                blk = term if blk is None else blk + term
            if blk is None:
                blk = mult(_zero_fun(dom))
            blocks[i][j] = blk
            var_orders[j] = max(var_orders[j], order)
    return blocks, R, var_orders


def _bc_component_value(g_out, pt: float) -> float:
    """A BC component may be a chebfun (evaluate at the endpoint) or
    already a scalar (the condition contained the point evaluation
    itself, e.g. ``N.bc = @(x,u) [u(-1); u(1)]``)."""
    if callable(g_out):
        return float(g_out(jnp.asarray(float(pt))))
    return float(g_out)


def _recover_functional(gfun, U, j, pt, K, dom):
    """Coefficients ``alpha_k`` with ``F[v] = sum alpha_k v^(k)(pt)``
    of the linearized boundary condition, by monomial probing."""
    h2 = 2.0 / (float(dom[-1]) - float(dom[0]))
    a, b = float(dom[0]), float(dom[-1])
    y0 = (2.0 * pt - (a + b)) / (b - a)
    n_probe = K + 2
    F = np.zeros(n_probe)
    for p in range(n_probe):
        v = _mono(dom, p)
        Up = list(U)
        Um = list(U)
        Up[j] = U[j] + _EPS_FD * v
        Um[j] = U[j] - _EPS_FD * v
        dg = (gfun(Up) - gfun(Um)) * (0.5 / _EPS_FD)
        F[p] = _bc_component_value(dg, pt)
    # v_p^(k)(pt) = h2^k y0^(p-k) / (p-k)!  (0 for k > p).
    M = np.zeros((K + 1, K + 1))
    for p in range(K + 1):
        for k in range(p + 1):
            M[p, k] = (h2 ** k) * (y0 ** (p - k)) / factorial(p - k)
    alphas = np.linalg.solve(M, F[: K + 1])
    # Validation probe: does the recovered functional predict F[K+1]?
    pred = sum(alphas[k] * (h2 ** k) * (y0 ** (K + 1 - k))
               / factorial(K + 1 - k) for k in range(K + 1))
    scale = max(np.max(np.abs(F)), 1.0)
    ok = abs(pred - F[K + 1]) < 1e-4 * scale
    return alphas, ok


def _functional_entry(alphas, pt, dom, threshold):
    from chebfunjax.operators.blocks import D as _D
    from chebfunjax.operators.blocks import eval_at, zero_functional
    entry = None
    for k, al in enumerate(alphas):
        if abs(al) <= threshold:
            continue
        ev = eval_at(float(pt), dom)
        term = (float(al) * ev if k == 0
                else float(al) * (ev * (_D(dom) ** k)))
        entry = term if entry is None else entry + term
    return entry if entry is not None else zero_functional(dom)


def _bc_residuals(N, U, dom):
    """Absolute boundary-condition residual values at ``U`` (no
    linearization -- used by the damping line search)."""
    a, b = float(dom[0]), float(dom[-1])
    m = len(U)
    out = []

    def add(spec, pt):
        o = (spec(*U) if _op_arity(spec) == m
             else spec(_identity(dom), *U))
        comps = list(o) if isinstance(o, (list, tuple)) else [o]
        for c in comps:
            if callable(c):
                out.append(abs(_bc_component_value(c, pt)))
            else:
                out.append(abs(float(c)))

    for spec, pt in ((getattr(N, "lbc", None), a),
                     (getattr(N, "rbc", None), b)):
        if spec is None:
            continue
        if callable(spec):
            add(spec, pt)
        else:
            vals = (list(spec) if isinstance(spec, (list, tuple))
                    else [spec])
            for j, v in enumerate(vals):
                out.append(abs(float(U[j](jnp.asarray(pt)))
                               - float(v)))
    bc = getattr(N, "bc", None)
    if bc is not None and callable(bc):
        add(bc, a)
    return out


def _collect_bcs(N, U, var_orders, dom):
    """Linearized boundary-condition rows ``(row_list, -residual)``."""
    a, b = float(dom[0]), float(dom[-1])
    m = len(U)
    rows = []

    def add_callable(spec, pts_to_try):
        out = spec(*U) if _op_arity(spec) == m else spec(_identity(dom),
                                                         *U)
        comps = list(out) if isinstance(out, (list, tuple)) else [out]
        for ci in range(len(comps)):
            def g(W, _ci=ci, _spec=spec):
                o = (_spec(*W) if _op_arity(_spec) == m
                     else _spec(_identity(dom), *W))
                return o[_ci] if isinstance(o, (list, tuple)) else o

            placed = False
            for pt in pts_to_try:
                row_list = []
                worst_scale = 0.0
                all_ok = True
                per_var = []
                for j in range(m):
                    K = max(var_orders[j] - 1, 0) + 1
                    alphas, ok = _recover_functional(
                        g, U, j, pt, K, dom)
                    per_var.append(alphas)
                    worst_scale = max(worst_scale,
                                      float(np.max(np.abs(alphas))))
                    all_ok = all_ok and ok
                if not all_ok or worst_scale == 0.0:
                    continue
                thr = 1e-6 * max(worst_scale, 1.0)
                for j in range(m):
                    row_list.append(_functional_entry(
                        per_var[j], pt, dom, thr))
                val = -_bc_component_value(comps[ci], pt)
                rows.append((row_list, val))
                placed = True
                break
            if not placed:
                raise NotImplementedError(
                    "chebop altdisc: could not linearize a boundary "
                    "condition into endpoint functionals.")

    for spec, pt in ((getattr(N, "lbc", None), a),
                     (getattr(N, "rbc", None), b)):
        if spec is None:
            continue
        if callable(spec):
            add_callable(spec, (pt,))
        else:
            from chebfunjax.operators.blocks import (
                eval_at,
                zero_functional,
            )
            vals = (list(spec) if isinstance(spec, (list, tuple))
                    else [spec])
            for j, v in enumerate(vals):
                row_list = [eval_at(pt, dom) if jj == j
                            else zero_functional(dom)
                            for jj in range(m)]
                res = float(U[j](jnp.asarray(pt))) - float(v)
                rows.append((row_list, -res))
    bc = getattr(N, "bc", None)
    if bc is not None and callable(bc):
        add_callable(bc, (a, b))
    return rows


def solve_bvp_altdisc(N, f=0.0, discretization: str = "ultraS",
                      n: int | None = None, tol: float = 1e-10,
                      max_iter: int = 30):
    """Solve the chebop BVP with a Newton iteration whose linear solves
    run under the requested discretization.

    Provenance
    ----------
    MATLAB source : @chebop/solvebvp.m with prefs.discretization
    Chebfun commit: 7574c77
    """
    from chebfunjax.operators.blocklinop import linop as _mk_linop
    from chebfunjax.operators.chebmatrix import ChebMatrix

    dom = tuple(float(v) for v in N.domain)
    m = N._n_vars()
    if isinstance(f, (list, tuple)):
        f_list = list(f)
    else:
        f_list = [f] * m

    # Seed the Newton iteration.  A converged chebcolloc2 solution is
    # the natural continuation seed (the iteration then refines to the
    # fixed point of the REQUESTED discretization -- the last Newton
    # corrections are solved entirely in ultraS/chebcolloc1 space);
    # fall back to N.init or zero functions.
    from chebfunjax.chebfun1d.chebfun import Chebfun as _Chebfun
    U = None
    try:
        sol = N.solve(f)
        if isinstance(sol, _Chebfun):
            cand = [sol]
        else:
            cand = [sol[i] for i in range(m)]
        if len(cand) == m and all(hasattr(c, "domain")
                                  for c in cand):
            U = cand
    except Exception:
        U = None
    if U is None and N.init is not None:
        init = (list(N.init) if isinstance(N.init, (list, tuple))
                else [N.init])
        U = [u if not isinstance(u, (int, float))
             else _zero_fun(dom) + float(u) for u in init]
        while len(U) < m:
            U.append(_zero_fun(dom))
    elif U is None:
        U = [_zero_fun(dom) for _ in range(m)]

    nn = int(n) if n is not None else 65

    def total_residual(Ut):
        base = _apply_op(N, Ut)
        R = [base[i] - (f_list[i] if i < len(f_list) else 0.0)
             for i in range(len(base))]
        return _res_norm(R, dom) + sum(_bc_residuals(N, Ut, dom))

    from chebfunjax.operators.altdisc import system_matrices

    # Chord Newton: the discretized Jacobian is refreshed only when the
    # step quality degrades; between refreshes each iteration reuses the
    # factor-ready SystemDisc, so it costs one residual evaluation and
    # one dense solve.
    sd = None
    var_orders = [_MAXK] * m
    maxk = None
    res_prev = None
    stale = True
    for _it in range(max_iter):
        if stale:
            blocks, R, var_orders = _frechet_blocks(
                N, U, f_list, dom, maxk=maxk)
            maxk = [max(o, 1) for o in var_orders]
            L = _mk_linop(ChebMatrix(blocks))
            bc_rows = _collect_bcs(N, U, var_orders, dom)
            for row_list, _val in bc_rows:
                L = L.add_constraint(row_list, 0.0)
            sd = system_matrices(L, nn, discretization)
            n_cont = len(sd.L.continuity)
            stale = False
        else:
            base = _apply_op(N, U)
            R = [base[i] - (f_list[i] if i < len(f_list) else 0.0)
                 for i in range(len(base))]
            bc_rows = _collect_bcs(N, U, var_orders, dom)
        bc_vals = [val for _row, val in bc_rows]
        res_now = _res_norm(R, dom) + sum(abs(v) for v in bc_vals)
        scale = max([_vscale(u) for u in U] + [1.0])
        if res_now < tol * max(scale, 1.0):
            break
        sd.con_vals = [0.0] * n_cont + bc_vals
        b = sd.rhs([-r for r in R])
        v = np.linalg.solve(np.asarray(sd.A), b)
        dU = sd.recover(v)
        # Negligible correction: the iterate is the fixed point of this
        # discretization to rounding; stop before finite-difference
        # noise in the correction pollutes the residual floor.
        if max(_vscale(d) for d in dU) < 1e-12 * scale:
            break
        lam = 1.0
        while lam > 1.0 / 64:
            trial = [U[j] + lam * dU[j] for j in range(m)]
            try:
                res_trial = total_residual(trial)
            except Exception:
                res_trial = np.inf
            if res_trial < (1.0 - 0.25 * lam) * res_now \
                    or res_trial < tol:
                break
            lam *= 0.5
        if lam < 1.0:
            stale = True
        U = [(U[j] + lam * dU[j]).simplify()
             if hasattr(U[j] + lam * dU[j], "simplify")
             else U[j] + lam * dU[j] for j in range(m)]
        if res_prev is not None and res_now > 0.5 * res_prev:
            # Slow progress on the chord: refresh the Jacobian at the
            # new iterate rather than giving up -- near a solution the
            # refreshed Newton step restores quadratic contraction.
            stale = True
        res_prev = res_now
    return U


def eigs_generalized_altdisc(N, B, k: int, n: int,
                             discretization: str, sort="LR"):
    """Generalized chebop eigenproblem under ultraS / chebcolloc1.

    Provenance
    ----------
    MATLAB source : @chebop/eigs.m with prefs.discretization
    Chebfun commit: 7574c77
    """
    import scipy.linalg as sla

    from chebfunjax.operators.altdisc import system_matrices
    from chebfunjax.operators.blocklinop import linop as _mk_linop
    from chebfunjax.operators.blocks import OperatorBlock
    from chebfunjax.operators.chebmatrix import ChebMatrix

    dom = tuple(float(v) for v in N.domain)
    m = N._n_vars()
    U0 = [_zero_fun(dom) for _ in range(m)]
    blocksA, _RA, var_orders = _frechet_blocks(N, U0, [0.0] * m, dom)
    LA = _mk_linop(ChebMatrix(blocksA))
    for row_list, _val in _collect_bcs(N, U0, var_orders, dom):
        LA = LA.add_constraint(row_list, 0.0)
    blocksB, _RB, _vo = _frechet_blocks(B, U0, [0.0] * m, dom)
    LB = _mk_linop(ChebMatrix(blocksB))
    rmin = [max((blk.order for blk in LB.A.blocks[i]
                 if isinstance(blk, OperatorBlock)), default=0)
            for i in range(LB.nrows)]

    def spectrum(nn):
        sd = system_matrices(LA, nn, discretization,
                             row_order_min=rmin)
        lam = sla.eig(np.asarray(sd.A), np.asarray(sd.mass(LB)),
                      right=False)
        return lam[np.isfinite(lam) & (np.abs(lam) < 1e8)], sd

    # Spurious-mode removal: keep only eigenvalues that reappear at a
    # second resolution (the same two-resolution agreement filter the
    # chebcolloc2 generalized path uses -- spurious barycentric modes
    # move with n, physical ones do not).
    lam1, sd1 = spectrum(n)
    lam2, _sd2 = spectrum(max(48, (3 * n) // 4))
    keep = np.asarray([np.min(np.abs(lam2 - lv))
                       < 1e-3 * (1.0 + np.abs(lv)) for lv in lam1])
    lam = lam1[keep]
    if isinstance(sort, str) and sort.upper() == "LR":
        order = np.argsort(-lam.real)
    elif isinstance(sort, str) and sort.upper() == "SR":
        order = np.argsort(lam.real)
    else:
        order = np.argsort(np.abs(lam))
    lam = lam[order[:k]]
    A1 = np.asarray(sd1.A)
    B1 = np.asarray(sd1.mass(LB))
    vecs = LA._altdisc_vecs(lam, A1, B1, sd1)
    return vecs, jnp.asarray(lam)


class LinearizedChebop:
    """The Frechet derivative of a chebop about a state ``U``, ready to
    solve ``J du = r`` (MATLAB ``linearize(N, u)`` followed by
    ``mldivide``).  Assembled once on the requested discretization; each
    ``solve`` is a dense linear solve with homogeneous linearized
    boundary conditions.

    Provenance
    ----------
    MATLAB source : @chebop/linearize.m, @linop/mldivide.m
    Chebfun commit: 7574c77
    """

    def __init__(self, N, U, discretization: str = "ultraS",
                 n: int = 257):
        from chebfunjax.operators.altdisc import system_matrices
        from chebfunjax.operators.blocklinop import linop as _mk_linop
        from chebfunjax.operators.chebmatrix import ChebMatrix

        dom = tuple(float(v) for v in N.domain)
        m = N._n_vars()
        if not isinstance(U, (list, tuple)):
            U = [U]
        U = list(U) + [_zero_fun(dom)] * (m - len(U))
        f_list = [0.0] * m
        blocks, _R, var_orders = _frechet_blocks(N, U, f_list, dom)
        L = _mk_linop(ChebMatrix(blocks))
        bc_rows = _collect_bcs(N, U, var_orders, dom)
        for row_list, _val in bc_rows:
            L = L.add_constraint(row_list, 0.0)
        self._sd = system_matrices(L, int(n), discretization)
        self._n_cont = len(self._sd.L.continuity)
        self._n_bc = len(bc_rows)
        self._m = m
        self._dom = dom

    def solve(self, r):
        """Solve ``J du = r`` with homogeneous linearized BCs."""
        rs = list(r) if isinstance(r, (list, tuple)) else [r]
        rs = [(_zero_fun(self._dom) + float(v))
              if isinstance(v, (int, float)) else v for v in rs]
        self._sd.con_vals = [0.0] * (self._n_cont + self._n_bc)
        b = self._sd.rhs(rs)
        v = np.linalg.solve(np.asarray(self._sd.A), b)
        dU = self._sd.recover(v)
        return dU[0] if self._m == 1 else dU

    def __truediv__(self, r):
        return self.solve(r)


def linearize_about(N, U, discretization: str = "ultraS", n: int = 257):
    """MATLAB ``linearize(N, u)``: the Frechet derivative about ``u`` as
    a solvable :class:`LinearizedChebop`."""
    return LinearizedChebop(N, U, discretization, n)
