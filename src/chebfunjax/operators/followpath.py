"""Pseudo-arclength path following for parameter-dependent chebop BVPs.

``followpath(N, lam0, ...)`` traces the solution branch of
``N(x, u, lam) = 0`` starting from ``lam = lam0``: at each step a tangent
``(t, tau)`` to the branch is computed from the augmented linearized
system, a predictor ``(u + sl*t, lam + sl*tau)`` is corrected by Newton
on the augmented system (Frechet derivative plus the tangent-normal
constraint), and the steplength is adapted (halved on retraction, doubled
after fast Newton convergence), exactly following MATLAB's algorithm.

The linear algebra runs in collocation VALUE space: the operator's
Jacobian with respect to ``u`` (finite-difference columns of the
collocation residual) and with respect to ``lam`` are assembled on an
``n``-point Chebyshev grid, the boundary conditions replace the last rows,
and the tangent-normal row uses Clenshaw-Curtis weights for the
``t' * du`` inner product.

Provenance
----------
MATLAB source : @chebop/followpath.m, @chebop/tangentBVP.m,
    @chebop/newtonBVP.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np  # uses-numpy: dense continuation linear algebra (host side)

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.domain import Domain
from chebfunjax.utils.quadrature import chebpts, chebweights

__all__ = ["followpath"]


class _Problem:
    """Collocation residual/Jacobian machinery for ``N(x, u, lam)``."""

    def __init__(self, N, n: int):
        self.N = N
        self.n = int(n)
        self.a = float(N.domain[0])
        self.b = float(N.domain[-1])
        self.dom = Domain((self.a, self.b))
        x = np.asarray(chebpts(self.n))
        self.xp = self.a + (self.b - self.a) * (x + 1.0) / 2.0
        self.w = np.asarray(chebweights(self.n)) * (self.b - self.a) / 2.0
        self.x_fun = Chebfun.identity(self.dom)
        # Boundary conditions: callables of (u, lam) (or of u), numbers,
        # or None; each yields one scalar condition at its endpoint.
        self.bcs = []
        for raw, pt in ((N._lbc_raw, self.a), (N._rbc_raw, self.b)):
            if raw is None:
                continue
            self.bcs.append((raw, pt))

    # -- evaluation helpers ------------------------------------------
    def to_fun(self, uv):
        return Chebfun.from_values(jnp.asarray(uv, dtype=jnp.float64),
                                   self.dom)

    def op_vals(self, uv, lam):
        u = self.to_fun(uv)
        out = self.N.op(self.x_fun, u, float(lam))
        if isinstance(out, (list, tuple)):
            out = out[0]
        return np.asarray(out(jnp.asarray(self.xp)), dtype=float)

    def bc_vals(self, uv, lam):
        u = self.to_fun(uv)
        vals = []
        for raw, pt in self.bcs:
            if callable(raw):
                try:
                    g = raw(u, float(lam))
                except TypeError:
                    g = raw(u)
                gs = list(g) if isinstance(g, (list, tuple)) else [g]
                for gi in gs:
                    if hasattr(gi, "funs"):
                        vals.append(float(gi(jnp.asarray(pt))))
                    else:
                        vals.append(float(gi))
            else:
                vals.append(float(u(jnp.asarray(pt))) - float(raw))
        return np.asarray(vals, dtype=float)

    def residual(self, uv, lam):
        """Collocation residual with BC rows replacing the last rows."""
        r = self.op_vals(uv, lam)
        b = self.bc_vals(uv, lam)
        nb = b.size
        if nb:
            r = r.copy()
            r[self.n - nb:] = b
        return r

    def jacobian(self, uv, lam, eps: float = 1e-7):
        """Finite-difference Jacobian [dR/du | dR/dlam]."""
        r0 = self.residual(uv, lam)
        n = self.n
        J = np.zeros((n, n + 1))
        scale = max(1.0, float(np.max(np.abs(uv))))
        h = eps * scale
        for j in range(n):
            up = np.array(uv, dtype=float)
            um = np.array(uv, dtype=float)
            up[j] += h
            um[j] -= h
            J[:, j] = (self.residual(up, lam) - self.residual(um, lam)) / (2 * h)
        hl = eps * max(1.0, abs(float(lam)))
        J[:, n] = (self.residual(uv, lam + hl)
                   - self.residual(uv, lam - hl)) / (2 * hl)
        return J, r0

    def augmented_solve(self, J, rhs, t, tau):
        """Solve [J; t'*w, tau] [du; dlam] = rhs."""
        n = self.n
        A = np.zeros((n + 1, n + 1))
        A[:n, :] = J
        A[n, :n] = np.asarray(t, dtype=float) * self.w
        A[n, n] = float(tau)
        sol = np.linalg.solve(A, np.asarray(rhs, dtype=float))
        return sol[:n], float(sol[n])


def followpath(N, lam0, *, uinit=None, direction: float = 1.0,
               measure=None, maxstepno: int = 20, stepmax: float = 0.5,
               stepmin: float = 1e-4, stopfun=None, n: int = 64,
               plotting: bool = False):
    """Trace the solution branch of ``N(x, u, lam) = 0`` from ``lam0``.

    Parameters
    ----------
    N : Chebop
        Operator ``lambda x, u, lam: ...`` with ``lbc``/``rbc`` given as
        callables of ``(u, lam)`` (or ``u``) or numbers.
    lam0 : float
        Starting parameter value.
    uinit : Chebfun, optional
        Starting solution; computed by solving ``N`` at ``lam0`` when
        omitted.
    direction : {1, -1}
        Initial tangent orientation.
    measure : callable, optional
        ``measure(u) -> float`` recorded along the branch.
    maxstepno : int
        Maximum number of accepted continuation steps.
    stepmax, stepmin : float
        Steplength bounds.
    stopfun : callable, optional
        ``stopfun(u, lam) -> bool`` terminates the run when true.
    n : int
        Collocation size for the continuation linear algebra.

    Returns
    -------
    (uquasi, lamvec, mvec, lamfun, mfun)
        Solutions along the branch (list of Chebfuns), the parameter
        values, the measures (empty without ``measure``), and Chebfun
        splines of ``lam`` and the measure over ``[0, 1]``.

    Provenance
    ----------
    MATLAB source : @chebop/followpath.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.operators.chebop import Chebop

    P = _Problem(N, n)
    lam = float(lam0)

    # Initial solution.
    if uinit is None:
        Ninit = Chebop(lambda x, u: N.op(x, u, lam), domain=N.domain)
        lb, rb = N._lbc_raw, N._rbc_raw

        def _fix(raw):
            if callable(raw):
                def _g(u, _raw=raw):
                    try:
                        return _raw(u, lam)
                    except TypeError:
                        return _raw(u)
                return _g
            return raw
        Ninit.lbc = _fix(lb)
        Ninit.rbc = _fix(rb)
        uinit = Ninit.solve(0.0)
        if not hasattr(uinit, "funs"):
            uinit = uinit[0]
    uv = np.asarray(uinit(jnp.asarray(P.xp)), dtype=float)

    def _measure(u):
        if measure is None:
            return 0.0
        return float(measure(u))

    uquasi = [P.to_fun(uv)]
    lamvec = [lam]
    mvec = [_measure(uquasi[0])]

    told = np.zeros(P.n)
    tauold = 1.0
    sl = stepmax
    retract = False
    counter = 1
    t = told
    tau = tauold
    uold, lamold = uv, lam
    upred, lampred = uv, lam

    while counter < maxstepno:
        if not retract:
            # Tangent: [J; told'*w, tauold] [t; tau] = [0; 1].
            J, _ = P.jacobian(uold, lamold)
            rhs = np.zeros(P.n + 1)
            rhs[P.n] = 1.0
            t, tau = P.augmented_solve(J, rhs, told, tauold)
            if counter == 1:
                t, tau = direction * t, direction * tau
            scale = np.sqrt(float(np.sum(P.w * t * t)) + tau ** 2)
            t, tau = t / scale, tau / scale
            upred = uold + sl * t
            lampred = lamold + sl * tau

        # Newton on the augmented system (newtonBVP).
        u_it, lam_it = np.array(upred), float(lampred)
        normu = max(np.sqrt(float(np.sum(P.w * u_it * u_it))), 1e-30)
        accept = False
        newton_iter = 0
        retract = False
        while not accept:
            J, res = P.jacobian(u_it, lam_it)
            rhs = np.concatenate([-res, [0.0]])
            du, dlam = P.augmented_solve(J, rhs, t, tau)
            u_it = u_it + du
            lam_it = lam_it + dlam
            newton_iter += 1
            if np.sqrt(float(np.sum(P.w * du * du))) / normu < 1e-3:
                accept = True
            elif newton_iter >= 5:
                retract = True
                break

        if retract:
            sl = sl / 4.0
            upred = uold + sl * t
            lampred = lamold + sl * tau
            if sl < stepmin:
                warnings.warn("followpath: steplength fell below stepmin.",
                              RuntimeWarning, stacklevel=2)
                break
            continue

        counter += 1
        u_fun = P.to_fun(u_it)
        uquasi.append(u_fun)
        lamvec.append(lam_it)
        mvec.append(_measure(u_fun))

        if newton_iter <= 3:
            sl = min(sl * 2.0, stepmax)
        told, tauold = t, tau
        uold, lamold = u_it, lam_it
        if stopfun is not None and stopfun(u_fun, lam_it):
            break

    lamvec = np.asarray(lamvec, dtype=float)
    mvec = np.asarray(mvec, dtype=float)
    s = np.linspace(0.0, 1.0, len(lamvec))
    from chebfunjax.chebfun1d.chebfun import chebfun as _mk

    def _spline(vals):
        if len(vals) < 2:
            return _mk(lambda x: 0.0 * x + float(vals[0]),
                       domain=(0.0, 1.0))
        return _mk(lambda x: jnp.interp(x, jnp.asarray(s),
                                        jnp.asarray(vals)),
                   domain=(0.0, 1.0), n=max(len(vals), 2))
    lamfun = _spline(lamvec)
    mfun = _spline(mvec) if measure is not None else _spline(np.zeros_like(lamvec))
    return uquasi, lamvec, mvec, lamfun, mfun
