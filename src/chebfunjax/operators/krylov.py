"""Operator Krylov methods for ODEs: pcg and minres on chebops.

Port of MATLAB Chebfun's ``@chebop/pcg.m`` and ``@chebop/minres.m``:
integral-preconditioned Krylov iterations applied directly to chebfuns.
The chebop L must be a self-adjoint second-order operator
``L(u) = (a(x)u')' + c(x)u`` (indefinite ``c`` allowed for minres) with
scalar Dirichlet boundary conditions; the preconditioner is the
indefinite integral ``R1 = cumsum`` with adjoint
``R2(u) = sum(u) - cumsum(u)``.

Provenance
----------
MATLAB source : @chebop/pcg.m, @chebop/minres.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

__all__ = ["pcg", "minres", "gmres"]


def _ip(f, g):
    """L2 inner product of two chebfuns."""
    return float((f * g).sum())


def _setup(L, f, tol):
    """Shared setup: coefficient mining, preconditioned operator T,
    and the inhomogeneity shift z (MATLAB pcg/minres preamble)."""
    import inspect

    from chebfunjax.chebfun1d.chebfun import Chebfun
    from chebfunjax.domain import Domain

    dom = (float(f.domain.a), float(f.domain.b))
    xf = Chebfun.identity(Domain(dom))

    left_bc = 0.0
    right_bc = 0.0
    op = L
    if hasattr(L, "op"):
        if L._lbc_raw is not None:
            if not isinstance(L._lbc_raw, (int, float)):
                raise ValueError(
                    "pcg/minres support only Dirichlet (scalar) "
                    "boundary conditions")
            left_bc = float(L._lbc_raw)
        if L._rbc_raw is not None:
            if not isinstance(L._rbc_raw, (int, float)):
                raise ValueError(
                    "pcg/minres support only Dirichlet (scalar) "
                    "boundary conditions")
            right_bc = float(L._rbc_raw)
        op = L.op
    nargs = len(inspect.signature(op).parameters)
    Lop = (lambda u: op(xf, u)) if nargs == 2 else op

    def R1(u):
        return u.cumsum()

    def R2(u):
        return u.sum() - u.cumsum()

    def Pi(g):
        return g - g.sum() * (1.0 / (dom[1] - dom[0]))

    # Data-mine the coefficients: c = L[1], and a from L[x^2/2], L[x].
    one = 1.0 + 0.0 * xf
    c = Lop(one)
    a = (-Lop(xf**2 * 0.5) - (-Lop(xf) + c * xf) * xf
         + c * (xf**2 * 0.5))

    def Lc(v):
        return -(a * v.diff()).diff() + c * v

    def T(v):
        return Pi(R2(Lc(R1(v))))

    # Inhomogeneity shift z so the working RHS lies in the right space.
    R2f = R2(f)
    PiR2f = Pi(R2f)
    endpts = jnp.asarray([dom[0], dom[1]])
    need_shift = (float((R2f - PiR2f).norm()) > tol
                  or abs(left_bc) > tol or abs(right_bc) > tol)
    if need_shift:
        basis = [xf**k for k in range(5)]
        A = np.zeros((4, 5))
        for j, bj in enumerate(basis):
            v = R1(R2(Lc(bj)))
            A[0:2, j] = np.asarray(v(endpts))
            A[2:4, j] = np.asarray(bj(endpts))
        b = np.concatenate([np.asarray(R1(R2f)(endpts)),
                            [left_bc, right_bc]])
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        z = None
        for cj, bj in zip(coef, basis):
            t = float(cj) * bj
            z = t if z is None else z + t
        g = Pi(R2f - R2(Lc(z)))
    else:
        g = PiR2f
        z = 0.0 * f
    return T, R1, Pi, g, z, c, a


def pcg(L, f, tol: float = 1e-10, maxit: int = 100, u0=None,
        full_output: bool = False):
    """Preconditioned conjugate gradients for a self-adjoint chebop.

    Solves ``L(u) = f`` with Dirichlet BCs, iterating directly on
    chebfuns with the indefinite-integral preconditioner.  Returns the
    solution chebfun (MATLAB ``pcg(L, f)``).
    """
    n2f = float(f.norm())
    T, R1, Pi, g, z, _c, _a = _setup(L, f, tol)

    u = 0.0 * f if u0 is None else u0
    Tu = T(u) if u0 is not None else u

    tolf = tol * float(g.norm())
    r = g - Tu
    p = r
    normr = float(r.norm())
    resvec = [normr]
    if normr <= tolf:
        out = R1(Pi(u)) + z
        return ((out, 0, normr / max(n2f, 1e-300), 0, resvec)
                if full_output else out)

    umin, normrmin = u, normr
    stag = 0
    moresteps = 0
    rho = _ip(r, r)
    flag = 1
    for _ii in range(1, maxit + 1):
        Lp = T(p)
        denom = _ip(p, Lp)
        if denom == 0 or not math.isfinite(denom):
            break
        alpha = rho / denom
        # Simplify each iterate: MATLAB's chebfun arithmetic trims
        # trailing coefficients automatically; without it the Krylov
        # vectors' lengths accumulate and iterations crawl.
        u = (u + alpha * p).simplify()
        r = (r - alpha * Lp).simplify()
        rho_new = _ip(r, r)
        beta = rho_new / rho
        p = (r + beta * p).simplify()
        rho = rho_new
        if rho == 0 or not math.isfinite(rho):
            break
        if float(p.norm()) * abs(alpha) < 2.2e-16 * float(u.norm()):
            stag += 1
        else:
            stag = 0
        normr = math.sqrt(max(rho, 0.0))
        resvec.append(normr)
        if normr <= tolf or stag >= 3 or moresteps:
            r = g - T(u)
            normr_act = float(r.norm())
            if normr_act <= tolf:
                flag = 0
                break
            if stag >= 3 and moresteps == 0:
                stag = 0
            moresteps += 1
            if moresteps >= 5:
                break
        if normr < normrmin:
            normrmin, umin = normr, u
        if stag >= 3:
            break
    it = _ii
    if flag != 0:
        r_comp = g - T(umin)
        if float(r_comp.norm()) <= normr:
            u = umin
            normr = float(r_comp.norm())
    out = R1(u) + z
    if full_output:
        return out, flag, normr / max(n2f, 1e-300), it, resvec
    return out


def minres(L, f, tol: float = 1e-10, maxit: int = 100,
           full_output: bool = False):
    """MINRES for a (possibly indefinite) self-adjoint chebop.

    Same preconditioning framework as :func:`pcg`; the Lanczos/Givens
    recurrence follows MATLAB @chebop/minres.m.
    """
    T, R1, Pi, g, z, _c, _a = _setup(L, f, tol)

    n2f = float(f.norm())
    u = 0.0 * f
    tolg = tol * float(g.norm())
    r = g
    normr = float(r.norm())
    resvec = [normr]
    if normr <= tolg:
        out = R1(Pi(u)) + z
        return ((out, 0, normr / max(n2f, 1e-300), 0, resvec)
                if full_output else out)

    vold = r
    v = vold
    beta1 = _ip(vold, v)
    if beta1 <= 0:
        return R1(Pi(u)) + z
    beta1 = math.sqrt(beta1)
    snprod = beta1
    vv = v * (1.0 / beta1)
    v = T(vv)
    Amvv = v
    alpha = _ip(vv, v)
    v = v - (alpha / beta1) * vold
    # Local reorthogonalization
    numer = _ip(vv, v)
    denom = _ip(vv, vv)
    v = v - (numer / denom) * vv
    volder = vold
    vold = v
    betaold = beta1
    beta = _ip(v, v)
    if beta < 0:
        return R1(Pi(u)) + z
    beta = math.sqrt(beta)
    gammabar = alpha
    epsilon = 0.0
    deltabar = beta
    gamma = math.sqrt(gammabar**2 + beta**2)
    mold = 0.0 * f
    Amold = mold
    m = vv * (1.0 / gamma)
    Am = Amvv * (1.0 / gamma)
    cs = gammabar / gamma
    sn = beta / gamma
    u = u + (snprod * cs) * m
    snprod = snprod * sn
    normr = abs(snprod)
    resvec.append(normr)
    if normr <= tolg:
        out = R1(Pi(u)) + z
        return ((out, 0, normr / max(n2f, 1e-300), 1, resvec)
                if full_output else out)

    stag = 0
    for _ii in range(2, maxit + 1):
        vv = (v * (1.0 / beta)).simplify()
        v = T(vv)
        Amolder = Amold
        Amold = Am
        Am = v
        v = v - (beta / betaold) * volder
        alpha = _ip(vv, v)
        v = (v - (alpha / beta) * vold).simplify()
        volder = vold
        vold = v
        betaold = beta
        beta = _ip(v, v)
        if beta < 0:
            break
        beta = math.sqrt(beta)
        delta = cs * deltabar + sn * alpha
        molder = mold
        mold = m
        m = (vv - delta * mold - epsilon * molder).simplify()
        Am = (Am - delta * Amold - epsilon * Amolder).simplify()
        gammabar = sn * deltabar - cs * alpha
        epsilon = sn * beta
        deltabar = -cs * beta
        gamma = math.sqrt(gammabar**2 + beta**2)
        m = m * (1.0 / gamma)
        Am = Am * (1.0 / gamma)
        cs = gammabar / gamma
        sn = beta / gamma
        u = (u + (snprod * cs) * m).simplify()
        snprod = snprod * sn
        normr = abs(snprod)
        resvec.append(normr)
        if normr <= tolg:
            break
        if abs(snprod * cs) * float(m.norm()) \
                < 2.2e-16 * float(u.norm()):
            stag += 1
            if stag >= 3:
                break
        else:
            stag = 0
    out = R1(Pi(u)) + z
    if full_output:
        flag = 0 if normr <= tolg else 1
        return out, flag, normr / max(n2f, 1e-300), _ii, resvec
    return out


def gmres(L, f, tol: float = 1e-10, maxit: int = 100):
    """GMRES for a chebop via the same integral preconditioning.

    Provenance
    ----------
    MATLAB source : @chebop/gmres.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.chebfun1d.chebfun import gmres as _cheb_gmres

    T, R1, Pi, g, z, _c, _a = _setup(L, f, tol)
    u, _flag = _cheb_gmres(T, g, tol=tol, maxiter=maxit)
    return R1(Pi(u)) + z
