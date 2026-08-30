"""Function-space Krylov solvers for self-adjoint second-order chebops.

MATLAB Chebfun's @chebop/pcg.m, minres.m, and gmres.m run Krylov
iterations directly on chebfuns: the operator
``L(u) = -(a(x) u')' + c(x) u`` with Dirichlet conditions is
preconditioned by the indefinite integral ``R1 = cumsum`` and its
adjoint ``R2 = sum - cumsum``, giving the bounded, self-adjoint
``T = Pi R2 L R1`` (``Pi`` projects out the mean).  The iterations use
L2 inner products of chebfuns; the solution is ``z + R1(Pi(v))``.

Provenance
----------
MATLAB source : @chebop/pcg.m, @chebop/minres.m, @chebop/gmres.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def _setup(N, f):
    """Extract a, c, the preconditioned operator T, and the shifted
    right-hand side g (with polynomial correction z when f is not in
    the preconditioned space or the BCs are inhomogeneous)."""
    from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
    from chebfunjax.domain import Domain

    dom = tuple(float(v) for v in N.domain)
    a0, b0 = dom[0], dom[-1]
    x = Chebfun.identity(Domain(N.domain))
    one = chebfun(lambda t: 1.0 + 0.0 * t, domain=(a0, b0))

    def L_of(u):
        return N.feval(u)

    # Data-mine the coefficients (MATLAB gmres.m, non-divergence form):
    # c = L(1); (b - a') = L(x) - c x; a = -L(x^2/2) + (b-a') x + c x^2/2.
    c = L_of(one)
    bminusa = L_of(x) - c * x
    a = (-1.0) * L_of(x ** 2 / 2) + bminusa * x + c * (x ** 2 / 2)
    b = bminusa + a.diff()

    def Lc(v):
        return ((-1.0) * (a * v.diff()).diff() + b * v.diff()
                + c * v)

    def R1(v):
        return v.cumsum()

    def R2(v):
        return float(v.sum()) - v.cumsum()

    def Pi(g):
        return g - float(g.sum()) / (b0 - a0)

    def T(v):
        return Pi(R2(Lc(R1(v))))

    lbc = float(N.lbc) if isinstance(N.lbc, (int, float)) else 0.0
    rbc = float(N.rbc) if isinstance(N.rbc, (int, float)) else 0.0

    R2f = R2(f)
    PiR2f = Pi(R2f)
    tolz = 1e-12 * max(1.0, _norm2(f))
    if (_norm2(R2f - PiR2f) > tolz or abs(lbc) > tolz
            or abs(rbc) > tolz):
        # Correct with a low-degree polynomial z (MATLAB basis x.^(0:4)).
        basis = [x ** j for j in range(5)]
        A = np.zeros((4, 5))
        ends = jnp.asarray([a0, b0])
        for j, bj in enumerate(basis):
            w = R1(R2(Lc(bj)))
            A[0:2, j] = np.asarray(w(ends))
            A[2:4, j] = np.asarray(bj(ends))
        rhs = np.concatenate([np.asarray(R1(R2f)(ends)),
                              [lbc, rbc]])
        coef = np.linalg.lstsq(A, rhs, rcond=None)[0]
        z = basis[0] * float(coef[0])
        for j in range(1, 5):
            z = z + basis[j] * float(coef[j])
        g = Pi(R2f - R2(Lc(z)))
    else:
        g = PiR2f
        z = 0.0 * f
    return T, R1, Pi, g, z


def _norm2(u):
    return float(jnp.sqrt(jnp.abs(jnp.asarray(u.inner(u)))))


def _ip(u, v):
    return float(jnp.asarray(u.inner(v)))


def pcg(N, f, tol: float = 1e-10, maxit: int = 100):
    """Preconditioned conjugate gradients on chebfuns (MATLAB pcg).

    Provenance
    ----------
    MATLAB source : @chebop/pcg.m
    Chebfun commit: 7574c77
    """
    T, R1, Pi, g, z = _setup(N, f)
    u = 0.0 * f
    r = g - T(u)
    p = r
    tolf = tol * max(_norm2(g), 1e-30)
    rho = _ip(r, r)
    for _ in range(maxit):
        if np.sqrt(rho) <= tolf:
            break
        Lp = T(p)
        alpha = rho / _ip(p, Lp)
        u = (u + alpha * p).simplify()
        r = (r - alpha * Lp).simplify()
        rho_new = _ip(r, r)
        p = (r + (rho_new / rho) * p).simplify()
        rho = rho_new
    return z + R1(Pi(u))


def minres(N, f, tol: float = 1e-10, maxit: int = 100):
    """MINRES on chebfuns for the preconditioned self-adjoint operator
    (implemented via the Lanczos-based residual minimization over the
    Krylov space; equivalent to MATLAB @chebop/minres.m).

    Provenance
    ----------
    MATLAB source : @chebop/minres.m
    Chebfun commit: 7574c77
    """
    return _arnoldi_solve(N, f, tol, maxit)


def gmres(N, f, tol: float = 1e-10, maxit: int = 60):
    """GMRES on chebfuns for the preconditioned operator (MATLAB
    @chebop/gmres.m).

    Provenance
    ----------
    MATLAB source : @chebop/gmres.m
    Chebfun commit: 7574c77
    """
    return _arnoldi_solve(N, f, tol, maxit)


def _arnoldi_solve(N, f, tol, maxit):
    """GMRES/MINRES in function space, discretized on a fixed fine
    Clenshaw-Curtis grid: the Krylov vectors live as value arrays (so
    orthogonalization is cheap numpy work) while each operator
    application T = Pi R2 L R1 runs through the chebfun calculus.
    Keeping the Q basis as chebfuns made the k-term Gram-Schmidt walk
    ever-growing representations (a 30-minute iteration by k ~ 20).
    """
    from chebfunjax.utils.quadrature import chebpts, chebweights

    T, R1, Pi, g, z = _setup(N, f)
    dom = tuple(float(v) for v in N.domain)
    a0, b0 = dom[0], dom[-1]
    n = 1024
    xg = np.array(chebpts(n))
    wq = np.array(chebweights(n)) * (b0 - a0) / 2.0
    xs = a0 + (b0 - a0) * (xg + 1.0) / 2.0
    xj = jnp.asarray(xs)

    def to_vals(u):
        return np.asarray(u(xj), dtype=float)

    from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
    from chebfunjax.domain import Domain
    from chebfunjax.tech.chebtech import Chebtech2

    def to_fun(v):
        # Direct Chebyshev fit + assembly: routing through the adaptive
        # constructor re-sampled the polynomial hundreds of times per
        # Krylov iteration (~12 s/iter -> GMRES minutes per case).
        c = np.polynomial.chebyshev.chebfit(xg, v, min(n - 1, 260))
        tol_c = 1e-14 * max(1.0, float(np.max(np.abs(c))))
        keep = np.nonzero(np.abs(c) > tol_c)[0]
        c = c[: (keep[-1] + 1)] if keep.size else c[:1]
        tech = Chebtech2.from_coeffs(jnp.asarray(c, dtype=jnp.float64))
        return Chebfun(funs=[_Piece(tech=tech, interval=(a0, b0))],
                       domain=Domain((a0, b0)))

    def ip(u, v):
        return float(np.sum(wq * u * v))

    gv = to_vals(g)
    beta = float(np.sqrt(ip(gv, gv)))
    if beta == 0.0:
        return z + 0.0 * f
    Q = [gv / beta]
    H = np.zeros((maxit + 1, maxit))
    tolf = tol * beta
    k_used = 0
    for k in range(maxit):
        w = to_vals(T(to_fun(Q[k]).simplify()))
        for j in range(k + 1):
            H[j, k] = ip(Q[j], w)
            w = w - H[j, k] * Q[j]
        H[k + 1, k] = float(np.sqrt(max(ip(w, w), 0.0)))
        k_used = k + 1
        e1 = np.zeros(k + 2)
        e1[0] = beta
        y, _, _, _ = np.linalg.lstsq(H[:k + 2, :k + 1], e1, rcond=None)
        resid = float(np.linalg.norm(H[:k + 2, :k + 1] @ y - e1))
        if resid <= tolf or H[k + 1, k] < 1e-14 * beta:
            break
        Q.append(w / H[k + 1, k])
    e1 = np.zeros(k_used + 1)
    e1[0] = beta
    y, _, _, _ = np.linalg.lstsq(H[:k_used + 1, :k_used], e1, rcond=None)
    uv = sum(float(y[j]) * Q[j] for j in range(k_used))
    u = to_fun(uv).simplify()
    return z + R1(Pi(u))
