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
import numpy as np  # uses-numpy: Arnoldi orthogonalization on fixed value grids (host-side, non-JIT)


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


def pcg(N, f, tol: float = 1e-10, maxit: int = 100, full_output: bool = False):
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
    g_norm = max(_norm2(g), 1e-30)
    tolf = tol * g_norm
    rho = _ip(r, r)
    resvec = [float(np.sqrt(rho))]
    it = 0
    flag = 1
    for _ in range(maxit):
        if np.sqrt(rho) <= tolf:
            flag = 0
            break
        Lp = T(p)
        alpha = rho / _ip(p, Lp)
        u = (u + alpha * p).simplify()
        r = (r - alpha * Lp).simplify()
        rho_new = _ip(r, r)
        p = (r + (rho_new / rho) * p).simplify()
        rho = rho_new
        it += 1
        resvec.append(float(np.sqrt(rho)))
    else:
        flag = 0 if np.sqrt(rho) <= tolf else 1
    sol = z + R1(Pi(u))
    if full_output:
        # MATLAB [u, flag, relres, iter, resvec] = pcg(...)
        return sol, flag, resvec[-1] / g_norm, it, np.asarray(resvec)
    return sol


def _minres_empty(value):
    """Concrete MATLAB empty input adapter for eager MINRES setup."""
    return (value is None or isinstance(value, (list, tuple)) and len(value) == 0
            or getattr(value, "size", None) == 0
            or hasattr(value, "isempty") and value.isempty())


def _minres_basic_correction(A, rhs):
    """Basic column-pivoted QR correction for source MINRES A\\b.

    A source-equivalent basic solution keeps non-pivot coordinates zero;
    minimum-norm SVD least squares can change the correction polynomial.
    Rank choices near the numerical threshold need a matched MATLAB fixture.
    """
    from jax.scipy.linalg import qr, solve_triangular

    Q, R, permutation = qr(A, mode="economic", pivoting=True)
    diagonal = jnp.abs(jnp.diag(R))
    threshold = max(A.shape)*jnp.finfo(A.dtype).eps*diagonal[0]
    rank = int(jnp.sum(diagonal > threshold))
    coeffs = jnp.zeros(A.shape[1], dtype=A.dtype)
    if rank:
        basic = solve_triangular(R[:rank, :rank], (Q.T.conj()@rhs)[:rank])
        coeffs = coeffs.at[permutation[:rank]].set(basic)
    return coeffs


def _setup_minres(N, f, tol):
    """Source MINRES divergence coefficients and range/Dirichlet correction."""
    from numbers import Real

    from chebfunjax.chebfun1d.chebfun import Chebfun

    dom = f.domain
    x = Chebfun.identity(dom)
    if not N._is_linear():
        raise ValueError('CHEBFUN:CHEBOP:pcg:nonlinear: MINRES supports only linear CHEBOP instances.')
    if N.linop().blocks[0][0].order != 2:
        raise ValueError('CHEBFUN:CHEBOP:pcg:DiffOrder: MINRES supports only second-order ODEs.')
    bcs = []
    for side in ('left', 'right'):
        value = getattr(N, 'lbc' if side == 'left' else 'rbc')
        if value is None:
            value = 0.
        if not isinstance(value, Real):
            raise ValueError('CHEBFUN:CHEBOP:pcg:' + side + 'bc: Currently, we require Dirichlet boundary conditions. Please supply N.' + ('lbc' if side == 'left' else 'rbc') + ' = double.')
        bcs.append(value)
    one = 1 + 0*x
    c = N.feval(one)
    halfx2 = x*x/2
    a = -N.feval(halfx2) - (-N.feval(x) + c*x)*x + c*halfx2

    def L(v):
        return -(a*v.diff()).diff() + c*v

    def R1(v):
        return v.cumsum()

    def R2(v):
        return v.sum() - v.cumsum()

    def Pi(v):
        return v - v.sum()/(dom.b-dom.a)

    def T(v):
        return Pi(R2(L(R1(v))))

    R2f = R2(f)
    PiR2f = Pi(R2f)
    if _norm2(R2f-PiR2f) > tol or any(abs(bc) > tol for bc in bcs):
        basis = [x**j for j in range(5)]
        ends = jnp.asarray([dom.a, dom.b])
        A = jnp.stack([jnp.concatenate((R1(R2(L(bj)))(ends), bj(ends)))
                       for bj in basis], axis=1)
        rhs = jnp.concatenate((R1(R2f)(ends), jnp.asarray(bcs)))
        coeffs = _minres_basic_correction(A, rhs)
        z = sum((bj*coeffs[j] for j,bj in enumerate(basis)), 0*f)
        g = Pi(R2f-R2(L(z)))
    else:
        g, z = PiR2f, 0*f
    return T, R1, Pi, g, z


def minres(N, f, tol: float | None = None, maxit: int | None = None,
           full_output: bool = False, *, R1=None, R2=None, u0=None):
    """Source function-space MINRES with Lanczos and plane rotations.

    Numerical vectors remain adaptive Chebfuns. ``full_output`` returns the
    solution, flag, relative residual, iteration and a JAX residual vector.
    Only the source indefinite-integral preconditioners are supported.
    The outer adaptive iteration uses eager scalar control flow.

    Provenance
    ----------
    MATLAB source : @chebop/minres.m, @chebfun/normest.m
    Chebfun commit: 7574c77
    """
    import warnings

    from chebfunjax.chebpref import ChebopPref

    prefs = ChebopPref()
    tol = prefs.bvpTol if _minres_empty(tol) else tol
    maxit = prefs.maxIter if _minres_empty(maxit) else maxit
    eps = float(jnp.finfo(jnp.float64).eps)
    warned = tol <= eps or tol >= 1
    if warned:
        warnings.warn('CHEBFUN:CHEBOP:pcg: tolerance must lie between eps and 1.',
                      RuntimeWarning, stacklevel=2)
        tol = max(eps, min(tol, 1-eps))
    if not _minres_empty(R1) or not _minres_empty(R2):
        raise ValueError('chebop:pcg:OnlyDefaultPreconditionerAllowed')
    T, R1, Pi, g, z = _setup_minres(N, f, tol)
    u0 = None if _minres_empty(u0) else u0
    u = 0*f if u0 is None else u0
    if u0 is not None:
        from chebfunjax.chebfun1d.chebfun import _hscale
        ends_f = jnp.asarray([f.domain.a, f.domain.b])
        ends_u = jnp.asarray([u0.domain.a, u0.domain.b])
        error = jnp.abs(ends_f-ends_u)
        threshold = 1e-15*max(_hscale(f), _hscale(u0))
        if not bool(jnp.all((error < threshold) | jnp.isnan(error))):
            raise ValueError('chebop:pcg:WrongInitGuessDomain')
    Tu = 0*f if u0 is None else T(u)
    n2f = _norm2(f)
    tolg = tol*sum(float(piece.tech.normest()) for piece in g.funs)
    r = g-Tu
    normr = _norm2(r)
    normr_act = normr
    resvec = [normr]

    def output(v, flag, relres, iteration):
        return (v, flag, relres, iteration, jnp.asarray(resvec)) if full_output else v

    def relative(value):
        # MATLAB permits the zero-rhs 0/0 NaN diagnostic.
        return float(jnp.asarray(value)/jnp.asarray(n2f))

    if normr <= tolg:
        # Literal source early-return branch omits the polynomial correction z.
        return output(R1(Pi(u)), 0, relative(normr), 0)
    flag, iteration = 1, 0
    umin, imin, normrmin = u, 0, normr
    vold = r
    beta1 = _ip(vold, vold)
    if beta1 <= 0:
        return output(R1(Pi(u)), 5, relative(normr), 0)
    beta1 = float(jnp.sqrt(beta1))
    snprod = beta1
    vv = vold/beta1
    v = T(vv)
    Amvv = v
    alpha = _ip(vv, v)
    v = v-(alpha/beta1)*vold
    numer, denom = _ip(vv, v), _ip(vv, vv)
    v = v-(numer/denom)*vv
    volder, vold, betaold = vold, v, beta1
    beta = _ip(v, v)
    if beta < 0:
        return output(R1(Pi(u)), 5, relative(normr), 0)
    iteration = 1
    beta = float(jnp.sqrt(beta))
    gammabar, epsilon, deltabar = alpha, 0., beta
    gamma = float(jnp.hypot(gammabar, beta))
    if gamma == 0 or not bool(jnp.isfinite(gamma)):
        return output(R1(Pi(u)), 4, relative(normr), 0)
    mold = Amold = 0*f
    m, Am = vv/gamma, Amvv/gamma
    cs, sn = gammabar/gamma, beta/gamma
    u = u+snprod*cs*m
    snprod *= sn
    normr = abs(snprod)
    resvec.append(normr)
    if normr <= tolg:
        # The source also uses the estimated residual and omits z here.
        return output(R1(Pi(u)), 0, relative(normr), 1)
    stag, moresteps = 0, 0
    maxmsteps = min(len(f)//50, 5, len(f)-maxit)
    ii = 1
    for ii in range(2, maxit+1):
        if beta == 0:
            flag = 4
            break
        vv = v/beta
        v = T(vv)
        Amolder, Amold, Am = Amold, Am, v
        v = v-(beta/betaold)*volder
        alpha = _ip(vv, v)
        v = v-(alpha/beta)*vold
        volder, vold, betaold = vold, v, beta
        beta = _ip(v, v)
        if beta < 0:
            flag = 5
            break
        beta = float(jnp.sqrt(beta))
        delta = cs*deltabar+sn*alpha
        molder, mold = mold, m
        m = vv-delta*mold-epsilon*molder
        Am = Am-delta*Amold-epsilon*Amolder
        gammabar = sn*deltabar-cs*alpha
        epsilon, deltabar = sn*beta, -cs*beta
        gamma = float(jnp.hypot(gammabar, beta))
        if gamma == 0 or not bool(jnp.isfinite(gamma)):
            flag = 4
            break
        m, Am = m/gamma, Am/gamma
        cs, sn = gammabar/gamma, beta/gamma
        if snprod*cs == 0 or abs(snprod*cs)*_norm2(m) < eps*_norm2(u):
            stag += 1
        else:
            stag = 0
        u = u+snprod*cs*m
        snprod *= sn
        normr = abs(snprod)
        resvec.append(normr)
        if normr <= tolg or stag >= 3 or moresteps:
            normr_act = _norm2(g-T(u))
            resvec[-1] = normr_act
            if normr_act <= tolg:
                flag, iteration = 0, ii
                break
            if stag >= 3 and moresteps == 0:
                stag = 0
            moresteps += 1
            if moresteps >= maxmsteps:
                if not warned:
                    warnings.warn('tooSmallTolerance', RuntimeWarning, stacklevel=2)
                flag, iteration = 3, ii
                break
        if normr < normrmin:
            normrmin, umin, imin = normr, u, ii
        if stag >= 3:
            flag = 3
            break
    if flag == 0:
        relres = relative(normr_act)
    else:
        r_comp_norm = _norm2(g-T(u))
        # Retain the source's minimum-iterate selection, including its
        # comparison with normr_act and residual from the final iterate.
        if r_comp_norm <= normr_act:
            u, iteration, relres = umin, imin, relative(r_comp_norm)
        else:
            iteration, relres = ii, relative(normr_act)
    return output(R1(u)+z, flag, relres, iteration)

def gmres(N, f, tol: float = 1e-10, maxit: int = 60, full_output: bool = False):
    """GMRES on chebfuns for the preconditioned operator (MATLAB
    @chebop/gmres.m).

    Provenance
    ----------
    MATLAB source : @chebop/gmres.m
    Chebfun commit: 7574c77
    """
    return _arnoldi_solve(N, f, tol, maxit, full_output)


def _arnoldi_solve(N, f, tol, maxit, full_output=False):
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
        sol = z + 0.0 * f
        return (sol, 0, 0.0, 0, np.zeros(1)) if full_output else sol
    Q = [gv / beta]
    H = np.zeros((maxit + 1, maxit))
    tolf = tol * beta
    k_used = 0
    resvec = [beta]
    flag = 1
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
        resvec.append(resid)
        if resid <= tolf or H[k + 1, k] < 1e-14 * beta:
            flag = 0
            break
        Q.append(w / H[k + 1, k])
    e1 = np.zeros(k_used + 1)
    e1[0] = beta
    y, _, _, _ = np.linalg.lstsq(H[:k_used + 1, :k_used], e1, rcond=None)
    uv = sum(float(y[j]) * Q[j] for j in range(k_used))
    u = to_fun(uv).simplify()
    sol = z + R1(Pi(u))
    if full_output:
        # MATLAB [u, flag, relres, iter, resvec] = gmres/minres(...)
        return sol, flag, resvec[-1] / beta, k_used, np.asarray(resvec)
    return sol
