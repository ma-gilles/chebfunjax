"""Function-space Krylov solvers for second-order Chebop problems.

PCG and MINRES use the self-adjoint divergence operator
L(u)=-(a*u')'+c*u. GMRES also retains the nonsymmetric b*u' term.
The source iterations act on adaptive Chebfuns with continuous inner products;
indefinite-integral preconditioners and polynomial boundary/range correction
replace sampled-grid matrix approximations. Final exits follow each source
solver's output and residual conventions.

Provenance
----------
MATLAB source : @chebop/pcg.m, @chebop/minres.m, @chebop/gmres.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

import warnings
from numbers import Real
from typing import Any

import jax
import jax.numpy as jnp


def _norm2(u):
    return float(jnp.sqrt(jnp.abs(jnp.asarray(u.inner(u)))))


def _ip(u, v):
    return float(jnp.asarray(u.inner(v)))


def _prepare_pcg_operator(N, f):
    """Validate source PCG operator before processing options."""
    from numbers import Real

    from chebfunjax.chebfun1d.chebfun import Chebfun

    if not N._is_linear():
        raise ValueError('CHEBFUN:CHEBOP:pcg:nonlinear: PCG supports only linear CHEBOP instances.')
    if N.linop().blocks[0][0].order != 2:
        raise ValueError('CHEBFUN:CHEBOP:pcg:DiffOrder: PCG supports only second-order ODEs.')
    n2f = _norm2(f)
    dom = f.domain
    bcs = []
    for side in ('left', 'right'):
        value = getattr(N, 'lbc' if side == 'left' else 'rbc')
        if value is None:
            value = 0.
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError('CHEBFUN:CHEBOP:pcg:' + side + 'bc: PCG only supports Dirichlet boundary conditions. Please supply N.' + ('lbc' if side == 'left' else 'rbc') + ' = double.')
        bcs.append(value)
    if N._op_nargs() not in (1, 2):
        raise ValueError("chebop:pcg:DiffOpNargin")
    return dom, Chebfun.identity(dom), bcs, n2f


def _setup_pcg(N, f, tol, R2, u0, prepared):
    """Source PCG divergence operator and absolute-tolerance range correction."""
    import warnings

    dom, x, bcs, _ = prepared
    if _minres_empty(R2):
        def R2(v):
            return v.sum() - v.cumsum()

    def apply(value):
        return N.op(value) if N._op_nargs() == 1 else N.op(x, value)

    c = apply(1 + 0*x)
    if c.min()[1] < -tol:
        warnings.warn('Chebfun:chebop:pcg:Eigenvalues: Does the differential operator have nonnegative eigenvalues?', RuntimeWarning, stacklevel=3)
    halfx2 = x*x/2
    a = -apply(halfx2) - (-apply(x) + c*x)*x + c*halfx2
    if a.min()[1] < -tol:
        warnings.warn('Chebfun:chebop:pcg:elliptic: Is the differential operator uniformly elliptic?', RuntimeWarning, stacklevel=3)

    def L(v):
        return -(a*v.diff()).diff() + c*v

    def R1(v):
        return v.cumsum()

    def Pi(v):
        return v - v.sum()/(dom.b-dom.a)

    def T(v):
        return Pi(R2(L(R1(v))))

    if _minres_empty(u0):
        u = 0*f
        Tu = u
    else:
        from chebfunjax.chebfun1d.chebfun import _hscale
        ends_f = jnp.asarray([f.domain.a, f.domain.b])
        ends_u = jnp.asarray([u0.domain.a, u0.domain.b])
        error = jnp.abs(ends_f-ends_u)
        threshold = 1e-15*max(_hscale(f), _hscale(u0))
        if not bool(jnp.all((error < threshold) | jnp.isnan(error))):
            raise ValueError('chebop:pcg:WrongInitGuessDomain')
        u = u0
        Tu = T(u)
    R2f = R2(f)
    PiR2f = Pi(R2f)
    if _norm2(R2f-PiR2f) > tol or any(abs(bc) > tol for bc in bcs):
        basis = [x**j for j in range(5)]
        ends = jnp.asarray([dom.a, dom.b])
        A = jnp.stack([jnp.concatenate((R1(R2(L(bj)))(ends), bj(ends))) for bj in basis], axis=1)
        rhs = jnp.concatenate((R1(R2f)(ends), jnp.asarray(bcs)))
        coeffs = _minres_basic_correction(A, rhs)
        z = sum((bj*coeffs[j] for j,bj in enumerate(basis)), 0*f)
        g = Pi(R2f-R2(L(z)))
    else:
        g, z = PiR2f, 0*f
    return T, R1, Pi, g, z, u, Tu


def pcg(N, f, tol: float | None = None, maxit: int | None = None,
        R1=None, R2=None, u0=None, *, full_output: bool = False):
    """Source PCG recurrence on adaptive Chebfuns with actual residual checks.

    Provenance
    ----------
    MATLAB source : @chebop/pcg.m
    Chebfun commit: 7574c77
    """
    import warnings

    from chebfunjax.chebpref import ChebopPref

    prepared = _prepare_pcg_operator(N, f)
    n2f = prepared[3]
    prefs = ChebopPref()
    tol = prefs.bvpTol if _minres_empty(tol) else tol
    maxit = prefs.maxIter if _minres_empty(maxit) else maxit
    eps = jnp.finfo(jnp.float64).eps
    warned = tol <= eps or tol >= 1
    if warned:
        warnings.warn('CHEBFUN:CHEBOP:pcg: tolerance must lie between eps and 1.', RuntimeWarning, stacklevel=2)
        tol = max(eps, min(tol, 1-eps))
    if not _minres_empty(R1):
        raise ValueError('chebop:pcg:OnlyDefaultPreconditionerAllowed')
    T, R1, Pi, g, z, u, Tu = _setup_pcg(N, f, tol, R2, u0, prepared)
    flag = 1
    umin, imin = u, 0
    tolf = tol*_norm2(g)
    r = g - Tu
    p = r
    normr = _norm2(r)
    normr_act = normr
    if normr <= tolf:
        sol = R1(Pi(u))
        relres = jnp.asarray(normr)/n2f
        return (sol, 0, relres, 0, jnp.asarray([normr])) if full_output else sol
    resvec = jnp.zeros(maxit+1).at[0].set(normr)
    normrmin = normr
    stag = moresteps = 0
    rho = jnp.asarray(r.inner(r))
    iteration = ii = 0
    for ii in range(1, maxit+1):
        Lp = T(p)
        alpha = rho/jnp.asarray(p.inner(Lp))
        u = u + alpha*p
        r = r - alpha*Lp
        rho_new = jnp.asarray(r.inner(r))
        beta = rho_new/rho
        p = r + beta*p
        rho = rho_new
        if rho == 0 or jnp.isinf(rho):
            flag = 4
            normr_act = float(jnp.sqrt(rho))
            break
        if jnp.isinf(alpha):
            flag = 4
            break
        if beta == 0 or jnp.isinf(beta):
            flag = 4
            break
        if _norm2(p)*abs(alpha) < eps*_norm2(u):
            stag += 1
        else:
            stag = 0
        normr = float(jnp.sqrt(rho))
        normr_act = normr
        resvec = resvec.at[ii].set(normr)
        if normr <= tolf or stag >= 3 or moresteps:
            r = g - T(u)
            normr_act = _norm2(r)
            resvec = resvec.at[ii].set(normr_act)
            if normr_act <= tolf:
                flag = 0
                iteration = ii
                break
            if stag >= 3 and moresteps == 0:
                stag = 0
            moresteps += 1
            if moresteps >= 5:
                if not warned:
                    warnings.warn('The tolerance is probably too small.', RuntimeWarning, stacklevel=2)
                flag = 3
                iteration = ii
                break
        if normr_act < normrmin:
            normrmin, umin, imin = normr_act, u, ii
        if stag >= 3:
            flag = 3
            break
    if flag == 0:
        relres = jnp.asarray(normr_act)/n2f
    else:
        r_comp = g - T(umin)
        norm_comp = _norm2(r_comp)
        if norm_comp <= normr_act:
            u, iteration = umin, imin
            relres = jnp.asarray(norm_comp)/n2f
        else:
            iteration = ii
            relres = jnp.asarray(normr_act)/n2f
    sol = R1(u) + z
    resvec = resvec[:ii+1] if flag <= 1 or flag == 3 else resvec[:ii]
    return (sol, flag, relres, iteration, resvec) if full_output else sol


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


def _prepare_minres_operator(N, f):
    """Validate source MINRES operator and mine divergence coefficients."""
    from numbers import Real

    from chebfunjax.chebfun1d.chebfun import Chebfun

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
    dom = f.domain
    x = Chebfun.identity(dom)
    one = 1 + 0*x
    c = N.feval(one)
    halfx2 = x*x/2
    a = -N.feval(halfx2) - (-N.feval(x) + c*x)*x + c*halfx2

    def L(v):
        return -(a*v.diff()).diff() + c*v

    return dom, x, L, bcs


def _setup_minres(N, f, tol, *, prepared=None):
    """Source MINRES range and Dirichlet correction after option validation."""
    dom, x, L, bcs = (_prepare_minres_operator(N, f)
                       if prepared is None else prepared)

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

    prepared = _prepare_minres_operator(N, f)
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
    T, R1, Pi, g, z = _setup_minres(N, f, tol, prepared=prepared)
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





def _gmres_is_empty(value: Any) -> bool:
    """Eager adapter for MATLAB isempty on optional Python arguments."""
    if value is None:
        return True
    if isinstance(value, (list, tuple)) and not value:
        return True
    if getattr(value, "size", None) == 0:
        return True
    return bool(getattr(value, "isempty", lambda: False)())


def _gmres_norm(u) -> float:
    """Continuous Chebfun L2 norm, converted to a host control scalar."""
    return float(jnp.sqrt(jnp.abs(jnp.asarray(u.inner(u)))))


def _gmres_ip(u, v):
    """MATLAB ``Q(:,k)'*v`` inner product (conjugates the first argument)."""
    return jnp.asarray(u.inner(v))


def _gmres_endpoint_values(f, endpoints):
    values = jnp.asarray(f(jnp.asarray(endpoints, dtype=jnp.float64)))
    return jnp.reshape(values, (-1,))


def _gmres_domain_pair(f):
    domain = f.domain.breakpoints
    if len(domain) < 2:
        raise ValueError("GMRES requires a nonempty interval domain.")
    return domain[0], domain[-1]


def _gmres_same_domain(f, u0) -> bool:
    """MATLAB domainCheck endpoint comparison with its relative hscale."""
    from chebfunjax.chebfun1d.chebfun import _hscale

    try:
        ends_f = jnp.asarray([f.domain.a, f.domain.b], dtype=jnp.float64)
        ends_u = jnp.asarray([u0.domain.a, u0.domain.b], dtype=jnp.float64)
        error = jnp.abs(ends_f - ends_u)
        threshold = 1e-15 * max(_hscale(f), _hscale(u0))
        # MATLAB domainCheck accepts NaN endpoint differences (notably Inf-Inf).
        return bool(jnp.all((error < threshold) | jnp.isnan(error)))
    except (AttributeError, TypeError, ValueError):
        return False


def _gmres_add_scaled(basis, coefficients):
    """Form a function-space linear combination without sampled vectors."""
    out = 0.0 * basis[0]
    for j, coefficient in enumerate(coefficients):
        out = out + coefficient * basis[j]
    return out


def _gmres_warning(message):
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _gmres_apply_source_op(N, x, value, nargs):
    """Apply a one- or two-argument operator on the RHS domain."""
    if nargs == 1:
        return N.op(value)
    if nargs == 2:
        return N.op(x, value)
    raise ValueError("chebop:pcg:DiffOpNargin")


def _prepare_gmres_operator(N, f, *, validate=True):
    """Validate and mine the source divergence-form second-order operator.

    The coefficient identities are copied from ``@chebop/gmres.m``. In
    particular ``b*diff(v)`` is retained, so nonsymmetric source operators
    are not silently converted to self-adjoint form.
    """
    from chebfunjax.chebfun1d.chebfun import Chebfun

    if validate and not N._is_linear():
        raise ValueError(
            "CHEBFUN:CHEBOP:pcg:nonlinear: GMRES supports only linear CHEBOP instances."
        )
    if validate and N.linop().blocks[0][0].order != 2:
        raise ValueError(
            "CHEBFUN:CHEBOP:pcg:DiffOrder: GMRES supports only second-order ODEs."
        )
    if validate and f is None:
        raise ValueError("CHEBFUN:CHEBOP:gmres:NotEnoughInputs")

    a0, b0 = _gmres_domain_pair(f)
    x = Chebfun.identity(f.domain)
    one = 1.0 + 0.0 * x
    # Source accepts L(u) and L(x,u), and mines on domain(f), not domain(N).
    nargs = N._op_nargs()
    if nargs not in (1, 2):
        raise ValueError("chebop:pcg:DiffOpNargin")

    def source_op(value):
        return _gmres_apply_source_op(N, x, value, nargs)

    c = source_op(one)
    b_minus_a = source_op(x) - c * x
    a = -source_op(x**2 / 2.0) + b_minus_a * x + c * x**2 / 2.0
    b = b_minus_a + a.diff()

    def lhat(v):
        return -(a * v.diff()).diff() + b * v.diff() + c * v

    return x, lhat, (a0, b0)


def _gmres_source(
    N,
    f,
    restart=None,
    tol=None,
    maxit=None,
    R1=None,
    R2=None,
    u0=None,
    *,
    full_output=False,
):
    """Source-shaped GMRES kernel; numerical qualification is pending.

    Parameters follow MATLAB ``gmres(N,f,restart,tol,maxit,R1,R2,u0)``.
    ``full_output`` is a Python keyword adapter for MATLAB's multiple returns.
    The returned ``iteration`` is always represented by a two-entry integer
    JAX array when that output is assigned by the source.
    """
    from chebfunjax.chebpref import ChebopPref
    from chebfunjax.operators.krylov import _minres_basic_correction

    # Match source validation order: linearity/order checks precede missing f.
    if not N._is_linear():
        raise ValueError(
            "CHEBFUN:CHEBOP:pcg:nonlinear: GMRES supports only linear CHEBOP instances."
        )
    if N.linop().blocks[0][0].order != 2:
        raise ValueError(
            "CHEBFUN:CHEBOP:pcg:DiffOrder: GMRES supports only second-order ODEs."
        )
    if f is None:
        raise ValueError("CHEBFUN:CHEBOP:gmres:NotEnoughInputs")

    n2f = _gmres_norm(f)
    # MATLAB accepts only numeric scalar Dirichlet endpoint data here.
    left_bc, right_bc = getattr(N, "lbc", None), getattr(N, "rbc", None)
    if _gmres_is_empty(left_bc):
        left_bc = 0.0
    elif not isinstance(left_bc, Real) or isinstance(left_bc, bool):
        raise ValueError(
            "CHEBFUN:CHEBOP:pcg:leftbc: GMRES only supports Dirichlet boundary "
            "conditions. Please supply N.lbc = double."
        )
    if _gmres_is_empty(right_bc):
        right_bc = 0.0
    elif not isinstance(right_bc, Real) or isinstance(right_bc, bool):
        # Preserve the pinned source's literal (left-boundary) wording.
        raise ValueError(
            "CHEBFUN:CHEBOP:pcg:rightbc: GMRES only supports Dirichlet boundary "
            "conditions. Please supply N.lbc = double."
        )
    left_bc, right_bc = float(left_bc), float(right_bc)

    # MATLAB callback arity handling and operator coefficient mining occur
    # before tolerance/restart/preconditioner option processing.
    x, lhat, (a0, b0) = _prepare_gmres_operator(N, f, validate=False)

    # MATLAB uses nargin, so None/list empties stand in for omitted parameters.
    restarted = not _gmres_is_empty(restart)
    pref = ChebopPref()
    if _gmres_is_empty(tol):
        tol = float(pref.bvpTol)
    else:
        tol = float(tol)
    warned = False
    eps = float(jnp.finfo(jnp.float64).eps)
    if tol < eps:
        _gmres_warning("CHEBFUN:CHEBOP:gmres:tooSmallTolerance")
        warned = True
        tol = eps
    elif tol >= 1.0:
        _gmres_warning("CHEBFUN:CHEBOP:gmres:tooBigTolerance")
        warned = True
        tol = 1.0 - eps
    if _gmres_is_empty(maxit):
        maxit = int(pref.maxIter)
    maxit = int(maxit)
    if restarted:
        outer, inner = maxit, int(restart)
    else:
        outer, inner = 1, maxit
    if not _gmres_is_empty(R1):
        raise ValueError("chebop:gmres:OnlyDefaultPreconditionerAllowed")
    if not _gmres_is_empty(R2):
        raise ValueError("chebop:gmres:OnlyDefaultPreconditionerAllowed")

    def r1(v):
        return v.cumsum()

    def r2(v):
        return v.sum() - v.cumsum()

    def pi(v):
        return v - v.mean()

    def apply_t(v):
        return pi(r2(lhat(r1(v))))

    if not _gmres_is_empty(u0):
        if not _gmres_same_domain(f, u0):
            raise ValueError("chebop:pcg:WrongInitGuessDomain")
        u = u0
        tu = apply_t(u)
    else:
        u = 0.0 * f
        tu = u

    # This helper is called with a Python signature, so only positional source
    # arity can be handled by its wrapper. It intentionally does not invent a
    # test for source nargin > 8.
    R2f = r2(f)
    PiR2f = pi(R2f)
    if (
        _gmres_norm(R2f - PiR2f) > tol
        or abs(left_bc) > tol
        or abs(right_bc) > tol
    ):
        basis = [x**j for j in range(5)]
        endpoints = jnp.asarray([a0, b0], dtype=jnp.float64)
        matrix_rows = []
        for basis_j in basis:
            top = _gmres_endpoint_values(r1(r2(lhat(basis_j))), endpoints)
            bottom = _gmres_endpoint_values(basis_j, endpoints)
            matrix_rows.append(jnp.concatenate((top, bottom)))
        matrix = jnp.stack(matrix_rows, axis=1)
        rhs = jnp.concatenate(
            (_gmres_endpoint_values(r1(R2f), endpoints),
             jnp.asarray([left_bc, right_bc], dtype=jnp.float64))
        )
        z_coeff = _minres_basic_correction(matrix, rhs)
        z = _gmres_add_scaled(basis, z_coeff)
        g = pi(R2f - r2(lhat(z)))
    else:
        g = PiR2f
        z = 0.0 * f

    flag = 1
    umin = u
    imin = jnp.asarray(0, dtype=jnp.int32)
    jmin = jnp.asarray(0, dtype=jnp.int32)
    tolg = tol * _gmres_norm(g)
    stag = 0
    moresteps = 0
    maxmsteps = 5
    maxstagsteps = 3
    minupdated = False

    r = g - tu
    normr = _gmres_norm(r)
    normr_act = normr
    if normr <= tolg:
        # Source's first exit omits ITER and returns scalar RESVEC. The tuple
        # adapter uses [0,0] and a one-entry residual array for stable Python.
        flag = 0
        # Keep source 0/0 behavior for a homogeneous zero RHS. Python-float
        # division would raise instead of returning NaN.
        relres = jnp.asarray(normr, dtype=jnp.float64) / jnp.asarray(
            n2f, dtype=jnp.float64
        )
        resvec = jnp.asarray([normr], dtype=jnp.float64)
        sol = r1(pi(u)) + z
        result = (sol, flag, relres, jnp.asarray([0, 0]), resvec)
        return result if full_output else sol

    normr = _gmres_norm(r)
    n2g = _gmres_norm(g)
    tolg = tol * n2g
    if normr <= tolg:
        # Deliberately source-shaped: this second exit does not undo R1/add z.
        flag = 0
        relres = normr / n2g
        result = (u, flag, relres, jnp.asarray([0, 0]),
                  jnp.asarray([n2g], dtype=jnp.float64))
        return result if full_output else u

    resvec = jnp.zeros(inner * outer + 1, dtype=jnp.float64)
    resvec = resvec.at[0].set(normr)
    normrmin = normr
    resvec_index = 1
    iter_out = 0
    iter_in = 0
    for outiter in range(1, outer + 1):
        qtb = jnp.asarray([_gmres_norm(r)], dtype=jnp.float64)
        Q = [r / qtb[0]]
        H = jnp.zeros((inner + 1, inner), dtype=jnp.float64)
        P = None
        R = None
        cycle_last = 0
        for initer in range(1, inner + 1):
            q = Q[initer - 1]
            v = apply_t(q)
            for k in range(1, initer + 1):
                hki = _gmres_ip(Q[k - 1], v)
                H = H.at[k - 1, initer - 1].set(jnp.real(hki))
                v = v - hki * Q[k - 1]
            beta_v = _gmres_norm(v)
            H = H.at[initer, initer - 1].set(beta_v)
            qtb = jnp.concatenate((qtb, jnp.zeros((1,), dtype=qtb.dtype)))
            Qnew, Rfull = jnp.linalg.qr(H[:initer + 1, :initer], mode="complete")
            P, R = Qnew, Rfull
            # Literal source divides even in a breakdown column.
            Q.append(v / beta_v)
            normr = float(jnp.abs(P[0, initer] * qtb[0]))
            resvec = resvec.at[resvec_index].set(normr)
            resvec_index += 1
            normr_act = normr
            cycle_last = initer
            iter_out, iter_in = outiter, initer

            if normr <= tolg or stag >= maxstagsteps or moresteps:
                triangular_rhs = P[:, :initer].conj().T @ qtb
                y = jax.scipy.linalg.solve_triangular(
                    R[:initer, :initer], triangular_rhs[:initer], lower=False
                )
                additive = _gmres_add_scaled(Q[:initer], y)
                if _gmres_norm(additive) < eps * _gmres_norm(u):
                    stag += 1
                else:
                    stag = 0
                um = u + additive
                r = g - apply_t(um)
                normr_act = _gmres_norm(r)
                resvec = resvec.at[resvec_index - 1].set(normr_act)
                if normr_act <= normrmin:
                    normrmin = normr_act
                    imin = jnp.asarray(outiter, dtype=jnp.int32)
                    jmin = jnp.asarray(initer, dtype=jnp.int32)
                    umin = um
                    minupdated = True
                if normr_act <= tolg:
                    u = um
                    flag = 0
                    break
                if stag >= maxstagsteps and moresteps == 0:
                    stag = 0
                moresteps += 1
                if moresteps >= maxmsteps:
                    if not warned:
                        _gmres_warning("chebop:gmres:tooSmallTolerance")
                    flag = 3
                    break

            if normr_act <= normrmin:
                normrmin = normr_act
                imin = jnp.asarray(outiter, dtype=jnp.int32)
                jmin = jnp.asarray(initer, dtype=jnp.int32)
                minupdated = True
            if stag >= maxstagsteps:
                flag = 3
                break

        if flag != 0:
            idx = int(jmin) if minupdated else cycle_last
            idx = max(1, idx)
            triangular_rhs = P[:, :idx].conj().T @ qtb
            y = jax.scipy.linalg.solve_triangular(
                R[:idx, :idx], triangular_rhs[:idx], lower=False
            )
            additive = _gmres_add_scaled(Q[:idx], y)
            u = u + additive
            umin = u
            r = g - apply_t(u)
            normr_act = _gmres_norm(r)

        if normr_act <= normrmin:
            umin = u
            normrmin = normr_act
            imin = jnp.asarray(outiter, dtype=jnp.int32)
            jmin = jnp.asarray(cycle_last, dtype=jnp.int32)
        if flag == 3:
            break
        if normr_act <= tolg:
            flag = 0
            iter_out, iter_in = outiter, cycle_last
            break
        minupdated = False

    if flag == 0:
        relres = normr_act / n2g
        iteration = jnp.asarray([iter_out, iter_in], dtype=jnp.int32)
    else:
        u = umin
        iteration = jnp.asarray([imin, jmin], dtype=jnp.int32)
        relres = normr_act / n2g
    sol = r1(u) + z
    resvec = resvec[:resvec_index]
    result = (sol, flag, relres, iteration, resvec)
    return result if full_output else sol


def gmres(
    N,
    f,
    restart=None,
    tol=None,
    maxit=None,
    R1=None,
    R2=None,
    u0=None,
    *,
    full_output: bool = False,
):
    """Source-shaped scalar Chebop GMRES entry point.

    Provenance
    ----------
    MATLAB source : @chebop/gmres.m
    Chebfun commit: 7574c77

    Omitted ``tol`` and ``maxit`` use the current ``ChebopPref`` values.
    ``full_output`` is the Python keyword adapter for MATLAB's multiple return
    values; ``iteration`` remains the source two-entry outer/inner pair.
    """
    return _gmres_source(
        N,
        f,
        restart=restart,
        tol=tol,
        maxit=maxit,
        R1=R1,
        R2=R2,
        u0=u0,
        full_output=full_output,
    )


def _arnoldi_solve(N, f, tol, maxit, full_output=False):
    """Private legacy call adapter to source function-space JAX GMRES.

    Existing MINRES regression guards monkeypatch this name to reject an
    accidental Arnoldi fallback. Numerical work uses the source GMRES path.
    """
    return gmres(N, f, tol=tol, maxit=maxit, full_output=full_output)
