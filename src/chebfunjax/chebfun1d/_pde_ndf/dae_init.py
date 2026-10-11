"""Private real-binary64, dense nonvectorized FD and full constant-mass IC.

Source: MathWorks R2025b ode15s/odenumjac/daeic12/condest/normest1,
Copyright 1984-2024 The MathWorks, Inc. Chebfun caller is pinned7574c77.
Sparse/vectorized/complex Jacobians and >4 algebraic variables are unsupported.

Provenance
----------
MATLAB R2025b ode15s.m and private/odenumjac.m, private/daeic12.m,
condest.m and normest1.m.
Chebfun native commit: 7574c77680d7e82b79626300bf255498271a72df.
Native ODE runtime/source release: MATLAB R2025b.
"""

import jax.numpy as jnp
import jax.scipy.linalg as jl

EPS = float(jnp.finfo(jnp.float64).eps)


def dense_numjac(fun, y, f, thresh, fac=None):
    y, f = jnp.asarray(y), jnp.asarray(f)
    if y.dtype != jnp.float64 or f.dtype != jnp.float64 or y.ndim != 1 or f.ndim != 1:
        raise NotImplementedError("Only real binary64 vector inputs are supported")
    n = y.size
    thresh = jnp.broadcast_to(jnp.asarray(thresh), y.shape)
    if bool(jnp.any(thresh <= 0)):
        raise ValueError("Positive source thresholds required")
    fac = jnp.full_like(y, EPS**0.5) if fac is None else fac
    br, bl, bu, facmin, facmax = EPS**0.875, EPS**0.75, EPS**0.25, EPS**0.78, 0.1
    scale = jnp.maximum(jnp.abs(y), thresh)
    delta = (y + fac * scale) - y
    for k in range(n):
        while float(delta[k]) == 0:
            if float(fac[k]) < facmax:
                fac = fac.at[k].set(jnp.minimum(100 * fac[k], facmax))
                delta = delta.at[k].set((y[k] + fac[k] * scale[k]) - y[k])
            else:
                delta = delta.at[k].set(thresh[k])
                break
    if f.size == n:
        delta = (-1 + 2 * (f >= 0)) * jnp.abs(delta)
    calls = []

    def evaluate(z):
        value = fun(z)
        calls.append(z)
        return value

    fdel = jnp.stack([evaluate(y.at[k].add(delta[k])) for k in range(n)], axis=1)
    fdiff = fdel - f[:, None]
    jac = fdiff / delta[None, :]
    rows = jnp.argmax(jnp.abs(fdiff), axis=0)
    difmax = jnp.max(jnp.abs(fdiff), axis=0)
    for k in range(n):
        row = int(rows[k])
        a = float(jnp.abs(fdel[row, k]))
        b = float(jnp.abs(f[row]))
        d = float(difmax[k])
        if (a != 0 and b != 0) or d == 0:
            fscale = max(a, b)
            fk = float(fac[k])
            if d <= br * fscale:
                tmpfac = min(fk**0.5, facmax)
                inc = (y[k] + tmpfac * scale[k]) - y[k]
                if tmpfac != fk and float(inc) != 0:
                    if f.size == n:
                        inc = jnp.abs(inc) if float(f[k]) >= 0 else -jnp.abs(inc)
                    fresh = evaluate(y.at[k].set(y[k] + inc))
                    diff = fresh - f
                    tmp = diff / inc
                    rm = int(jnp.argmax(jnp.abs(diff)))
                    dm = float(jnp.max(jnp.abs(diff)))
                    nt = float(jnp.max(jnp.abs(tmp)))
                    if tmpfac * nt >= float(jnp.max(jnp.abs(jac[:, k]))):
                        if nt > 0:
                            jac = jac.at[:, k].set(tmp)
                        fs = max(float(jnp.abs(fresh[rm])), float(jnp.abs(f[rm])))
                        if dm <= bl * fs:
                            fac = fac.at[k].set(min(10 * tmpfac, facmax))
                        elif dm > bu * fs:
                            fac = fac.at[k].set(max(0.1 * tmpfac, facmin))
                        else:
                            fac = fac.at[k].set(tmpfac)
            elif d <= bl * fscale:
                fac = fac.at[k].set(min(10 * fk, facmax))
            if d > bu * fscale:
                fac = fac.at[k].set(max(0.1 * fk, facmin))
    return jac, fac, tuple(calls)


def small_condest(a):
    """Literal normest1 n<=4 branch, without stochastic larger-size estimator."""
    n = a.shape[0]
    if n > 4:
        raise NotImplementedError("condest n>4 is not implemented")
    lu, _ = jl.lu_factor(a)
    upper = jnp.triu(lu)
    lower = jnp.tril(lu, -1) + jnp.eye(n, dtype=a.dtype)
    if bool(jnp.any(jnp.diag(upper) == 0)):
        return float("inf")
    inverse_product = jl.solve_triangular(
        upper,
        jl.solve_triangular(lower, jnp.eye(n, dtype=a.dtype), lower=True, unit_diagonal=True),
        lower=False,
    )
    return float(
        jnp.max(jnp.sum(jnp.abs(inverse_product), axis=0)) * jnp.max(jnp.sum(jnp.abs(a), axis=0))
    )


def initialize_full_mass(fun, mass, y, *, reltol=1e-6, abstol=1e-6):
    """Source ICtype2, default zero InitialSlope, numerical dense Jacobian."""
    f = fun(y)
    jac, fac, calls = dense_numjac(fun, y, f, abstol)
    nfe, nje = len(calls), 1
    u, d, vh = jnp.linalg.svd(mass, full_matrices=True)
    v = vh.T
    cutoff = y.size * jnp.max(d) * EPS
    alg = jnp.where(d <= cutoff)[0]
    d = jnp.where(d <= cutoff, 0.0, d)
    dif = jnp.where(d != 0)[0]
    transformed = u.T @ f
    dfdy = (u.T @ jac) @ v
    Y = v.T @ y
    yp = v.T @ jnp.zeros_like(y)
    log = []

    def finish(y, f, F, reason):
        slope = v @ yp.at[dif].set(F[dif] / d[dif])
        return dict(
            y=y,
            yp=slope,
            f=f,
            jac=jac,
            fac=fac,
            nfe=nfe,
            nje=nje,
            log=log,
            algebraic=alg,
            svd_cutoff=cutoff,
            reason=reason,
        )

    if alg.size == 0:
        return finish(y, f, transformed, "nonsingular")
    J = dfdy[jnp.ix_(alg, alg)]
    nonzero = int(jnp.count_nonzero(J))
    if nonzero == 0 or EPS * nonzero * small_condest(J) > 1:
        raise ValueError("Source IndexGTOne")
    norm = jnp.linalg.norm
    if float(norm(transformed[alg])) <= 1000 * EPS * float(norm(transformed)):
        return finish(y, f, transformed, "initial_consistent")
    factor = jl.lu_factor(J)
    need = False
    for iteration in range(1, 16):
        if need:
            jac, fac, calls = dense_numjac(fun, y, f, abstol, fac)
            nfe += len(calls)
            nje += 1
            dfdy = (u.T @ jac) @ v
            J = dfdy[jnp.ix_(alg, alg)]
            factor = jl.lu_factor(J)
        delta = jl.lu_solve(factor, -transformed[alg])
        res = float(norm(delta))
        step = 1.0
        Yn = Y
        for probe in range(1, 4):
            Yn = Yn.at[alg].set(Y[alg] + step * delta)
            yn = v @ Yn
            fn = fun(yn)
            Fn = u.T @ fn
            nfe += 1
            log.append(
                dict(
                    iteration=iteration,
                    probe=probe,
                    step=step,
                    residual=float(norm(Fn[alg])),
                    full_residual=float(norm(Fn)),
                )
            )
            if float(norm(Fn[alg])) <= 1e-3 * reltol * float(norm(Fn)):
                return finish(yn, fn, Fn, "residual")
            resnew = float(norm(jl.lu_solve(factor, Fn[alg])))
            if resnew < 0.9 * res:
                break
            step *= 0.5
        ynorm = max(float(norm(Y[alg])), float(norm(Yn[alg]))) or EPS
        Y = Yn
        y = v @ Yn
        f = fn
        transformed = Fn
        if resnew <= 1e-3 * reltol * ynorm:
            return finish(y, f, transformed, "correction")
        need = resnew > 0.1 * res
    raise ValueError("Source NeedBetterY0")
