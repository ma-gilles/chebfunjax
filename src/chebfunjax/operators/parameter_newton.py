"""Mixed function/scalar Newton discretization for parameter BVPs."""

import warnings

import jax.numpy as jnp

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
from chebfunjax.operators.blocks import ChebColloc2Disc
from chebfunjax.utils.interpolation import barymat
from chebfunjax.utils.quadrature import chebpts


def solve_parameter_fixed(op, f, n, max_iter, initial=None):
    """Solve one differential equation with trailing scalar parameters.

    Provenance
    ----------
    MATLAB source: @chebop/linearize.m, @chebop/solvebvpNonlinear.m,
        @chebcolloc/reduce.m.

    Adaptations
    -----------
    Uses existing system solver residual/update bounds (1e-11/1e-12) and
    30-step halving with minimum factor 1e-6. This is not the full MATLAB
    dampingErrorBased predictor-corrector preference algorithm.
    Residuals, Jacobians and scalar parameters preserve complex arithmetic.
    Chebfun commit: 7574c77
    """
    from chebfunjax.operators.chebop import SystemSolution

    domain = tuple(op.domain)
    disc = ChebColloc2Disc(n, domain)
    points = disc.points()
    x = chebfun(lambda t: t, domain=domain)
    m = op._n_vars()
    flags = (True,) + (False,) * (m - 1)
    guess = initial if initial is not None else op.init
    if guess is None:
        guess = [0.0] * m
    values = jnp.concatenate(
        [
            jnp.broadcast_to(
                jnp.asarray(guess[0](points) if callable(guess[0]) else guess[0]), (n,)
            ),
            jnp.asarray([g(domain[0]) if callable(g) else g for g in guess[1:]]).reshape(-1),
        ]
    )

    def funs(v):
        return [Chebfun.from_values(v[:n], domain)] + [
            chebfun(v[n + j].item(), domain=domain) for j in range(m - 1)
        ]

    def conditions(us):
        result = []
        for callback, point in ((op._lbc_raw, domain[0]), (op._rbc_raw, domain[-1])):
            if callback is not None:
                rows = callback(*us)
                for row in rows if isinstance(rows, (tuple, list)) else [rows]:
                    result.append(row(point) if callable(row) else row)
        if op._bc_general is not None:
            rows = op._bc_general(*us)
            result.extend(rows if isinstance(rows, (tuple, list)) else [rows])
        return result

    nbc = len(conditions(funs(values)))
    nrows = n + m - 1 - nbc
    projection = barymat(chebpts(nrows, kind=1), chebpts(n, kind=2))
    rhs = jnp.broadcast_to(jnp.asarray(f(points) if callable(f) else f), (n,))

    def evaluate(v, jacobian=False):
        us = funs(v)
        if jacobian:
            us = [ADChebfun(u).seed(i + 1, flags) for i, u in enumerate(us)]
        out = op._call_op(x, us)
        if isinstance(out, (list, tuple)):
            out = out[0]
        bc = conditions(us)
        primal = out.func if jacobian else out
        r = jnp.concatenate(
            [
                projection @ (primal(points) - rhs),
                jnp.asarray([c.func if isinstance(c, ADChebfun) else c for c in bc]).reshape(-1),
            ]
        )
        if not jacobian:
            return r
        matrix = out.jacobian.matrix(n, domain)[0]
        rows = [c.jacobian.matrix(n, domain)[0] for c in bc]
        return r, jnp.concatenate([projection @ matrix, *rows], axis=0)

    history = []
    op._last_info = {"normDelta": history}
    for _ in range(max_iter):
        residual, matrix = evaluate(values, True)
        if float(jnp.max(jnp.abs(residual))) < 1e-11:
            break
        step = jnp.linalg.solve(matrix, residual)
        norm = jnp.linalg.norm(step)
        lam = 1.0
        for _ in range(30):
            candidate = values - lam * step
            try:
                trial = evaluate(candidate)
                simplified = jnp.linalg.solve(matrix, trial)
                if bool(jnp.all(jnp.isfinite(trial))) and (
                    float(jnp.linalg.norm(simplified)) < float(norm) or lam < 1e-6
                ):
                    break
            except (ValueError, FloatingPointError):
                pass
            lam *= 0.5
        values = values - lam * step
        delta = funs(lam * step)
        history.append(float(jnp.sqrt(delta[0].norm() ** 2 + jnp.sum(jnp.abs(lam * step[n:]) ** 2))))
        op._last_info = {"normDelta": history}
        if history[-1] <= 1e-12 * max(1.0, float(jnp.linalg.norm(values))):
            break
    else:
        warnings.warn(
            "chebop parameter Newton: max iterations reached", RuntimeWarning, stacklevel=2
        )
    # linsolve converts only function blocks to Chebfuns; scalar blocks stay numeric.
    return SystemSolution([funs(values)[0], *values[n:].tolist()])
