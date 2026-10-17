"""Source linear solves with genuine numeric parameter columns.

Provenance
----------
MATLAB source: @chebop/linearize.m, @linop/linsolve.m,
@chebop/solvebvpLinear.m, @adchebfun/adchebfun.m (seed).
Chebfun commit: 7574c77
"""

import inspect
import math

import jax.numpy as jnp

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
from chebfunjax.operators._coupled_newton import _entries, _value
from chebfunjax.operators._linear_altdisc import solve_operator
from chebfunjax.operators.blocklinop import BlockLinop
from chebfunjax.operators.blocks import eval_at, zero_functional
from chebfunjax.operators.chebmatrix import ChebMatrix


def prepare(op):
    """Return a typed linearization or None for unsupported legacy syntax.

    Function/parameter incidence comes from the existing source-typed linop
    extraction. Parameters occupy one numeric unknown, with no added p'=0
    equation. AD verifies both differential and constraint linearity.
    """
    if (op._n_vars() < 2 or op._periodic or len(op.domain) != 2
            or not all(math.isfinite(float(x)) for x in op.domain)):
        return None
    try:
        flags = tuple(op.linop().is_fun_variable())
    except (NotImplementedError, TypeError, AttributeError):
        return None
    if all(flags) or not any(flags):
        return None
    domain = tuple(op.domain)
    zero = chebfun(0., domain=domain)
    seeds = [ADChebfun(zero).seed(i+1, flags) for i in range(len(flags))]
    for seed, flag in zip(seeds, flags):
        if not flag:
            seed.func = 0.
    x = chebfun(lambda t: t, domain=domain)
    outputs = _entries(op._call_op(x, seeds))
    groups = _conditions(op, seeds, flags, x)
    if (not all(isinstance(row, ADChebfun) and row.is_linear for row in outputs)
            or not all(not isinstance(row, ADChebfun) or row.is_linear
                       for group in groups for row in group)):
        return None
    rows = [row.jacobian.blocks[0] for row in outputs]
    L = BlockLinop(ChebMatrix(rows, domain=domain), domain=domain)
    # Restrict this route to differential function rows. Algebraic/functional
    # equations and alternative discretizations retain their existing routes.
    if any(L._row_is_functional(i) for i in range(L.nrows)):
        return None
    for group in groups:
        for condition in group:
            row = (list(condition.jacobian.blocks[0])
                   if isinstance(condition, ADChebfun) else
                   [zero_functional(domain) if flag else 0. for flag in flags])
            for j, flag in enumerate(flags):
                if not flag and isinstance(row[j], Chebfun):
                    row[j] = _constant(row[j], domain)
            L = L.add_constraint(row, -_constant(_value(condition), domain))
    return L, [_value(row) for row in outputs], flags


def _conditions(op, seeds, flags, x):
    """Native endpoint evaluation and general-BC parameter contraction.

    @chebop/linearize.m246-279 contracts scalar AD derivatives before the
    general boundary callback. This prevents functional+zero-operator
    promotion from changing an actual scalar condition into a function row.
    """
    domain = tuple(op.domain)
    groups = []
    for callback, point in ((op._lbc_raw, domain[0]), (op._rbc_raw, domain[-1])):
        if callback is None:
            groups.append([])
            continue
        if callable(callback):
            rows = _entries(callback(*seeds))
        else:
            values = jnp.asarray(callback).reshape(-1)
            if len(values) != len(seeds):
                raise ValueError('numeric system boundary conditions require one value per unknown')
            rows = [seed-v for seed, v in zip(seeds, values)]
        groups.append([row(point) if isinstance(_value(row), Chebfun) else row
                       for row in rows])
    if op._bc_general is not None:
        contracted = []
        evaluator = eval_at(domain[0], domain)
        for seed, flag in zip(seeds, flags):
            if flag:
                contracted.append(seed)
            else:
                scalar = seed + 0.
                scalar.jacobian = evaluator * seed.jacobian
                contracted.append(scalar)
        args = [x, *contracted]
        if len(inspect.signature(op._bc_general).parameters) == len(seeds):
            args = contracted
        groups.append(_entries(op._bc_general(*args)))
    return groups


def _constant(value, domain):
    """Scalarize an actual constant constraint, never an arbitrary function."""
    if isinstance(value, Chebfun):
        coefficients = [jnp.asarray(piece.tech.coeffs) for piece in value.funs]
        if (not coefficients or any(c[0].size != 1 or not bool(jnp.all(c[1:] == 0))
                                    for c in coefficients)
                or any(not bool(jnp.all(c[0] == coefficients[0][0]))
                       for c in coefficients)):
            raise ValueError('General boundary conditions must return scalar residuals')
        constant = coefficients[0][0].reshape(())
        points = getattr(value, '_point_values', None)
        if points is not None and not bool(jnp.all(points == constant)):
            raise ValueError('General boundary conditions must return scalar residuals')
        return constant.item()
    return jnp.asarray(value).reshape(()).item()


def solve(op, f, prepared, *, n, n_min, n_max, tol):
    """Adaptive source C2 solve and numeric parameter output conversion."""
    from chebfunjax.operators.chebop import SystemSolution

    L, values, flags = prepared
    forcing = _entries(f) if isinstance(f, (list, tuple, ChebMatrix)) else [f]*L.nrows
    if len(forcing) != L.nrows:
        raise ValueError('Mixed linear RHS must have one entry per equation')
    forcing = [chebfun(v, domain=op.domain) if callable(v) and not isinstance(v, Chebfun)
               else v for v in forcing]
    result, disc, converged = solve_operator(
        L, [rhs-value for rhs, value in zip(forcing, values)],
        backend='chebcolloc2', n=n, n_min=n_min, n_max=n_max, tol=tol)
    op._last_info = {'converged': converged, 'linearDimensions': disc.dimension_history,
                     'isFun': flags, 'bvpTol': tol}
    return SystemSolution(result)
