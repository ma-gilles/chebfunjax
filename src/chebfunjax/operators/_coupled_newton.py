"""Exact coupled function linearization and source error-controlled Newton.

Provenance: Chebfun 7574c77 @chebop/linearize.m, @linop/fitBCs.m,
@chebop/solvebvp.m, @chebop/solvebvpNonlinear.m and dampingErrorBased.m.
Scalar, parameter and periodic solver implementations are unchanged.
"""
import inspect
import math
import warnings

import jax.numpy as jnp

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
from chebfunjax.domain import Domain
from chebfunjax.operators._linear_altdisc import solve_operator
from chebfunjax.operators.blocklinop import BlockLinop
from chebfunjax.operators.blocks import zero_functional, zeros_op
from chebfunjax.operators.chebmatrix import ChebMatrix
from chebfunjax.operators.scalar_newton import _add, _constraint_solve, _divide, _norm


class CoupledParameterVariables(NotImplementedError):
    """Exact source parameter detection requests the established adapter."""


def supported(op):
    """Finite coupled function-only route; no scalar parameters or periodic BCs."""
    return (op._n_vars() >= 2 and not op._periodic
            and not op._has_explicit_scalar_parameters()
            and op._n_params() == 0
            and all(math.isfinite(float(x)) for x in op.domain))


def _entries(value):
    if isinstance(value, ChebMatrix):
        return [row[0] for row in value.blocks]
    return list(value) if isinstance(value, (tuple, list)) else [value]


def _value(value):
    return value.func if isinstance(value, ADChebfun) else value


def _initial(op, state=None):
    count = op._n_vars()
    values = [0]*count if state is None else _entries(state)
    if len(values) != count:
        raise ValueError('Coupled initial state must have one entry per function variable')
    return [value if isinstance(value, Chebfun) else chebfun(value, domain=op.domain)
            for value in values]


def _conditions(op, seeds):
    groups = [[], [], []]
    for k, (callback, point) in enumerate(((op._lbc_raw, op.domain[0]),
                                          (op._rbc_raw, op.domain[-1]))):
        if callback is None:
            continue
        if callable(callback):
            rows = _entries(callback(*seeds))
        else:
            targets = jnp.asarray(callback).reshape(-1)
            if len(targets) != len(seeds):
                raise ValueError('Coupled endpoint values require one target per variable')
            rows = [u-target for u, target in zip(seeds, targets)]
        for row in rows:
            groups[k].append(row(point) if isinstance(_value(row), Chebfun) else row)
    if op._bc_general is not None:
        callback = op._bc_general
        args = [chebfun(lambda t: t, domain=op.domain), *seeds]
        if len(inspect.signature(callback).parameters) == len(seeds):
            args = seeds
        groups[2] = _entries(callback(*args))
    return groups


def linearize(op, state=None):
    """Omitted state means zero, independently of op.init (native public API)."""
    if not supported(op):
        raise NotImplementedError('Exact coupled linearization requires finite function variables')
    initial = _initial(op, state)
    count = len(initial)
    seeds = [ADChebfun(u).seed(j+1, (True,)*count) for j, u in enumerate(initial)]
    x = chebfun(lambda t: t, domain=op.domain)
    outputs = _entries(op._call_op(x, seeds))
    rows = [out.jacobian.blocks[0] if isinstance(out, ADChebfun)
            else [zeros_op(op.domain) for _ in seeds] for out in outputs]
    # Native linearize.m isParam: only classify undifferentiated columns
    # as parameters when another column is differentiated/integrated.
    incidence = [[getattr(block, 'isnotdiffint', True) for block in row]
                 for row in rows]
    if (any(not item for row in incidence for item in row)
            and any(all(row[j] for row in incidence) for j in range(count))):
        raise CoupledParameterVariables(
            'Native linearize detected scalar parameter columns; delegate to parameter route')
    L = BlockLinop(ChebMatrix(rows), domain=op.domain)
    flags = [all(not isinstance(out, ADChebfun) or out.is_linear for out in outputs)]
    for group in _conditions(op, seeds):
        flags.append(all(not isinstance(row, ADChebfun) or row.is_linear for row in group))
        for row in group:
            jacobian = (row.jacobian.blocks[0] if isinstance(row, ADChebfun)
                        else [zero_functional(op.domain) for _ in seeds])
            # Retain structurally zero rows. fitBCs removes trivial numeric
            # rows locally; public linearize/matrix must preserve their count.
            L = L.add_constraint(jacobian, _value(row))
    return L, [_value(out) for out in outputs], tuple(flags)


def fit_bcs(L):
    """Source C2 fitBCs: incidence degrees, five rank attempts, basic solve.

    Trivial numerical rows are removed only from the fit. The source fifth
    attempt returns zeros even if its final rank test would have succeeded.
    No pseudoinverse is substituted for a rank-deficient constraint matrix.
    """
    domain = tuple(L.domain)
    zero = [chebfun(0, domain=domain) for _ in range(L.ncols)]
    if all(getattr(b, 'isnotdiffint', True) for row in L.A.blocks for b in row):
        return zero
    L = L.derive_continuity(domain) if not L.continuity else L
    degrees = [sum(not getattr(row[j], 'iszero', row[j] == 0)
                   for row, _ in L.constraint) for j in range(L.ncols)]
    if L.continuity:
        degrees = [max(d, order-1) for d, order in zip(degrees, L._var_orders())]
    degrees = [max(1, d) for d in degrees]
    for attempt in range(5):
        sizes = [attempt]*(len(domain)-1)
        rows, values = L._constraint_rows(sizes, domain, degrees, [True]*L.ncols)
        width = sum((attempt+d)*(len(domain)-1) for d in degrees)
        matrix = jnp.concatenate(rows, axis=0) if rows else jnp.zeros((0, width))
        values = jnp.asarray(values).reshape(-1)
        keep = jnp.any(matrix != 0, axis=1)
        matrix, values = matrix[keep], values[keep]
        if matrix.shape[0] == 0:
            return zero
        if int(jnp.linalg.matrix_rank(matrix)) == matrix.shape[0]:
            if attempt == 4:
                break
            vector = _constraint_solve(matrix, -values)
            result, pos = [], 0
            for degree in degrees:
                pieces = []
                n = attempt+degree
                for a, b in zip(domain[:-1], domain[1:]):
                    pieces.append(Chebfun.from_values(vector[pos:pos+n], (a, b)).funs[0])
                    pos += n
                result.append(Chebfun(funs=pieces, domain=Domain(domain)))
            return result
    warnings.warn('CHEBFUN:LINOP:fitBCs:failure: Unable to construct a suitable '
                  'initial guess. Using a zero guess.', RuntimeWarning, stacklevel=2)
    return zero


def solve_coupled(op, f=0, n=None, max_iter=25, bvp_tol=5e-13,
                  n_min=32, n_max=4096, backend='chebcolloc2'):
    """Source Newton on coupled function blocks using the shared adaptive solve."""
    # Native solvebvpNonlinear.m75 captures this preference once per solve.
    from chebfunjax.chebpref import ChebopPref

    lambda_min = ChebopPref().lambdaMin
    domain = tuple(op.domain)
    u = _initial(op, op.init)
    L0, _, flags = linearize(op, u)
    if op.init is None and not all(flags):
        u = fit_bcs(L0)
    forcing = _entries(f) if isinstance(f, (list, tuple, ChebMatrix)) else [f]*len(u)
    forcing = [chebfun(v, domain=domain) if callable(v) and not isinstance(v, Chebfun)
               else v for v in forcing]
    if all(flags):
        L, values, _ = linearize(op, u)
        entries, _, _ = solve_operator(L, [v-rhs for v, rhs in zip(values, forcing)],
                                       backend=backend, n=n, n_min=n_min, n_max=n_max,
                                       tol=bvp_tol, vscale=[float(v.vscale) for v in u])
        return _add(u, entries, -1)
    x = chebfun(lambda t: t, domain=domain)
    err_tol = 200*bvp_tol
    grid_history, discretization_history = [], []

    def operator(us):
        return _entries(op._call_op(x, us))

    def residual(us):
        return [v-rhs for v, rhs in zip(operator(us), forcing)]

    def scale(us):
        return [float(v.vscale) for v in us]

    def assemble(us):
        L, values, _ = linearize(op, us)
        return L, [v-rhs for v, rhs in zip(values, forcing)], {}

    def linsolve(linear, rhs, scales, start=None):
        L, _, cache = linear
        entries, disc, happy = solve_operator(
            L, rhs, backend=backend, n=n, n_min=n_min, n_max=n_max,
            tol=bvp_tol, vscale=scales,
            disc=cache.get('disc') if start is not None else None)
        cache['disc'] = disc
        for dimensions in disc.dimension_history:
            grid_history.append(list(dimensions))
            discretization_history.append({'dimensions': list(dimensions),
                'functionPoints': [[size+r for size in dimensions] for r in disc.orders],
                'backend': backend})
        return [-entry for entry in entries], max(disc.dimensions), happy

    history, damping_history = [], []
    damping = bool(getattr(op, "damping", True))
    pref_damping = damping
    lam = 1.0
    give_up = 0
    success = False
    error = float('inf')
    delta_bar = None
    norm_bar = None
    norm_old = None
    counter = 0
    while True:
        linear = assemble(u)
        delta, size, resolved = linsolve(linear, linear[1], scale(u))
        if not resolved:
            give_up = 1
            break
        norm_delta = _norm(delta)
        if counter == 0 and _divide(norm_delta, sum(scale(u))) < bvp_tol:
            history = [norm_delta]
            error = float('nan')
            success = True
            break
        if damping:
            contraction = float("nan")
            prediction = True
            while True:
                if counter > 0 and prediction:
                    mu = _divide(norm_old * norm_bar,
                                 _norm(_add(delta_bar, delta, -1)) * norm_delta) * lam
                    lam = min(1, mu)
                    prediction = False
                if lam < lambda_min:
                    trial = _add(u, delta)
                    lam = 1
                    give_up += .5
                    contraction = float('nan')
                    break
                trial = _add(u, delta, lam)
                delta_bar, size, resolved = linsolve(
                    linear, residual(trial), scale(u), start=size)
                if not resolved:
                    give_up = 1
                    break
                norm_bar = _norm(delta_bar)
                contraction = _divide(norm_bar, norm_delta)
                mu_prime = _divide(.5 * norm_delta * lam ** 2,
                                   _norm(_add(delta_bar, delta, -(1 - lam))))
                if contraction >= 1:
                    lam = min(mu_prime, .5 * lam)
                    continue
                lam_prime = min(1, mu_prime)
                if lam_prime == 1 and norm_bar < err_tol:
                    # Source's provisional trial+deltaBar is overwritten by
                    # its final u=uTrial assignment. Preserve actual behavior.
                    success = True
                    give_up = 0
                    break
                damping = not (lam_prime == 1 and contraction < .5)
                if lam_prime >= 4 * lam:
                    lam = lam_prime
                    continue
                give_up = 0
                break
            u = trial
            error = float('nan')
        else:
            contraction = float('nan') if counter == 0 else _divide(norm_delta, norm_old)
            if counter > 0:
                if contraction >= 1:
                    damping = pref_damping
                    if damping:
                        continue
                    u = _add(u, delta)
                else:
                    error = norm_delta / (1 - contraction ** 2)
                    lam = 1
                    u = _add(u, delta)
        counter += 1
        history.append(norm_delta)
        damping_history.append({'lambda': lam, 'contraction': contraction,
                                'dimension': size, 'damping': damping})
        norm_old = norm_delta
        if error < err_tol:
            success = True
        if success or counter > max_iter or give_up == 1:
            break
    if not success:
        warnings.warn('chebop coupled Newton iteration failed: ' +
                      ('linear solve or damping did not converge' if give_up else
                       'maximum number of iterations exceeded'), RuntimeWarning,
                      stacklevel=2)
    op._last_info = {'normDelta': history, 'error': error, 'converged': success,
                     'dampingHistory': damping_history, 'linearDimensions': grid_history,
                     'linearDiscretizations': discretization_history,
                     'boundaryResidual': [_value(c) for group in _conditions(op, u) for c in group],
                     'bvpTol': bvp_tol, 'newtonTolerance': err_tol}
    if backend != 'chebcolloc2':
        op._last_info['discretization'] = backend
    return [v.simplify(tol=bvp_tol) for v in u]
