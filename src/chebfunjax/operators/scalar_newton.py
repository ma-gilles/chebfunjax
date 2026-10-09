"""Source error-controlled Newton iteration for scalar finite-interval BVPs.

Provenance: Chebfun 7574c77 @chebop/solvebvpNonlinear.m,
@chebop/dampingErrorBased.m, @linop/linsolve.m and
@opDiscretization/testConvergence.m. The public adapter currently supports
one scalar function on one finite interval. The Newton loop follows the
qualified parameter_newton implementation; that separate path is unchanged.

Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/.
"""

import inspect
import math
import warnings

import jax.numpy as jnp
from jax.scipy.linalg import lu_factor, lu_solve

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.utils.interpolation import barymat
from chebfunjax.utils.quadrature import chebpts


def _norm(blocks):
    """CHEBMATRIX Frobenius norm: continuous L2 and numeric scalar blocks."""
    return float(jnp.sqrt(sum(b.norm(2) ** 2 if isinstance(b, Chebfun)
                              else abs(b) ** 2 for b in blocks)))


def _add(left, right, factor=1):
    return [a + factor * b for a, b in zip(left, right)]


def _divide(a, b):
    # Match IEEE MATLAB arithmetic, including zero-denominator predictions.
    return float(jnp.asarray(a) / jnp.asarray(b))



def _conditions(op, u):
    """Source parseBC/linearize: evaluate endpoint and general conditions."""
    result = []
    for callback, point in ((op._lbc_raw, op.domain[0]),
                            (op._rbc_raw, op.domain[-1])):
        if callback is None:
            continue
        if callable(callback):
            rows = callback(u)
        else:
            rows = [u.diff(k) - target if k else u - target
                    for k, target in enumerate(jnp.asarray(callback).reshape(-1))]
        for row in rows if isinstance(rows, (tuple, list)) else [rows]:
            is_function = (isinstance(row.func, Chebfun) if
                           isinstance(row, ADChebfun) else callable(row))
            result.append(row(point) if is_function else row)
    if op._bc_general is not None:
        callback = op._bc_general
        if len(inspect.signature(callback).parameters) > 1:
            rows = callback(chebfun(lambda t: t, domain=op.domain), u)
        else:
            rows = callback(u)
        result.extend(rows if isinstance(rows, (tuple, list)) else [rows])
    return result


def _constraint_solve(matrix, values):
    """MATLAB backslash: square solve or basic full-row-rank solution.

    Column-pivoted orthogonalization selects independent columns for the
    underdetermined case; unconstrained coefficients are zero. This preserves
    the basic-solution semantics, without claiming LAPACK pivot-bit identity.
    """
    rows, columns = matrix.shape
    if rows == columns:
        return jnp.linalg.solve(matrix, values)
    work = matrix
    selected = []
    for _ in range(rows):
        norms = jnp.sum(jnp.abs(work)**2, axis=0)
        if selected:
            norms = norms.at[jnp.asarray(selected)].set(-1)
        pivot = int(jnp.argmax(norms))
        selected.append(pivot)
        unit = work[:, pivot] / jnp.linalg.norm(work[:, pivot])
        work = work - unit[:, None] * (jnp.conj(unit) @ work)[None, :]
    indices = jnp.asarray(selected)
    answer = jnp.zeros(columns, dtype=jnp.result_type(matrix, values))
    return answer.at[indices].set(jnp.linalg.solve(matrix[:, indices], values))


def fit_scalar_bcs(op):
    """@linop/fitBCs: lowest constraint-count degree, five rank attempts.

    Provenance
    ----------
    MATLAB source: @linop/fitBCs.m.
    Chebfun commit: 7574c77

    Single finite interval only; no continuity rows are
    required. Linearize conditions at zero, remove trivial rows, solve in
    Chebcolloc2 values. Source's fifth attempt still returns the zero fallback.
    """
    domain = tuple(op.domain)
    if len(domain) != 2 or not all(math.isfinite(x) for x in domain):
        raise ValueError('scalar fitBCs requires one finite interval')
    zero = chebfun(0, domain=domain)
    seed = ADChebfun(zero).seed(1, (True,))
    linear = op._call_op(chebfun(lambda t: t, domain=domain), [seed])
    if isinstance(linear, ADChebfun) and linear.jacobian.isnotdiffint:
        return zero
    rows = _conditions(op, seed)
    degree = max(1, sum(isinstance(row, ADChebfun) and not row.jacobian.iszero
                        for row in rows))
    for attempt in range(5):
        size = degree + attempt
        matrices = [row.jacobian.matrix(size).reshape(-1)
                    if isinstance(row, ADChebfun) else jnp.zeros(size)
                    for row in rows]
        matrix = jnp.stack(matrices) if matrices else jnp.zeros((0, size))
        values = jnp.asarray([row.func if isinstance(row, ADChebfun) else row
                              for row in rows]).reshape(-1)
        keep = jnp.any(matrix != 0, axis=1)
        matrix, values = matrix[keep], values[keep]
        if matrix.shape[0] == 0:
            return zero
        if int(jnp.linalg.matrix_rank(matrix)) == matrix.shape[0]:
            if attempt == 4:
                break
            vals = _constraint_solve(matrix, -values)
            return Chebfun.from_values(vals, domain)
    warnings.warn('CHEBFUN:LINOP:fitBCs:failure: Unable to construct a suitable '
                  'initial guess. Using a zero guess.', RuntimeWarning, stacklevel=2)
    return zero


def _validate_initial(op, domain, n_min):
    """Preserve the legacy source zero-seed validation before fitting BCs.

    No AD rules are changed here. This checks the same op evaluation and
    linearize_op errors as the former scalar Chebop method.
    """
    from chebfunjax.autodiff.adchebfun import linearize_op
    from chebfunjax.operators.blocks import ChebColloc2Disc
    from chebfunjax.operators.chebop import _chebfun_to_values

    size = max(n_min, 16)
    disc = ChebColloc2Disc(size, domain)
    initial = op.init if op.init is not None else 0
    u = initial if isinstance(initial, Chebfun) else chebfun(initial, domain=domain)
    x = chebfun(lambda t: t, domain=domain)
    try:
        valid = bool(jnp.all(jnp.isfinite(_chebfun_to_values(op._apply_op(x, u), disc))))
    except (ValueError, FloatingPointError):
        valid = False
    if valid:
        try:
            linearize_op(op.op, u, domain=domain)
        except ValueError as exc:
            if 'nansInfs' in str(exc) or 'divisionByZero' in str(exc):
                valid = False
        except Exception:
            pass
    if not valid:
        raise ValueError('CHEBFUN:CHEBOP:linearize:invalidInitialGuess: '
                         'the operator cannot be evaluated at the initial '
                         'guess (non-finite residual); supply N.init.')


def solve_scalar(op, f=0, n=None, max_iter=25, initial=None,
                    bvp_tol=5e-13, n_min=32, n_max=4096, backend="chebcolloc2"):
    """Resolve each correction before source damping and L2 error stopping.

    Provenance
    ----------
    MATLAB source: @chebop/solvebvpNonlinear.m,
        @chebop/dampingErrorBased.m, @linop/linsolve.m.
    Chebfun commit: 7574c77
    Fixed n is an explicit nonadaptive extension; n counts equation points.
    """
    if backend not in ("chebcolloc2", "chebcolloc1", "ultraS"):
        raise ValueError(f"Unsupported scalar Newton backend {backend!r}")
    domain = tuple(op.domain)
    _validate_initial(op, domain, n_min)
    x = chebfun(lambda t: t, domain=domain)
    m = 1
    flags = (True,)
    guess = initial if initial is not None else op.init
    if guess is None:
        guess = fit_scalar_bcs(op)
    u = [guess if isinstance(guess, Chebfun) else chebfun(guess, domain=domain)]
    err_tol = 200 * bvp_tol
    low, high = math.log2(n_min), math.log2(n_max)
    if low > high:
        raise ValueError("minimum discretization exceeds maximum discretization")
    powers = []
    p = low
    while p <= min(high, 9):
        powers.append(p)
        p += 1
    p = low if low >= 9 else 9.5
    if high > 9:
        while p <= high:
            if p not in powers:
                powers.append(p)
            p += .5
    dimensions = [int(math.floor(2 ** p + .5)) for p in powers]
    if callable(f) and not isinstance(f, Chebfun):
        f = chebfun(f, domain=domain)
    grid_history = []
    discretization_history = []

    def conditions(us):
        return _conditions(op, us[0])

    def operator(us):
        out = op._call_op(x, us)
        return out[0] if isinstance(out, (list, tuple)) else out

    def linearize(us):
        seeds = [ADChebfun(v if isinstance(v, Chebfun) else
                           chebfun(v, domain=domain)).seed(i + 1, flags)
                 for i, v in enumerate(us)]
        return operator(seeds), conditions(seeds), {}

    def value(row):
        return row.func if isinstance(row, ADChebfun) else row

    def scale(us):
        return [float(v.vscale) if isinstance(v, Chebfun) else abs(v) for v in us]

    def linsolve(linear, residual, vscale, start=None):
        """Resolve a correction with frozen differential and BC Jacobians."""
        out, bc, cache = linear
        order = max(0, out.jacobian.diff_order)
        if backend != "chebcolloc2":
            from chebfunjax.operators._linear_altdisc import solve_operator
            from chebfunjax.operators.blocklinop import BlockLinop
            from chebfunjax.operators.blocks import zero_functional

            if "alternate_operator" not in cache:
                operator = BlockLinop(out.jacobian, domain=domain)
                for condition in bc:
                    row = (condition.jacobian if isinstance(condition, ADChebfun)
                           else zero_functional(domain))
                    operator = operator.add_constraint(row, value(condition))
                cache["alternate_operator"] = operator
            # The source passes the previous discretization only for the
            # simplified Newton trial, preserving the frozen operator/BC RHS.
            previous = cache.get("alternate_disc") if start is not None else None
            entries, actual_disc, happy = solve_operator(
                cache["alternate_operator"], [residual], backend=backend,
                n=n, n_min=n_min, n_max=n_max, tol=bvp_tol, vscale=[vscale],
                disc=previous)
            cache["alternate_disc"] = actual_disc
            for dimensions_used in actual_disc.dimension_history:
                size_used = int(dimensions_used[0])
                grid_history.append(size_used)
                discretization_history.append({
                    'dimension': size_used, 'functionPoints': size_used + order,
                    'equationPoints': size_used, 'diffOrder': order,
                    'backend': backend})
            return [-entry for entry in entries], int(actual_disc.dimensions[0]), happy
        if len(bc) != order + m - 1:
            raise ValueError("scalar linearization is not square: "
                             "boundary count must equal differential order "
                             "plus number of scalar parameters")
        schedule = ([int(n)] if n is not None else dimensions if start is None
                    else [start] + [d for d in dimensions if d > start])
        for size in schedule:
            grid_history.append(size)
            # Source dimension counts OUTPUT values. The function unknown
            # grid has dimAdjust=diffOrder additional second-kind points.
            unknown_size = size + order
            discretization_history.append({'dimension': size,
                                           'functionPoints': unknown_size,
                                           'equationPoints': size,
                                           'diffOrder': order})
            if size not in cache:
                reference_unknown = chebpts(unknown_size, kind=2)
                reference_equation = chebpts(size, kind=1)
                points = (domain[-1] * (reference_equation + 1) / 2
                          + domain[0] * (1 - reference_equation) / 2)
                projection = barymat(
                    reference_equation, reference_unknown,
                    Chebtech2.barywts(unknown_size),
                    Chebtech1.angles(size), Chebtech2.angles(unknown_size),
                    do_flip=True)
                matrix = out.jacobian.matrix(unknown_size)
                rows = [c.jacobian.matrix(unknown_size).reshape(1, -1)
                        if isinstance(c, ADChebfun) else
                        jnp.zeros((1, unknown_size + m - 1)) for c in bc]
                # Source constraints precede differential-equation rows.
                matrix = jnp.concatenate([*rows, projection @ matrix], axis=0)
                # @valsDiscretization/mldivide: row scaling before LU.
                scaling = 1 / jnp.maximum(1, jnp.max(jnp.abs(matrix), axis=1))
                cache[size] = (points, projection, scaling,
                               lu_factor(scaling[:, None] * matrix))
            points, projection, scaling, factors = cache[size]
            # @valsDiscretization/rhs evaluates the continuous residual on
            # equation points directly; it does not project sampled RHS data.
            rhs = residual(points) if callable(residual) else jnp.full(size, residual)
            rhs = jnp.concatenate([jnp.asarray([value(c) for c in bc]).reshape(-1), rhs])
            vals = -lu_solve(factors, scaling * rhs)
            if not bool(jnp.all(jnp.isfinite(vals))):
                return None, size, False
            projected = projection @ vals[:unknown_size]
            # linsolve applies P*v before both happiness and output conversion.
            # chebcolloc2/toFunctionOut converts these FIRST-kind values to
            # coefficients, then to SECOND-kind Chebfun constructor values.
            coefficients = Chebtech1.vals2coeffs(projected)
            full = Chebfun.from_values(Chebtech2.coeffs2vals(coefficients), domain)
            tech = full.funs[0].tech
            happy, cutoff = Chebtech2.happiness_check(
                tech.coeffs, tech.values, tol=bvp_tol, vscale=vscale,
                sample_test=False)
            if happy or n is not None or size == schedule[-1]:
                keep = cutoff if happy else size
                function = Chebfun.from_values(
                    Chebtech2.coeffs2vals(coefficients[:keep]), domain)
                update = [function, *vals[unknown_size:].tolist()]
                return update, size, happy or n is not None
            vscale = max(vscale, float(full.vscale))

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
        linear = linearize(u)
        residual = value(linear[0]) - f
        delta, size, resolved = linsolve(linear, residual, scale(u)[0])
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
                if lam < 1e-6:
                    trial = _add(u, delta)
                    lam = 1
                    give_up += .5
                    contraction = float('nan')
                    break
                trial = _add(u, delta, lam)
                delta_bar, size, resolved = linsolve(
                    linear, operator(trial) - f, scale(u)[0], start=size)
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
        warnings.warn('chebop scalar Newton iteration failed: ' +
                      ('linear solve or damping did not converge' if give_up else
                       'maximum number of iterations exceeded'), RuntimeWarning,
                      stacklevel=2)
    op._last_info = {'normDelta': history, 'error': error, 'converged': success,
                     'dampingHistory': damping_history, 'linearDimensions': grid_history,
                     'linearDiscretizations': discretization_history,
                     'boundaryResidual': [value(c) for c in conditions(u)],
                     'bvpTol': bvp_tol, 'newtonTolerance': err_tol}
    if backend != 'chebcolloc2':
        op._last_info['discretization'] = backend
    return u[0].simplify(tol=bvp_tol)
