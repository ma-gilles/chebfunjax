"""Draft R2017a qpsub finite-box path and its literal phase-I LP.

Source: shared/optimlib/qpsub.m, matlab/matfun/{qrinsert,qrdelete,planerot}.m.
Bound by native_box_active_set_source_20261009/SOURCE_BINDINGS_v1.json.
Real float64 array arithmetic is JAX; dynamic working sets use host control.
Not JIT/AD, not a complete SQP optimizer; never registered with public extrema.
The only root constraints supported are the four finite two-dimensional box
faces. Phase I is the native augmented three-dimensional five-constraint LP.
"""
from typing import NamedTuple

import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from chebfunjax.utils._active_set_qp import (
    NEGATIVE_CURVATURE,
    NEWTON,
    STEEPEST_DESCENT,
    _real64,
    compdir,
    find_blocking_constraint,
)

ZERO_STEP, SINGULAR = 3, 4
EPS = jnp.finfo(jnp.float64).eps


class QPResult(NamedTuple):
    x: object
    multipliers: object
    exitflag: int
    iterations: int
    violation: object
    how: str
    active: tuple


def _indices(mask):
    return [i for i, value in enumerate(mask.tolist()) if value]


def _planerot(x):
    if bool(x[1] != 0):
        radius = jnp.linalg.norm(x)
        rotation = jnp.stack((x, jnp.stack((-x[1], x[0])))) / radius
        return rotation, jnp.stack((radius, jnp.asarray(0., dtype=x.dtype)))
    return jnp.eye(2, dtype=x.dtype), x


def _insert(q, r, index, column):
    """Native column qrinsert; zero-based position, square Q."""
    if r.shape[1] == 0:
        return jnp.linalg.qr(column[:, None], mode='complete')
    r = jnp.concatenate((r[:, :index], (q.T @ column)[:, None], r[:, index:]), axis=1)
    n = r.shape[1]
    for k in range(q.shape[0]-2, index-1, -1):
        rotation, values = _planerot(r[k:k+2, index])
        r = r.at[k:k+2, index].set(values)
        if k+1 < n:
            r = r.at[k:k+2, k+1:n].set(rotation @ r[k:k+2, k+1:n])
        q = q.at[:, k:k+2].set(q[:, k:k+2] @ rotation.T)
    return q, r


def _delete(q, r, index):
    """Native column qrdelete for square Q; source Givens order."""
    r = jnp.concatenate((r[:, :index], r[:, index+1:]), axis=1)
    m, n = r.shape
    for k in range(index, min(n, m-1)):
        rotation, values = _planerot(r[k:k+2, k])
        r = r.at[k:k+2, k].set(values)
        if k+1 < n:
            r = r.at[k:k+2, k+1:n].set(rotation @ r[k:k+2, k+1:n])
        q = q.at[:, k:k+2].set(q[:, k:k+2] @ rotation.T)
    return q, r


def _multipliers(q, r, gradient):
    # R is n-by-k upper trapezoidal, k<=n: the lower least-squares residual
    # does not change the full-column-rank triangular solution.
    k = r.shape[1]
    return -solve_triangular(r[:k, :k], (q.T @ gradient)[:k], lower=False)


def _block(a, step, residual, threshold, active):
    if a.shape == (4, 2):
        result = find_blocking_constraint(a, step, residual, threshold, active)
        return bool(jnp.any(result.eligible)), int(result.index), result.distance
    # Literal same native helper for the augmented phase-I shape.
    directional = a @ step
    eligible = (directional > threshold*jnp.linalg.norm(step)) & ~active
    ind = _indices(eligible)
    if not ind:
        return False, -1, jnp.asarray(1e16)
    distances = jnp.abs(residual[jnp.asarray(ind)]) / directional[jnp.asarray(ind)]
    nearest = jnp.min(distances)
    first = _indices(distances == nearest)[0]
    return True, ind[first], nearest


def _pinv_source(matrix):
    # MATLAB pinv default threshold is max(size(A))*eps(norm(A)); eps here
    # is spacing at the largest singular value, NOT eps*norm(A).
    u, singular, vh = jnp.linalg.svd(matrix, full_matrices=False)
    largest = singular[0]
    _, exponent = jnp.frexp(largest)
    spacing = jnp.where(largest == 0, jnp.asarray(float.fromhex('0x0.0000000000001p-1022')),
                        jnp.ldexp(jnp.asarray(1.), exponent-53))
    keep = singular > max(matrix.shape)*spacing
    inv = jnp.where(keep, 1/jnp.where(keep, singular, 1), 0)
    return (vh.T * inv) @ u.T


def box_qp(hessian, gradient, lower, upper, *, initial=None, active=(),
           max_iterations=40, tolerance=1e-6, native_randn=None, observer=None):
    """Native qpsub step for a finite 2D box; unqualified implementation draft.

    native_randn is an explicit native-stream dependency only if the source's
    singular-working-set branch is reached. Its absence raises, never replaces
    the branch with a deterministic perturbation. observer receives events only.
    """
    h = _real64(hessian, 'hessian')
    g = _real64(gradient, 'gradient')
    lo, hi = _real64(lower, 'lower'), _real64(upper, 'upper')
    x = jnp.zeros(2, dtype=jnp.float64) if initial is None else _real64(initial, 'initial')
    if h.shape != (2, 2) or any(v.shape != (2,) for v in (g, lo, hi, x)):
        raise ValueError('box_qp requires two variables')
    if not bool(jnp.all(jnp.isfinite(lo) & jnp.isfinite(hi) & (lo < hi))):
        raise ValueError('box_qp requires finite positive-width bounds')
    if len(set(active)) != len(active) or any(i not in range(4) for i in active):
        raise ValueError('active indices must be distinct box-face indices')
    a = jnp.concatenate((-jnp.eye(2), jnp.eye(2)))
    b = jnp.concatenate((-lo, hi))
    return _qpsub(h, g, a, b, x, list(active), max_iterations, tolerance,
                  False, native_randn, observer)


def _qpsub(h, f, a, b, x, active, maxiter, tolcon, phase, randn, observer):
    n, nc = x.size, b.size
    if (phase and (n, nc) != (3, 5)) or (not phase and (n, nc) != (2, 4)):
        raise ValueError('Only root 2D box and native 3D phase-I LP are supported')
    is_qp = h is not None and bool(jnp.linalg.norm(h, ord=jnp.inf) != 0)
    normf = jnp.asarray(1.)
    if not phase and not is_qp:
        normf = jnp.linalg.norm(f)
        if bool(normf > 0):
            f = f/normf
    norms = jnp.ones(nc)
    if not phase:
        for i in range(nc):
            norm = jnp.linalg.norm(a[i])
            if bool(norm != 0):
                a = a.at[i].set(a[i]/norm)
                b = b.at[i].set(b[i]/norm)
                norms = norms.at[i].set(norm)
    threshold = jnp.asarray(0.01)*jnp.sqrt(EPS)
    multipliers = jnp.zeros(nc)
    how, flag, iteration = 'ok', 1, 0
    violation = jnp.asarray(0.)

    def emit(event, **values):
        if observer is not None:
            observer(event, iteration, tuple(active), values)

    def done():
        return QPResult(x, multipliers, flag, iteration, violation, how, tuple(active))

    def mask():
        return jnp.zeros(nc, dtype=jnp.bool_).at[jnp.asarray(active, dtype=jnp.int32)].set(True)

    # eqnsolv specialization proof: no equalities; root n=2 truncates warm
    # active set to at most1; a retained box row has rank1 and one nonzero.
    if active and not phase:
        active = active[:n-1]
        aset = a[jnp.asarray(active)]
        q, r = jnp.linalg.qr(aset.T, mode='complete')
        z = q[:, len(active):]
        delta = solve_triangular(r[:1, :1].T, b[jnp.asarray(active)]-aset@x, lower=True)
        x = x + q[:, :1] @ delta
        errors = a@x-b
        if bool(jnp.any(errors > EPS)):
            coordinate = int(jnp.argmax(jnp.abs(aset[0])))
            basic = jnp.zeros(n).at[coordinate].set(b[active[0]]/aset[0, coordinate])
            if bool(jnp.max(a@basic-b) < jnp.max(errors)):
                x = basic
        emit('warm_start_projected', x=x)
    elif not active:
        q, r, z = jnp.eye(n), jnp.empty((n, 0)), jnp.eye(n)
    else:
        q, r = jnp.linalg.qr(a[jnp.asarray(active)].T, mode='complete')
        z = q[:, len(active):]
    residual = a@x-b
    violation = jnp.max(residual)
    scaled = tolcon/norms[int(jnp.argmax(residual))]
    if bool(violation > scaled):
        if phase:
            raise ValueError('Phase-I initial feasibility invariant failed; nested dimension unsupported')
        emit('phase_one_enter', x=x, violation=violation)
        augmented = jnp.concatenate((jnp.column_stack((a, -jnp.ones(nc))),
                                     jnp.asarray([[0., 0., -1.]])))
        result = _qpsub(None, jnp.asarray([0., 0., 1.]), augmented,
                        jnp.concatenate((b, jnp.asarray([1e-5]))),
                        jnp.concatenate((x, (violation+1)[None])), [],
                        10*max(n, nc), tolcon, True, randn, observer)
        slack, x = result.x[-1], result.x[:-1]
        residual = a@x-b
        violation = jnp.max(residual)
        scaled = tolcon/norms[int(jnp.argmax(residual))]
        if bool(slack > scaled):
            how = 'infeasible' if bool(slack > 1e-8) else 'overly constrained'
            flag = -2
            multipliers = normf*(result.multipliers[:nc]/norms)
            active = []
            violation = jnp.empty((0,))
            return done()
        active = []
        # Literal source qpsub313: Q=zeros after successful phase I, Z=1.
        q, r, z = jnp.zeros((n, n)), jnp.empty((n, 0)), jnp.eye(n)
        emit('phase_one_return', x=x, slack=slack)
    simplex = len(active) >= n-1
    gf = h@x+f if is_qp else f
    if is_qp:
        sd, kind = compdir(z, h, gf)
        kind = int(kind)
    else:
        sd, kind = (-z@z.T)@gf, STEEPEST_DESCENT
    oldind = -1
    lind = -1
    rlambda = jnp.empty((0,))
    while iteration < maxiter:
        iteration += 1
        eligible, ind, step = _block(a, sd, residual, threshold, mask())
        delete = False
        if eligible and bool(jnp.isfinite(step)):
            if kind == NEWTON and bool(step > 1):
                step, delete = jnp.asarray(1.), True
            x = x + step*sd
        elif kind == NEWTON:
            step, delete = jnp.asarray(1.), True
            x = x+sd
        else:
            if not is_qp or kind == NEGATIVE_CURVATURE:
                if bool(jnp.linalg.norm(sd) > threshold):
                    step = jnp.abs((x[-1]+1e-5)/(sd[-1]+EPS)) if phase else jnp.asarray(1e16)
                    x = x+step*sd
                    how, flag = 'unbounded', -3
                else:
                    how, flag = 'ill posed', -7
                return done()
            projected = (z.T@h)@z
            zg = z.T@gf
            psd = _pinv_source(projected)@(-zg)
            if bool(jnp.linalg.norm(projected@psd+zg) > 10*EPS*(jnp.linalg.norm(projected, ord=2)+jnp.linalg.norm(zg))):
                if bool(jnp.linalg.norm(sd) > threshold):
                    step = jnp.abs((x[-1]+1e-5)/(sd[-1]+EPS)) if phase else jnp.asarray(1e16)
                    x = x+step*sd
                    how, flag = 'unbounded', -3
                else:
                    how, flag = 'ill posed', -7
                return done()
            sd = z@psd
            if bool(gf@sd > 0):
                sd = -sd
            kind = SINGULAR
            eligible, ind, step = _block(a, sd, residual, threshold, mask())
            if bool(step > 1):
                step, delete = jnp.asarray(1.), True
            x = x+step*sd
        if is_qp:
            gf = h@x+f
        residual = a@x-b
        violation = jnp.max(residual)
        emit('move', x=x, direction=sd, step=step, kind=kind)
        if delete:
            if not active:
                return done()
            rlambda = _multipliers(q, r, gf)
            negative = _indices(rlambda < 0)
            if not negative:
                multipliers = multipliers.at[jnp.asarray(active)].set(normf*(rlambda/norms[jnp.asarray(active)]))
                return done()
            lind = min(negative, key=lambda i: active[i])
            q, r = _delete(q, r, lind)
            del active[lind]
            simplex, ind = False, -1
            emit('delete_at_minimum', removed_position=lind)
        if phase and bool(x[-1] < EPS):
            return done()
        if bool(violation > 1e5*threshold):
            how, flag = 'unreliable', -2
        else:
            how, flag = 'ok', 1
        if ind >= 0:
            q, r = _insert(q, r, len(active), a[ind])
            active.append(ind)
            emit('insert', index=ind)
        if not simplex:
            z = q[:, len(active):]
            if len(active) == n-1:
                simplex = True
            oldind = -1
        else:
            rlambda = _multipliers(q, r, gf)
            if bool(jnp.isneginf(rlambda[0])):
                if randn is None:
                    raise RuntimeError('Native singular-set RNG stream is required by qpsub640')
                perturbation = _real64(randn((len(active), n)), 'native_randn result')
                if perturbation.shape != (len(active), n):
                    raise ValueError('native_randn returned wrong shape')
                perturbed = (a[jnp.asarray(active)]+jnp.sqrt(EPS)*perturbation).T
                if perturbed.shape[0] != perturbed.shape[1]:
                    raise RuntimeError('Native rectangular singular-set backslash remains unimplemented')
                rlambda = jnp.linalg.solve(perturbed, -gf)
            negative = _indices(rlambda < 0)
            if not negative:
                multipliers = multipliers.at[jnp.asarray(active)].set(normf*(rlambda/norms[jnp.asarray(active)]))
                return done()
            lind = int(jnp.argmin(rlambda)) if bool(step > threshold) else min(negative, key=lambda i: active[i])
            oldind = active[lind]
            q, r = _delete(q, r, lind)
            del active[lind]
            z = q[:, -1:]
            emit('delete_simplex', index=oldind)
        if is_qp:
            zg = z.T@gf
            if zg.size and bool(jnp.linalg.norm(zg) < 1e-15):
                sd, kind = jnp.zeros(n), ZERO_STEP
            else:
                sd, kind = compdir(z, h, gf)
                kind = int(kind)
        else:
            if not simplex:
                sd = -z@(z.T@gf)
                gradsd = jnp.linalg.norm(sd)
            else:
                gradsd = (z.T@gf).reshape(())
                sd = (-z if bool(gradsd > 0) else z).reshape(n)
            if bool(jnp.abs(gradsd) < 1e-10):
                if oldind < 0:
                    rlambda = _multipliers(q, r, gf)
                    temporary, qt, rt = list(active), q, r
                else:
                    temporary = list(active)
                    temporary.insert(lind, oldind)
                    qt, rt = _insert(q, r, lind, a[oldind])
                negative = _indices(rlambda < threshold)
                multipliers = multipliers.at[jnp.asarray(temporary, dtype=jnp.int32)].set(normf*(rlambda/norms[jnp.asarray(temporary, dtype=jnp.int32)]))
                if not negative:
                    return done()
                m = len(temporary)
                for candidate in negative:
                    if not bool(jnp.abs(gradsd) < 1e-10):
                        break
                    lind = candidate
                    q, r = _delete(qt, rt, lind)
                    z = q[:, m-1:n]
                    if m != n:
                        sd = (-z@z.T)@gf
                        gradsd = jnp.linalg.norm(sd)
                    else:
                        gradsd = (z.T@gf).reshape(())
                        sd = (-z if bool(gradsd > 0) else z).reshape(n)
                if bool(jnp.abs(gradsd) < 1e-10):
                    return done()
                active = list(temporary)
                del active[lind]
                multipliers = jnp.zeros(nc)
                emit('lp_null_direction_delete', removed_position=lind)
    flag, how = 0, 'MaxSQPIter'
    return done()
