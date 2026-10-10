"""Draft literal R2017a nlconst finite-box SQP with unresolved FD injection.

Source shared/optimlib/nlconst.m and fmincon.m, bound R2017a installation.
No public registration. A finite_difference provider MUST supply the protected
native helper semantics; no default approximation or analytic substitution.
Host-controlled JAX array arithmetic; no complete JIT/AD or MATLAB bit claim.
"""
from typing import NamedTuple

import jax.numpy as jnp

from chebfunjax.utils._active_set_box_qp import EPS, box_qp
from chebfunjax.utils._active_set_qp import (
    _dot2,
    _matvec2,
    _real64,
    _scalar_matrix_divide,
)


class SQPResult(NamedTuple):
    x: object
    value: object
    multipliers: object
    exitflag: int
    iterations: int
    evaluations: int
    gradient: object
    hessian: object
    optimality: object


def _optimality(gradient, a, multipliers, residual):
    lagrangian = jnp.linalg.norm(gradient+a.T@multipliers, ord=jnp.inf)
    complementarity = jnp.linalg.norm(multipliers*residual, ord=jnp.inf)
    return jnp.where(jnp.isfinite(lagrangian) & jnp.isfinite(complementarity),
                     jnp.maximum(lagrangian, complementarity), jnp.inf)


def _merit2(value, violation, bad_qp):
    if bool(violation > 0):
        result = violation
    elif bool(value >= 0):
        result = -1/(value+1)
    else:
        result = jnp.asarray(0.)
    if not bad_qp and bool(value < 0):
        result = result+value-1
    return result


def _convergence(violation, step, a, gradient, active, residual, old_lambda,
                 old_error, tolcon):
    """nlconst.testConvergence: real box, no equalities/semifinite callback."""
    if not bool(violation < tolcon):
        return -2, old_lambda, old_error
    multipliers = jnp.zeros(4)
    if active:
        rows = a[jnp.asarray(active)]
        if len(active) == 1:
            # Native 2x1 backslash for a signed coordinate unit vector.
            values = -(rows@gradient)
        elif len(active) == 2:
            values = jnp.linalg.solve(rows.T, -gradient)
        else:
            raise RuntimeError('Unsupported dependent native convergence working set')
        multipliers = multipliers.at[jnp.asarray(active)].set(values)
    multipliers = jnp.maximum(0, multipliers)
    error = _optimality(gradient, a, multipliers, residual)
    if bool(error < EPS):
        flag = 1
    elif bool(jnp.linalg.norm(step, ord=jnp.inf) < 2*EPS):
        flag = 4
    else:
        flag = 5
    return flag, multipliers, error


def active_set_box(fun, initial, lower, upper, *, finite_difference,
                   native_randn=None, observer=None):
    """Source caller options fixed: TolFun/TolFunValue/TolX=eps, box2D.

    finite_difference(x, fun, lower, upper, base_value, options) returns
    (gradient, num_extra_evaluations). This provider is presently unavailable;
    injected scripted gradients in future controls do not qualify native FD.
    """
    if finite_difference is None:
        raise RuntimeError('Protected R2017a finite-difference provider is unavailable')
    x = _real64(initial, 'initial')
    lo, hi = _real64(lower, 'lower'), _real64(upper, 'upper')
    if any(v.shape != (2,) for v in (x, lo, hi)):
        raise ValueError('active_set_box requires two variables')
    if not bool(jnp.all(jnp.isfinite(lo) & jnp.isfinite(hi) & (lo < hi))):
        raise ValueError('active_set_box requires finite positive-width bounds')
    # fmincon455: clamp only violated initial bound coordinates.
    x = jnp.where(x < lo, lo, jnp.where(x > hi, hi, x))
    value = _real64(fun(x), 'objective')
    if value.shape != ():
        raise ValueError('Objective must be scalar')
    a = jnp.concatenate((-jnp.eye(2), jnp.eye(2)))
    b = jnp.concatenate((-lo, hi))
    residual = a@x-b
    violation = jnp.max(residual)
    h = jnp.eye(2)
    lamb = jnp.zeros(4)
    lambda_nlp = jnp.zeros(4)
    old_a, old_g = jnp.zeros((4, 2)), jnp.zeros(2)
    old_x, old_c = x, residual
    matx, old_value = x, value
    step = jnp.ones(2)
    steplength = jnp.asarray(1.)
    evaluations, gradients, iteration = 1, 1, 0
    best_value, best = jnp.inf, None
    active = ()
    # nlconst.m284: iteration-zero optimality is empty; best restoration
    # must preserve that output state rather than invent scalar infinity.
    error = jnp.empty((0,), dtype=jnp.float64)
    fd_options = {'FinDiffType': 'forward', 'FinDiffRelStep': jnp.full(2, jnp.sqrt(EPS)),
                  'TypicalX': jnp.ones(2), 'DiffMinChange': 0., 'DiffMaxChange': jnp.inf,
                  'fwdFinDiff': True, 'scaleObjConstr': False, 'chkFunEval': False,
                  'chkComplexObj': False, 'isGrad': True}
    tolcon = 1e-6
    done = False
    flag = 1

    def emit(event, **state):
        if observer is not None:
            observer(event, iteration, state)

    while not done:
        g, count = finite_difference(x, fun, lo, hi, value, fd_options)
        g = _real64(g, 'finite-difference gradient')
        if g.shape != (2,) or not isinstance(count, int) or count < 0:
            raise ValueError('Invalid protected finite-difference provider result')
        evaluations += count
        if iteration > 0:
            error = _optimality(g, a, lambda_nlp, residual)
            if bool(error < EPS) and bool(violation < tolcon):
                flag, done = 1, True
            else:
                if evaluations > 200:
                    x, value, g = matx, old_value, old_g
                    flag, done = 0, True
                if iteration >= 400:
                    flag, done = 0, True
        if done:
            break
        iteration += 1
        if gradients > 1:
            y = (g+a.T@lamb) - (old_g+old_a.T@lamb)
            displacement = x-old_x
            if bool(_dot2(y, displacement) < steplength**2*1e-3):
                while bool(_dot2(y, displacement) < -1e-5):
                    index = int(jnp.argmin(y*displacement))
                    y = y.at[index].set(y[index]/2)
                if bool(_dot2(y, displacement) < EPS*jnp.linalg.norm(h, ord='fro')):
                    factor = a.T@residual-old_a.T@old_c
                    factor = factor*(displacement*factor > 0)*(y*displacement <= EPS)
                    weight = jnp.asarray(1e-2)
                    if bool(jnp.max(jnp.abs(factor)) == 0):
                        factor = 1e-5*jnp.sign(displacement)
                    while bool(_dot2(y, displacement) < EPS*jnp.linalg.norm(h, ord='fro')) and bool(weight < 1/EPS):
                        y = y+weight*factor
                        weight = weight*2
            if bool(_dot2(y, displacement) > EPS):
                h = (h+_scalar_matrix_divide(jnp.outer(y, y), _dot2(y, displacement))
                     -_scalar_matrix_divide(
                         jnp.outer(_matvec2(h, displacement), _matvec2(h, displacement)),
                         _dot2(_matvec2(h.T, displacement), displacement)))
            emit('bfgs', hessian=h, corrected_y=y, displacement=displacement)
        else:
            old_lambda = jnp.full(4, EPS+g@g)/(jnp.sum(a.T*a.T, axis=0)+EPS)
            active = ()
        gradients += 1
        previous_lambda = lamb
        old_a, old_g, old_c, old_value, old_x = a, g, residual, value, x
        h = (h+h.T)*0.5
        qp = box_qp(h, g, lo-x, hi-x, active=active, max_iterations=40,
                    tolerance=tolcon, native_randn=native_randn)
        step, active = qp.x, qp.active
        lambda_nlp = jnp.zeros(4).at[jnp.asarray(active, dtype=jnp.int32)].set(
            qp.multipliers[jnp.asarray(active, dtype=jnp.int32)])
        lamb = qp.multipliers
        old_lambda = jnp.maximum(lamb, 0.5*(lamb+old_lambda))
        product = g@step
        bad_qp = qp.how in ('infeasible', 'ill posed', 'MaxSQPIter')
        matx = x
        matl = value+jnp.sum(old_lambda*(residual > 0)*residual)+1e-30
        matl2 = _merit2(value, violation, bad_qp)
        if bool(violation < EPS) and bool(value < best_value):
            best_value = value
            best = (x, h, g, qp.multipliers, violation, error)
        search = True
        trial = jnp.asarray(2.)
        emit('qp', result=qp, evaluations=evaluations)
        while search and evaluations < 200:
            trial = trial/2
            if bool(trial < 1e-4):
                trial = -trial
            if (bool(jnp.linalg.norm(step, ord=jnp.inf) < 2*EPS)
                    or bool(jnp.abs(trial*product) < EPS)) and (bool(violation < tolcon) or bad_qp):
                flag, lambda_nlp, error = _convergence(
                    violation, step, a, g, active, residual, lambda_nlp, error, tolcon)
                done = True
                break
            x = matx+trial*step
            value = _real64(fun(x), 'objective')
            evaluations += 1
            residual = a@x-b
            violation = jnp.max(residual)
            merit = value+jnp.sum(old_lambda*(residual > 0)*residual)
            merit2 = _merit2(value, violation, bad_qp)
            search = bool(merit2 > matl2) and bool(merit > matl)
            emit('line_search', x=x, value=value, trial=trial, evaluations=evaluations)
        steplength = trial
        if not done:
            magnitude = jnp.abs(steplength)
            lamb = magnitude*lamb+(1-magnitude)*previous_lambda
    if bool(value > best_value):
        x, h, g, _best_qp_lambda, violation, error = best
        value = best_value
    return SQPResult(x, value, lambda_nlp, flag, iteration, evaluations, g, h, error)
