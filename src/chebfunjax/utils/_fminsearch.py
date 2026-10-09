"""Real two-variable R2017a fminsearch fallback policy.

This is the fallback dependency for pinned separableApprox.minandmax2, not an
active-set implementation or an assertion about the historical page's branch.
Tolerance is exactly binary64 eps as set by the Chebfun caller. Output/display
callbacks and general dimension/type option parsing are outside this adapter.

Provenance
----------
R2017a toolbox/matlab/optimfun/fminsearch.m, simplex and iteration sections.
Caller: Chebfun @separableApprox/minandmax2.m at7574c77.
"""
from typing import NamedTuple

import jax.numpy as jnp


class SearchResult(NamedTuple):
    x: object
    fun: object
    exitflag: int
    iterations: int
    evaluations: int


def _threshold(value):
    """max(eps,10*MATLAB_eps(value)), for caller's fixed eps tolerance.

    Spacings below2^-56 cannot affect this maximum. Clamping that exponent
    avoids needing gradual subnormal arithmetic without changing the threshold.
    Nonfinite behavior is unqualified; no native max(tol,NaN) parity is
    inferred from this adapter's comparisons.
    """
    _, exponent = jnp.frexp(jnp.abs(value))
    spacing = jnp.ldexp(jnp.asarray(1., dtype=jnp.float64),
                        jnp.maximum(jnp.where(value == 0, -56, exponent-53), -56))
    result = jnp.maximum(jnp.finfo(jnp.float64).eps, 10*spacing)
    return jnp.where(jnp.isfinite(value), result, jnp.nan)


def _converged(vertices, values):
    fspread = jnp.max(jnp.abs(values[0]-values[1:]))
    xspread = jnp.max(jnp.abs(vertices[:, 1:]-vertices[:, :1]))
    # Native uses eps(max(best coordinates)), not eps(max(abs(coordinates))).
    return bool((fspread <= _threshold(values[0]))
                & (xspread <= _threshold(jnp.max(vertices[:, 0]))))


def fminsearch(fun, x0, *, max_iterations=400, max_evaluations=400):
    """Real 2D minimization; explicit caps correspond to native option values.

    The source checks limits only at loop entry, allowing a completed branch
    to cross the evaluation limit. Zero exitflag still returns the best point.
    Objective exceptions propagate for the caller's native fallback handling.
    """
    x = jnp.asarray(x0, dtype=jnp.float64)
    if x.shape != (2,):
        raise ValueError('This source adapter requires a two-coordinate vector')
    vertices = jnp.zeros((2, 3), dtype=jnp.float64).at[:, 0].set(x)
    values = jnp.zeros(3, dtype=jnp.float64).at[0].set(fun(x))
    for j in range(2):
        y = x.at[j].set(jnp.where(x[j] != 0, (1+.05)*x[j], .00025))
        vertices = vertices.at[:, j+1].set(y)
        values = values.at[j+1].set(fun(y))
    order = jnp.argsort(values, stable=True)
    values, vertices = values[order], vertices[:, order]
    iterations, evaluations = 1, 3
    while evaluations < max_evaluations and iterations < max_iterations:
        if _converged(vertices, values):
            break
        centroid = jnp.sum(vertices[:, :2], axis=1)/2
        reflected = 2*centroid-vertices[:, -1]
        fr = fun(reflected)
        evaluations += 1
        shrink = False
        if bool(fr < values[0]):
            expanded = 3*centroid-2*vertices[:, -1]
            fe = fun(expanded)
            evaluations += 1
            point, value = (expanded, fe) if bool(fe < fr) else (reflected, fr)
            vertices, values = vertices.at[:, -1].set(point), values.at[-1].set(value)
        elif bool(fr < values[1]):
            vertices = vertices.at[:, -1].set(reflected)
            values = values.at[-1].set(fr)
        elif bool(fr < values[-1]):
            contracted = 1.5*centroid-.5*vertices[:, -1]
            fc = fun(contracted)
            evaluations += 1
            if bool(fc <= fr):
                vertices = vertices.at[:, -1].set(contracted)
                values = values.at[-1].set(fc)
            else:
                shrink = True
        else:
            contracted = .5*centroid+.5*vertices[:, -1]
            fc = fun(contracted)
            evaluations += 1
            if bool(fc < values[-1]):
                vertices = vertices.at[:, -1].set(contracted)
                values = values.at[-1].set(fc)
            else:
                shrink = True
        if shrink:
            for j in (1, 2):
                point = vertices[:, 0]+.5*(vertices[:, j]-vertices[:, 0])
                vertices = vertices.at[:, j].set(point)
                values = values.at[j].set(fun(point))
            evaluations += 2
        order = jnp.argsort(values, stable=True)
        values, vertices = values[order], vertices[:, order]
        iterations += 1
    flag = 0 if evaluations >= max_evaluations or iterations >= max_iterations else 1
    return SearchResult(vertices[:, 0], values[0], flag, iterations, evaluations)
