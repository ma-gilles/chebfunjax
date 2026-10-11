"""Default-double scalar root adapter for source example construction.

Provenance
----------
MATLAB R2025b toolbox/matlab/optimfun/optimfun/fzero.m, scalar interval
search and bisection/interpolation loop. Chebfun commit: 7574c77 uses
this MATLAB helper; it is not an upstream Chebfun API implementation.
Only finite real evaluations and default TolX are supported here.
"""

import jax.numpy as jnp


def _fzero_default(function, initial):
    """Return a root and observations for finite real default-double input.

    Provenance
    ----------
    MATLAB R2025b fzero.m: scalar search (344-437), interpolation (459-550).
    Chebfun commit: 7574c77; MATLAB helper used by the source examples.
    Options, callbacks, single precision and nonfinite-return recovery are
    outside this private adapter's scope.
    """
    x = jnp.asarray(initial)
    if jnp.iscomplexobj(x) or not bool(jnp.all(jnp.isfinite(x))):
        raise ValueError("MATLAB:fzero:InvalidInitialValue")
    x = x.astype(jnp.float64).reshape(-1)
    count = 0
    iterations = 0
    interval_iterations = 0

    def evaluate(point):
        value = jnp.asarray(function(point))
        if value.ndim != 0 or jnp.iscomplexobj(value):
            raise ValueError("MATLAB:fzero:ValueAtInitGuessComplexOrNotFinite")
        if not bool(jnp.isfinite(point) & jnp.isfinite(value)):
            raise ValueError("MATLAB:fzero:ValueAtInitGuessComplexOrNotFinite")
        return value.astype(jnp.float64)

    if x.size == 1:
        seed = x[0]
        fx = evaluate(seed)
        count = 1
        a = b = seed
        fa = fb = fx
        if bool(fx != 0):
            dx = seed / 50 if bool(seed != 0) else jnp.asarray(1 / 50)
            expansion = jnp.sqrt(jnp.asarray(2.0, dtype=jnp.float64))
            while bool((fa > 0) == (fb > 0)):
                interval_iterations += 1
                dx = expansion * dx
                a = seed - dx
                fa = evaluate(a)
                count += 1
                if bool((fa > 0) != (fb > 0)):
                    break
                b = seed + dx
                fb = evaluate(b)
                count += 1
    elif x.size == 2:
        a, b = x[0], x[1]
        fa, fb = evaluate(a), evaluate(b)
        count = 2
        if bool(fa != 0) and bool(fb != 0) and bool((fa > 0) == (fb > 0)):
            raise ValueError("MATLAB:fzero:ValuesAtEndPtsSameSign")
        if bool(fa == 0):
            b, fb = a, fa
    else:
        raise ValueError("MATLAB:fzero:LengthArg2")
    bracket = (a, b)
    bracket_values = (fa, fb)
    fc = fb
    c = a
    d = e = b - a
    tol = jnp.finfo(jnp.float64).eps
    while bool(fb != 0) and bool(a != b):
        if bool((fb > 0) == (fc > 0)):
            c, fc = a, fa
            d = b - a
            e = d
        if bool(jnp.abs(fc) < jnp.abs(fb)):
            # Native assignments are sequential: c receives the new a.
            a, fa = b, fb
            b, fb = c, fc
            c, fc = a, fa
        midpoint = 0.5 * (c - b)
        tolerance = 2.0 * tol * jnp.maximum(jnp.abs(b), 1.0)
        if bool(jnp.abs(midpoint) <= tolerance) or bool(fb == 0):
            break
        if bool(jnp.abs(e) < tolerance) or bool(jnp.abs(fa) <= jnp.abs(fb)):
            d = midpoint
            e = midpoint
        else:
            ratio = fb / fa
            if bool(a == c):
                p = 2.0 * midpoint * ratio
                q = 1.0 - ratio
            else:
                q = fa / fc
                r = fb / fc
                p = ratio * (2.0 * midpoint * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (ratio - 1.0)
            if bool(p > 0):
                q = -q
            else:
                p = -p
            if bool(2.0 * p < 3.0 * midpoint * q - jnp.abs(tolerance * q)) and bool(
                p < jnp.abs(0.5 * e * q)
            ):
                e = d
                d = p / q
            else:
                d = midpoint
                e = midpoint
        a, fa = b, fb
        if bool(jnp.abs(d) > tolerance):
            b = b + d
        elif bool(b > c):
            b = b - tolerance
        else:
            b = b + tolerance
        fb = evaluate(b)
        count += 1
        iterations += 1
    flag = (
        -5
        if bool(jnp.abs(fb) > jnp.maximum(jnp.abs(bracket_values[0]), jnp.abs(bracket_values[1])))
        else 1
    )
    return b, {
        "value": fb,
        "iterations": iterations,
        "intervaliterations": interval_iterations,
        "funcCount": count,
        "exitflag": flag,
        "bracket": bracket,
    }
