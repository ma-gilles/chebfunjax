"""Source root partition and point values for bounded real polynomial arrays.

Provenance
----------
MATLAB source: @chebfun/abs.m, addBreaksAtRoots.m, getRootsForBreaks.m,
addBreaks.m; @chebtech/abs.m. Chebfun commit7574c77.
Root computation and restriction use the existing public library primitives.
"""
import math

import jax.numpy as jnp

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


def bounded_real_array(f):
    """Whether the common polynomial representation supports source array abs."""
    return (f.n_columns > 1 and bool(f.funs) and all(
        isinstance(piece.tech, (Chebtech1, Chebtech2))
        and all(math.isfinite(float(v)) for v in piece.interval)
        and bool(jnp.all(jnp.imag(piece.tech.coeffs) == 0))
        for piece in f.funs))


def source_array_abs(f):
    """Keep array columns together through native union roots and simplify.

    Provenance: Chebfun7574c77 @chebfun/abs.m and addBreaksAtRoots.m.
    A union root zeros only the column(s) whose own root list contains it.
    """
    from chebfunjax.domain import Domain

    from .chebfun import Chebfun

    # Native pointValues are independent of row orientation. The Python
    # evaluation adapter transposes implicit row point values, so perform
    # the source representation operations in column orientation.
    is_transposed = f.is_transposed
    f = f.T if is_transposed else f
    eps = float(jnp.finfo(jnp.float64).eps)
    tolerance = float(ChebfunPref().techPrefs.chebfuneps)
    all_roots = jnp.asarray(f.roots(nojump=True, nozerofun=True))
    if all_roots.ndim != 2:
        raise ValueError("Array root computation must retain the column dimension")
    old = tuple(float(v) for v in f.domain.breakpoints)
    candidates = sorted(float(v) for v in all_roots.ravel()
                        if math.isfinite(float(v)))
    root_tol = max(eps * max(abs(old[0]), abs(old[-1])), tolerance)
    roots = [v for i, v in enumerate(candidates)
             if i == 0 or v-candidates[i-1] >= root_tol]
    break_tol = max(100*eps*max(min(b-a for a, b in zip(old[:-1], old[1:])), 1), tolerance)
    added = [v for v in roots if old[0] < v < old[-1]
             and all(abs(v-b) >= break_tol for b in old)]
    breaks = tuple(sorted(set(old + tuple(added))))
    restricted = f if breaks == old else f.restrict(breaks)
    values = jnp.abs(restricted.point_values)
    if breaks != old:
        for column in range(f.n_columns):
            own = set(float(v) for v in all_roots[:, column]
                      if math.isfinite(float(v)))
            for row, node in enumerate(breaks):
                if node in own:
                    values = values.at[row, column].set(0.)
    out = Chebfun(funs=[piece.abs() for piece in restricted.funs],
                  domain=Domain(breaks)).set_point_values(values)
    out = Chebfun._as_transposed(out, is_transposed)
    return out.simplify()
