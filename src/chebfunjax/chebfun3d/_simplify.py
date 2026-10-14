"""Chebfun3 factor and absolute HOSVD truncation (Chebfun 7574c77)."""

import jax.numpy as jnp

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.trigtech import Trigtech


def _factor_panel(factors, tol):
    """@chebfun/simplify globaltol on one array-valued factor panel."""
    if not factors:
        return []
    kind = type(factors[0])
    n = max(t.n for t in factors)
    coefficients = jnp.column_stack([t.prolong(n).coeffs for t in factors])
    kwargs = {'ishappy': all(t.ishappy for t in factors)}
    if isinstance(factors[0], Trigtech):
        kwargs['real_columns'] = tuple(t.real_columns[0] for t in factors)
    panel = kind(coeffs=coefficients, **kwargs)
    scales = panel.vscale_columns
    if callable(scales):
        scales = scales()
    # Literal source division retains zero-column Inf/NaN tolerances.
    panel = panel.simplify(tol * jnp.max(scales) / scales)
    output = []
    for j in range(len(factors)):
        column_kwargs = {'ishappy': panel.ishappy}
        if isinstance(panel, Trigtech):
            column_kwargs['real_columns'] = (panel.real_columns[j],)
        output.append(kind(coeffs=panel.coeffs[:, j], **column_kwargs))
    return output


def source_simplify(f, tol=None, *args):
    """Literal @chebfun3/simplify.m branches, including ignored rank flag."""
    from chebfunjax.chebfun3d.chebfun3 import Chebfun3

    if len(args) > 1:
        return f
    if args:
        if f.isempty():
            raise ValueError('HOSVD outputs are unassigned for an empty Chebfun3')
        values, orthogonal = f.hosvd()
        ranks = []
        for mode in values:
            retained = jnp.where(mode > tol)[0]
            ranks.append(int(retained[-1]) + 1 if retained.size else 0)
        x, y, z = ranks
        out = Chebfun3(orthogonal.cols[:x], orthogonal.rows[:y],
                       orthogonal.tubes[:z], orthogonal.core[:x, :y, :z],
                       orthogonal.domain)
        if not any(ranks):
            object.__setattr__(out, '_is_empty_object', True)
        return out
    if f.isempty() and hasattr(f, 'cols'):
        return f
    if f.isempty():
        raise TypeError('Native empty factors have no simplify method')
    if tol is None or (hasattr(tol, 'size') and tol.size == 0):
        tol = ChebfunPref().techPrefs.chebfuneps
    elif isinstance(tol, ChebfunPref):
        tol = tol.techPrefs.chebfuneps
    return Chebfun3(_factor_panel(f.cols, tol), _factor_panel(f.rows, tol),
                   _factor_panel(f.tubes, tol), f.core, f.domain)
