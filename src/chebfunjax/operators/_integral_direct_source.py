"""Direct adaptive Chebfun integral actions; no operator discretization.

Copyright 2017 by The University of Oxford and The Chebfun Developers.
"""

import jax.numpy as jnp


def adaptive_action(kernel, function, *, volterra):
    """Apply native scalar-x inner/outer adaptive construction to columns.

    Provenance
    ----------
    MATLAB source: @chebfun/fred.m fred_col, @chebfun/volt.m volt_col.
    Chebfun commit: 7574c77.
    Python array columns are treated as native quasimatrix columns;
    the existing transpose adapter is preserved.
    """
    from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun

    domain = tuple(float(value) for value in function.domain.breakpoints)
    column_function = function.transpose() if function.is_transposed else function
    results = []
    for index in reversed(range(column_function.n_columns)):
        column = column_function.extract_columns(index)
        normv = column.norm()

        def action(x, column=column, normv=normv):
            if volterra and bool(x == domain[0]):
                return jnp.asarray(0.)
            if volterra and not bool(x == domain[-1]):
                inner_domain = tuple(value for value in domain if bool(value < x)) + (float(x),)
            else:
                inner_domain = domain
            inner = chebfun(
                lambda y: column(y)*kernel(x, y), domain=inner_domain,
                resampling=not volterra, splitting=True, blowup=False,
                vscale=normv,
            )
            return inner.sum()

        options = dict(blowup=False, vectorize=True, vscale=normv)
        if volterra:
            options['extrapolate'] = 1
        else:
            options.update(sample_test=False, resampling=True)
        results.append(chebfun(action, domain=domain, **options))
    result = Chebfun.horzcat(list(reversed(results)))
    return Chebfun._as_transposed(result, function.is_transposed)
