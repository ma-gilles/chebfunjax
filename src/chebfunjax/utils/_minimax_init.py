"""Source CF, AAA-Lawson and CDF initialization for rational minimax.

Provenance
----------
MATLAB source : minimax.m (cfInit, cdfInit, pwiselin, refGen)
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
import warnings

import jax.numpy as jnp


def _cf_reference(source, handle, m, n):
    from chebfunjax.utils.cfpade import cf
    from chebfunjax.utils.minimax import _exchange_rat, minimax

    def target(x):
        return handle(x)
    target._minimax_source = source
    target._minimax_vscale = float(source.vscale)
    domain = (float(source.domain.a), float(source.domain.b))

    def attempt(degree=None):
        p, q, _, _ = cf(source, m, n, degree)
        return _exchange_rat(jnp.asarray([]), 0, 2, target,
                             lambda x: p(x)/q(x), m+n+2, *domain)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if len(source.funs) == 1:
            reference, _, flag = attempt()
        else:
            try:
                reference, _, flag = attempt(50*(m+n))
            except Exception:
                flag = 0
    if flag == 0:
        reference = minimax(source, m+n, domain=domain).xk
    return reference


def _piecewise_linear(x, y, queries):
    x, y, queries = jnp.asarray(x), jnp.asarray(y), jnp.asarray(queries)
    if x.size == 1:
        return jnp.full(queries.shape, y[0])
    index = jnp.searchsorted(x, queries, side='right')-1
    # MATLAB histcounts assigns out-of-range queries to bin0, then index1.
    index = jnp.where((queries < x[0]) | (queries > x[-1]), 0, index)
    index = jnp.minimum(index, x.size-2)
    weight = (queries-x[index])/(x[index+1]-x[index])
    return (1-weight)*y[index]+weight*y[index+1]


def _reference_from_cdf(source, reference, size, symmetry):
    reference = jnp.asarray(reference)
    x = jnp.linspace(-1, 1, reference.size)
    if symmetry == 0:
        return _piecewise_linear(x, reference, jnp.linspace(-1, 1, size))
    lo, hi = float(source.domain.a), float(source.domain.b)
    if symmetry == 1:
        half, count = reference.size//2, size//2
        if float(reference[0]) == lo:
            nodes = _piecewise_linear(x[half:], reference[half:],
                                      jnp.linspace(x[half], x[-1], count))
            return jnp.sort(jnp.concatenate((nodes, -nodes[1:], jnp.asarray([lo]))))
        if float(reference[-1]) == hi:
            nodes = _piecewise_linear(x[:half], reference[:half],
                                      jnp.linspace(x[0], x[half-1], count))
            return jnp.sort(jnp.concatenate((nodes, -nodes[:-1], jnp.asarray([hi]))))
    else:
        half, count = (reference.size-1)//2, (size-1)//2
        if float(reference[0]) == lo:
            nodes = _piecewise_linear(x[half+1:], reference[half+1:],
                                      jnp.linspace(x[half+1], x[-1], count))
            return jnp.sort(jnp.concatenate((nodes, -nodes, jnp.asarray([lo]))))
        if float(reference[-1]) == hi:
            nodes = _piecewise_linear(x[:half+1], reference[:half+1],
                                      jnp.linspace(x[0], x[half], count))
            return jnp.sort(jnp.concatenate((nodes, -nodes, jnp.asarray([hi]))))
    return _piecewise_linear(x, reference, jnp.linspace(-1, 1, size))


def _cdf_reference(source, handle, m, n, symmetry, kernel, step, silent):
    from chebfunjax.utils.minimax import minimax
    step_size = 2*step if symmetry > 0 else step
    domain = (float(source.domain.a), float(source.domain.b))
    if not silent:
        print(f'Trying CDF-based initialization with step size {step_size}...')
    counter = 0

    def announce(mm, nn):
        nonlocal counter
        counter += 1
        if not silent:
            if counter == 10:
                print()
                counter = 0
            print(f'({mm},{nn}) ', end='', flush=True)

    def polynomial(mm, nn):
        return minimax(source, mm+nn, domain=domain)

    if abs(m-n) <= 2:
        minimum = min(m, n)
        k = (minimum-minimum % step_size)//step_size-(3-step)
        mm, nn = m-step_size*k, n-step_size*k
        reference = _cf_reference(source, handle, mm, nn)
        status = kernel(mm, nn, reference, False)
        while mm < m-step_size and status.success:
            mm, nn = mm+step_size, nn+step_size
            announce(mm, nn)
            reference = _reference_from_cdf(source, status.xk, mm+nn+2, symmetry)
            status = kernel(mm, nn, reference, False)
        if status.success:
            reference = _reference_from_cdf(source, status.xk, m+n+2, symmetry)
            if not silent:
                print(f'({m},{n})')
        else:
            if not silent:
                print(f'\nInitialization failed using CDF with step size {step_size}...')
            reference = polynomial(m, n).xk
    elif m < n:
        k = (m-m % step_size)//step_size
        mm, nn = m-step_size*k, n-step_size*k
        hk = (nn-nn % step_size)//step_size
        start_n = nn-step_size*hk
        reference = _cf_reference(source, handle, mm, start_n)
        status = kernel(mm, start_n, reference, False)
        while start_n < nn-step_size and status.success:
            start_n += step_size
            reference = _reference_from_cdf(source, status.xk, mm+start_n+2, 0)
            announce(mm, start_n)
            status = kernel(mm, start_n, reference, False)
        if not status.success:
            status = polynomial(mm, nn)
        success = True
        while mm < m-step_size and success:
            mm, nn = mm+step_size, nn+step_size
            reference = _reference_from_cdf(source, status.xk, mm+nn+2, symmetry)
            announce(mm, nn)
            status = kernel(mm, nn, reference, False)
            success = status.success
        if success:
            reference = _reference_from_cdf(source, status.xk, m+n+2, 0)
            if not silent:
                print(f'({m},{n})')
        else:
            if not silent:
                print(f'\nInitialization failed using CDF with step size {step_size}')
            reference = polynomial(m, n).xk
    else:
        k = (n-n % step_size)//step_size
        mm, nn = m-step_size*k, n-step_size*k
        reference = _cf_reference(source, handle, mm, nn)
        status = kernel(mm, nn, reference, False)
        while mm < m-step_size and status.success:
            mm, nn = mm+step_size, nn+step_size
            announce(mm, nn)
            reference = _reference_from_cdf(source, status.xk, mm+nn+2, symmetry)
            status = kernel(mm, nn, reference, False)
        if status.success:
            reference = _reference_from_cdf(source, status.xk, m+n+2, symmetry)
            if not silent:
                print(f'({m},{n})')
        else:
            if not silent:
                print(f'\nInitialization failed using CDF with step size {step_size}')
            reference = polynomial(m, n).xk
    if not silent:
        print()
    return reference


def initialize_rational(source, handle, m, n, symmetry, kernel, *, silent=False):
    """Try the literal source sequence and preserve branch diagnostics.

    Provenance
    ----------
    MATLAB source : minimax.m (lines134–199)
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    from chebfunjax.utils.minimax import _aaa_lawson_init
    try:
        reference = _cf_reference(source, handle, m, n)
        result = kernel(m, n, reference)
    except Exception:
        if not silent:
            print('CF-based initialization failed, turning to AAA-Lawson...')
        result = None
    if result is None or not result.success:
        if not silent:
            print('Trying AAA-Lawson-based initialization...')
        reference = _aaa_lawson_init(handle, m, n, float(source.domain.a), float(source.domain.b))
        result = kernel(m, n, reference)
    for step in (1, 2):
        if result.success:
            break
        reference = _cdf_reference(source, handle, m, n, symmetry, kernel, step, silent)
        result = kernel(m, n, reference)
    if not result.success:
        raise RuntimeError('MINIMAX failed to produce the best approximant. If the accuracy is close to machine precision, it may be that what you have asked for is unachievable in floating-point arithmetic. Try reducing the degree to get a clean best approximant.')
    return result
