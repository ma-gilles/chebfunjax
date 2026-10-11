"""Source legpoly.m, Chebfun commit 7574c77; eager shape/API adapter."""
import jax
import jax.numpy as jnp

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.utils.quadrature import chebpts, chebweights
from chebfunjax.utils.transforms import leg2cheb


def source_legpoly(n, domain, normalize, method):
    """Translate literal source method selection and coefficient assembly."""
    from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun

    n = jnp.asarray(n)
    if n.size == 0:
        return Chebfun.empty()
    transpose = n.ndim == 2 and n.shape[0] > 1
    ns = n.reshape(-1).astype(jnp.int32)
    # MATLAB replaces empty normalize with 0 before routing the normalized
    # short form's third argument to method; empty explicit method is unset.
    if normalize is not None and not isinstance(normalize, str):
        if jnp.asarray(normalize).size == 0:
            normalize = 0
    if method is not None and jnp.asarray(method).size == 0:
        method = None
    if isinstance(domain, str):
        if method is None and normalize is not None:
            method = normalize
        normalize, domain = domain, (-1., 1.)
    if domain is None or len(domain) == 0:
        domain = (-1., 1.)
    if isinstance(normalize, str):
        normalize = normalize.startswith('norm')
    normalize = bool(normalize)
    dv = tuple(float(x) for x in domain)
    if bool(jnp.any(jnp.isinf(jnp.asarray(dv)))):
        raise ValueError('CHEBFUN:legpoly:infdomain: Legendre polynomials '
                         'are not defined over an unbounded domain.')
    pref = ChebfunPref()
    # The Python implementation has separate concrete technology classes;
    # MATLAB's shared ``chebtech`` superclass is not represented here.
    technology = pref.tech
    if isinstance(technology, str):
        name = technology.lower().lstrip('@')
        pref.tech = (Chebtech1 if name == 'chebtech1' else Chebtech2)
    elif technology not in (Chebtech1, Chebtech2):
        pref.tech = Chebtech2
    maximum = int(jnp.max(ns))
    count = maximum + 1
    if method is None:
        method = 3 if maximum > 1000 else 2
        if method == 3 and maximum < 5000 and ns.size > maximum/5:
            method = 1
    if method == 1:
        # MATLAB stores only the requested unique degrees during recurrence,
        # then uses unique's inverse map to restore input order/duplicates.
        # Keep that bounded workspace: O(count * requested degrees), not
        # O(count**2) for short high-degree lists.
        degree_values = tuple(int(k) for k in ns.tolist())
        unique_degrees = tuple(sorted(set(degree_values)))
        inverse = jnp.asarray(
            [unique_degrees.index(k) for k in degree_values], dtype=jnp.int32)
        column_for_degree = [-1] * count
        for column, degree in enumerate(unique_degrees):
            column_for_degree[degree] = column
        column_for_degree = jnp.asarray(column_for_degree, dtype=jnp.int32)
        x = chebpts(count, kind=2)
        def step(k, state):
            low, high, values = state
            column = column_for_degree[k-2]
            values = jax.lax.cond(
                column >= 0,
                lambda data: data.at[:, column].set(low),
                lambda data: data,
                values)
            next_high = (2-1/k)*x*high - (1-1/k)*low
            return high, next_high, values
        _, _, values = jax.lax.fori_loop(
            2, maximum+3, step,
            (jnp.ones(count), x,
             jnp.zeros((count, len(unique_degrees)), dtype=x.dtype)))
        values = values[:, inverse]
        if normalize:
            values = values*jnp.sqrt((2*ns+1)/(dv[-1]-dv[0]))[None, :]
        coefficients = Chebtech2.vals2coeffs(values)
    elif method == 2:
        pts = 2*count
        w = chebweights(pts, kind=2)
        theta = jnp.pi*jnp.arange(pts-1, -1, -1)/(pts-1)
        matrix = jnp.cos(theta[:, None]*jnp.arange(count)[None, :])
        q, _ = jnp.linalg.qr(jnp.sqrt(w)[:, None]*matrix, mode='reduced')
        values = q/jnp.sqrt(w)[:, None]
        scale = (jnp.sqrt(2/(dv[-1]-dv[0]))*jnp.sign(values[-1, ns])
                 if normalize else 1/values[-1, ns])
        coefficients = Chebtech2.vals2coeffs(values[:, ns]*scale[None, :])[:count]
    elif method == 3:
        legendre = jnp.zeros((count, ns.size)).at[ns, :].set(jnp.eye(ns.size))
        coefficients = leg2cheb(legendre, normalize=normalize)
    else:
        raise ValueError('legpoly method must be 1, 2 or 3')
    coefficients = coefficients[:, 0] if ns.size == 1 else coefficients
    result = chebfun(coefficients, domain=(dv[0], dv[-1]), pref=pref, coeffs=True)
    if len(dv) > 2:
        result = result._restrict_breaks(dv)
    return result.T if transpose else result
