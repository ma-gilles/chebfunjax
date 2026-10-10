"""Active native addition and internal exact component assembly.

Provenance
----------
MATLAB source : @chebfun3/{plus,minus,iszero,isPeriodicTech}.m
Chebfun commit: 7574c77
Public addition uses the active resampling branch, not compressed_plus.
The block assembler is an internal Python complex-constructor adapter only.
"""

import jax.numpy as jnp

from chebfunjax.chebfun3d._power import _domain_check, _evaluate, _is_empty, _source_vscale
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.trigtech import Trigtech
from chebfunjax.utils.quadrature import chebpts_ab


def _assemble_components(a, b):
    """Exact sum of two Tucker expansions, without public arithmetic.

    The disjoint core blocks and concatenated factors contain exactly the
    terms of a+b, with no cross terms. Keep complex-zero storage and every
    component; no compression, tolerance, or zero-component dropping.
    """
    from chebfunjax.chebfun3d.chebfun3 import Chebfun3

    ra, rb = a.core.shape, b.core.shape
    core = jnp.zeros(tuple(x+y for x, y in zip(ra, rb)),
                     dtype=jnp.result_type(a.core, b.core))
    core = core.at[:ra[0], :ra[1], :ra[2]].set(a.core)
    core = core.at[ra[0]:, ra[1]:, ra[2]:].set(b.core)
    return Chebfun3(a.cols+b.cols, a.rows+b.rows, a.tubes+b.tubes, core, a.domain)


def _source_iszero(f):
    """Literal core, 10-point tensor, then zero-factor source checks."""
    if bool(jnp.max(jnp.abs(f.core)) == 0):
        return True
    grids = [jnp.linspace(f.domain[2*i], f.domain[2*i+1], 10) for i in range(3)]
    if bool(jnp.max(jnp.abs(_evaluate(f, *jnp.meshgrid(*grids, indexing='ij')))) > 0):
        return False
    return any(all(bool(jnp.all(tech.coeffs == 0)) for tech in group)
               for group in (f.cols, f.rows, f.tubes))


def _is_periodic(f):
    return all(isinstance(tech, Trigtech) for group in (f.cols, f.rows, f.tubes)
               for tech in group)


def _double_data(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, complex)):
        return jnp.asarray(value, dtype=jnp.complex128 if isinstance(value, complex)
                           else jnp.float64)
    try:
        data = jnp.asarray(value)
    except (TypeError, ValueError):
        return None
    return data if data.dtype in (jnp.float64, jnp.complex128) else None


def _numeric_function(data, domain):
    """Native double-to-function dispatch through existing constructors.

    Split complex numeric tensors because the inherited values constructor
    casts real. This preserves the represented interpolant; it does not port
    the native chebfun3double ACA algorithm.
    """
    from chebfunjax.chebfun3d.chebfun3 import Chebfun3

    tol = ChebfunPref().cheb3Prefs.chebfun3eps
    if data.size == 1:
        return Chebfun3.from_function(lambda x, y, z: data.reshape(())+0*x,
                                     domain=domain, tol=tol)
    shape = data.shape+(1,)*max(0, 3-data.ndim)
    data = data.reshape(shape)
    if jnp.iscomplexobj(data):
        real = Chebfun3.from_values(jnp.real(data), domain=domain, tol=tol)
        imag = Chebfun3.from_values(jnp.imag(data), domain=domain, tol=tol)
        return _assemble_components(real, imag*1j)
    return Chebfun3.from_values(data, domain=domain, tol=tol)


def _sum_vscale(f, g):
    if _is_periodic(f) == _is_periodic(g):
        values = f.sample(51, 51, 51)+g.sample(51, 51, 51)
    else:
        panels = []
        for obj in (f, g):
            grids = [chebpts_ab(51, obj.domain[2*i], obj.domain[2*i+1], kind=2)
                     for i in range(3)]
            panels.append(_evaluate(obj, *jnp.meshgrid(*grids, indexing='ij')))
        values = panels[0]+panels[1]
    return jnp.max(jnp.abs(values))


def source_plus(f, g):
    """Native active plus branch order and condition-scaled resampling."""
    from chebfunjax.chebfun3d.chebfun3 import Chebfun3

    if not isinstance(f, Chebfun3):
        if isinstance(g, Chebfun3):
            return source_plus(g, f)
        raise ValueError('CHEBFUN:CHEBFUN3:plus:unknown: Unsupported input types.')
    if _is_empty(f) or _is_empty(g):
        return Chebfun3.empty()
    data = _double_data(g)
    if data is not None:
        return source_plus(f, _numeric_function(data, f.domain))
    if not isinstance(g, Chebfun3):
        raise ValueError('CHEBFUN:CHEBFUN3:plus:unknown: Unsupported input type.')
    if not _domain_check(f, g):
        raise ValueError('CHEBFUN:CHEBFUN3:plus:domain: Inconsistent domains.')
    if _source_iszero(f):
        return g
    if _source_iszero(g):
        return f
    hscale = _sum_vscale(f, g)
    vscale = jnp.asarray([_source_vscale(f), _source_vscale(g)])
    kappa = jnp.sum(vscale)/hscale
    tol = ChebfunPref().cheb3Prefs.chebfun3eps*kappa
    return Chebfun3.from_function(lambda x, y, z: _evaluate(f, x, y, z)
                                 + _evaluate(g, x, y, z),
                                 domain=f.domain, tol=float(tol))
