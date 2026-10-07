"""Public CF source dispatch for scalar, array-valued and quasimatrix inputs.

Provenance
----------
MATLAB source : @chebfun/cf.m, cheb2quasi.m, @chebfun/transpose.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Python cell-array adaptation: r is nested tuples, s a 2D JAX array.
"""
import math
import warnings

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2, _clenshaw
from chebfunjax.utils._cf_kernel import _cf_polynomial_jax, cf_rational_small_jax
from chebfunjax.utils.quadrature import chebpts
from chebfunjax.utils.transforms import _vals2coeffs_jax


def _evaluate(f, x):
    """Scalar smooth Chebyshev evaluation in JAX, including stored endpoints."""
    x = jnp.asarray(x)
    result = jnp.zeros_like(x, dtype=jnp.result_type(*[p.coeffs for p in f.funs], x))
    for index, piece in enumerate(f.funs):
        a, b = piece.interval
        values = _clenshaw(piece.coeffs, (2*x-a-b)/(b-a))
        if len(f.funs) == 1:
            result = values
        else:
            # @chebfun/feval.m extends the outer ownership intervals and
            # selects complex queries by their real part.
            left = -jnp.inf if index == 0 else a
            right = jnp.inf if index == len(f.funs)-1 else b
            mask = (jnp.real(x) >= left) & (jnp.real(x) < right)
            result = jnp.where(mask, values, result)
    points = f._breakpoint_values()
    for endpoint, value in zip(f.domain.breakpoints, points, strict=True):
        # This adapter handles scalar columns only (array inputs split above).
        result = jnp.where(x == endpoint, jnp.asarray(value).reshape(()), result)
    return result


def _make(coefficients, domain):
    return Chebfun(funs=[_Piece.from_coeffs(coefficients, *domain)], domain=Domain(domain))


def _one(f, m, n, degree):
    domain = (float(f.domain.a), float(f.domain.b))
    if any(math.isinf(x) for x in domain):
        raise ValueError('CHEBFUN:CHEBFUN:cf:unboundedDomain')
    if any(getattr(p.tech, 'exponents', None) is not None for p in f.funs):
        raise ValueError('CHEBFUN:cf:singularFunction')
    if not all(isinstance(p.tech, (Chebtech1, Chebtech2)) for p in f.funs):
        raise NotImplementedError('CF smooth Chebyshev input required')
    if len(f.funs) > 1:
        # Source global M+1 interpolation precedes the trivial m>=M return.
        nodes = chebpts(degree+1, kind=2)
        physical = domain[1]*(nodes+1)/2 + domain[0]*(1-nodes)/2
        f = _make(_vals2coeffs_jax(_evaluate(f, physical)), domain)
    if m >= degree:
        return f, _make(jnp.ones(1), domain), lambda x: _evaluate(f, x), jnp.asarray(0.)
    full = jnp.asarray(f.funs[0].coeffs)
    # Source indexes stored coefficients rather than silently padding an
    # unsupported explicit degree beyond the actual single-piece length.
    if degree >= full.shape[0]:
        raise ValueError('CF explicit M exceeds stored coefficient degree')
    coefficients = full[:degree+1]
    if bool(jnp.any(jnp.imag(coefficients) != 0)):
        warnings.warn('CHEBFUN:CHEBFUN:cf:complex: Taking real part.', RuntimeWarning, stacklevel=2)
    coefficients = jnp.asarray(jnp.real(coefficients), dtype=jnp.float64)
    if jnp.asarray(n).size == 0 or n == 0:
        pc, qc, error = _cf_polynomial_jax(coefficients, m)
    else:
        pc, qc, error = cf_rational_small_jax(
            coefficients, m, n, source_vscale=f.vscale,
            source_piece_count=1, source_full_coeffs=full)
    p, q = _make(pc, domain), _make(qc, domain)
    return p, q, lambda x: _evaluate(p, x)/_evaluate(q, x), error


def cf(f, m, n=0, M=None):
    """Source scalar/array/quasimatrix dispatch, with explicit cell orientation.

    Provenance
    ----------
    MATLAB source : @chebfun/cf.m (lines 53–90)
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    if isinstance(f, Quasimatrix):
        columns = f.cols
    elif isinstance(f, Chebfun):
        columns = [f] if f.n_columns == 1 else [f.extract_columns(i) for i in range(f.n_columns)]
        if f.is_transposed and len(columns)>1:
            columns = [column.transpose() for column in columns]
    else:
        raise TypeError('cf expects a Chebfun or Quasimatrix')
    degree = max(len(column) for column in columns)-1 if M is None else M
    results = [_one(column, m, n, degree) for column in columns]
    if len(columns)==1:
        return results[0]
    transposed = columns[0].is_transposed
    if any(column.is_transposed != transposed for column in columns):
        raise ValueError('CF quasimatrix columns have inconsistent orientation')
    ps, qs, rs, errors = zip(*results, strict=True)
    if transposed:
        ps, qs = [p.transpose() for p in ps], [q.transpose() for q in qs]
    p = Quasimatrix(list(ps), columns[0].domain)
    q = Quasimatrix(list(qs), columns[0].domain)
    r = tuple((entry,) for entry in rs) if transposed else (tuple(rs),)
    shape = (len(columns),1) if transposed else (1,len(columns))
    return p, q, r, jnp.asarray(errors).reshape(shape)
