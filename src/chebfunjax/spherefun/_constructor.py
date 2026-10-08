"""Source-facing Spherefun constructor input dispatch.

Provenance
----------
MATLAB source : @spherefun/{spherefun,constructor,sphf2cartf}.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
import ast
import inspect
import warnings

import jax.numpy as jnp

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.trigtech import Trigtech
from chebfunjax.utils._trigpts import global_trigpts_nodes, map_global_nodes
from chebfunjax.utils.matlab_expr import _FUNS, matlab_expression


def fix_rank(g, rank):
    """Apply the source CDR prefix rule, retaining source metadata behavior.

    Provenance
    ----------
    MATLAB source : @spherefun/constructor.m, fixTheRank
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    from chebfunjax.spherefun.spherefun import Spherefun
    if rank is None or g.isempty():
        return g
    cols, rows, pivots = list(g.cols), list(g.rows), g.pivots
    plus, minus = g.idx_plus, g.idx_minus
    def zero():
        return Trigtech(coeffs=jnp.zeros(1, dtype=jnp.complex128),
                        is_real=True, ishappy=True)
    if rank == 0:
        cols, rows, pivots = [zero()], [zero()], jnp.asarray([jnp.inf])
        plus, minus = (), (0,)
    elif rank < len(cols):
        cols, rows, pivots = cols[:rank], rows[:rank], pivots[:rank]
        plus = tuple(i for i in plus if i < rank)
        minus = tuple(i for i in minus if i < rank)
    elif rank > len(cols):
        count = rank - len(cols)
        cols += [zero() for _ in range(count)]
        rows += [zero() for _ in range(count)]
        pivots = jnp.concatenate((pivots, jnp.zeros(count)))
    return Spherefun(cols=cols, rows=rows, pivots=pivots,
                     idx_plus=plus, idx_minus=minus,
                     pivot_locations=g.pivot_locations,
                     nonzero_poles=g.nonzero_poles)


def _string_operator(expression):
    """Compile the bounded elementary subset of source str2op.

    Provenance
    ----------
    MATLAB source : @spherefun/constructor.m, str2op
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    # Restricted arithmetic/function grammar; no arbitrary Python execution.
    translated = expression.replace('.*', '*').replace('./', '/')
    translated = translated.replace('.^', '**').replace('^', '**')
    tree = ast.parse(translated, mode='eval')
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name,
               ast.Load, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div,
               ast.Pow, ast.USub, ast.UAdd)
    if any(not isinstance(n, allowed) for n in ast.walk(tree)):
        raise ValueError('Unsupported Spherefun string expression grammar')
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and (not isinstance(n.func, ast.Name)
                                       or n.func.id not in _FUNS):
            raise ValueError('Unsupported Spherefun string function')
    names = sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
                    and n.id not in _FUNS})
    if len(names) > 3:
        raise ValueError('CHEBFUN:SPHEREFUN:constructor:str2op:depvars')
    if not names:
        # Upstream str2op indexes a nonexistent dependent variable here.
        raise ValueError('Spherefun string input has no dependent variables')
    return matlab_expression(expression, tuple(names))


def _pointwise(op):
    """Scalar-call adapter for source vectorize, using JAX array assembly.

    Provenance
    ----------
    MATLAB source : @spherefun/constructor.m, evaluate
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    def evaluate(lam, theta):
        lam, theta = jnp.broadcast_arrays(jnp.asarray(lam), jnp.asarray(theta))
        values = [jnp.asarray(op(x, y)) for x, y in zip(lam.ravel(), theta.ravel())]
        return jnp.stack(values).reshape(lam.shape)
    return evaluate


def _vector_check(op, tol):
    """Detect scalar-only or incompatible array callbacks at source probes.

    Provenance
    ----------
    MATLAB source : @spherefun/constructor.m, vectorCheck
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    # Literal source interior 2x2 probe. Callback exceptions on array input
    # trigger scalar vectorization; exceptions on scalar input still propagate.
    dom = jnp.asarray([-jnp.pi, jnp.pi, 0., jnp.pi])
    x = dom[:2] / 3 + (dom[1] - dom[0]) / 3
    y = dom[2:] / 2 + (dom[3] - dom[2]) / 3
    xx, yy = jnp.meshgrid(x, y)
    try:
        a = jnp.asarray(op(xx, yy))
    except Exception:
        warnings.warn('CHEBFUN:SPHEREFUN:constructor:vectorize', stacklevel=3)
        return _pointwise(op)
    b = _pointwise(op)(xx, yy)
    if bool(jnp.any(jnp.abs(a - b) > min(1000 * tol, 1e-4))):
        warnings.warn('CHEBFUN:SPHEREFUN:constructor:vectorize', stacklevel=3)
        return _pointwise(op)
    return op


def spherefun(op=None, *options):
    """Construct from spherical/Cartesian callbacks, values, strings or CDR.

    Positional source options include fixed rank, [m,n], canonical domain,
    ChebfunPref, 'eps', 'vectorize', and 'coeffs'. Host dispatch is not JIT-safe;
    new arithmetic is JAX and existing construction backends are reused.
    Strings use a restricted elementary arithmetic grammar, not MATLAB eval.

    Provenance
    ----------
    MATLAB source : @spherefun/{spherefun,constructor,sphf2cartf}.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    from chebfunjax.spherefun.spherefun import Spherefun, _get_tol_sphere
    if (op is None or (isinstance(op, str) and not op)
            or (isinstance(op, Spherefun) and op.isempty())):
        return Spherefun.empty()
    if not callable(op) and not isinstance(op, str) and jnp.asarray(op).size == 0:
        return Spherefun.empty()
    if isinstance(op, str):
        op = _string_operator(op)
    if callable(op) and not isinstance(op, Spherefun):
        arity = len(inspect.signature(op).parameters)
        if arity <= 1:
            raise ValueError('CHEBFUN:SPHEREFUN:CONSTRUCTOR:toFewInputArgs')
        if arity == 3:
            cartesian = op
            def op(lam, theta):
                return cartesian(jnp.cos(lam) * jnp.sin(theta),
                                 jnp.sin(lam) * jnp.sin(theta), jnp.cos(theta))
    opts = list(options)
    rank, lengths = None, None
    while opts and not isinstance(opts[0], (str, ChebfunPref)):
        d = jnp.asarray(opts.pop(0)).ravel()
        if d.size == 4:
            if bool(jnp.any(d != jnp.asarray([-jnp.pi, jnp.pi, 0., jnp.pi]))):
                raise ValueError('CHEBFUN:SPHEREFUN:CONSTRUCTOR:domain')
        elif d.size == 2:
            lengths = d
        elif d.size == 1:
            rank = float(d[0])
        else:
            raise ValueError('CHEBFUN:SPHEREFUN:CONSTRUCTOR:domain')
    eps = float(jnp.finfo(jnp.float64).eps)
    if lengths is not None:
        if bool(jnp.any((lengths <= 0) | (jnp.abs(jnp.round(lengths)-lengths) > eps))):
            raise ValueError('CHEBFUN:SPHEREFUN:constructor:parseInputs:domain2')
        lengths = tuple(int(x) for x in lengths)
    if rank is not None and bool(jnp.isnan(rank)):
        rank = None
    if rank is not None:
        if not bool(jnp.isfinite(rank)) or rank < 0 or abs(round(rank)-rank) > eps:
            raise ValueError('CHEBFUN:SPHEREFUN:constructor:parseInputs:domain3')
        rank = int(rank)
    prefs = [p for p in opts if isinstance(p, ChebfunPref)]
    pref = prefs[0] if prefs else ChebfunPref()
    opts = [p for p in opts if not isinstance(p, ChebfunPref)]
    tol = pref.cheb2Prefs.chebfun2eps
    for i, p in enumerate(opts):
        if isinstance(p, str) and p.lower() == 'eps':
            tol = float(jnp.maximum(tol, opts[i+1]))
            del opts[i:i+2]
            break
    vectorize = any(isinstance(p, str) and p.lower().startswith('vectori') for p in opts)
    if callable(op) and not isinstance(op, Spherefun):
        op = _pointwise(op) if vectorize else _vector_check(op, pref.chebfuneps)
        callback = op
        def op(lam, theta):
            value = jnp.asarray(callback(lam, theta))
            if jnp.iscomplexobj(value):
                warnings.warn("SPHEREFUN:CONSTRUCTOR:COMPLEX", stacklevel=2)
                value = jnp.real(value)
            if bool(jnp.any(jnp.isinf(value))):
                raise ValueError('CHEBFUN:SPHEREFUN:constructor:inf')
            if bool(jnp.any(jnp.isnan(value))):
                raise ValueError('CHEBFUN:SPHEREFUN:constructor:nan')
            return jnp.broadcast_to(jnp.real(value), jnp.broadcast_shapes(lam.shape, theta.shape))
    if any(isinstance(p, str) and p.lower() == 'coeffs' for p in opts):
        op = Spherefun.coeffs2spherefun(op)
    if lengths is not None:
        m, n = lengths
        x = map_global_nodes(global_trigpts_nodes(2*n), -jnp.pi, jnp.pi)
        y = jnp.linspace(0., jnp.pi, m+1)
        xx, yy = jnp.meshgrid(x, y)
        op = op(xx, yy)
    if isinstance(op, Spherefun):
        return fix_rank(op, rank)
    if callable(op):
        return Spherefun.from_function(op, tol=tol,
                                       max_rank=pref.cheb2Prefs.maxRank,
                                       max_sample=2**16,
                                       sample_test=pref.cheb2Prefs.sampleTest,
                                       _fixed_rank=rank)
    values = jnp.asarray(op)
    if jnp.iscomplexobj(values):
        warnings.warn('SPHEREFUN:CONSTRUCTOR:COMPLEX', stacklevel=2)
        values = jnp.real(values)
    if values.size == 1:
        # Source numeric scalar recursion discards supplied preferences and
        # re-enters callable construction before the OUTER fixed-rank rule.
        # Reuse public dispatch so Inf/NaN preserve source error identifiers.
        value = values.reshape(())
        return fix_rank(spherefun(lambda lam, theta: value + 0 * lam), rank)
    # Explicit eps affects numeric tolerance after source pole/column checks.
    numeric_tol = None
    if values.ndim == 2 and values.shape[0] > 1 and values.shape[1] % 2 == 0:
        n, m = values.shape
        numeric_tol = _get_tol_sphere(values, 2*jnp.pi/m, jnp.pi/(n-1), tol)[0]
    return fix_rank(Spherefun.from_values(values, tol=numeric_tol), rank)
