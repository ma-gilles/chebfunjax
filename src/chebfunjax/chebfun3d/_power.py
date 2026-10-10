"""Native Chebfun3 power dispatch and private source predicates.

Provenance
----------
MATLAB source : @chebfun3/power.m, private/singleSignTest.m,
    @chebfun3/{domainCheck,isreal,vscale}.m, @chebfun/{domainCheck,hscale}.m
Chebfun commit: 7574c77

Dispatch and numeric promotion are eager. New numerical operations use JAX;
the adaptive constructor retains its existing host implementation.
"""

import equinox as eqx
import jax.numpy as jnp

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.domain import _linear_inverse_map
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2, _clenshaw
from chebfunjax.tech.trigtech import Trigtech, _trig_eval
from chebfunjax.utils.quadrature import chebpts_ab, trigpts


def _is_empty(value):
    if hasattr(value, 'isempty'):
        return value.isempty()
    if isinstance(value, (list, tuple, str)):
        return len(value) == 0
    return hasattr(value, 'size') and value.size == 0


def _double_scalar(value):
    # Python numeric literals adapt MATLAB doubles; typed arrays retain class.
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, complex)):
        return jnp.asarray(value, dtype=jnp.complex128 if isinstance(value, complex)
                           else jnp.float64)
    try:
        array = jnp.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.size == 1 and array.dtype in (jnp.float64, jnp.complex128):
        return array.reshape(())
    return None


def _source_isreal(f):
    """Literal stored core and factor realness, including complex-zero data."""
    if jnp.iscomplexobj(f.core):
        return False
    for axis in (f.cols, f.rows, f.tubes):
        for tech in axis:
            if isinstance(tech, Trigtech):
                if not tech.is_real:
                    return False
            elif jnp.iscomplexobj(tech.coeffs):
                return False
    return True


def _domain_check(f, g):
    """Three independent strict source Chebfun domain predicates."""
    for axis in range(3):
        a = jnp.asarray(f.domain[2*axis:2*axis+2])
        b = jnp.asarray(g.domain[2*axis:2*axis+2])
        ha, hb = jnp.max(jnp.abs(a)), jnp.max(jnp.abs(b))
        ha = jnp.where(jnp.isinf(ha), 1., ha)
        hb = jnp.where(jnp.isinf(hb), 1., hb)
        delta = a-b
        if not bool(jnp.all((jnp.abs(delta) < 1e-15*jnp.maximum(ha, hb))
                            | jnp.isnan(delta))):
            return False
    return True


def _factor_values(f, axis, point):
    group = (f.cols, f.rows, f.tubes)[axis]
    a, b = f.domain[2*axis:2*axis+2]
    t = _linear_inverse_map(point, a, b)
    return jnp.stack([_trig_eval(tech.coeffs, t, tech.is_real)
                      if isinstance(tech, Trigtech) else _clenshaw(tech.coeffs, t)
                      for tech in group], axis=-1)


def _txm(core, matrix, axis):
    return jnp.moveaxis(jnp.tensordot(matrix, core, axes=(1, axis)), 0, axis)


@eqx.filter_jit
def _evaluate_grid(f, vectors, axes, order):
    # Source tensorCase: permute the core to physical tensor axes, then
    # contract sequentially in the literal x/y/z (meshgrid: y/x/z) order.
    permutation = tuple(axes.index(i) for i in range(3))
    result = jnp.transpose(f.core, permutation)
    for coordinate in order:
        result = _txm(result, _factor_values(f, coordinate, vectors[coordinate]),
                      axes[coordinate])
    return result


@eqx.filter_jit
def _evaluate_vector(f, points):
    # Source vectorCase: x contraction, pagewise y contraction, then z sum.
    panels = [_factor_values(f, axis, point.reshape(-1))
              for axis, point in enumerate(points)]
    result = _txm(f.core, panels[0], 0)
    result = jnp.einsum('pjk,pj->pk', result, panels[1])
    return jnp.sum(result * panels[2], axis=1).reshape(points[0].shape)


def _evaluate(f, x, y, z):
    """Native tensor-grid recognition and sequential vector contractions.

    Eager exact grid recognition precedes persistent JAX contraction kernels.
    Grid factors are evaluated only on their one-dimensional coordinate lists.
    Matrix/unstructured inputs use the source vector contraction order.
    """
    points = jnp.broadcast_arrays(jnp.asarray(x), jnp.asarray(y), jnp.asarray(z))
    if points[0].size == 0:
        return jnp.empty(points[0].shape, dtype=f.core.dtype)
    if points[0].ndim == 3:
        for axes, order in (((0, 1, 2), (0, 1, 2)),
                            ((2, 0, 1), (0, 1, 2)),
                            ((1, 2, 0), (0, 1, 2)),
                            ((1, 0, 2), (1, 0, 2))):
            vectors = []
            for point, axis in zip(points, axes):
                index = tuple(slice(None) if i == axis else 0 for i in range(3))
                vector = point[index]
                shape = tuple(vector.size if i == axis else 1 for i in range(3))
                if not bool(jnp.max(jnp.abs(point-vector.reshape(shape))) == 0):
                    break
                vectors.append(vector)
            else:
                return _evaluate_grid(f, tuple(vectors), axes, order)
    return _evaluate_vector(f, points)


def _source_vscale(f):
    """First-column technology chooses all three native coarse grids."""
    tech = f.cols[0]
    counts = [min(max(n, 9), 41) for n in f.length()]
    grids = []
    for axis, count in enumerate(counts):
        a, b = f.domain[2*axis:2*axis+2]
        if isinstance(tech, Trigtech):
            grid = trigpts(count, (a, b))[0]
        elif isinstance(tech, Chebtech1):
            grid = chebpts_ab(count, a, b, kind=1)
        elif isinstance(tech, Chebtech2):
            grid = chebpts_ab(count, a, b, kind=2)
        else:
            raise ValueError('CHEBFUN:CHEBFUN3:vscale:mypoints:techType')
        grids.append(grid)
    points = jnp.meshgrid(*grids, indexing='ij')
    return jnp.max(jnp.abs(_evaluate(f, *points)))


def _single_sign_test(f):
    """Native strict fixed-grid sign predicate; no extremum search."""
    tol = 2 * ChebfunPref().cheb3Prefs.chebfun3eps
    values = f.sample()
    threshold = tol * _source_vscale(f)
    positive = bool(jnp.all(values > -threshold))
    single = positive or bool(jnp.all(values < threshold))
    return single, bool(jnp.any(values == 0)), positive


def _fractional(exponent):
    # round(n)~=n: round complex components separately. For this equality
    # predicate, floor detects precisely the same finite integer set.
    real = jnp.real(exponent)
    imag = jnp.imag(exponent)
    return bool((jnp.floor(real) != real) | (jnp.floor(imag) != imag))


def _value_power(base, exponent):
    """Real integer arithmetic or principal complex power as MATLAB requires.

    Sampling is eager: promotion depends on the paired sampled values.
    This is not a globally traceable dtype-changing primitive.
    """
    base, exponent = jnp.asarray(base), jnp.asarray(exponent)
    if jnp.iscomplexobj(base) or jnp.iscomplexobj(exponent):
        return jnp.power(base.astype(jnp.complex128), exponent)
    needs_complex = jnp.any((base < 0) & (jnp.floor(exponent) != exponent))
    if bool(needs_complex):
        return jnp.power(base.astype(jnp.complex128), exponent)
    return jnp.power(base, exponent)


def _construct(op, domain, *, complex_possible=False, preserve_complex=False):
    """Use explicit components when one real probe cannot classify the output.

    The inherited constructor tests one point before choosing its real path.
    Power can be real there and complex elsewhere. Explicit real/imaginary
    construction avoids that dtype loss without changing the constructor.
    Exact zero imaginary output from real inputs retains a real representation;
    explicitly complex operands retain the existing complex assembly adapter.
    """
    from chebfunjax.chebfun3d.chebfun3 import Chebfun3

    tol = ChebfunPref().cheb3Prefs.chebfun3eps
    if not complex_possible:
        return Chebfun3.from_function(op, domain=domain, tol=tol)
    real = Chebfun3.from_function(
        lambda x, y, z: jnp.real(op(x, y, z)), domain=domain, tol=tol)
    imag = Chebfun3.from_function(
        lambda x, y, z: jnp.imag(op(x, y, z)), domain=domain, tol=tol)
    if not preserve_complex and imag.iszero():
        return real
    from chebfunjax.chebfun3d._plus import _assemble_components

    return _assemble_components(real, imag * 1j)


def source_power(base, exponent):
    """Full native power branch order with Python scalar/operator adapters.

    Provenance
    ----------
    MATLAB source : @chebfun3/power.m
    Chebfun commit: 7574c77

    Typed numeric arrays must be binary64 real/complex singletons; Python
    int/float/complex literals adapt double. Empty operands precede validation.
    Nonfinite sampled results retain the constructor's existing limitations.
    """
    from chebfunjax.chebfun3d.chebfun3 import Chebfun3

    if _is_empty(base) or _is_empty(exponent):
        return Chebfun3.empty()
    scalar_base, scalar_exponent = _double_scalar(base), _double_scalar(exponent)
    if scalar_base is not None and isinstance(exponent, Chebfun3):
        return _construct(lambda x, y, z: _value_power(
            scalar_base, _evaluate(exponent, x, y, z)), exponent.domain,
            complex_possible=(jnp.iscomplexobj(scalar_base) or
                              not _source_isreal(exponent) or bool(jnp.real(scalar_base) < 0)),
            preserve_complex=jnp.iscomplexobj(scalar_base) or not _source_isreal(exponent))
    if isinstance(base, Chebfun3) and scalar_exponent is not None:
        if _fractional(scalar_exponent) and _source_isreal(base):
            if not _single_sign_test(base)[0]:
                raise ValueError('CHEBFUN:CHEBFUN3:power:fractional: '
                                 'Sign change detected. Unable to represent the result.')
        return _construct(lambda x, y, z: _value_power(
            _evaluate(base, x, y, z), scalar_exponent), base.domain,
            complex_possible=(not _source_isreal(base) or
                              jnp.iscomplexobj(scalar_exponent) or _fractional(scalar_exponent)),
            preserve_complex=not _source_isreal(base) or jnp.iscomplexobj(scalar_exponent))
    if isinstance(base, Chebfun3) and isinstance(exponent, Chebfun3):
        if not _domain_check(base, exponent):
            raise ValueError('CHEBFUN:CHEBFUN3:power:domain: Domains must be the same.')
        return _construct(lambda x, y, z: _value_power(
            _evaluate(base, x, y, z), _evaluate(exponent, x, y, z)), base.domain,
            complex_possible=True,
            preserve_complex=not _source_isreal(base) or not _source_isreal(exponent))
    raise ValueError('CHEBFUN:CHEBFUN3:power:inputs: '
                     'Inputs must be either CHEBFUN3 objects or scalars.')
