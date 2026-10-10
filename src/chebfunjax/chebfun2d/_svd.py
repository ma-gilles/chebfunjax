"""Continuous factor QR and small-core SVD, native pin 7574c77.

Sources: @separableApprox/{svd,norm,cdr}.m, @chebfun/{qr,mtimes}.m,
@chebfun2/outerProduct.m. Python lists/vector S adapt native quasimatrices
and diagonal S. Empty values-only uses an empty array instead of Chebfun.
Host shape/zero dispatch is intentional; global JIT/AD is not promised.
"""
from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech, _trig_inner_product_jax, _trig_qr_collate


@eqx.filter_jit
def _axis_panel(factors):
    """Native horizontal collation without turning an array into a quasi.

    @chebtech/horzcat.m and @trigtech/horzcat.m (7574c77): first input
    technology/happiness; Trig coefficient AND value caches are collated.
    """
    if all(isinstance(t, Trigtech) for t in factors):
        return _trig_qr_collate(factors)
    if all(isinstance(t, (Chebtech1, Chebtech2)) for t in factors):
        n = max(t.n for t in factors)
        return type(factors[0])(
            coeffs=jnp.column_stack([jnp.pad(t.coeffs, (0, n-t.n))
                                    for t in factors]),
            ishappy=factors[0].ishappy)
    raise NotImplementedError(
        'SVD axis requires source collation for heterogeneous or singular '
        'factor technologies; no polynomial reinterpretation is provided.')


@eqx.filter_jit
def _divide_axis(panel, denominator):
    """Source scalar division; denominator is the nonzero real QR norm.

    @chebfun/qr.m and @trigtech/rdivide.m (7574c77). This private path
    avoids inherited eager transform dispatch without changing scalar QR.
    """
    if isinstance(panel, Trigtech):
        return Trigtech(coeffs=panel.coeffs/denominator,
                        real_columns=panel.real_columns,
                        ishappy=panel.ishappy,
                        _values=panel.values/denominator)
    return type(panel)(coeffs=panel.coeffs/denominator,
                       ishappy=panel.ishappy)


def _constant_panel(cls, value):
    """One-column constant after source global simplification."""
    values = jnp.asarray([[value]], dtype=jnp.float64)
    if cls is Trigtech:
        return Trigtech(coeffs=values.astype(jnp.complex128),
                        real_columns=(True,), ishappy=True, _values=values)
    return cls(coeffs=values, ishappy=True)


def _as_fun(panel, interval):
    return Chebfun(funs=[_Piece(tech=panel, interval=interval)],
                   domain=Domain(interval))


@eqx.filter_jit
def _polynomial_scalar_inner(panel):
    return jnp.reshape(panel.inner(panel), ())


def _scalar_inner(panel):
    """Shared native self-product with its actual post-prolong equality.

    Python scalar storage may become real only when the source real-pair
    or equality predicate applies. NaN coefficients make equality false.
    """
    if not isinstance(panel, Trigtech):
        return _polynomial_scalar_inner(panel)
    matrix, equal = _trig_inner_product_jax(panel, panel)
    value = jnp.reshape(matrix, ())
    return jnp.real(value) if panel.is_real or bool(equal) else value


def _require_finite(value, stage):
    """Explicit Python capability boundary, not a native exception claim.

    Source formulas are preserved until the named stage. Nonfinite native
    provider behavior is unobserved; do not silently accept JAX NaN output
    or replace it with a fabricated answer.
    """
    if not bool(jnp.all(jnp.isfinite(value))):
        raise NotImplementedError(
            f'Nonfinite {stage}: native SVD/QR provider semantics remain unqualified.')


def _finite_panel_storage(panel, stage):
    _require_finite(panel.coeffs, stage)
    if isinstance(panel, Trigtech) and panel._values is not None:
        _require_finite(panel._values, stage)


def _axis_qr(panel, interval):
    """Native continuous QR including one-column zero normalization.

    @chebfun/qr.m27-36 and @bndfun/qr.m (7574c77). Scalar arithmetic is
    JAX; the nonzero predicate is a host adapter, not a differentiable branch.
    """
    if panel.coeffs.ndim == 2 and panel.coeffs.shape[1] != 1:
        _finite_panel_storage(panel, 'multicolumn QR input')
        q, r = _as_fun(panel, interval).qr()
        _require_finite(r, 'multicolumn QR output')
        return q, r
    scale = (interval[1]-interval[0])/2
    r = jnp.sqrt(_scalar_inner(panel)*scale)
    _require_finite(r, 'scalar QR normalization')
    if bool(r != 0):
        q = _divide_axis(panel, r)
    else:
        q = _constant_panel(type(panel), 1/jnp.sqrt(interval[1]-interval[0]))
    return _as_fun(q, interval), jnp.reshape(r, (1, 1))


@eqx.filter_jit
def _action(panel, matrix):
    """Native FUN numeric mtimes, retaining Trig cached values."""
    return panel @ matrix


@eqx.filter_jit
def _conjugate(panel):
    """Literal native Tech conjugation, including even-Trig source quirk."""
    return panel.conj()


@eqx.filter_jit
def _core_matrix(left, weights, right):
    """@separableApprox/svd.m: LEFT*D*RIGHT.'; no Gram/cutoff."""
    return (left @ jnp.diag(weights)) @ right.T


@eqx.filter_jit
def _core_provider(core):
    u, s, vh = jnp.linalg.svd(core, full_matrices=False)
    return u, s, jnp.conj(vh.T)


def _core_svd(left, weights, right):
    core = _core_matrix(left, weights, right)
    _require_finite(core, 'small-core SVD input')
    u, s, v = _core_provider(core)
    for value in (u, s, v):
        _require_finite(value, 'small-core SVD output')
    return u, s, v


def _zero_constant_class(factors):
    """Source 1+0*f promotes via compose/constructor with axis periodicity.

    The constant-one ACA has one unit pivot and unit column/row slices;
    global simplify reduces each slice to one coefficient. Nonperiodic
    axes select the session Tech, as constructor.m672-713 requires.
    """
    from chebfunjax.chebpref import ChebfunPref

    pref = ChebfunPref()
    if all(isinstance(t, Trigtech) for t in factors):
        pref.tech = Trigtech
    if callable(pref.techPrefs.happinessCheck) or callable(pref.techPrefs.refinementFunction):
        raise NotImplementedError(
            'Full zero SVD requires native constructor callback execution '
            'for custom happiness/refinement preferences.')
    tech = pref.tech
    key = (tech.__name__ if isinstance(tech, type) else str(tech)).lower().lstrip('@')
    classes = {'chebtech': Chebtech2, 'chebtech1': Chebtech1,
               'chebtech2': Chebtech2, 'trigtech': Trigtech,
               'trig': Trigtech, 'periodic': Trigtech}
    if key not in classes:
        raise NotImplementedError('Constant SVD factors require a supported session Tech.')
    return classes[key]


def source_svd(approx, *, full=False, operator=False, as_array=False):
    """Native factor operations with explicit Python output adapters.

    ``operator`` represents L=C*D*R' before QR, as norm.m requires. The
    full complex V is literal Qright*V, not a corrected reconstruction
    factor. Rank-deficient singular vectors have provider-dependent phases.
    """
    # Native SVD retains array-valued CHEBFUNs through subsequent mtimes.
    # Lists are only the existing Python public-output adapter.
    def full_output(left, singular, right):
        if as_array:
            return left, singular, right
        return left.mat2cell(), singular, right.mat2cell()

    empty = jnp.empty((0,), dtype=jnp.float64)
    if approx is None or not approx.cols:
        if full and as_array:
            return Chebfun.empty(), empty, Chebfun.empty()
        return ([], empty, []) if full else empty
    xa, xb, ya, yb = approx.domain
    weights = jnp.asarray(approx.pivots)
    if not operator and bool(jnp.linalg.norm(weights) == 0):
        s = jnp.zeros((1,), dtype=jnp.float64)
        if not full:
            return s
        # The constant simplification of native 1+0*f is valid only for
        # finite factor storage. Values-only returned above without reading
        # factors, exactly as the native zero-weight branch requires.
        for axis in (approx.cols, approx.rows):
            for factor in axis:
                _finite_panel_storage(factor, 'full-zero factor construction')
        # Literal native width/height scaling, even on nonsquare domains.
        left = _constant_panel(_zero_constant_class(approx.cols), 1/jnp.sqrt(xb-xa))
        right = _constant_panel(_zero_constant_class(approx.rows), 1/jnp.sqrt(yb-ya))
        if as_array:
            return _as_fun(left, (ya, yb)), s, _as_fun(right, (xa, xb))
        return [_as_fun(left, (ya, yb))], s, [_as_fun(right, (xa, xb))]
    left = _axis_panel(tuple(approx.cols))
    right = _axis_panel(tuple(approx.rows))
    if operator:
        # @chebfun/mtimes -> outerProduct: columns=C*D, rows=conj(R),
        # pivotValues=ones. Do not move D outside the first QR.
        left = _action(left, jnp.diag(weights))
        right = _conjugate(right)
        weights = jnp.ones(weights.shape, dtype=jnp.float64)
    qleft, rleft = _axis_qr(left, (ya, yb))
    qright, rright = _axis_qr(right, (xa, xb))
    u, s, v = _core_svd(rleft, weights, rright)
    if not full:
        return s
    left = _action(qleft.funs[0].tech, u)
    right = _action(qright.funs[0].tech, v)
    return full_output(_as_fun(left, (ya, yb)), s, _as_fun(right, (xa, xb)))


def source_frobenius(approx):
    """Native norm empty branch, then sqrt(sum(svd(f).^2))."""
    if not approx.cols:
        return jnp.empty((0,), dtype=jnp.float64)
    values = source_svd(approx)
    return jnp.sqrt(jnp.sum(values**2))
