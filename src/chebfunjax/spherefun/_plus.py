"""Real sphere factor addition; variable-rank host dispatch, JAX arithmetic.

Provenance
----------
MATLAB source : @spherefun/plus.m, @spherefun/extractPole.m,
    @separableApprox/plus.m, @separableApprox/uminus.m
Chebfun commit: 7574c77
"""

import jax
import jax.numpy as jnp

from chebfunjax.spherefun._cdr import inverse_pivots
from chebfunjax.spherefun._factor_assembly import stack_factor_coefficients
from chebfunjax.tech.trigtech import (
    _trig_coeffs2vals_impl,
    _trig_eval,
    _trig_prolong_coeffs,
    _trig_vals2coeffs_impl,
)


@jax.jit
def real_trig_matrix_qr(coefficients):
    """Source builtin array-trig QR followed by physical [-pi,pi] scaling.

    Provenance
    ----------
    MATLAB source : @trigtech/qr.m (qr_builtin), @bndfun/qr.m
    Chebfun commit: 7574c77
    Requires >=2 real-valued factor columns; scalar/empty paths belong to caller.
    """
    nf, count = coefficients.shape
    if count < 2:
        raise ValueError("caller must use source scalar/empty QR branches")
    size = max(nf, count)
    padded = _trig_prolong_coeffs(coefficients, size)
    values = jnp.real(_trig_coeffs2vals_impl(padded))
    q, r = jnp.linalg.qr(values, mode="reduced")
    signs = jnp.sign(jnp.diag(r))
    signs = jnp.where(signs == 0, 1, signs)
    q = q * signs[None, :]
    r = signs[:, None] * r
    weight = jnp.sqrt(jnp.asarray(2.0) / size)
    q = q / weight
    r = weight * r
    qc = _trig_prolong_coeffs(_trig_vals2coeffs_impl(q), nf)
    # bndfun physical domain rescaleFactor=pi; retain source sequential stages.
    return qc / jnp.sqrt(jnp.pi), r * jnp.sqrt(jnp.pi)


def compressed_parity_core(columns, rows, pivots, vscale_f, vscale_g):
    """Source small-SVD compression for already-concatenated non-pole factors.

    Provenance
    ----------
    MATLAB source : @separableApprox/plus.m (compression_plus)
    Chebfun commit: 7574c77
    Caller handles operand pivot-order swap, parity groups, zero/empty and poles.
    """
    qc, rc = real_trig_matrix_qr(columns)
    qr, rr = real_trig_matrix_qr(rows)
    d = jnp.diag(1 / pivots)
    u, s, vh = jnp.linalg.svd((rc @ d) @ rr.T, full_matrices=False)
    threshold = 10 * jnp.finfo(jnp.float64).eps * (2 * jnp.maximum(vscale_f, vscale_g))
    keep = int(jnp.sum(s > threshold))
    if keep == 0:
        return None  # caller must implement source 0*f representation
    return qc @ u[:, :keep], qr @ vh[:keep, :].T, 1 / s[:keep]


def _class():
    from chebfunjax.spherefun.spherefun import Spherefun

    return Spherefun


def _tech(c):
    from chebfunjax.tech.trigtech import Trigtech

    return Trigtech(coeffs=c, is_real=True, ishappy=True)


def _make(cols, rows, pivots, plus, pole=False, locations=None):
    if not cols:
        return _class().empty()
    return _class()(
        cols=list(cols),
        rows=list(rows),
        pivots=jnp.asarray(pivots),
        idx_plus=tuple(range(plus)),
        idx_minus=tuple(range(plus, len(cols))),
        nonzero_poles=pole,
        pivot_locations=(
            tuple(locations)
            if locations is not None
            else tuple((float("nan"), float("nan")) for _ in cols)
        ),
    )


def _zero():
    return _make([_tech(jnp.zeros(1))], [_tech(jnp.ones(1))], [1.0], 1)


def _stack(techs):
    return stack_factor_coefficients(techs)


@jax.jit
def _real_sample_values(coefficients):
    """Persistently stage the unchanged real part of the source FFT transform.

    Provenance
    ----------
    MATLAB source : @trigtech/coeffs2vals.m, @spherefun/sample.m
    Chebfun commit: 7574c77
    Shape/dtype define the compilation cache; coefficient values remain dynamic.
    Disabled JIT executes the same pure JAX transform. Sampling and aliasing stay
    in the caller, with their original sequential arithmetic.
    """
    return jnp.real(_trig_coeffs2vals_impl(coefficients))


def _sample(techs, m):
    """Real factor samples using source coefficient aliasing and FFT.

    Provenance
    ----------
    MATLAB source : @trigtech/sample.m, @trigtech/alias.m, @spherefun/sample.m
    Chebfun commit: 7574c77
    The single-point branch preserves source reversed negative-mode dot and
    positive-mode dot before their sum with the constant coefficient.
    """
    # Source sample aliases when its vscale sampling cap is below stored length.
    # The existing generic alias helper uses NumPy; this bounded matrix port
    # keeps the source update order and JAX arithmetic, including disabled JIT.
    c = _stack(techs)
    n = c.shape[0]
    if m >= n:
        c = _trig_prolong_coeffs(c, m)
    else:
        if n % 2 == 0:
            c = c.at[0].multiply(0.5)
            c = jnp.concatenate((c, c[:1]), axis=0)
            n += 1
        n2 = (n - 1) // 2
        if m == 1:
            # alias.m's dedicated one-point branch, evaluated at x=-1.
            const = c[n2]
            negative = c[:n2][::-1]
            positive = c[n2 + 1 :]
            signs = jnp.where(jnp.arange(n2) % 2 == 0, -1.0, 1.0)
            a = (const + (signs @ negative + signs @ positive))[None, :]
        elif m % 2:
            m2 = (m - 1) // 2
            a = c[n2 - m2 : n2 + m2 + 1]
            for j in range(-n2, -m2):
                k = (j + m2 + 1) % (-m) + m2
                sign = (-1) ** ((j + k) % 2)
                a = a.at[k + m2].add(sign * c[j + n2])
                a = a.at[-k + m2].add(sign * c[-j + n2])
        else:
            m2 = m // 2
            a = c[n2 - m2 : n2 + m2]
            a = jnp.concatenate((a, -a[:1]), axis=0)
            for j in range(-n2, -m2 + 1):
                k = (j + m2) % (-m) + m2
                a = a.at[k + m2].add(c[j + n2])
                a = a.at[-k + m2].add(c[-j + n2])
            a = a.at[0].add(a[-1])[:-1]
        c = a
    return _real_sample_values(c)



def _scale(f):
    # Source separableApprox.vscale -> spherefun.sample; preserve physical half.
    if f.isempty() or not f.cols:
        return jnp.asarray(0.0)
    m = min(max(max(t.coeffs.shape[0] for t in f.rows), 9), 2000)
    n = min(max(max(t.coeffs.shape[0] for t in f.cols), 9), 2000)
    cv = _sample(f.cols, 2 * n - 2)
    cv = jnp.concatenate((cv[n - 1 : 2 * n - 2], cv[:1]), axis=0)
    rv = _sample(f.rows, m)
    return jnp.max(jnp.abs((cv @ jnp.diag(inverse_pivots(f.pivots))) @ rv.T))


@jax.jit
def _real_factor_values(coefficients, points):
    """Persistent JAX staging boundary for unchanged real Horner evaluation.

    Provenance
    ----------
    MATLAB source : @trigtech/horner.m, @separableApprox/iszero.m
    Chebfun commit: 7574c77
    No padding, coefficient regrouping or change to the source zero predicate.
    The wrapper caches one executable per input shape/dtype; disabled JIT still
    executes the same JAX kernel. This is not whole-Spherefun JIT support.
    """
    return _trig_eval(coefficients, points, is_real=True)


def _iszero(f):
    if f.isempty() or not f.cols:
        return True
    if bool(jnp.max(jnp.abs(1 / f.pivots)) == 0):
        return True
    # Source iszero uses 10x10 physical endpoint-inclusive samples, exact >0.
    lam = jnp.linspace(-1.0, 1.0, 10)
    th = jnp.linspace(0.0, 1.0, 10)
    c = jnp.stack([_real_factor_values(t.coeffs, th) for t in f.cols], axis=1)
    r = jnp.stack([_real_factor_values(t.coeffs, lam) for t in f.rows], axis=1)
    if bool(
        jnp.max(jnp.abs((c @ jnp.diag(inverse_pivots(f.pivots))) @ r.T)) > 0
    ):
        return False
    return all(bool(jnp.all(t.coeffs == 0)) for t in f.cols) or all(
        bool(jnp.all(t.coeffs == 0)) for t in f.rows
    )


def negate(f):
    """Source structural unary minus, preserving all factor metadata.

    Provenance
    ----------
    MATLAB source : @separableApprox/uminus.m
    Chebfun commit: 7574c77
    """
    if f.isempty():
        return f
    return _class()(
        cols=list(f.cols),
        rows=list(f.rows),
        pivots=-f.pivots,
        idx_plus=f.idx_plus,
        idx_minus=f.idx_minus,
        pivot_locations=f.pivot_locations,
        nonzero_poles=f.nonzero_poles,
    )


def _equal(f, g):
    if f.isempty() or g.isempty():
        return f.isempty() and g.isempty()
    return (
        f.pivots.shape == g.pivots.shape
        and bool(jnp.array_equal(f.pivots, g.pivots))
        and all(
            a.coeffs.shape == b.coeffs.shape and bool(jnp.array_equal(a.coeffs, b.coeffs))
            for a, b in zip(f.cols + f.rows, g.cols + g.rows, strict=True)
        )
    )


def _select(f, indices, plus, pole=False):
    ids = list(indices)
    if not ids:
        return _class().empty()
    loc = [f.pivot_locations[i] for i in ids] if len(f.pivot_locations) == len(f.cols) else None
    return _make(
        [f.cols[i] for i in ids],
        [f.rows[i] for i in ids],
        f.pivots[jnp.asarray(ids)],
        len(ids) if plus else 0,
        pole,
        loc,
    )


def _extract(f):
    if not f.nonzero_poles:
        return f, _class().empty()
    # Source extractPole owns the first factor, not the first surviving plus tag.
    pole = _select(f, [0], True, True)
    keep = list(range(1, len(f.cols)))
    if not keep:
        return _class().empty(), pole
    rest = _class()(
        cols=list(f.cols[1:]),
        rows=list(f.rows[1:]),
        pivots=f.pivots[1:],
        idx_plus=tuple(i - 1 for i in f.idx_plus if i != 0),
        idx_minus=tuple(i - 1 for i in f.idx_minus),
        pivot_locations=tuple(f.pivot_locations[1:]),
        nonzero_poles=False,
    )
    return rest, pole


def _partition(f, plus):
    if f.isempty():
        return f
    return _select(f, f.idx_plus if plus else f.idx_minus, plus)


def _compress(f, g, plus):
    if f.isempty():
        return g
    if g.isempty():
        return f
    if _iszero(f):
        return g
    if _iszero(g):
        return f
    if bool(jnp.min(jnp.abs(f.pivots)) < jnp.min(jnp.abs(g.pivots))):
        f, g = g, f
    cols = list(f.cols) + list(g.cols)
    rows = list(f.rows) + list(g.rows)
    result = compressed_parity_core(
        _stack(cols), _stack(rows), jnp.concatenate((f.pivots, g.pivots)), _scale(f), _scale(g)
    )
    if result is None:
        z = _zero()
        return _make(z.cols, z.rows, z.pivots, 1 if plus else 0)
    c, r, piv = result
    return _make(
        [_tech(c[:, i]) for i in range(c.shape[1])],
        [_tech(r[:, i]) for i in range(r.shape[1])],
        piv,
        c.shape[1] if plus else 0,
    )


def _column_norm(c):
    # Literal trigtech.innerProduct: prolong to sum of lengths, trapezium
    # weights, weighted dot, physical bndfun scaling, then chebfun.norm.
    n = 2 * c.shape[0]
    v = jnp.real(_trig_coeffs2vals_impl(_trig_prolong_coeffs(c, n)))
    inner = ((jnp.asarray(2.0) / n) * v) @ v
    return jnp.sqrt(jnp.abs(jnp.abs(inner) * jnp.pi))


def _add_poles(f, g, tol):
    if g.isempty():
        return f
    if f.isempty():
        return g
    fc, gc = f.cols[0], g.cols[0]
    size = max(fc.coeffs.shape[0], gc.coeffs.shape[0])
    fm = f.rows[0].coeffs[f.rows[0].coeffs.shape[0] // 2]
    gm = g.rows[0].coeffs[g.rows[0].coeffs.shape[0] // 2]
    fv = jnp.real(_trig_coeffs2vals_impl(_trig_prolong_coeffs(fc.coeffs, size)))
    gv = jnp.real(_trig_coeffs2vals_impl(_trig_prolong_coeffs(gc.coeffs, size)))
    values = (fm / f.pivots[0]) * fv + (gm / g.pivots[0]) * gv
    coeffs = _trig_vals2coeffs_impl(values)
    if bool(_column_norm(coeffs) <= tol):
        # Literal addPoles zero branch: zero both multipliers, pivot=0.
        values = (0 / f.pivots[0]) * fv + (0 / g.pivots[0]) * gv
        return _make([_tech(_trig_vals2coeffs_impl(values))], [_tech(jnp.ones(1))], [0.0], 1, False)
    from chebfunjax.tech.trigtech import _trig_eval

    endpoints = _trig_eval(coeffs, jnp.asarray([0.0, 1.0]), is_real=True)
    pole = bool(jnp.any(jnp.abs(endpoints) > tol))
    return _make([_tech(coeffs)], [_tech(jnp.ones(1))], [1.0], 1, pole)


def eligible(f):
    """Real canonical scalar factors eligible for this source addition path.

    Provenance
    ----------
    MATLAB source : @spherefun/plus.m
    Chebfun commit: 7574c77
    Complex objects retain the legacy public route; no complex parity claim.
    """
    return f.isempty() or (
        all(t.is_real for t in f.cols + f.rows) and not jnp.iscomplexobj(f.pivots)
    )


def plus(f, g):
    """Source real sphere addition with pole extraction and parity compression.

    Provenance
    ----------
    MATLAB source : @spherefun/plus.m, @separableApprox/plus.m
    Chebfun commit: 7574c77
    Host decisions determine rank; all new numerical operations use JAX.
    """
    if f.isempty():
        return g
    if not isinstance(g, _class()):
        scalar = jnp.asarray(g)
        if scalar.ndim != 0 or jnp.iscomplexobj(scalar):
            raise TypeError("SPHEREFUN:plus:unknown")
        g = (
            _zero()
            if bool(scalar == 0)
            else _make([_tech(jnp.atleast_1d(scalar))], [_tech(jnp.ones(1))], [1.0], 1, True)
        )
    if f.isempty():
        return g
    if g.isempty():
        return f
    if not eligible(f) or not eligible(g):
        raise NotImplementedError("real source addition only")
    if _iszero(f):
        return g
    if _iszero(g):
        return f
    if _equal(f, negate(g)):
        return _zero()
    fr, fp = _extract(f)
    gr, gp = _extract(g)
    hp = _compress(_partition(fr, True), _partition(gr, True), True)
    hm = _compress(_partition(fr, False), _partition(gr, False), False)
    # Source resets parity and locations even for passthrough/zero operands.
    if not hp.isempty():
        hp = _make(hp.cols, hp.rows, hp.pivots, len(hp.cols))
    if not hm.isempty():
        hm = _make(hm.cols, hm.rows, hm.pivots, 0)
    if not fp.isempty() or not gp.isempty():
        tol = (
            10
            * jnp.finfo(jnp.float64).eps
            * jnp.max(jnp.stack([_scale(x) for x in (fr, fp, gr, gp)]))
        )
        pole = _add_poles(fp, gp, tol)
        if _iszero(pole) and not hp.isempty():
            pole = _class().empty()
        pieces = [x for x in (pole, hp) if not x.isempty()]
        if pieces:
            cols = [t for x in pieces for t in x.cols]
            rows = [t for x in pieces for t in x.rows]
            hp = _make(
                cols,
                rows,
                jnp.concatenate([x.pivots for x in pieces]),
                len(cols),
                False if pole.isempty() else pole.nonzero_poles,
            )
    parts = [x for x in (hp, hm) if not x.isempty()]
    if not parts:
        return _class().empty()
    if len(parts) == 1:
        result = parts[0]
    else:
        result = _make(
            hp.cols + hm.cols,
            hp.rows + hm.rows,
            jnp.concatenate((hp.pivots, hm.pivots)),
            len(hp.cols),
            hp.nonzero_poles or hm.nonzero_poles,
        )
    return result.projectOntoBMCI()
