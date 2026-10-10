"""Native disk weighted QR and singular functions with JAX arithmetic.

Provenance
----------
MATLAB source : @diskfun/svd.m, @chebtech/restrict.m, @chebfun/qr.m,
    @trigtech/qr.m, @bndfun/qr.m, @separableApprox/cdr.m
Chebfun commit: 7574c77
"""

import equinox as eqx
import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun2d._svd import (
    _action,
    _as_fun,
    _axis_panel,
    _core_svd,
    _require_finite,
)
from chebfunjax.domain import _linear_inverse_map
from chebfunjax.spherefun._svd import _row_r
from chebfunjax.spherefun._svd_uv import _signed_qr
from chebfunjax.tech._trig_constructor import _values
from chebfunjax.tech.chebtech import Chebtech2, _clenshaw
from chebfunjax.tech.trigtech import (
    Trigtech,
    _trig_coeffs2vals_impl,
    _trig_project_values,
    _trig_prolong_coeffs,
    _trig_vals2coeffs_impl,
)
from chebfunjax.utils.quadrature import chebpts, legpts
from chebfunjax.utils.transforms import legvals2chebvals, vals2coeffs


@eqx.filter_jit
def _restricted_radial_panel(cols):
    """Collate first, then restrict to [0,1], without host Clenshaw dispatch."""
    panel = _axis_panel(cols)
    x = chebpts(panel.n, kind=2)
    # @chebtech/restrict: .5*[1-x,1+x]*[0;1]. No simplification.
    values = _clenshaw(panel.coeffs, 0.5 * (1 + x))
    return Chebtech2(coeffs=vals2coeffs(values), ishappy=panel.ishappy)


def _disk_qr(cols):
    """Literal diskQR, including unweight-before-sign and values construction."""
    panel = _restricted_radial_panel(tuple(cols))
    r, w = legpts(panel.n + 1, interval=(0.0, 1.0))
    values = _clenshaw(panel.coeffs, _linear_inverse_map(r, 0.0, 1.0))
    wr = jnp.sqrt(w * r)
    raw_q, raw_r = jnp.linalg.qr(wr[:, None] * values, mode="reduced")
    signs = jnp.sign(jnp.diag(raw_r))
    signs = jnp.where(signs == 0, 1, signs)
    # Native invWR*discreteQ*S: retain the reciprocal then multiply order.
    discrete_q = ((1.0 / wr)[:, None] * raw_q) * signs[None, :]
    rc = signs[:, None] * raw_r
    _require_finite(discrete_q, 'disk weighted QR output')
    qvalues = legvals2chebvals(discrete_q)
    # Numeric finite-data populate applies vals2coeffs, no simplify/chop.
    return Chebfun.from_values(qvalues, domain=(0.0, 1.0)), rc


@eqx.filter_jit
def _angular_panel(rows):
    """Native horzcat preserving supplied caches; transforms stay JAX-only."""
    n = max(row.n for row in rows)
    coeffs = [_trig_prolong_coeffs(row.coeffs, n) for row in rows]
    values = [_values(row) if row.n == n else
              _trig_project_values(_trig_coeffs2vals_impl(c), row.real_columns)
              for row, c in zip(rows, coeffs)]
    return Trigtech(coeffs=jnp.column_stack(coeffs),
                   real_columns=tuple(row.is_real for row in rows),
                   ishappy=rows[0].ishappy, _values=jnp.column_stack(values))


def _angular_qr(rows):
    """Continuous angular QR, not coefficient QR; preserve native cache order.

    The existing sphere scalar normalization and signed dense QR are reused.
    Local array handling avoids inherited concrete NumPy FFT dispatch even
    with JIT disabled. The physical domain is [-pi,pi].
    """
    panel = _angular_panel(tuple(rows))
    nf, count = panel.coeffs.shape
    if count == 1:
        rr = _row_r(rows)
        norm = rr[0, 0]
        _require_finite(norm, 'disk angular scalar QR normalization')
        if bool(norm != 0):
            q = Trigtech(coeffs=panel.coeffs / norm,
                         real_columns=panel.real_columns, ishappy=panel.ishappy,
                         _values=_values(panel) / norm)
        else:
            values = jnp.asarray([[1 / jnp.sqrt(2 * jnp.pi)]])
            q = Trigtech(coeffs=values.astype(jnp.complex128),
                         real_columns=(True,), ishappy=True, _values=values)
        return _as_fun(q, (-jnp.pi, jnp.pi)), rr
    n = max(nf, count)
    if n == nf:
        values = _values(panel)
    else:
        c = _trig_prolong_coeffs(panel.coeffs, n)
        values = _trig_project_values(_trig_coeffs2vals_impl(c), panel.real_columns)
    if panel.is_real:
        values = jnp.real(values)
    q, rr = _signed_qr(values)
    weight = jnp.sqrt(jnp.asarray(2.0 / n))
    q, rr = q / weight, weight * rr
    c = _trig_vals2coeffs_impl(q)
    real_columns = (panel.is_real,) * count
    if nf != n:
        c = _trig_prolong_coeffs(c, nf)
        q = _trig_project_values(_trig_coeffs2vals_impl(c), real_columns)
    scale = jnp.sqrt(jnp.pi)
    qtech = Trigtech(coeffs=c / scale, real_columns=real_columns,
                    ishappy=panel.ishappy, _values=q / scale)
    return _as_fun(qtech, (-jnp.pi, jnp.pi)), rr * scale


def source_svd(f, *, return_uv=False):
    """Native disk SVD; optional U, diagonal S, V on source physical domains.

    Provenance
    ----------
    MATLAB source : @diskfun/svd.m, @separableApprox/cdr.m
    Chebfun commit: 7574c77

    Empty native varargout contains one empty output before nargout dispatch;
    both Python modes therefore return one empty array. Nonfinite providers
    retain the shared explicit unqualified capability boundary. Adaptive
    construction and output selection remain eager; no global AD guarantee.
    """
    if f.isempty() or not f.cols:
        return jnp.empty((0,), dtype=jnp.float64)
    qc, rc = _disk_qr(f.cols)
    qr, rr = _angular_qr(f.rows)
    reciprocal = 1.0 / jnp.asarray(f.pivots)
    reciprocal = jnp.where(jnp.isinf(jnp.abs(reciprocal)), 0, reciprocal)
    u, s, v = _core_svd(rc, reciprocal, rr)
    if not return_uv:
        return s
    left = _action(qc.funs[0].tech, u)
    right = _action(qr.funs[0].tech, v)
    return (_as_fun(left, (0.0, 1.0)), jnp.diag(s),
            _as_fun(right, (-jnp.pi, jnp.pi)))
