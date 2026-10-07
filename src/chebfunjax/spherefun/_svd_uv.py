"""Source sphere SVD singular functions; additive to the values-only draft.

Provenance
----------
MATLAB source : @spherefun/svd.m, @chebfun/qr.m, @trigtech/qr.m,
               @bndfun/qr.m, legvals2chebvals.m, legcoeffs2chebvals.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.domain import Domain
from chebfunjax.spherefun._svd import _row_r, _values_at_length
from chebfunjax.tech.trigtech import (
    Trigtech,
    _trig_eval,
    _trig_prolong_coeffs,
    _trig_vals2coeffs_impl,
)
from chebfunjax.utils.quadrature import legpts
from chebfunjax.utils.transforms import (
    _coeffs2vals_jax,
    _legendre_idlt,
    _vals2coeffs_jax,
    leg2cheb,
)


def _signed_qr(a):
    q, r = jnp.linalg.qr(a, mode="reduced")
    diagonal = jnp.diag(r)
    phase = jnp.where(diagonal == 0, jnp.ones_like(diagonal), jnp.sign(diagonal))
    return q * phase[None, :], phase[:, None] * r


def _row_qr(rows):
    nf = max(row.coeffs.shape[0] for row in rows)
    if len(rows) == 1:
        r = _row_r(rows)
        norm = r[0, 0]
        c = _trig_prolong_coeffs(rows[0].coeffs, nf)
        safe = jnp.where(norm == 0, 1, norm)
        constant = jnp.zeros_like(c).at[nf // 2].set(1 / jnp.sqrt(2 * jnp.pi))
        return jnp.where(norm == 0, constant, c / safe)[:, None], r
    n = max(nf, len(rows))
    values = jnp.column_stack([_values_at_length(row, n) for row in rows])
    if all(row.is_real for row in rows):
        values = jnp.real(values)
    q, r = _signed_qr(values)
    weight = jnp.sqrt(jnp.asarray(2.0 / n))
    q = q / weight
    r = weight * r
    coeffs = _trig_vals2coeffs_impl(q) if n > 1 else q.astype(jnp.complex128)
    coeffs = _trig_prolong_coeffs(coeffs, nf)
    return coeffs / jnp.sqrt(jnp.pi), r * jnp.sqrt(jnp.pi)


def decomposition_coefficients(f):
    """Return U Chebyshev coefficients, s, V Fourier coefficients.

    Provenance
    ----------
    MATLAB source : @spherefun/svd.m, legvals2chebvals.m,
                   @chebtech/populate.m, @chebtech/mtimes.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df

    Domains are U:[0,pi], V:[-pi,pi]. All transforms are unconditional JAX.
    This retains the literal source complex phase/output convention rather
    than promising complex U*S*V' reconstruction beyond the source contract.
    """
    n = max(col.coeffs.shape[0] for col in f.cols) + 9
    x, w = legpts(n, (0.0, jnp.pi))
    values = jnp.column_stack([
        _trig_eval(col.coeffs, x / jnp.pi, is_real=col.is_real)
        for col in f.cols
    ])
    wr = jnp.sqrt(w * jnp.sin(x))
    q, rc = _signed_qr(wr[:, None] * values)
    q = q / wr[:, None]
    # Literal legvals2chebvals = IDLT -> leg2cheb -> coeffs2vals.
    # The subsequent numeric Chebfun construction applies vals2coeffs.
    # Preserve that FFT round trip; do not replace it by identity.
    q_cheb_values = _coeffs2vals_jax(leg2cheb(_legendre_idlt(q)))
    qc = _vals2coeffs_jax(q_cheb_values)
    qr, rr = _row_qr(f.rows)
    reciprocal = 1.0 / jnp.asarray(f.pivots)
    reciprocal = jnp.where(jnp.isinf(jnp.abs(reciprocal)), 0, reciprocal)
    u, s, vh = jnp.linalg.svd(rc @ jnp.diag(reciprocal) @ rr.T,
                            full_matrices=False)
    return qc @ u, s, qr @ jnp.conj(vh).T


def singular_functions(f):
    """Construct source-domain quasimatrices, preserving Python vector s API.

    Provenance
    ----------
    MATLAB source : @spherefun/svd.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df

    Quasimatrix construction remains a host operation; the coefficient
    kernel is separately traceable. Source multioutput empty returns only
    one empty output; preserve the existing Python empty-array convention.
    """
    if f.isempty() or not f.cols:
        return jnp.empty((0,), dtype=jnp.float64)
    uc, s, vc = decomposition_coefficients(f)
    udomain = Domain((0.0, jnp.pi))
    vdomain = Domain((-jnp.pi, jnp.pi))
    us = [Chebfun.from_coeffs(uc[:, j], domain=udomain)
          for j in range(uc.shape[1])]
    # Avoid the public trig constructor's concrete NumPy transform path.
    real_output = (all(t.is_real for t in f.cols + f.rows)
                   and not jnp.iscomplexobj(f.pivots))
    vs = [Chebfun(funs=[_Piece(tech=Trigtech.from_coeffs(vc[:, j], is_real=real_output),
                              interval=(-jnp.pi, jnp.pi))], domain=vdomain)
          for j in range(vc.shape[1])]
    return Quasimatrix(us, udomain), s, Quasimatrix(vs, vdomain)
