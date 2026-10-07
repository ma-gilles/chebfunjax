"""Source-domain reconstruction controls, invariant to singular-vector signs.

Provenance
----------
MATLAB source : @spherefun/svd.m and its sphereQR helper
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Analytic orthogonal factors supply independent expectations, not solver data.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.tech.chebtech import _clenshaw
from chebfunjax.tech.trigtech import Trigtech, _trig_eval, _trig_prolong_coeffs
from chebfunjax.utils.quadrature import legpts


def _field(complex_column=False, resolved=False):
    c1 = Trigtech.from_coeffs(jnp.array([1.0]), is_real=True)
    # sin(2*theta), zero at poles and orthogonal to 1 in the surface measure.
    c2 = Trigtech.from_coeffs(jnp.array([0.5j, 0, 0, 0, -0.5j]), is_real=True)
    if resolved:
        c2 = Trigtech.from_coeffs(_trig_prolong_coeffs(c2.coeffs, 33), is_real=True)
    if complex_column:
        c2 = Trigtech.from_coeffs(1j * c2.coeffs, is_real=False)
    r2 = Trigtech.from_coeffs(jnp.array([0.5, 0, 0.5]), is_real=True)
    return Spherefun(cols=[c1, c2], rows=[c1, r2], pivots=jnp.ones(2),
                     idx_plus=(0, 1), idx_minus=())


@pytest.mark.parametrize("disable", [False, True])
@pytest.mark.parametrize("complex_column", [False, True])
@pytest.mark.parametrize("resolved", [False, True])
def test_source_factor_reconstruction_and_weighted_orthogonality(
    disable, complex_column, resolved
):
    from chebfunjax.spherefun._svd_uv import decomposition_coefficients

    f = _field(complex_column, resolved)
    with jax.disable_jit(disable):
        uc, s, vc = decomposition_coefficients(f)
        # Short source polynomial output is checked at its original Gauss
        # nodes. The fixed33-coefficient resolved fixture additionally checks
        # independent80-node integration, without changing production grids.
        count = 80 if resolved else max(c.coeffs.shape[0] for c in f.cols) + 9
        th, w = legpts(count, (0.0, jnp.pi))
        lam = -jnp.pi + 2 * jnp.pi * jnp.arange(80) / 80
        u = _clenshaw(uc, 2 * th / jnp.pi - 1)
        v = _trig_eval(vc, lam / jnp.pi, is_real=False)
        actual = (u * s[None, :]) @ jnp.conj(v).T
        amplitude = 1j if complex_column else 1.0
        expected = 1 + amplitude * jnp.sin(2 * th[:, None]) * jnp.cos(lam[None, :])
        ug = jnp.conj(u).T @ ((w * jnp.sin(th))[:, None] * u)
        vg = jnp.conj(v).T @ v * (2 * jnp.pi / 80)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12)
    np.testing.assert_allclose(ug, np.eye(2), rtol=0, atol=2e-12)
    np.testing.assert_allclose(vg, np.eye(2), rtol=0, atol=2e-12)
    np.testing.assert_allclose(s, np.sqrt([4 * np.pi, 16 * np.pi / 15]),
                               rtol=0, atol=2e-12)


@pytest.mark.parametrize("disable", [False, True])
def test_public_output_domains_and_vector_s(disable):
    with jax.disable_jit(disable):
        u, s, v = _field().svd(return_uv=True)
    assert (u.domain.a, u.domain.b) == (0.0, jnp.pi)
    assert (v.domain.a, v.domain.b) == (-jnp.pi, jnp.pi)
    assert s.shape == (2,)
    assert len(u.cols) == len(v.cols) == 2


@pytest.mark.parametrize("disable", [False, True])
def test_complex_row_literal_source_output_convention(disable):
    # Rank-one source: R=(1+i), Qr=(1+i)/sqrt(4*pi), positive Rr.
    # Source V=Qr*Vsmall and nonconjugating reduced core imply U*s*V'
    # equals conj(R), not R. Record this source limitation explicitly;
    # do not impose an unsupported general complex reconstruction claim.
    one = Trigtech.from_coeffs(jnp.array([1.0]), is_real=True)
    row = Trigtech.from_coeffs(jnp.array([1 + 1j]), is_real=False)
    f = Spherefun(cols=[one], rows=[row], pivots=jnp.ones(1),
                  idx_plus=(0,), idx_minus=())
    from chebfunjax.spherefun._svd_uv import decomposition_coefficients

    with jax.disable_jit(disable):
        uc, s, vc = decomposition_coefficients(f)
        u = _clenshaw(uc, jnp.array([0.0]))
        v = _trig_eval(vc, jnp.array([0.0]), is_real=False)
        product = (u * s) @ jnp.conj(v).T
    np.testing.assert_allclose(product, [[1 - 1j]], rtol=0, atol=2e-12)
