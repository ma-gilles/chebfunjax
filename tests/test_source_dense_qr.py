"""Native CPU QR arithmetic and full-rank differential regression tests.

Provenance: MATLAB R2025b qr(A,0), used by Chebfun 7574c77 constructors.
The original capture scripts and metadata accompany the immutable fixtures.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils._source_qr import qr


@pytest.mark.parametrize('index', range(79))
def test_native_real_qr(index):
    with np.load(Path(__file__).parent / 'data/source_dense_qr/native_controls.npz') as data:
        a, expected_q, expected_r = (data[f'case{index}_{key}'] for key in ('A', 'Q', 'R'))
    q, r = qr(jnp.asarray(a))
    for actual, expected in ((q, expected_q), (r, expected_r)):
        # Signed zero and subnormal results matter at subsequent source decisions.
        np.testing.assert_array_equal(
            np.ascontiguousarray(actual).view(np.uint64),
            np.ascontiguousarray(expected).view(np.uint64),
        )


@pytest.mark.parametrize('shape', [(17, 1), (17, 2), (9, 5), (5, 5), (33, 7), (129, 9)])
def test_full_rank_thin_qr_differential(shape):
    m, n = shape
    rng = np.random.default_rng(9817 + m * 97 + n)
    a = jnp.asarray(rng.normal(size=shape) + 3 * np.eye(m, n))
    da = jnp.asarray(rng.normal(size=shape))
    (q, r), (dq, dr) = jax.jvp(qr, (a,), (da,))
    (q0, _), (dq0, dr0) = jax.jvp(lambda x: jnp.linalg.qr(x, mode='reduced'), (a,), (da,))
    signs = jnp.sign(jnp.sum(q * q0, axis=0))
    dq0, dr0 = dq0 * signs, dr0 * signs[:, None]
    assert float(jnp.max(jnp.abs(da - dq @ r - q @ dr))) < 1e-11
    assert float(jnp.max(jnp.abs(q.T @ dq + dq.T @ q))) < 1e-11
    assert max(float(jnp.max(jnp.abs(dq - dq0))), float(jnp.max(jnp.abs(dr - dr0)))) < 1e-11
    wq, wr = (jnp.asarray(rng.normal(size=x.shape)) for x in (q, r))

    def loss(x):
        qx, rx = qr(x)
        return jnp.sum(qx * wq) + jnp.sum(rx * wr)

    gradient = jax.grad(loss)(a)
    directional = jnp.sum(gradient * da)
    assert float(jnp.abs(directional - jnp.sum(wq * dq) - jnp.sum(wr * dr))) < 1e-10
    h = 1e-5
    assert float(jnp.abs((loss(a + h * da) - loss(a - h * da)) / (2 * h) - directional)) < 1e-6
    jax.clear_caches()
