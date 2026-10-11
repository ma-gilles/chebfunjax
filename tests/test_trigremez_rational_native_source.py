"""Independent rational controls, native Chebfun7574c77680d7e82b79626300bf255498271a72df.

R2025b records preserve source trial matrices/reference and public outputs.
All comparison bounds are the original test_trigremez.m tolerance1e-12.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.utils._trigremez_rational_source import rational_trial

FIXTURES = Path(__file__).parent / "fixtures/trigremez_rational_source_2025b"


@pytest.mark.parametrize("case,m,n", [(1, 2, 2), (2, 6, 6), (3, 1, 1), (4, 0, 1)])
def test_native_finite_pencil_trial(case, m, n):
    with np.load(FIXTURES / f"case{case}.npz") as native:
        x = jnp.asarray(native["trial_reference"])
        values = jnp.asarray(native["trial_values"])
        p, q, r, h = rational_trial(values, x, m, n)
        assert q.roots().size == 0
        assert len(p) == 2 * m + 1 and len(q) == 2 * n + 1
        sigma = jnp.where(jnp.arange(x.size) % 2, -1.0, 1.0)
        np.testing.assert_allclose(r(x), values - h * sigma, rtol=0, atol=1e-12)
        # A generalized eigenvector's common p/q sign/scale is arbitrary.
        # Compare its represented quotient, not a fitted output normalization.
        ntp = cj.chebfun(
            jnp.asarray(native["trial_p_coeffs"]), domain=(-jnp.pi, jnp.pi), coeffs=True, trig=True
        )
        ntq = cj.chebfun(
            jnp.asarray(native["trial_q_coeffs"]), domain=(-jnp.pi, jnp.pi), coeffs=True, trig=True
        )
        query = jnp.asarray(native["trial_query"])
        np.testing.assert_allclose(r(query), ntp(query) / ntq(query), rtol=0, atol=1e-12)
        np.testing.assert_allclose(h, native["trial_level"], rtol=0, atol=1e-12)


@pytest.mark.parametrize("case,m,n", [(3, 1, 1), (4, 0, 1)])
def test_exact_rational_public_source_controls(case, m, n):
    if case == 3:
        f = cj.chebfun(
            lambda x: (
                (1 + 0.2 * jnp.cos(2 * jnp.pi * (x - 2) / 4))
                / (1 + 0.3 * jnp.sin(2 * jnp.pi * (x - 2) / 4))
            ),
            domain=(2, 6),
            trig=True,
        )
    else:
        f = cj.chebfun(lambda x: 1 / (2 + jnp.cos(jnp.pi * x)), trig=True)
    p, q, r, error, status = cj.trigremez(f, m, n)
    assert status["iter"] > 0
    assert set(status) == {"iter", "delta", "diffx", "xk"}
    assert q.roots().size == 0
    assert len(p) == 2 * m + 1 and len(q) == 2 * n + 1
    with np.load(FIXTURES / f"case{case}.npz") as native:
        x = jnp.asarray(native["sample_x"])
        np.testing.assert_allclose(r(x), native["r_values"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(r(x), f(x), rtol=0, atol=1e-12)
    assert float(error) < 1e-12
