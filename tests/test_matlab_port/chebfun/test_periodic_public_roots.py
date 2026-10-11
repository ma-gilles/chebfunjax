import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj


@pytest.mark.parametrize("flag", ["complex_roots", "all_roots"])
@pytest.mark.parametrize("domain", [(-1.0, 1.0), (0.0, 2.0), (2.0, 6.0)])
def test_periodic_public_complex_roots(flag, domain):
    a, b = domain
    f = cj.chebfun(
        lambda x: 2 + jnp.cos(jnp.pi * (2 * x - a - b) / (b - a)), domain=domain, trig=True
    )
    roots = f.roots(**{flag: True})
    assert roots.shape == (2,)
    z = (2 * roots - a - b) / (b - a)
    expected = np.log(2 + np.sqrt(3)) / np.pi
    np.testing.assert_allclose(
        np.sort(np.imag(z)), [-expected, expected], rtol=0, atol=4 * np.finfo(float).eps
    )
    np.testing.assert_allclose(np.real(z), [1.0, 1.0], rtol=0, atol=4 * np.finfo(float).eps)
    assert float(jnp.max(jnp.abs(f(roots)))) < 10 * np.finfo(float).eps * float(f.vscale)
