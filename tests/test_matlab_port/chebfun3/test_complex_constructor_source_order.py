"""Complex-valued Chebfun3 source-order construction controls."""

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import Chebfun3


def test_real_source_order_constructor_control():
    f = Chebfun3.from_function(lambda x, y, z: 1.0 + x * y + z)
    x, y, z = jnp.meshgrid(
        jnp.linspace(-0.8, 0.7, 4),
        jnp.linspace(-0.6, 0.9, 5),
        jnp.linspace(-0.5, 0.75, 3),
        indexing="ij",
    )
    expected = 1.0 + x * y + z
    assert not jnp.iscomplexobj(f.core)
    assert jnp.allclose(f(x, y, z), expected, rtol=0.0, atol=2e-12)


def test_complex_nonseparable_constructor_conj_and_product():
    def polynomial(x, y, z):
        return ((1.25 - 0.5j) + x + (2j) * y + (1.0 - 1j) * z
                + (0.5 + 0.25j) * x * y - 0.75j * y * z
                + (0.25 + 0.5j) * x * z)

    f = Chebfun3.from_function(polynomial)
    x, y, z = jnp.meshgrid(
        jnp.linspace(-0.8, 0.7, 4),
        jnp.linspace(-0.6, 0.9, 5),
        jnp.linspace(-0.5, 0.75, 3),
        indexing="ij",
    )
    expected = polynomial(x, y, z)
    assert jnp.iscomplexobj(f.core)
    assert jnp.allclose(f(x, y, z), expected, rtol=0.0, atol=2e-11)

    conjugate = f.conj()
    assert jnp.allclose(conjugate(x, y, z), jnp.conj(expected),
                        rtol=0.0, atol=5e-10)

    magnitude_squared = f * conjugate
    assert jnp.allclose(magnitude_squared(x, y, z), jnp.abs(expected) ** 2,
                        rtol=0.0, atol=2e-9)
