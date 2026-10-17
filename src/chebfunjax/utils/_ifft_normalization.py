"""CPU binary64 inverse FFT normalization for source value transforms.

Native @chebtech2/vals2coeffs.m (pin7574c77) uses IFFT, whose
normalization divides each unnormalized component by the transform length.
Keep the backend FFT algorithm; prevent replacing that division with a
rounded reciprocal multiplication. This does not reproduce native FFT plans.
"""

import jax
import jax.numpy as jnp

from chebfunjax.utils._ordered_binary import binary


def source_ifft(values):
    """IFFT on axis zero, preserving the CPU binary64 division operation."""
    def cpu(a):
        # JAX IFFT is already normalized inside its primitive; norm="forward"
        # multiplies that rounded result back by N. Use the Fourier conjugacy
        # identity to obtain an unnormalized inverse before dividing once.
        raw = jnp.conj(jnp.fft.fft(jnp.conj(a), axis=0))
        length = jnp.asarray(a.shape[0], dtype=jnp.float64)
        real = binary(jnp.real(raw), length, 3)
        imag = binary(jnp.imag(raw), length, 3)
        return jax.lax.complex(real, imag)

    def generic(a):
        return jnp.fft.ifft(a, axis=0)

    if values.dtype not in (jnp.float64, jnp.complex128):
        return generic(values)
    return jax.lax.platform_dependent(values, cpu=cpu, default=generic)
