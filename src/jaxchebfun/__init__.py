"""jaxchebfun — Chebfun in Python, powered by JAX."""

__version__ = "0.1.0"

# Enable float64 by default — spectral methods need double precision
import jax
jax.config.update("jax_enable_x64", True)
