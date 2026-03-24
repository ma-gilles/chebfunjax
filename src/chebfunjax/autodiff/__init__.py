"""chebfunjax.autodiff — documentation for JAX AD compatibility.

JAX automatic differentiation works through all chebfunjax evaluation
methods.  See ``tests/test_autodiff.py`` for examples and tests.

Key AD-compatible entry points:
  - :func:`~chebfunjax.chebfun1d.chebfun.Chebfun.__call__`
  - :func:`~chebfunjax.chebfun2d.chebfun2.Chebfun2.__call__`
  - :func:`~chebfunjax.chebfun2d.separable_approx.SeparableApprox.__call__`
  - :func:`~chebfunjax.chebfun3d.chebfun3.Chebfun3.__call__`

All these are ``jit``-safe, ``grad``-safe, and ``vmap``-safe.
"""

__all__: list = []
