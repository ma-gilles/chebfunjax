"""chebfunjax.autodiff — Fréchet AD and JAX AD compatibility.

Two complementary differentiation mechanisms are provided:

**JAX AD** (``jax.grad``, ``jax.jit``, ``jax.vmap``)
    Works through all Chebfun evaluation methods (JIT-safe).  See
    ``tests/test_autodiff.py`` for examples.

**Fréchet / operator AD** (:mod:`.adchebfun` + :mod:`.treevar`)
    Symbolic linearization of nonlinear differential operators.  Use
    :func:`~chebfunjax.autodiff.adchebfun.linearize_op` to obtain a
    :class:`~chebfunjax.operators.blocks.OperatorBlock` representing
    ``dN[u0]`` for an operator ``N`` and linearization point ``u0``.
    This is used internally by :class:`~chebfunjax.operators.chebop.Chebop`
    for Newton iteration.

Key AD-compatible entry points:
  - :func:`~chebfunjax.chebfun1d.chebfun.Chebfun.__call__`
  - :func:`~chebfunjax.chebfun2d.chebfun2.Chebfun2.__call__`
  - :func:`~chebfunjax.chebfun2d.separable_approx.SeparableApprox.__call__`
  - :func:`~chebfunjax.chebfun3d.chebfun3.Chebfun3.__call__`

Key operator-AD entry points:
  - :func:`~chebfunjax.autodiff.adchebfun.linearize_op`
  - :func:`~chebfunjax.autodiff.adchebfun.detect_linearity`
  - :class:`~chebfunjax.autodiff.adchebfun.ADChebfun`
  - :class:`~chebfunjax.autodiff.treevar.TreeVar`
  - :func:`~chebfunjax.autodiff.treevar.linearize_tree`
"""

from chebfunjax.autodiff.adchebfun import ADChebfun, linearize_op, detect_linearity
from chebfunjax.autodiff.treevar import TreeVar, linearize_tree

__all__ = [
    "ADChebfun",
    "linearize_op",
    "detect_linearity",
    "TreeVar",
    "linearize_tree",
]
