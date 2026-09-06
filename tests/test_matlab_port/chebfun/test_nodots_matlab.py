"""Port of MATLAB Chebfun tests/chebfun/test_nodots.m (Fable 5).

MATLAB distinguishes ``x*x`` (matrix product) from ``x.*x``
(elementwise) and the string constructor vectorizes the former; Python
operators are elementwise already, so the two spellings coincide.  The
final MATLAB case (``'vectorcheck', 'off'`` forcing an inner-dimension
error) has no Python counterpart and is asserted as the elementwise
result instead.

Provenance
----------
MATLAB source : tests/chebfun/test_nodots.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import chebfun

jax.config.update("jax_enable_x64", True)

TOL = 1e-14


def _n(f):
    # MATLAB norm of an array-valued chebfun: Frobenius over the columns.
    return float(jnp.linalg.norm(jnp.atleast_1d(jnp.asarray(f.norm(2)))))


class TestChebfunNodots:
    def test_all_matlab_assertions(self):
        x = chebfun("x")
        a = x * x
        b = chebfun(lambda t: t * t)
        c = chebfun("x*x")
        assert float((a - b).norm(2)) + float((a - c).norm(2)) < TOL   # pass(1)

        a = x / (2 + x)
        b = chebfun(lambda t: t / (2 + t))
        c = chebfun("x/(2+x)")
        assert float((a - b).norm(2)) + float((a - c).norm(2)) < TOL   # pass(2)

        a = x ** 2
        b = chebfun(lambda t: t ** 2)
        c = chebfun("x^2")
        assert float((a - b).norm(2)) + float((a - c).norm(2)) < TOL   # pass(3)

        # Array-valued [x x].
        xx = chebfun(lambda t: jnp.stack([t, t], axis=-1))
        assert _n(xx * xx - xx * xx) < TOL                # pass(4)
        assert _n(xx / (2 + xx) - xx / (2 + xx)) < TOL    # pass(5)
        assert _n(xx ** 2 - xx ** 2) < TOL                # pass(6)

        # 'vectorcheck', 'off': no inner-dimension error in Python.
        assert float((chebfun("x*x") - x ** 2).norm(2)) < TOL          # pass(7)
