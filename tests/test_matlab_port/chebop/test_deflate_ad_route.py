"""Public Chebop deflation retains the source nonlocal AD derivative.

Provenance
----------
MATLAB source: @chebop/deflate.m, @adchebfun/adchebfun.m (deflationFun).
Chebfun commit: 7574c77.
Supplemental independent product-rule controls; tolerances follow the existing
AD deflation controls, not the literal Painleve residual bound.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.autodiff.adchebfun import ADChebfun
from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.blocks import ChebColloc2Disc
from chebfunjax.operators.chebop import Chebop, deflate


@pytest.mark.parametrize("kind", ["L2", "H1"])
def test_public_deflate_nonlocal_jacobian(kind):
    domain = (2.0, 5.0)
    x = chebfun(lambda t: t, domain=domain)
    u = chebfun(lambda t: 0.5 + 0.02*t, domain=domain)
    h = chebfun(lambda t: 0.03 + 0.001*t**2, domain=domain)
    root = chebfun(0.1, domain=domain)
    operator = Chebop(lambda x, v: v.diff(2) - v*v + x, domain)
    modified = deflate(operator, root, 3, 0.1, kind)
    actual = modified.op(x, ADChebfun(u))
    delta = u - root
    square = (delta*delta).sum()
    inner = (delta*h).sum()
    if kind == "H1":
        square += (delta.diff()**2).sum()
        inner += (delta.diff()*h.diff()).sum()
    residual = u.diff(2) - u*u + x
    factor = square**(-1.5)
    expected = (factor + 0.1)*(h.diff(2) - 2*u*h)
    expected -= residual*(3*factor*inner/square)
    scale = max(1.0, float(expected.norm(jnp.inf)))
    assert float((actual.jacobian.apply(h) - expected).norm(jnp.inf)) < 5e-12*scale
    disc = ChebColloc2Disc(24, domain)
    nodes = disc.points()
    matrix_action = actual.jacobian.matrix(disc) @ h(nodes)
    assert float(jnp.max(jnp.abs(matrix_action - expected(nodes)))) < 5e-10*scale
