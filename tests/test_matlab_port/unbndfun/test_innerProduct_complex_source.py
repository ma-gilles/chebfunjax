"""Complex first-operand conjugation in native Unbndfun innerProduct.

Authored controls for @unbndfun/innerProduct.m and @classicfun/conj.m,
Chebfun7574c77. The existing product/sum implementation must be retained.
"""
import jax.numpy as jnp
import numpy as np

from chebfunjax.domain import Domain
from chebfunjax.fun.unbndfun import Unbndfun


def test_complex_smooth_first_operand_conjugated():
    domain = Domain((1., np.inf))
    f = Unbndfun.from_function(lambda x: (1+1j)*jnp.exp(-x), domain)
    g = Unbndfun.from_function(lambda x: (2-1j)*jnp.exp(-x), domain)
    tol = 2e8 * np.finfo(float).eps * max(f.vscale, g.vscale)
    ff, fg, gf = f.inner(f), f.inner(g), g.inner(f)
    assert np.isfinite(tol)
    assert abs(complex(ff)-np.exp(-2)) < tol
    assert abs(complex(f.norm())-np.exp(-1)) < tol
    assert abs(complex(fg)-(1-3j)*np.exp(-2)/2) < tol
    assert abs(complex(gf)-complex(fg).conjugate()) < tol


def test_complex_explicit_singular_first_operand_conjugated():
    domain = Domain((1., np.inf))
    f = Unbndfun.from_function(lambda x: (1+1j)*x, domain, exps=(0, 1))
    g = Unbndfun.from_function(lambda x: (2-1j)*jnp.exp(-x), domain)
    tol = 2e8 * np.finfo(float).eps * max(f.vscale, g.vscale)
    assert np.isfinite(tol)
    assert abs(complex(f.inner(g))-(1-3j)*2*np.exp(-1)) < tol
