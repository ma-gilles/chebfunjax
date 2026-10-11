"""Unbounded roots preserve the piece-to-onefun recursion option.

Provenance
----------
MATLAB @unbndfun/roots.m, @classicfun/roots.m, @singfun/roots.m.
Chebfun commit: 7574c77.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.singfun import Singfun
from chebfunjax.fun.unbndfun import Unbndfun
from chebfunjax.tech.chebtech import Chebtech2


@pytest.mark.parametrize("singular", [False, True])
@pytest.mark.parametrize("norecursion", [False, True])
def test_mapped_real_root_and_recursion_option(singular, norecursion, monkeypatch):
    # Linear reference polynomial has root s=1/4, mapped to finite x=4/3.
    smooth = Chebtech2(coeffs=jnp.asarray([-.25, 1.]))
    onefun = Singfun(smooth, (-.5, -.5)) if singular else smooth
    piece = Unbndfun(onefun=onefun, domain=Domain((-jnp.inf, jnp.inf)),
                     mapping_type="both_inf")
    original = Chebtech2.roots
    calls = []

    def observe(self, *args, **kwargs):
        calls.append(kwargs.get("recurse", True))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Chebtech2, "roots", observe)
    got = piece.roots(norecursion=norecursion)
    np.testing.assert_allclose(got, [4/3], atol=1e-14, rtol=0)
    assert calls == [not norecursion]
