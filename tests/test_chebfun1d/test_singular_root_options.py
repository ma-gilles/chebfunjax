"""Native Singfun root option forwarding and endpoint insertion.

Provenance
----------
MATLAB @singfun/roots.m, @classicfun/roots.m, @chebfun/roots.m.
Chebfun commit: 7574c77.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import _Piece
from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech2


@pytest.mark.parametrize("options", [dict(all_roots=True, recurse=False),
                                     dict(complex_roots=True, recurse=False),
                                     dict(all_roots=True, prune=True, recurse=False)])
@pytest.mark.parametrize("exponents", [(-.5, -.25), (.5, .25)])
def test_complex_roots_and_endpoint_insertion(options, exponents):
    # T0*1.5 + T2*.5 = 1+x**2, with roots +/-i.
    f = Singfun(Chebtech2(coeffs=jnp.asarray([1.5, 0., .5])), exponents)
    got = np.asarray(f.roots(**options))
    exact = np.array([-1j, 1j] if exponents[0] < 0 else [-1., -1j, 1j, 1.])
    np.testing.assert_allclose(np.sort_complex(got), np.sort_complex(exact),
                               atol=20*np.finfo(float).eps, rtol=0)
    assert np.iscomplexobj(got)


@pytest.mark.parametrize("norecursion", [False, True])
def test_bounded_singular_piece_recursion_option(norecursion, monkeypatch):
    f = Singfun(Chebtech2(coeffs=jnp.asarray([-.25, 1.])), (-.5, -.5))
    piece = _Piece(tech=f, interval=(-2., 6.))
    original = Chebtech2.roots
    calls = []

    def observe(self, *args, **kwargs):
        calls.append(kwargs.get("recurse", True))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Chebtech2, "roots", observe)
    np.testing.assert_allclose(piece.roots(norecursion=norecursion), [3.], atol=1e-14, rtol=0)
    assert calls == [not norecursion]


@pytest.mark.parametrize("option", ["all_roots", "complex_roots"])
def test_public_singular_chebfun_complex_roots(option):
    f = cj.chebfun(lambda x: (1+x)**(-.5)*(1+x*x), exps=(-.5, 0.))
    got = np.asarray(f.roots(**{option: True}, norecursion=True))
    np.testing.assert_allclose(np.sort_complex(got), [-1j, 1j], atol=2e-14, rtol=0)
