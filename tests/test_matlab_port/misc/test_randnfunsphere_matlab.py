"""Port of MATLAB Chebfun tests/misc/test_randnfunsphere.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_randnfunsphere.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.randfuns import randnfunsphere

jax.config.update("jax_enable_x64", True)


class TestMiscRandnfunsphere:
    def test_all_matlab_assertions(self):
        np.random.seed(0)
        f = randnfunsphere(.2)
        assert abs(float((f ** 2).mean2()) - 1) < .1                # pass(1)
        assert abs(float(f.mean2())) < .1                           # pass(2)
        f = randnfunsphere(1e6)
        assert float(f.diff().norm("fro")) < 1e-4                   # pass(3)
        f = randnfunsphere(3.1)
        assert f.rank == 4                                        # pass(4)
        f = randnfunsphere(3.1, "mono")
        assert f.rank == 3                                        # pass(5)


@pytest.mark.parametrize("args", [("white",), (1, "white"), (1, 5), ("mono", "mono")])
def test_native_argument_errors(args):
    with pytest.raises(ValueError):
        randnfunsphere(*args)


@pytest.mark.parametrize("mono", [False, True])
def test_source_fixed_grid(monkeypatch, mono):
    from chebfunjax.spherefun.spherefun import Spherefun
    from chebfunjax.utils.quadrature import trigpts
    from chebfunjax.utils.random import _sph_harm_sum, _sph_harm_sum_fixed_deg

    draws = np.arange(1., 6. if mono else 10.)
    monkeypatch.setattr(np.random, "randn", lambda n: draws.copy())
    captured = []
    original = Spherefun.from_values.__func__

    def capture(cls, values, *args, **kwargs):
        captured.append(np.asarray(values))
        return original(cls, values, *args, **kwargs)

    monkeypatch.setattr(Spherefun, "from_values", classmethod(capture))
    f = randnfunsphere(3.1, "mono") if mono else randnfunsphere(3.1)
    # Use the specified source grid API, rather than a host-rounded replacement.
    ll = trigpts(6, interval=(-jnp.pi, jnp.pi))[0]
    tt = jnp.linspace(0, jnp.pi, 6)
    c = np.sqrt(4*np.pi/np.count_nonzero(draws))*draws
    harmonic_sum = _sph_harm_sum_fixed_deg if mono else _sph_harm_sum
    expected = harmonic_sum(ll, tt, 2, c)
    assert captured[0].shape == (6, 6)
    np.testing.assert_array_equal(captured[0], expected)
    assert f.rank == (3 if mono else 4)
