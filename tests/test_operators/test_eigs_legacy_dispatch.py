"""C1/C2 source policy dispatch; retain the other legacy backend adapter.

Source: @linop/eigs.m, Chebfun7574c77. C1 now shares native adaptivity.
"""
import jax.numpy as jnp

from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators import _eigs_source
from chebfunjax.operators.blocklinop import BlockLinop
from chebfunjax.operators.blocks import D


def test_legacy_alternative_default_and_c1_c2_adaptive_dispatch(monkeypatch):
    calls = []

    def alternative(self, k, sigma, n, backend, B=None):
        calls.append((backend, n))
        return jnp.asarray([1.]), []

    def collocation(self, **kwargs):
        calls.append((kwargs['backend'], kwargs['n']))
        return jnp.asarray([1.]), []

    monkeypatch.setattr(BlockLinop, '_eigs_altdisc', alternative)
    monkeypatch.setattr(_eigs_source, 'solve', collocation)
    operator = BlockLinop(D((-1., 1.), 2))
    for backend in ('chebcolloc1', 'ultraS'):
        operator.eigs(k=1, discretization=backend)
        operator.eigs(k=1, n=29, discretization=backend)
        operator.eigs(k=1, pref=ChebopPref(discretization=backend))
    operator.eigs(k=1)
    assert calls == [('chebcolloc1', None), ('chebcolloc1', 29),
                     ('chebcolloc1', None), ('ultraS', 65), ('ultraS', 29),
                     ('ultraS', 65), ('chebcolloc2', None)]
