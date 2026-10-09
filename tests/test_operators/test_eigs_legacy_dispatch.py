"""Preserve existing fixed alternative-backend API; no adaptive parity claim."""
import jax.numpy as jnp

from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators import _eigs_source
from chebfunjax.operators.blocklinop import BlockLinop
from chebfunjax.operators.blocks import D


def test_legacy_alternative_default_and_c2_adaptive_dispatch(monkeypatch):
    calls = []

    def alternative(self, k, sigma, n, backend, B=None):
        calls.append((backend, n))
        return jnp.asarray([1.]), []

    def c2(self, **kwargs):
        calls.append(('chebcolloc2', kwargs['n']))
        return jnp.asarray([1.]), []

    monkeypatch.setattr(BlockLinop, '_eigs_altdisc', alternative)
    monkeypatch.setattr(_eigs_source, 'solve', c2)
    operator = BlockLinop(D((-1., 1.), 2))
    for backend in ('chebcolloc1', 'ultraS'):
        operator.eigs(k=1, discretization=backend)
        operator.eigs(k=1, n=29, discretization=backend)
        operator.eigs(k=1, pref=ChebopPref(discretization=backend))
    operator.eigs(k=1)
    assert calls == [('chebcolloc1', 65), ('chebcolloc1', 29),
                     ('chebcolloc1', 65), ('ultraS', 65), ('ultraS', 29),
                     ('ultraS', 65), ('chebcolloc2', None)]
