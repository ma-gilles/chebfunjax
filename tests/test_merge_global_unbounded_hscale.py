"""Global source hscale across finite and infinite merge trials.

Provenance
----------
MATLAB source : @chebfun/hscale.m, @chebfun/merge.m, @bndfun/bndfun.m,
    @unbndfun/unbndfun.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.fun.unbndfun import Unbndfun
from chebfunjax.tech.chebtech import Chebtech2


def test_global_unbounded_hscale_survives_finite_intermediate_union(monkeypatch):
    # Independent constant field: no adaptive constructor data or source counts.
    ends = (0., 10., 20., float('inf'))
    one = Chebtech2.from_coeffs(jnp.array([1.]))
    f = Chebfun(funs=[_Piece(one, ends[:2]), _Piece(one, ends[1:3]),
                     Unbndfun.from_chebtech(one, Domain(ends[2:]))],
                domain=Domain(ends))
    fit = Chebtech2.from_function
    scales = []

    def record(cls, op, **kwargs):
        scales.append(kwargs['hscale'])
        return fit(op, **kwargs)

    monkeypatch.setattr(Chebtech2, 'from_function', classmethod(record))
    result = f.merge(max_length=33)
    # The global scale is1 even for the first bounded union[0,20].
    # Bndfun divides it by20; the subsequent unbounded union retains1.
    assert scales == [1/20, 1.]
    assert result.domain.breakpoints == (0., float('inf'))
