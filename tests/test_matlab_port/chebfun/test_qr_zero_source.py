"""Separate zero-QR source branch controls; pin7574c77 @chebfun/qr.m32.

Piecewise zero follows the literal numeric-vector expansion in plus.m;
no global orthogonality predicate is imposed on that native corner.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("cls", [Chebtech1,Chebtech2])
@pytest.mark.parametrize("breaks", [(2.,5.),(2.,3.,5.)])
def test_source_zero_metadata_and_expansion(cls,breaks):
    dom = Domain(breaks)
    f = Chebfun(funs=[_Piece(cls.from_coeffs(jnp.array([0.])),(a,b))
                     for a,b in zip(breaks[:-1],breaks[1:])],domain=dom)
    Q,R = f.qr()
    assert isinstance(Q,Chebfun)
    assert all(type(p.tech) is cls for p in Q.funs)
    assert Q.domain.breakpoints == breaks
    assert not Q.is_transposed
    assert float(R[0,0]) == 0
    expected = jnp.array([1/(b-a)**.5 for a,b in zip(breaks[:-1],breaks[1:])])
    assert Q.n_columns == len(breaks)-1
    eps = float(jnp.finfo(jnp.float64).eps)
    for p in Q.funs:
        assert float(jnp.max(jnp.abs(p.tech.coeffs[0]-expected))) < 2*eps
        assert p.tech.ishappy
    target = jnp.broadcast_to(expected,(len(breaks),len(expected)))
    observed = Q.point_values.reshape((len(breaks),len(expected)))
    assert float(jnp.max(jnp.abs(observed-target))) < 2*eps
    if len(breaks) == 2:
        # Unchanged original QR15 acceptance bound.
        assert abs(complex(jnp.squeeze(Q.H@Q))-1) < 1e-13
    else:
        # This is source numeric-row-vector expansion, not a normalized basis.
        assert float(Q.mat2cell()[0].norm()) > 1
