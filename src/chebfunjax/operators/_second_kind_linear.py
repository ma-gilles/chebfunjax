"""Native second-kind input stacks sharing the adaptive linear solver.

Provenance: Chebfun 7574c77 @valsDiscretization/instantiate.m,
@chebcolloc/reduce.m, @chebcolloc2/toFunctionOut.m, @linop/linsolve.m.
"""
import math

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.operators._linear_altdisc import LinearDiscretization, _block_diagonal
from chebfunjax.operators.blocks import ChebColloc2Disc, OperatorBlock
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.utils.interpolation import barymat
from chebfunjax.utils.quadrature import chebpts


class SecondKindDiscretization(LinearDiscretization):
    """C2 function points n+r; C1 equation/output points n, per interval."""

    def __init__(self, L, dimensions, backend='chebcolloc2', domain=None):
        self.backend = backend
        self.domain = tuple(L.domain if domain is None else domain)
        if not all(math.isfinite(x) for x in self.domain):
            raise ValueError('CHEBFUN:LINOP:linsolve:infDom -- Unbounded domain')
        self.L = L.derive_continuity(self.domain) if not L.continuity else L
        self.dimensions = tuple(dimensions)
        if not all(self.L.is_fun_variable()):
            raise NotImplementedError('Second-kind adaptive assembly requires function variables')
        self.orders = tuple(self.L.proj_order())
        self.dimension_history = [self.dimensions]
        self._factor = self._scaling = None
        self.tech = Chebtech2
        self.intervals = tuple(zip(self.domain[:-1], self.domain[1:]))
        self.col_dimensions = tuple(sum(n+r for n in self.dimensions) for r in self.orders)
        self.row_orders = tuple(max(0, max(getattr(b, 'order', 0) for b in row))
                                for row in self.L.A.blocks)
        self.projections = [_block_diagonal([
            barymat(chebpts(n, kind=1), chebpts(n+r, kind=2),
                    Chebtech2.barywts(n+r), Chebtech1.angles(n),
                    Chebtech2.angles(n+r), do_flip=True)
            for n in self.dimensions]) for r in self.orders]
        rows, self.constraint_values = self.L._constraint_rows(
            self.dimensions, self.domain, self.orders, [True]*self.L.ncols)
        self.continuity_count = len(self.L.continuity)
        for blocks in self.L.A.blocks:
            columns = []
            for j, block in enumerate(blocks):
                if not isinstance(block, OperatorBlock):
                    raise NotImplementedError('Second-kind function equations require operator blocks')
                disc = ChebColloc2Disc([n+self.orders[j] for n in self.dimensions], self.domain)
                columns.append(self.projections[j] @ block.matrix(disc))
            rows.append(jnp.concatenate(columns, axis=1))
        self.A = jnp.concatenate(rows, axis=0)
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError('CHEBFUN:LINOP:linsolve:notSquare -- Operator may not have the correct number of boundary conditions.')

    @property
    def is_factored(self):
        # Native valsDiscretization compares with equation sizes, not offsets.
        return (self._factor is not None and self._factor[0].shape[0]
                == self.L.ncols*sum(self.dimensions))

    def coefficient_data(self, vector):
        by_variable = []
        offset = 0
        for width, projection in zip(self.col_dimensions, self.projections):
            projected = projection @ vector[offset:offset+width]
            offset += width
            pieces, pos = [], 0
            for n in self.dimensions:
                pieces.append(Chebtech1.vals2coeffs(projected[pos:pos+n]))
                pos += n
            by_variable.append(pieces)
        return [jnp.stack([v[k] for v in by_variable], axis=1)
                for k in range(len(self.dimensions))]

    def recover(self, vector, cutoffs=None):
        data = self.coefficient_data(vector)
        cuts = self.dimensions if cutoffs is None else cutoffs
        return [Chebfun(funs=[_Piece(
            tech=Chebtech2.from_values(Chebtech2.coeffs2vals(c[:int(cut), j])),
            interval=interval) for c, cut, interval in zip(data, cuts, self.intervals)],
            domain=Domain(self.domain)) for j in range(self.L.ncols)]
