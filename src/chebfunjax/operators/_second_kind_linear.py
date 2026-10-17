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
    """C2 function points n+r, C1 output n; scalar columns stay one value.

    Only function variables are projected, checked for happiness and converted
    to Chebfuns. Numeric parameters occupy one global scalar column and retain
    their numeric values (@linop/linsolve.m108-180).
    """

    def __init__(self, L, dimensions, backend='chebcolloc2', domain=None):
        self.backend = backend
        self.domain = tuple(L.domain if domain is None else domain)
        if not all(math.isfinite(x) for x in self.domain):
            raise ValueError('CHEBFUN:LINOP:linsolve:infDom -- Unbounded domain')
        self.L = L.derive_continuity(self.domain) if not L.continuity else L
        self.dimensions = tuple(dimensions)
        self.isfun = tuple(self.L.is_fun_variable())
        self.function_indices = tuple(i for i, flag in enumerate(self.isfun) if flag)
        self.orders = tuple(self.L.proj_order())
        self.dimension_history = [self.dimensions]
        self._factor = self._scaling = None
        self.tech = Chebtech2
        self.intervals = tuple(zip(self.domain[:-1], self.domain[1:]))
        self.col_dimensions = tuple(sum(n+r for n in self.dimensions) if flag else 1
                                    for r, flag in zip(self.orders, self.isfun))
        self.row_orders = tuple(max(0, max(getattr(b, 'order', 0) for b in row))
                                for row in self.L.A.blocks)
        self.projections = [_block_diagonal([
            barymat(chebpts(n, kind=1), chebpts(n+r, kind=2),
                    Chebtech2.barywts(n+r), Chebtech1.angles(n),
                    Chebtech2.angles(n+r), do_flip=True)
            for n in self.dimensions]) if flag else jnp.ones((1, 1))
            for r, flag in zip(self.orders, self.isfun)]
        rows, self.constraint_values = self.L._constraint_rows(
            self.dimensions, self.domain, self.orders, self.isfun)
        self.continuity_count = len(self.L.continuity)
        for blocks in self.L.A.blocks:
            columns = []
            for j, block in enumerate(blocks):
                if self.isfun[j]:
                    if not isinstance(block, OperatorBlock):
                        raise NotImplementedError('Second-kind function equations require operator blocks')
                    disc = ChebColloc2Disc([n+self.orders[j] for n in self.dimensions], self.domain)
                    columns.append(self.projections[j] @ block.matrix(disc))
                else:
                    points = jnp.concatenate([
                        (b*(chebpts(n, kind=1)+1) + a*(1-chebpts(n, kind=1)))/2
                        for n, (a, b) in zip(self.dimensions, self.intervals)])
                    values = block(points) if isinstance(block, Chebfun) else jnp.full(points.shape, block)
                    columns.append(jnp.asarray(values).reshape(-1, 1))
            rows.append(jnp.concatenate(columns, axis=1))
        self.A = jnp.concatenate(rows, axis=0)
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError('CHEBFUN:LINOP:linsolve:notSquare -- Operator may not have the correct number of boundary conditions.')

    def rhs(self, entries, constraint_values=None):
        """Evaluate continuous RHS rows at the actual equation points.

        Provenance: @valsDiscretization/rhs.m, Chebfun 7574c77. Continuity
        values precede constraints; each function block then uses all panels.
        A tolerance-merged domain need not retain every tiny RHS FUN.
        """
        entries = self.L._normalize_rhs(entries)
        constraints = (self.constraint_values if constraint_values is None
                       else constraint_values)
        points = jnp.concatenate([
            (b*(chebpts(n, kind=1)+1)+a*(1-chebpts(n, kind=1)))/2
            for n, (a, b) in zip(self.dimensions, self.intervals)])
        pieces = [jnp.asarray(constraints).reshape(-1)]
        for entry in entries:
            values = entry(points) if callable(entry) else jnp.full(points.shape, entry)
            if jnp.ndim(values) == 0:
                values = jnp.full(points.shape, values)
            pieces.append(jnp.asarray(values).reshape(-1))
        return jnp.concatenate(pieces)

    @property
    def is_factored(self):
        # Native valsDiscretization/isFactored counts function equation sizes
        # plus one value per scalar variable, without derivative offsets.
        length = sum(self.isfun)*sum(self.dimensions) + self.isfun.count(False)
        return self._factor is not None and self._factor[0].shape[0] == length

    def coefficient_data(self, vector):
        by_variable = []
        offset = 0
        for width, projection, flag in zip(self.col_dimensions, self.projections, self.isfun):
            segment = vector[offset:offset+width]
            offset += width
            if not flag:
                continue
            projected = projection @ segment
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
        functions = [Chebfun(funs=[_Piece(
            tech=Chebtech2.from_values(Chebtech2.coeffs2vals(c[:int(cut), j])),
            interval=interval) for c, cut, interval in zip(data, cuts, self.intervals)],
            domain=Domain(self.domain)) for j in range(len(self.function_indices))]
        out, offset, index = [], 0, 0
        for width, flag in zip(self.col_dimensions, self.isfun):
            out.append(functions[index] if flag else vector[offset].item())
            index += int(flag)
            offset += width
        return out
