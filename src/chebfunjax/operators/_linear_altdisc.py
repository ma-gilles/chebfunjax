"""Exact differential blocks and adaptive alternative discretizations.

Provenance
----------
MATLAB source: @linop/linsolve.m, @opDiscretization/matrix.m,
    @chebcolloc/reduce.m, @ultraS/reduce.m, @opDiscretization/testConvergence.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
"""
from __future__ import annotations

import math
import warnings

import jax.numpy as jnp
from jax.scipy.linalg import lu_factor, lu_solve

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece, chebfun
from chebfunjax.discretization.ultras import convertmat, multmat
from chebfunjax.discretization.ultras import diffmat as ultra_diffmat
from chebfunjax.domain import Domain
from chebfunjax.operators._native_values import FirstKindDisc
from chebfunjax.operators.blocks import OperatorBlock
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2, _clenshaw
from chebfunjax.utils.diffmat import _cheb1_barywts, diffmat
from chebfunjax.utils.interpolation import barymat
from chebfunjax.utils.quadrature import chebpts


def _block_diagonal(blocks):
    rows = sum(block.shape[0] for block in blocks)
    cols = sum(block.shape[1] for block in blocks)
    out = jnp.zeros((rows, cols), dtype=jnp.result_type(*blocks))
    i = j = 0
    for block in blocks:
        r, c = block.shape
        out = out.at[i:i+r, j:j+c].set(block)
        i, j = i+r, j+c
    return out


def _coefficients(value, interval):
    """Obtain existing polynomial coefficients without operator probes."""
    if isinstance(value, Chebfun):
        restricted = value.restrict(interval)
        if len(restricted.funs) != 1:
            raise ValueError("Coefficient breakpoints must be included in the discretization")
        tech = restricted.funs[0].tech
        if not isinstance(tech, (Chebtech1, Chebtech2)):
            raise NotImplementedError(
                "Exact alternative assembly requires polynomial Chebtech1/Chebtech2 data; "
                "native conversion of other technologies is not implemented")
        return jnp.ravel(tech.coeffs)
    if callable(value):
        return _coefficients(chebfun(value, domain=interval), interval)
    return jnp.asarray(value).reshape(1)


def _prolong(coefficients, length):
    coefficients = jnp.asarray(coefficients)
    return jnp.pad(coefficients[:length], (0, max(0, length-len(coefficients))))


def _dimension_values(minimum, maximum, backend):
    """Literal coeffs/valsDiscretization.dimensionValues schedules."""
    lo, hi = math.log2(minimum), math.log2(maximum)
    if lo > hi:
        raise ValueError("Minimum discretization specified is greater than maximum discretization specified")
    def interval(start, stop, step):
        count = max(0, int(math.floor((stop-start)/step + 1e-14))+1)
        return [start+step*i for i in range(count)]

    if backend == 'ultraS' or hi <= 9:
        powers = interval(lo, hi, 1.)
    elif lo >= 9:
        powers = interval(lo, hi, .5)
    else:
        powers = interval(lo, 9, 1.)+interval(9.5, hi, .5)
    return tuple(int(math.floor(2**p + 0.5)) for p in powers)



def _native_capability(block, backend):
    """Source values-stack capability, with the native ultraS rejection."""
    if backend == 'ultraS':
        raise TypeError(
            "COEFFSDISCRETIZATION:instantiate:fail -- Cannot represent this "
            "operator. Suggest you use VALSDISCRETIZATION.")
    capability = block._values_capability
    if backend != 'chebcolloc1' or capability is None:
        raise TypeError("No native first-kind stack capability for this operator")
    return capability


class LinearDiscretization:
    """Source input offsets, projection and solve for function-valued blocks.

    Dimensions are equation dimensions, one per interval. Column j has
    dimension n_i + r_j before source projection back to n_i.

    Provenance
    ----------
    MATLAB source: @opDiscretization/matrix.m, @opDiscretization/getDimAdjust.m,
        @linop/getProjOrder.m, @valsDiscretization/mldivide.m.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    """

    def __init__(self, L, dimensions, backend, domain=None):
        if backend not in ('chebcolloc1', 'ultraS'):
            raise ValueError(f"Unsupported differential backend {backend!r}")
        self.backend = backend
        self.domain = tuple(L.domain if domain is None else domain)
        if not all(math.isfinite(x) for x in self.domain):
            raise ValueError("CHEBFUN:LINOP:linsolve:infDom -- Unbounded domains are not supported.")
        self.dimensions = tuple(int(n) for n in dimensions)
        if len(self.dimensions) != len(self.domain)-1:
            raise ValueError("CHEBFUN:opDiscretization:matrix:subIntDim -- Must specify one dimension value for each subinterval.")
        self.L = L.derive_continuity(self.domain) if not L.continuity else L
        if not all(self.L.is_fun_variable()):
            raise NotImplementedError("Exact alternative assembly currently requires function-valued variables")
        self.orders = tuple(self.L.proj_order())
        self.dimension_history = [self.dimensions]
        self._factor = None
        self._scaling = None
        self.tech = Chebtech1 if backend == 'chebcolloc1' else Chebtech2
        self.intervals = tuple(zip(self.domain[:-1], self.domain[1:]))
        self.col_dimensions = tuple(sum(n+r for n in self.dimensions)
                                    for r in self.orders)
        self.row_orders = tuple(max(0, max(getattr(block, 'order', 0)
                                          for block in row))
                                for row in self.L.A.blocks)
        self.projections = []
        input_to_second_kind = []
        for r in self.orders:
            projections, transfers = [], []
            for n in self.dimensions:
                m = n+r
                if backend == 'chebcolloc1':
                    projections.append(barymat(chebpts(n, kind=1),
                                                chebpts(m, kind=1),
                                                _cheb1_barywts(m),
                                                Chebtech1.angles(n),
                                                Chebtech1.angles(m), True))
                    transfers.append(Chebtech2.coeffs2vals(
                        Chebtech1.vals2coeffs(jnp.eye(m))))
                else:
                    projections.append(jnp.eye(n, m))
                    transfers.append(Chebtech2.coeffs2vals(jnp.eye(m)))
            self.projections.append(_block_diagonal(projections))
            input_to_second_kind.append(_block_diagonal(transfers))
        constraint_rows, self.constraint_values = self.L._constraint_rows(
            self.dimensions, self.domain, self.orders,
            [True]*self.L.ncols,
        )
        transfer = _block_diagonal(input_to_second_kind)
        rows = [row @ transfer for row in constraint_rows]
        self.continuity_count = len(self.L.continuity)
        for i, blocks in enumerate(self.L.A.blocks):
            row = [self._operator_column(block, j, self.row_orders[i])
                   for j, block in enumerate(blocks)]
            rows.append(jnp.concatenate(row, axis=1))
        self.A = jnp.concatenate(rows, axis=0)
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("CHEBFUN:LINOP:linsolve:notSquare -- Operator may not have the correct number of boundary conditions.")

    def _operator_column(self, block, column, output_order):
        if isinstance(block, OperatorBlock):
            if block._coeff_fn is None:
                capability = _native_capability(block, self.backend)
                sizes = tuple(n+self.orders[column] for n in self.dimensions)
                matrix = capability.realize(FirstKindDisc(sizes, self.domain))
                return self.projections[column] @ matrix
            coefficients = block.coeff_list()
        elif isinstance(block, (int, float, complex)):
            coefficients = [block]
        else:
            raise NotImplementedError("Exact alternative assembly requires differential coefficient realizations")
        pieces = []
        for n, interval in zip(self.dimensions, self.intervals):
            m = n+self.orders[column]
            a, b = interval
            terms = []
            for order, value in enumerate(reversed(coefficients)):
                c = _coefficients(value, interval)
                if self.backend == 'ultraS':
                    derivative = (2/(b-a))**order * ultra_diffmat(m, order)
                    term = (convertmat(m, order, output_order-1)
                            @ multmat(m, c, order) @ derivative)
                else:
                    values = _clenshaw(c, chebpts(m, kind=1))
                    derivative = diffmat(m, order, domain=interval, kind=1)
                    term = values[:, None]*derivative
                terms.append(term)
            matrix = sum(terms)
            if self.backend == 'ultraS':
                pieces.append(matrix[:n, :])
            else:
                projection = barymat(chebpts(n, kind=1), chebpts(m, kind=1),
                                     _cheb1_barywts(m), Chebtech1.angles(n),
                                     Chebtech1.angles(m), True)
                pieces.append(projection @ matrix)
        return _block_diagonal(pieces)

    def rhs(self, entries, constraint_values=None):
        """Discretize function rows after continuity and endpoint data.

        Provenance
        ----------
        MATLAB source: @ultraS/rhs.m, @valsDiscretization/rhs.m.
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        entries = self.L._normalize_rhs(entries)
        constraints = (self.constraint_values if constraint_values is None
                       else constraint_values)
        pieces = [jnp.asarray(constraints).reshape(-1)]
        for i, entry in enumerate(entries):
            for n, interval in zip(self.dimensions, self.intervals):
                c = _coefficients(entry, interval)
                if self.backend == 'ultraS':
                    pieces.append(convertmat(n, 0, self.row_orders[i]-1)
                                  @ _prolong(c, n))
                else:
                    pieces.append(_clenshaw(c, chebpts(n, kind=1)))
        return jnp.concatenate(pieces)

    @property
    def is_factored(self):
        """Source compatibility predicate, including its equation-size check.

        Provenance
        ----------
        MATLAB source: @valsDiscretization/isFactored.m.
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return (self.backend == 'chebcolloc1' and self._factor is not None
                and self._factor[0].shape[0]
                == self.L.ncols*sum(self.dimensions))

    def solve(self, rhs):
        """C1 scaled LU or the source unscaled coefficient-space solve.

        Provenance
        ----------
        MATLAB source: @valsDiscretization/mldivide.m,
            @opDiscretization/opDiscretization.m (mldivide).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        if self.backend == 'ultraS':
            return jnp.linalg.solve(self.A, rhs)
        if not self.is_factored:
            self._scaling = 1/jnp.maximum(1, jnp.max(jnp.abs(self.A), axis=1))
            self._factor = lu_factor(self._scaling[:, None]*self.A)
        return lu_solve(self._factor, self._scaling*rhs)

    def coefficient_data(self, vector):
        """Project first; expose unsimplified per-interval coefficient arrays.

        Provenance
        ----------
        MATLAB source: @linop/linsolve.m, @opDiscretization/partition.m.
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        by_variable = []
        offset = 0
        for width, projection in zip(self.col_dimensions, self.projections):
            projected = projection @ vector[offset:offset+width]
            offset += width
            pieces = []
            pos = 0
            for n in self.dimensions:
                values = projected[pos:pos+n]
                pieces.append(Chebtech1.vals2coeffs(values)
                              if self.backend == 'chebcolloc1' else values)
                pos += n
            by_variable.append(pieces)
        return [jnp.stack([var[k] for var in by_variable], axis=1)
                for k in range(len(self.dimensions))]

    def recover(self, vector, cutoffs=None):
        """Source projection/output conversion; no premature simplification.

        Provenance
        ----------
        MATLAB source: @chebcolloc1/toFunctionOut.m, @ultraS/toFunctionOut.m.
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        data = self.coefficient_data(vector)
        cuts = self.dimensions if cutoffs is None else cutoffs
        return [Chebfun(funs=[_Piece(
            tech=(Chebtech1.from_values(Chebtech1.coeffs2vals(c[:int(cut), j]))
                  if self.backend == 'chebcolloc1'
                  else Chebtech2.from_coeffs(c[:int(cut), j])), interval=interval,
        ) for c, cut, interval in zip(data, cuts, self.intervals)],
            domain=Domain(self.domain)) for j in range(self.L.ncols)]


def solve_operator(L, rhs, *, backend, n=None, n_min=32, n_max=4096,
                   tol=5e-13, vscale=None, disc=None, happiness_check='standard'):
    """Adapt only unresolved intervals and return entries, disc, converged.

    Explicit n is the Python fixed-equation-dimension extension. A supplied
    disc assumes the same frozen operator, as in source linsolve; constraint
    values and RHS may change. Arrays and linear algebra remain in JAX.

    Provenance
    ----------
    MATLAB source: @linop/linsolve.m, @opDiscretization/testConvergence.m,
        @coeffsDiscretization/coeffsDiscretization.m (dimensionValues),
        @valsDiscretization/valsDiscretization.m (dimensionValues).
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    """
    entries = L._normalize_rhs(rhs)
    coefficient_functions = []
    native_domains = []
    for row in L.A.blocks:
        for block in row:
            if not isinstance(block, OperatorBlock):
                continue
            if block._coeff_fn is None:
                native_domains.extend(_native_capability(block, backend).domain)
            else:
                coefficient_functions.extend(
                    coefficient for coefficient in block.coeff_list()
                    if isinstance(coefficient, Chebfun))
    domain = L._merged_domain(entries+coefficient_functions)
    if native_domains:
        domain = tuple(sorted(set(domain).union(native_domains)))
    schedule = _dimension_values(n_min, n_max, backend)
    if n is not None:
        dimensions = tuple(L._sizes(n, domain))
        schedule = ()
    elif disc is None:
        dimensions = (schedule[0],)*(len(domain)-1)
        schedule = schedule[1:]
    else:
        dimensions = disc.dimensions
        schedule = tuple(size for size in schedule if size > max(dimensions))
    scales = jnp.zeros(L.ncols) if vscale is None else jnp.broadcast_to(
        jnp.asarray(vscale), (L.ncols,))
    history = []
    for next_dimension in (*schedule, None):
        if disc is None or disc.dimensions != dimensions or not disc.is_factored:
            disc = LinearDiscretization(L, dimensions, backend, domain)
        history.append(dimensions)
        current_constraints = [val for _, val in disc.L.continuity]
        current_constraints += [val for _, val in L.constraint]
        vector = disc.solve(disc.rhs(entries, current_constraints))
        if n is not None:
            disc.dimension_history = history
            return disc.recover(vector), disc, True
        data = disc.coefficient_data(vector)
        values = [disc.tech.coeffs2vals(c) for c in data]
        # Source testConvergence calls toFunctionOut before reading coeffs.
        # C1 therefore includes the values-constructor transform here too.
        if backend == 'chebcolloc1':
            data = [Chebtech1.vals2coeffs(value) for value in values]
        for value in values:
            scales = jnp.maximum(scales, jnp.max(jnp.abs(value), axis=0))
        checks = [disc.tech.happiness_check(
            c, value, tol=tol, vscale=scales, hscale=1.,
            check=happiness_check, sample_test=False,
        ) for c, value in zip(data, values)]
        done = [bool(happy) for happy, _ in checks]
        if all(done) or next_dimension is None:
            break
        dimensions = tuple(size if happy else next_dimension
                           for size, happy in zip(dimensions, done))
    if not all(done):
        warnings.warn("CHEBFUN:LINOP:linsolve:noConverge -- Linear system solution may not have converged.",
                      UserWarning, stacklevel=2)
    disc.dimension_history = history
    return disc.recover(vector, [cut for _, cut in checks]), disc, all(done)
