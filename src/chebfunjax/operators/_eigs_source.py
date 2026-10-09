"""C2 projected eigensystems and the native adaptive selection policy.

MATLAB source: @linop/eigs.m, @chebcolloc/reduce.m,
@chebcolloc2/toFunctionOut.m, @opDiscretization/testConvergence.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
New numerical policy is JAX; the inherited generalized LAPACK boundary remains
operators.blocklinop._geig and is not a JAX eigendecomposition claim.
"""
from __future__ import annotations

import math
import warnings

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.chebpref import ChebopPref
from chebfunjax.domain import Domain
from chebfunjax.operators._linear_altdisc import _dimension_values
from chebfunjax.operators.blocks import ChebColloc2Disc, OperatorBlock, _blkdiag
from chebfunjax.operators.chebmatrix import ChebMatrix
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.utils.interpolation import barymat
from chebfunjax.utils.quadrature import chebpts


def preferences(pref=None):
    """Resolve source operator preferences without changing session defaults."""
    return ChebopPref() if pref is None else ChebopPref(pref)


def projection(n, order):
    """Native reduceOne: canonical six-argument C2-to-C1 barycentric map."""
    m = n + order
    return barymat(chebpts(n, kind=1), chebpts(m, kind=2),
                   Chebtech2.barywts(m), Chebtech1.angles(n),
                   Chebtech2.angles(m), True)


def output_coefficients(values, cutoff=None):
    """toFunctionOut followed by numeric constructor's coefficient transform."""
    coeffs = Chebtech1.vals2coeffs(values)
    if cutoff is not None:
        coeffs = coeffs[:cutoff]
    return Chebtech2.vals2coeffs(Chebtech2.coeffs2vals(coeffs))


class C2Pencil:
    """Finite function-variable block pencil; constraints belong only to A."""

    def __init__(self, operator, dimensions, mass=None, domain=None):
        from chebfunjax.operators.blocklinop import BlockLinop

        dom = tuple(operator.domain if domain is None else domain)
        if mass is not None:
            mass = mass if isinstance(mass, BlockLinop) else BlockLinop(mass)
            if dom[0] != mass.domain[0] or dom[-1] != mass.domain[-1]:
                raise ValueError('Generalized eigs requires matching domain endpoints')
            dom = tuple(sorted(set(dom).union(mass.domain)))
        if not all(math.isfinite(x) for x in dom):
            raise ValueError('CHEBFUN:LINOP:eigs:infDom -- Unbounded domains are not supported')
        if operator.nrows != operator.ncols:
            raise ValueError('CHEBFUN:LINOP:eigs:notSquare')
        if not all(operator.is_fun_variable()):
            raise NotImplementedError('Projected C2 eigs currently requires function variables')
        self.domain = dom
        self.dimensions = tuple(operator._sizes(dimensions, dom))
        self.operator = operator.derive_continuity(dom) if not operator.continuity else operator
        self.orders = operator.proj_order()
        if mass is not None:
            self.orders = [max(a, b) for a, b in zip(self.orders, mass.proj_order())]
        self.projections = [_blkdiag([projection(n, order) for n in self.dimensions])
                            for order in self.orders]
        self.P = _blkdiag(self.projections)
        self.discs = [ChebColloc2Disc([n+r for n in self.dimensions], dom)
                      for r in self.orders]
        constraints, _ = self.operator._constraint_rows(
            self.dimensions, dom, self.orders, [True]*operator.ncols)
        self.constraint_count = len(constraints)
        arows = self._rows(operator.A)
        self.A = jnp.concatenate([*constraints, *arows], axis=0)
        source_b = operator.A.identity() if mass is None else mass.A
        brows = jnp.concatenate(self._rows(source_b), axis=0)
        self.B = jnp.concatenate([
            jnp.zeros((self.constraint_count, brows.shape[1]), dtype=brows.dtype), brows])
        if self.A.shape[0] != self.A.shape[1] or self.B.shape != self.A.shape:
            raise ValueError('Eigs pencil is not square after source constraints')

    def _rows(self, matrix):
        rows = []
        for row in matrix.blocks:
            parts = []
            for j, block in enumerate(row):
                if not isinstance(block, OperatorBlock):
                    raise NotImplementedError('Projected C2 eigs requires differential/function blocks')
                parts.append(self.projections[j] @ block.matrix(self.discs[j]))
            rows.append(jnp.concatenate(parts, axis=1))
        return rows

    def coefficients(self, projected, cutoffs=None):
        """Return variable -> interval -> coefficient arrays, without simplify."""
        result, offset = [], 0
        for _ in self.orders:
            pieces = []
            for i, n in enumerate(self.dimensions):
                cut = None if cutoffs is None else cutoffs[i]
                pieces.append(output_coefficients(projected[offset:offset+n], cut))
                offset += n
            result.append(pieces)
        return result

    def functions(self, projected, cutoffs=None):
        result = []
        for coefficients in self.coefficients(projected, cutoffs):
            pieces = [_Piece.from_coeffs(c, a, b) for c, (a, b) in
                      zip(coefficients, zip(self.domain[:-1], self.domain[1:]))]
            result.append(Chebfun(funs=pieces, domain=Domain(self.domain)))
        return result


def nearest(lam, sigma, count):
    """Source deflate and nearest; count is the number of constraint rows."""
    lam = jnp.asarray(lam)
    deflated = lam.at[jnp.argsort(-jnp.abs(lam), stable=True)[:count]].set(jnp.inf)
    if isinstance(sigma, str):
        criteria = {'LM': -jnp.abs(deflated), 'SM': jnp.abs(deflated),
                    'LR': -jnp.real(deflated), 'SR': jnp.real(deflated),
                    'LI': -jnp.imag(deflated), 'SI': jnp.imag(deflated)}
        if sigma.upper() not in criteria:
            raise ValueError(f'CHEBFUN:LINOP:eigs:sigma -- {sigma!r}')
        key = criteria[sigma.upper()]
    else:
        key = -jnp.abs(deflated) if math.isinf(abs(complex(sigma))) else jnp.abs(deflated-sigma)
    ordered = jnp.argsort(key, stable=True)
    return [int(i) for i in ordered if bool(jnp.isfinite(deflated[i]))]


def filter_modes(indices, projected, k, pencil):
    """Literal coefficient-energy queue from @linop/eigs.m/filter."""
    n = min(pencil.dimensions)
    queue = list(range(min(k, n, len(indices))))
    keep = set(queue)
    ten = math.ceil(n/10)
    while queue:
        j = queue.pop(0)
        coefficients = pencil.coefficients(projected[:, indices[j]])
        length = max(len(c) for pieces in coefficients for c in pieces)
        energy = jnp.zeros(length)
        for pieces in coefficients:
            for c in pieces:
                energy = energy.at[:len(c)].add(jnp.real(c*jnp.conj(c)))
        modes = jnp.sqrt(energy)
        last90 = jnp.linalg.norm(modes[ten-1:n])
        last10 = jnp.linalg.norm(modes[n-ten-1:n])
        first10 = jnp.linalg.norm(modes[:ten])
        if bool((last10 > .5*last90) & (last90 > 1e-8*first10)):
            keep.remove(j)
            # Source appends after the last queued index (including current).
            next_index = (queue[-1] if queue else j)+1
            if next_index < len(indices):
                queue.append(next_index)
                keep.add(next_index)
    return jnp.asarray([indices[i] for i in sorted(keep)], dtype=jnp.int32)


def eigenvalues(pencil, k, sigma):
    from chebfunjax.operators.blocklinop import _geig

    lam, vectors = _geig(pencil.A, pencil.B)
    lam, vectors = jnp.asarray(lam), jnp.asarray(vectors)
    projected = pencil.P @ vectors
    selected = filter_modes(nearest(lam, sigma, pencil.constraint_count), projected, k, pencil)
    return lam[selected], vectors[:, selected]


def automatic_target(coarse, fine, fine_coefficients):
    """Source33/65 choice from raw, unnormalized projected fine functions."""
    delta = jnp.min(jnp.abs(fine[:, None]-coarse[None, :]), axis=0)
    large = delta > 1e-12*jnp.max(jnp.abs(coarse))
    coarse_b = jnp.where(large, 0, coarse)
    large = large | (delta > 1e-3*jnp.max(jnp.abs(coarse_b)))
    if bool(jnp.all(large)):
        return complex(coarse[jnp.argmin(delta)])
    coeffnorm = sum(jnp.sum(jnp.abs(c), axis=0) for c in fine_coefficients)
    return complex(fine[jnp.argmin(coeffnorm)])


def convergence(pencil, projected, pref):
    coefficients = pencil.coefficients(projected)
    # Source toFunctionOut concatenates variables; vscale is global per column.
    scales = jnp.asarray([max(float(jnp.max(jnp.abs(Chebtech2.coeffs2vals(c))))
                             for c in pieces) for pieces in coefficients])
    done, cutoffs = [], []
    for i in range(len(pencil.dimensions)):
        c = jnp.column_stack([pieces[i] for pieces in coefficients])
        v = Chebtech2.coeffs2vals(c)
        ok, cut = Chebtech2.happiness_check(
            c, v, tol=float(pref.bvpTol), vscale=scales,
            check=str(pref.happinessCheck), sample_test=False)
        done.append(bool(ok))
        cutoffs.append(pencil.dimensions[i] if cut is None else int(cut))
    return done, cutoffs


def solve(operator, *, k=6, sigma=None, n=None, mass=None, domain=None, pref=None):
    """Native C2 automatic targeting/refinement; n is a fixed-size extension."""
    pref = preferences(pref)
    dims = _dimension_values(float(pref.minDimension), float(pref.maxDimension), 'chebcolloc2')
    dom = tuple(operator.domain if domain is None else domain)
    if mass is not None:
        dom = tuple(sorted(set(dom).union(mass.domain)))
    current = [dims[0]]*(len(dom)-1) if n is None else operator._sizes(n, dom)
    if sigma is None:
        p1 = C2Pencil(operator, 33, mass, dom)
        coarse, _ = eigenvalues(p1, 33, 0)
        p2 = C2Pencil(operator, 65, mass, dom)
        fine, vectors = eigenvalues(p2, 33, 0)
        projected = p2.P @ vectors
        width = sum(p2.dimensions)
        # Native adds variable value blocks before the output transform.
        combined_values = sum(projected[j*width:(j+1)*width]
                              for j in range(len(p2.orders)))
        combined, offset = [], 0
        for size in p2.dimensions:
            combined.append(output_coefficients(combined_values[offset:offset+size]))
            offset += size
        sigma = automatic_target(coarse, fine, combined)
        if n is None:
            current = [65]*(len(dom)-1)
    done = [False]*len(current)
    for next_dimension in (*dims, None) if n is None else (None,):
        pencil = C2Pencil(operator, current, mass, dom)
        lam, vectors = eigenvalues(pencil, k, sigma)
        if not len(lam):
            raise ValueError('Eigs returned no finite modes')
        composite = vectors @ (1/(2*jnp.arange(1, len(lam)+1)))
        done, cutoffs = convergence(pencil, pencil.P @ composite, pref)
        if all(done) or n is not None:
            break
        if next_dimension is not None:
            current = [old if happy else next_dimension for old, happy in zip(current, done)]
    if n is None and not all(done):
        warnings.warn('LINOP:EIGS:convergence -- Maximum dimension reached; solution may not have converged.', stacklevel=2)
    if len(lam) < k:
        warnings.warn(f'CHEBFUN:LINOP:eigs:rank -- Input has finite rank, only {len(lam)} eigenvalues returned.', stacklevel=2)
    order = (jnp.argsort(jnp.real(lam), stable=True) if bool(jnp.all(jnp.imag(lam) == 0))
             else jnp.lexsort((jnp.angle(lam), jnp.abs(lam))))
    lam, vectors = lam[order], vectors[:, order]
    output = []
    for j in range(len(lam)):
        functions = [f.simplify() for f in pencil.functions(pencil.P @ vectors[:, j], cutoffs)]
        point = dom[0]+(dom[-1]-dom[0])*.500023981
        sign = jnp.sign(jnp.sign(jnp.real(functions[0](point)))+.1)
        normsq = sum(jnp.asarray((f*f.conj()).sum()) for f in functions)
        scale = sign/jnp.sqrt(normsq)
        output.append(ChebMatrix([[f*scale] for f in functions], domain=dom))
    return lam, output
