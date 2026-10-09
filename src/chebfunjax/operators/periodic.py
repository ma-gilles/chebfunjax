"""Periodic spectral operator assembly.

Provenance
----------
MATLAB source : @chebop/eigs.m, @linop/eigs.m, @linop/deriveContinuity.m,
    @chebcolloc/reduce.m, @chebcolloc2/equationPoints.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.operators.blocks import ChebColloc2Disc, I
from chebfunjax.utils.diffmat import diffmat
from chebfunjax.utils.interpolation import barymat
from chebfunjax.utils.quadrature import chebpts


def _system_blocks(op, count):
    """Seed function variables in both members of the operator pencil.

    MATLAB generalized linearization passes paramReshape=false to B; a
    variable without derivatives in B remains a function when A differentiates it.
    """
    from chebfunjax.operators.chebop import _LinopVar, _op_arity

    seeds = []
    for var in range(count):
        row = [None]*count
        row[var] = I(op.domain)
        seeds.append(_LinopVar(row, op.domain))
    x = Chebfun.identity(Domain(op.domain))
    result = (op.op(x, *seeds) if _op_arity(op.op, count+1) > count
              else op.op(*seeds))
    if not isinstance(result, (list, tuple)):
        result = [result]
    if len(result) != count or any(not isinstance(r, _LinopVar) for r in result):
        raise ValueError('Periodic generalized eigs requires a square linear system')
    return [r.jac for r in result]


def _pencil(a, b, n, orders):
    """Rectangular source column projections, then continuity and wrap rows."""
    count = len(orders)
    dom = tuple(float(x) for x in a.domain)
    pieces = len(dom)-1
    sizes = [n+q for q in orders]
    widths = [pieces*s for s in sizes]
    offsets = [sum(widths[:j]) for j in range(count)]
    width = sum(widths)
    blocks_a = _system_blocks(a, count)
    blocks_b = _system_blocks(b, count)
    rows_a, rows_b = [], []
    for eq in range(count):
        columns_a, columns_b = [], []
        for var, size in enumerate(sizes):
            disc = ChebColloc2Disc(size, dom)
            local = barymat(chebpts(n, kind=1), chebpts(size))
            projection = jnp.kron(jnp.eye(pieces), local)
            zero = jnp.zeros((pieces*size, pieces*size))
            aa = blocks_a[eq][var]
            bb = blocks_b[eq][var]
            columns_a.append(projection @ (aa.matrix(disc) if aa is not None else zero))
            columns_b.append(projection @ (bb.matrix(disc) if bb is not None else zero))
        rows_a.append(jnp.concatenate(columns_a, axis=1))
        rows_b.append(jnp.concatenate(columns_b, axis=1))
    constraints = []
    for var, (order, size) in enumerate(zip(orders, sizes)):
        for deriv in range(order):
            d = [diffmat(size, deriv, domain=dom[p:p+2]) for p in range(pieces)]
            for p in range(pieces):
                # Interior interfaces and the final-to-first periodic interface.
                right = (p+1) % pieces
                row = jnp.zeros(width)
                start = offsets[var]+p*size
                other = offsets[var]+right*size
                row = row.at[start:start+size].add(d[p][-1])
                row = row.at[other:other+size].add(-d[right][0])
                constraints.append(row)
    ca = jnp.stack(constraints)
    aa = jnp.concatenate((*rows_a, ca), axis=0)
    bb = jnp.concatenate((*rows_b, jnp.zeros_like(ca)), axis=0)
    return aa, bb, sizes, offsets


def generalized_periodic_system(a, b, k, n, sort):
    """Generalized periodic system with piecewise Chebyshev eigenfunctions.

    MATLAB solves a dense generalized pencil. JAX's standard dense eigensolver
    receives the algebraically equivalent inverse pencil A^-1 B, removing its
    zero eigenvalues (infinite generalized eigenvalues). This adapter requires
    an invertible A; singular A is reported instead of silently dropping modes.
    """
    from chebfunjax.operators.chebop import SystemSolution

    count = a._n_vars()
    orders = a._piecewise_orders(count)
    dom = tuple(float(x) for x in a.domain)
    if tuple(b.domain) != tuple(a.domain):
        raise ValueError('Generalized periodic operators must share a domain')
    if not all(q > 0 for q in orders):
        raise NotImplementedError('Periodic generalized parameter variables are unsupported')
    # Source starts with a modest dimension and refines the selected modes.
    n = min(max(int(n), 8), 32)
    from chebfunjax.tech.chebtech import Chebtech2

    for _ in range(4):
        aa, bb, sizes, offsets = _pencil(a, b, n, orders)
        mu, vec = jnp.linalg.eig(jnp.linalg.solve(aa, bb))
        lam = 1/mu
        finite = jnp.isfinite(lam) & (jnp.abs(mu) > 100*jnp.finfo(jnp.float64).eps)
        indices = jnp.where(finite)[0]
        values = lam[indices]
        rank = -jnp.real(values) if sort == 'LR' else jnp.abs(values)
        selected = indices[jnp.argsort(rank)[:k]]
        values = lam[selected]
        if len(values) != k:
            raise RuntimeError('Periodic generalized eigs found too few finite modes')
        # @linop/eigs uses this nontrivial composite of the selected modes,
        # then @opDiscretization/testConvergence uses the source bvpTol.
        composite = vec[:, selected] @ (1/(2*jnp.arange(1, k+1)))
        cutoffs = []
        resolved = True
        for size, offset in zip(sizes, offsets):
            local = composite[offset:offset+(len(dom)-1)*size]
            vscale = float(jnp.max(jnp.abs(local)))
            cuts = []
            for panel in range(len(dom)-1):
                samples = local[panel*size:(panel+1)*size]
                coeffs = Chebtech2.vals2coeffs(samples)
                happy, cutoff = Chebtech2.happiness_check(
                    coeffs, samples, tol=5e-13, vscale=vscale, sample_test=False)
                resolved = resolved and happy
                cuts.append(cutoff if cutoff is not None else size)
            cutoffs.append(cuts)
        if resolved:
            break
        n *= 2
    else:
        raise RuntimeError('Periodic generalized eigenvalues did not resolve')
    vectors = []
    for index in selected.tolist():
        components = []
        for var, (size, offset) in enumerate(zip(sizes, offsets)):
            funs = []
            for panel in range(len(dom)-1):
                samples = vec[offset+panel*size:offset+(panel+1)*size, index]
                coeffs = Chebtech2.vals2coeffs(samples)[:cutoffs[var][panel]]
                funs.append(_Piece.from_coeffs(coeffs, dom[panel], dom[panel+1]))
            components.append(Chebfun(funs=funs, domain=Domain(dom)).simplify())
        scale = jnp.sqrt(sum(u.norm()**2 for u in components))
        sign = jnp.sign(jnp.sign(jnp.real(components[0](dom[0]+(dom[-1]-dom[0])*.500023981)))+.1)
        vectors.append(SystemSolution([u*sign/scale for u in components]))
    return vectors, values


class _CoefficientProxy:
    """Linear Fourier coefficient action with source Toeplitz multipliers.

    Provenance
    ----------
    MATLAB source : @trigspec/diffmat.m, @trigspec/multmat.m
    Chebfun commit: 7574c77
    """
    def __init__(self, matrix, grid, length, real=True):
        self.mat = matrix
        self.grid = grid
        self.length = length
        self.real = real

    def _wrap(self, matrix, real=None):
        return _CoefficientProxy(matrix, self.grid, self.length,
                                 self.real if real is None else real)

    def sum(self):
        from chebfunjax.operators._periodic_nonlocal import NonlocalPeriodicFunctional
        raise NonlocalPeriodicFunctional

    def diff(self, order=1):
        n = self.mat.shape[0]
        modes = jnp.arange(-(n//2), (n+1)//2, dtype=jnp.float64)
        if n % 2 == 0 and order % 2:
            modes = modes.at[0].set(0.)
        return self._wrap(((2j*jnp.pi*modes/self.length)**order)[:, None] * self.mat)

    def __add__(self, other):
        if not isinstance(other, _CoefficientProxy):
            return NotImplemented
        return self._wrap(self.mat+other.mat, self.real and other.real)

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self._wrap(-self.mat)

    def __mul__(self, other):
        from chebfunjax.operators.trigspec import multmat

        if callable(other):
            from chebfunjax.chebfun1d.chebfun import chebfun
            a = float(self.grid[0])
            coefficient = chebfun(lambda x: other(x), domain=(a, a+self.length), trig=True)
            coefficients = coefficient.funs[0].tech.coeffs
            modes = jnp.arange(-(len(coefficients)//2), (len(coefficients)+1)//2)
            coefficients = coefficients*jnp.where(modes % 2 == 0, 1., -1.)
            return self._wrap(multmat(self.mat.shape[0], coefficients) @ self.mat,
                              self.real and coefficient.isreal())
        values = jnp.asarray(other)
        real = self.real and bool(jnp.all(jnp.imag(values) == 0))
        if values.ndim == 0:
            return self._wrap(values*self.mat, real)
        coefficients = jnp.fft.fftshift(jnp.fft.fft(values)/len(values))
        return self._wrap(multmat(self.mat.shape[0], coefficients) @ self.mat, real)

    __rmul__ = __mul__


def solve_coefficients(op, rhs, n, n_max, tol, n_min=32):
    """Adaptive coefficient discretization selected by trigspec/coeffs.

    Provenance
    ----------
    MATLAB source : @trigspec/diffmat.m, @trigspec/multmat.m,
        @chebop/determineDiscretization.m, @linop/linsolve.m
    Chebfun commit: 7574c77
    """
    from chebfunjax.chebfun1d.chebfun import chebfun
    from chebfunjax.operators._periodic_nonlocal import NonlocalPeriodicFunctional

    a, b = float(op.domain[0]), float(op.domain[-1])
    length = b-a
    from chebfunjax.tech.trigtech import Trigtech

    rhs_function = chebfun(lambda x: rhs(x) if callable(rhs) else rhs,
                          domain=(a, b), trig=True)
    rhs_coeffs = rhs_function.funs[0].tech.coeffs
    rhs_modes = jnp.arange(-(len(rhs_coeffs)//2), (len(rhs_coeffs)+1)//2)
    rhs_coeffs = rhs_coeffs*jnp.where(rhs_modes % 2 == 0, 1., -1.)
    coordinate = Chebfun.identity(Domain((a, b)))
    size = 16 if n is None else int(n)
    while True:
        # 2N samples supply all convolution diagonals of the N-mode matrix.
        grid = a+length*jnp.arange(2*size)/(2*size)
        proxy = _CoefficientProxy(jnp.eye(size, dtype=jnp.complex128), grid, length)
        try:
            try:
                action = op._apply_op(coordinate, proxy)
            except TypeError:
                # Numeric-only JAX coordinate callbacks retain the existing API.
                # Function-valued callbacks above adaptively resolve coefficients.
                action = op._apply_op(grid, proxy)
        except NonlocalPeriodicFunctional:
            from chebfunjax.operators._periodic_nonlocal import prepare, solve
            data = prepare(op, rhs)
            if data is None:
                raise NotImplementedError("trigspec nonlocal adapter requires a linear equation")
            return solve(data, backend="trigspec", n=n, n_min=n_min,
                         n_max=n_max, tol=tol)
        if not isinstance(action, _CoefficientProxy):
            raise TypeError('trigspec requires a linear differential operator')
        modes = jnp.arange(-(size//2), (size+1)//2)
        source_index = modes+len(rhs_coeffs)//2
        rhs_vector = jnp.where((source_index >= 0) & (source_index < len(rhs_coeffs)),
                               rhs_coeffs[jnp.clip(source_index, 0, len(rhs_coeffs)-1)], 0.)
        solution = jnp.linalg.solve(action.mat, rhs_vector)
        canonical = solution*jnp.where(modes % 2 == 0, 1., -1.)
        happy, _ = Trigtech.happiness_check(
            canonical, Trigtech.coeffs2vals(canonical), tol=tol)
        if n is not None or happy or size >= n_max:
            break
        size *= 2
    modes = jnp.arange(-(size//2), (size+1)//2)
    real = action.real and rhs_function.isreal()

    def evaluate(x):
        theta = 2*jnp.pi*(jnp.asarray(x)-a)/length
        out = jnp.exp(1j*theta[..., None]*modes) @ solution
        return jnp.real(out) if real else out

    return chebfun(evaluate, domain=(a, b), trig=True)
