"""Native periodic values-stack realization and Fourier conversion.

Provenance
----------
MATLAB source: @trigcolloc/{trigcolloc,sum,toFunctionOut}.m,
    @trigspec/{convertOperator,toFunctionOut,toValues}.m,
    @linop/linsolve.m, @opDiscretization/testConvergence.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
"""
import math
import warnings
from dataclasses import dataclass

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece, chebfun
from chebfunjax.discretization.trigcolloc import TrigColloc
from chebfunjax.domain import Domain
from chebfunjax.tech.trigtech import Trigtech


class NonlocalPeriodicFunctional(Exception):
    """Internal route marker raised only by legacy periodic proxy sum."""


class PeriodicDisc:
    """One native trigcolloc interval; no dimension adjustment/projection."""

    def __init__(self, n, domain):
        self.domain = tuple(float(x) for x in domain)
        if len(self.domain) != 2 or not all(map(math.isfinite, self.domain)):
            raise ValueError("Periodic Fourier discretization needs one finite interval")
        self.n = int(n)
        self.sizes = (self.n,)
        self.intervals = (self.domain,)
        self._disc = TrigColloc(self.n, self.domain)
        self.points = self._disc.points()

    def native_derivative(self, order):
        return self._disc.diffmat(order)

    def native_integral(self):
        return self._disc.weights()

    def evaluation(self, location, direction):
        raise NotImplementedError(
            "Periodic nonlocal evaluation functional is outside this native sum adapter")


def convert_operator(matrix):
    """Literal trigspec nonlocal conversion, including native cleanup quirks."""
    matrix = Trigtech.vals2coeffs(Trigtech.coeffs2vals(matrix.T).T)
    tolerance = jnp.max(jnp.abs(matrix))*1e-10
    if bool(jnp.max(jnp.abs(jnp.imag(matrix))) < 10*tolerance):
        matrix = jnp.real(matrix)
    elif bool(jnp.max(jnp.abs(jnp.real(matrix))) < 10*tolerance):
        # Pinned source intentionally removes i here.
        matrix = jnp.imag(matrix)
    return jnp.where(jnp.abs(matrix) < tolerance, 0, matrix)


def output_function(data, domain, backend, cutoff=None):
    """Literal scalar toFunctionOut loops over initial numel(data).

    Even input appends the WHOLE coefficient tail. Source then declares N+1
    and indexes its centered first N+1 entries, so only the first appended
    entry survives an unchopped conversion. Each subsequent values iteration
    still performs its FFT roundtrip. Coefficient iterations retain indexing
    without an FFT roundtrip. This intentionally preserves pinned quirks.
    """
    values = jnp.asarray(data).reshape(-1)
    iterations = values.size
    retained = math.inf if cutoff is None else int(cutoff)+(int(cutoff) % 2 == 0)
    for _ in range(iterations):
        coefficients = (Trigtech.vals2coeffs(values) if backend == 'trigcolloc'
                        else values)
        size = len(coefficients)
        if size % 2 == 0:
            coefficients = jnp.concatenate((.5*coefficients[:1], coefficients[1:],
                                             .5*coefficients))
            size += 1
        count = min(size, retained)
        center = (size+1)//2
        start = center-(count+1)//2
        coefficients = coefficients[start:start+count]
        values = (Trigtech.coeffs2vals(coefficients) if backend == 'trigcolloc'
                  else coefficients)
    tech = (Trigtech.from_values(values) if backend == 'trigcolloc'
            else Trigtech.from_coeffs(values))
    return Chebfun(funs=[_Piece(tech=tech, interval=tuple(domain))], domain=Domain(domain))


@dataclass
class PeriodicData:
    block: object
    rhs: Chebfun
    initial: Chebfun | None
    domain: tuple


def prepare(op, forcing):
    """Select supported exact scalar linear nonlocal blocks, without probes."""
    from chebfunjax.operators.blocks import OperatorBlock
    from chebfunjax.operators.chebop_altdisc import _linearize_scalar_ad

    domain = tuple(float(x) for x in op.domain)
    if not all(map(math.isfinite, domain)):
        return None
    if callable(forcing) and not isinstance(forcing, Chebfun):
        forcing = chebfun(forcing, domain=domain)
    data = _linearize_scalar_ad(op, forcing, domain)
    if data is None:
        return None
    operator, rhs, initial = data
    block = operator.A.blocks[0][0]
    if not isinstance(block, OperatorBlock):
        raise NotImplementedError("Periodic equation must return a function-valued operator")
    if block._coeff_fn is not None:
        return None
    capability = block._values_capability
    if capability is None:
        raise NotImplementedError("Periodic nonlocal block has no native values capability")
    if tuple(capability.domain) != domain:
        raise ValueError("Periodic Fourier discretization cannot contain breakpoints")
    # @chebop/solvebvp changes RHS and initial technology before linsolve.
    source_rhs = rhs[0]
    periodic_rhs = chebfun(lambda x: source_rhs(x) if callable(source_rhs)
                          else source_rhs, domain=domain, trig=True)
    periodic_initial = (None if op.init is None else
                        chebfun(lambda x: initial(x), domain=domain, trig=True))
    return PeriodicData(block, periodic_rhs, periodic_initial, domain)


def assemble(data, n, backend):
    descriptor = PeriodicDisc(n, data.domain)
    matrix = data.block._values_capability.realize(descriptor)
    rhs = data.rhs(descriptor.points)
    if backend == 'trigspec':
        matrix = convert_operator(matrix)
        rhs = Trigtech.vals2coeffs(rhs)
    elif backend != 'trigcolloc':
        raise ValueError("Unknown periodic discretization")
    return matrix, rhs


def real_if_small(result, tol):
    """Source solvebvp periodic real projection uses a strict continuous norm."""
    return result.real() if result.imag().norm(jnp.inf) < tol*result.vscale else result


def solve(data, *, backend, n=None, n_min=32, n_max=4096, tol=5e-13):
    """Native adaptive scalar Fourier solve in the requested space."""
    from chebfunjax.operators._linear_altdisc import _dimension_values

    sizes = ((int(n),) if n is not None else
             _dimension_values(n_min, n_max, 'ultraS' if backend == 'trigspec'
                               else 'chebcolloc1'))
    vscale = 0.
    for size in sizes:
        matrix, rhs = assemble(data, size, backend)
        if backend == 'trigcolloc':
            scale = 1/jnp.maximum(1, jnp.max(jnp.abs(matrix), axis=1))
            values = jnp.linalg.solve(scale[:, None]*matrix, scale*rhs)
        else:
            values = jnp.linalg.solve(matrix, rhs)
        provisional = output_function(values, data.domain, backend)
        tech = provisional.funs[0].tech
        vscale = max(vscale, float(provisional.vscale))
        happy, cutoff = Trigtech.happiness_check(
            tech.coeffs, Trigtech.coeffs2vals(tech.coeffs), tol=tol, vscale=vscale)
        if happy or n is not None:
            break
    if not happy and n is None:
        warnings.warn("CHEBFUN:LINOP:linsolve:noConverge -- Linear system solution "
                      "may not have converged.", RuntimeWarning, stacklevel=2)
    result = output_function(values, data.domain, backend, cutoff)
    result = result if data.initial is None else (data.initial+result).simplify()
    return real_if_small(result, tol)
