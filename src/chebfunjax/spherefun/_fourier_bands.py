"""Isolated source sphere Fourier assembly in compact JAX bands.

Provenance: Chebfun7574c77680d7e82b79626300bf255498271a72df,
@spherefun/{poisson,helmholtz}.m, @trigspec/{diffmat,multmat}.m,
@coeffsDiscretization/sptoeplitz.m. No public solver is replaced here.
"""
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=('lower', 'upper'))
def band_apply(ab, rhs, *, lower, upper):
    """Multiply input bands by vector/matrix RHS; ignore reserved LU fill."""
    ab, rhs = jnp.asarray(ab), jnp.asarray(rhs)
    n = ab.shape[0]
    if ab.shape != (n, 2*lower+upper+1):
        raise ValueError('invalid row-band storage')
    if rhs.ndim not in (1, 2) or rhs.shape[0] != n:
        raise ValueError('invalid band RHS shape')
    vector = rhs.ndim == 1
    b = rhs[:, None] if vector else rhs
    offsets = jnp.arange(-lower, upper+1)
    columns = jnp.arange(n)[:, None]+offsets[None, :]
    valid = (columns >= 0) & (columns < n)
    values = b[jnp.clip(columns, 0, n-1)]
    terms = ab[:, :lower+upper+1, None]*values
    out = jnp.sum(jnp.where(valid[:, :, None], terms, 0), axis=1)
    return out[:, 0] if vector else out


@partial(jax.jit, static_argnames=('lower', 'upper'))
def band_dense(ab, *, lower, upper):
    """Dense reconstruction for bounded independent controls only."""
    ab = jnp.asarray(ab)
    n = ab.shape[0]
    if ab.shape != (n, 2*lower+upper+1):
        raise ValueError('invalid row-band storage')
    rows = jnp.arange(n)[:, None]
    cols = jnp.arange(n)[None, :]
    slots = cols-rows+lower
    valid = (slots >= 0) & (slots <= lower+upper)
    return jnp.where(valid, ab[rows, jnp.clip(slots, 0, ab.shape[1]-1)], 0)


@dataclass(frozen=True)
class BandMatrix:
    """Unfactored row storage compatible with the scratch pivoted solver."""
    ab: jax.Array
    lower: int
    upper: int

    def apply(self, rhs):
        return band_apply(self.ab, rhs, lower=self.lower, upper=self.upper)

    def dense(self):
        return band_dense(self.ab, lower=self.lower, upper=self.upper)

    def shifted(self, shift):
        return BandMatrix(self.ab.at[:, self.lower].add(shift),
                          self.lower, self.upper)


def _centered_coefficients(coeffs):
    coeffs = jnp.asarray(coeffs)
    if coeffs.ndim != 1 or coeffs.size < 1:
        raise ValueError('a nonempty Fourier coefficient vector is required')
    # Source multmat splits the existing even Nyquist term, without chopping.
    if coeffs.size % 2 == 0:
        coeffs = jnp.concatenate((coeffs[:1]/2, coeffs[1:], coeffs[:1]/2))
    return coeffs


def multiplier_bands(n, coeffs, *, radius=None):
    """Native Toeplitz multiplier, optionally padded to a common bandwidth."""
    if n < 1:
        raise ValueError('positive matrix size is required')
    coeffs = _centered_coefficients(coeffs)
    center = coeffs.size//2
    needed = min(n-1, center)
    radius = needed if radius is None else radius
    if radius < needed or radius > n-1:
        raise ValueError('common bandwidth must retain all physical bands')
    offsets = jnp.arange(-radius, radius+1)
    indices = center-offsets  # A[i,j] = coeff[center+i-j].
    diagonal = jnp.where((indices >= 0) & (indices < coeffs.size),
                         coeffs[jnp.clip(indices, 0, coeffs.size-1)], 0)
    columns = jnp.arange(n)[:, None]+offsets[None, :]
    core = jnp.where((columns >= 0) & (columns < n), diagonal[None, :], 0)
    # Solver allocates upper+lower live superdiagonals after pivoting.
    ab = jnp.pad(core, ((0, 0), (0, radius)))
    return BandMatrix(ab, radius, radius)


def differentiation_diagonal(n, order, *, nyquist=False):
    """Literal trigspec mode ordering, including sphere's odd Nyquist flag."""
    if n < 1 or order < 0:
        raise ValueError('positive size and nonnegative derivative order required')
    modes = jnp.arange(n, dtype=jnp.float64)-n//2
    if n % 2 == 0 and order % 2 == 1 and not nyquist:
        modes = modes.at[0].set(0)
    return (1j)**order * modes**order


def integration_weights(m):
    """Source en, retaining computed complex odd-mode terms except ±1."""
    if m < 3:
        raise ValueError('general sphere mean assembly requires latitude size >=3')
    zero = m//2
    modes = jnp.arange(m, dtype=jnp.float64)-zero
    weights = 2*jnp.pi*(1+jnp.exp(1j*jnp.pi*modes))/(1-modes**2)
    return weights.at[zero-1].set(0).at[zero+1].set(0)


@dataclass(frozen=True)
class SphereOperators:
    """Source operators/constraints only; zero-mode and public solves are external."""
    cs_coeffs: jax.Array
    sin2_coeffs: jax.Array
    dtheta1: jax.Array
    dtheta2: jax.Array
    dlambda2: jax.Array
    cs: BandMatrix
    sin2: BandMatrix
    laplace: BandMatrix
    weights: jax.Array
    zero_latitude: int
    zero_longitude: int

    def _rhs(self, coefficients):
        coefficients = jnp.asarray(coefficients)
        if coefficients.shape != (self.weights.size, self.dlambda2.size):
            raise ValueError('RHS must contain the full latitude/longitude coefficient matrix')
        return coefficients

    def poisson_rhs(self, coefficients):
        """Return weighted RHS, removed mean and literal zero constraint.

        The caller owns the native meanRHS warning and final additive constant.
        """
        coefficients = self._rhs(coefficients)
        mean = (self.weights @ coefficients[:, self.zero_longitude]
                / self.weights[self.zero_latitude])
        adjusted = coefficients.at[self.zero_latitude, self.zero_longitude].add(-mean)
        return self.sin2.apply(adjusted), mean, jnp.zeros_like(mean)

    def helmholtz(self, coefficients, k, c=1):
        """Return L, longitude shifts, weighted RHS and integral constraint.

        Precondition k!=0; native k==0 delegation/eigenvalue checks belong to
        the caller. Preserve the constraint computed from ORIGINAL coefficients.
        """
        coefficients = self._rhs(coefficients)
        scale = k**2
        integral = (self.weights @ coefficients[:, self.zero_longitude])/scale
        operator = BandMatrix(c*self.laplace.ab/scale+self.sin2.ab,
                              self.laplace.lower, self.laplace.upper)
        return (operator, c*self.dlambda2/scale,
                self.sin2.apply(coefficients)/scale, integral)

    def constraint_rows(self):
        """Source ii excludes only the latitude zero mode, retaining row order."""
        m = self.weights.size
        return jnp.concatenate((jnp.arange(self.zero_latitude),
                                jnp.arange(self.zero_latitude+1, m)))


def build_source_operators(m, n, *, cs_coeffs=None, sin2_coeffs=None):
    """Construct source-built multipliers without a dense matrix product.

    Coefficient injection supports independent controls; ordinary callers use
    the literal source Trigtech constructions, including their tiny odd bands.
    Native Helmholtz rounds m upward to even before this routine; Poisson does
    not. This helper never silently changes supplied dimensions.
    """
    if m < 3 or n < 1:
        raise ValueError('general sphere assembly requires m>=3 and n>=1')
    if cs_coeffs is None or sin2_coeffs is None:
        from chebfunjax.tech.trigtech import Trigtech
        if cs_coeffs is None:
            cs_coeffs = Trigtech.from_function(
                lambda t: jnp.sin(jnp.pi*t)*jnp.cos(jnp.pi*t)).coeffs
        if sin2_coeffs is None:
            sin2_coeffs = Trigtech.from_function(
                lambda t: jnp.sin(jnp.pi*t)**2).coeffs
    cs_coeffs, sin2_coeffs = jnp.asarray(cs_coeffs), jnp.asarray(sin2_coeffs)
    radius = min(m-1, max(_centered_coefficients(cs_coeffs).size//2,
                          _centered_coefficients(sin2_coeffs).size//2))
    cs = multiplier_bands(m, cs_coeffs, radius=radius)
    sin2 = multiplier_bands(m, sin2_coeffs, radius=radius)
    d1 = differentiation_diagonal(m, 1, nyquist=True)
    d2 = differentiation_diagonal(m, 2)
    dn = differentiation_diagonal(n, 2)
    columns = jnp.arange(m)[:, None]+jnp.arange(cs.ab.shape[1])[None, :]-radius
    valid = (columns >= 0) & (columns < m)
    # Multiplication on the RIGHT by a diagonal scales columns, not rows.
    lap = sin2.ab*d2[jnp.clip(columns, 0, m-1)] + cs.ab*d1[jnp.clip(columns, 0, m-1)]
    lap = jnp.where(valid, lap, 0)
    return SphereOperators(cs_coeffs, sin2_coeffs, d1, d2, dn, cs, sin2,
                           BandMatrix(lap, radius, radius), integration_weights(m),
                           m//2, n//2)
