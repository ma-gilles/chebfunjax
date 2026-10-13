"""Dense tensor Chebyshev approximation in three dimensions.

Construction is adaptive and eager; evaluation supports JAX transformations.
Tucker conversion methods are Python compatibility extensions.

Provenance
----------
MATLAB source : @chebfun3t/constructor.m, feval.m, vals2coeffs.m,
    coeffs2vals.m, sum3.m, ndf.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

import math
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp

from chebfunjax.chebfun3d._chebfun3t_arithmetic import Chebfun3TArithmeticMixin
from chebfunjax.chebfun3d._chebfun3t_utilities import Chebfun3TUtilitiesMixin
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.domain import _linear_inverse_map
from chebfunjax.tech.chebtech import Chebtech2, _clenshaw
from chebfunjax.utils.quadrature import chebpts_ab
from chebfunjax.utils.transforms import _coeffs2vals_jax, _vals2coeffs_jax

__all__ = ["Chebfun3T", "chebfun3t"]
_UNIT = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)


def _transform(tensor, transform):
    # Source unfold/fold transforms in mode1, mode2, mode3 order.
    for axis in range(3):
        moved = jnp.moveaxis(tensor, axis, 0)
        shape = moved.shape
        tensor = jnp.moveaxis(
            transform(moved.reshape((shape[0], -1), order="F")).reshape(shape, order="F"), 0, axis
        )
    return tensor


@jax.jit
def _vals2coeffs(values):
    return _transform(values, _vals2coeffs_jax)


@jax.jit
def _coeffs2vals(coeffs):
    return _transform(coeffs, _coeffs2vals_jax)


def _txm(tensor, matrix, axis):
    # Native txm multiplies on the right for modes2/3, left for mode1.
    if axis == 0:
        shape = (matrix.shape[0],) + tensor.shape[1:]
        return (matrix @ tensor.reshape((tensor.shape[0], -1), order="F")).reshape(shape, order="F")
    moved = jnp.moveaxis(tensor, axis, -1)
    shape = moved.shape[:-1] + (matrix.shape[0],)
    out = moved.reshape((-1, moved.shape[-1]), order="F") @ matrix.T
    return jnp.moveaxis(out.reshape(shape, order="F"), -1, axis)


def _grid_eval(coeffs, x, y, z):
    matrices = [_clenshaw(jnp.eye(n), q) for n, q in zip(coeffs.shape, (x, y, z))]
    out = _txm(coeffs, matrices[2], 2)
    out = _txm(out, matrices[1], 1)
    return _txm(out, matrices[0], 0)


def _scattered_eval(coeffs, x, y, z):
    def point(q):
        a, b, c = [_clenshaw(jnp.eye(n), t) for n, t in zip(coeffs.shape, q)]
        # Same source vector-point contraction order, bounded workspace.
        reduced = coeffs.reshape((-1, coeffs.shape[2]), order="F") @ c
        return a @ (reduced.reshape(coeffs.shape[:2], order="F") @ b)

    return jax.lax.map(point, (x.reshape(-1), y.reshape(-1), z.reshape(-1))).reshape(x.shape)


def _evaluate(coeffs, domain, x, y, z):
    x, y, z = jnp.broadcast_arrays(jnp.asarray(x), jnp.asarray(y), jnp.asarray(z))
    mapped = []
    for q, a, b in zip((x, y, z), domain[::2], domain[1::2]):
        mapped.append(q if (a, b) == (-1.0, 1.0) else _linear_inverse_map(q, a, b))
    x, y, z = mapped
    if x.ndim != 3:
        return _scattered_eval(coeffs, x, y, z)
    nd = jnp.all(x == x[:, :1, :1]) & jnp.all(y == y[:1, :, :1]) & jnp.all(z == z[:1, :1, :])
    mesh = jnp.all(x == x[:1, :, :1]) & jnp.all(y == y[:, :1, :1]) & jnp.all(z == z[:1, :1, :])

    def ndgrid(_):
        return _grid_eval(coeffs, x[:, 0, 0], y[0, :, 0], z[0, 0, :])

    def meshgrid(_):
        return _grid_eval(coeffs, x[0, :, 0], y[:, 0, 0], z[0, 0, :]).transpose(1, 0, 2)

    return jax.lax.cond(
        nd,
        ndgrid,
        lambda _: jax.lax.cond(mesh, meshgrid, lambda _: _scattered_eval(coeffs, x, y, z), None),
        None,
    )


class Chebfun3T(Chebfun3TUtilitiesMixin, Chebfun3TArithmeticMixin, eqx.Module):
    """Full tensor of Chebyshev coefficients on a rectangular domain.

    Broadcasting of query arrays is a Python extension to native vector and
    structured tensor queries. Construction is eager; evaluation is JIT safe.

    Provenance
    ----------
    MATLAB source : @chebfun3t/chebfun3t.m, constructor.m, feval.m
    Chebfun commit: 7574c77
    """

    coeffs: jax.Array
    vscale: jax.Array
    domain: tuple = eqx.field(static=True)
    _tucker: object = None

    def __init__(
        self, coeffs=None, domain=_UNIT, vscale=None, *, core=None, cols=None, rows=None, tubes=None
    ):
        bridge = None
        if core is not None:
            bridge = Chebfun3(
                core=core, cols=list(cols), rows=list(rows), tubes=list(tubes), domain=tuple(domain)
            )
            coeffs = bridge.chebcoeffs3() if core.size else jnp.zeros((0, 0, 0))
        if coeffs is None:
            coeffs = jnp.zeros((0, 0, 0), dtype=jnp.float64)
        coeffs = jnp.asarray(coeffs)
        if coeffs.ndim > 3:
            raise ValueError("Chebfun3T coefficients must have at most three dimensions")
        coeffs = coeffs.reshape(coeffs.shape + (1,) * (3 - coeffs.ndim))
        self.coeffs = coeffs
        self.domain = tuple(float(v) for v in domain)
        self.vscale = jnp.asarray(
            (jnp.max(jnp.abs(_coeffs2vals(coeffs))) if coeffs.size else 0.0)
            if vscale is None
            else vscale
        )
        self._tucker = bridge

    @classmethod
    def from_coeffs(cls, coeffs, domain=_UNIT, vscale=None):
        """Store dense coefficients without approximation or truncation.

        Provenance
        ----------
        MATLAB source : @chebfun3t/chebfun3t.m (coefficient storage)
        Chebfun commit: 7574c77
        Python construction extension; omitted scale uses the tensor grid.
        """
        return cls(coeffs, domain, vscale)

    @classmethod
    def empty(cls):
        """Construct an empty dense tensor.

        Provenance
        ----------
        MATLAB source : @chebfun3t/chebfun3t.m
        Chebfun commit: 7574c77
        """
        return cls()

    def isempty(self):
        """Whether the coefficient tensor is empty.

        Provenance
        ----------
        MATLAB source : @chebfun3t/isempty.m
        Chebfun commit: 7574c77
        """
        return self.coeffs.size == 0

    def ndf(self):
        """Return the number of stored coefficients.

        Provenance
        ----------
        MATLAB source : @chebfun3t/ndf.m
        Chebfun commit: 7574c77
        """
        return self.coeffs.size

    @classmethod
    def from_chebfun3(cls, f):
        """Expand Tucker coefficients into the dense canonical representation.

        Provenance
        ----------
        MATLAB source : @chebfun3t/chebfun3t.m
        Chebfun commit: 7574c77
        """
        return cls(core=f.core, cols=f.cols, rows=f.rows, tubes=f.tubes, domain=f.domain)

    @classmethod
    def from_arrays(cls, core, cols, rows, tubes, domain=_UNIT):
        """Python extension: expand explicit Tucker data without approximation.

        Provenance
        ----------
        MATLAB source : @chebfun3t/chebfun3t.m
        Chebfun commit: 7574c77
        """
        for n, factors in zip(core.shape, (cols, rows, tubes)):
            if n != len(factors):
                raise ValueError("Tucker core and factor dimensions disagree")
        return cls(core=jnp.asarray(core), cols=cols, rows=rows, tubes=tubes, domain=domain)

    def to_chebfun3(self):
        """Python extension: expose an exact Tucker representation.

        Provenance
        ----------
        MATLAB source : @chebfun3t/chebfun3t.m
        Chebfun commit: 7574c77
        """
        if self._tucker is not None:
            return self._tucker
        factors = [
            [Chebtech2.from_coeffs(jnp.eye(n)[:, i]) for i in range(n)] for n in self.coeffs.shape
        ]
        return Chebfun3(
            core=self.coeffs, cols=factors[0], rows=factors[1], tubes=factors[2], domain=self.domain
        )

    @property
    def core(self):
        """Python compatibility view of the Tucker core.

        Provenance
        ----------
        MATLAB source : @chebfun3t/chebfun3t.m
        Chebfun commit: 7574c77
        """
        return self._tucker.core if self._tucker is not None else self.coeffs

    @property
    def cols(self):
        """Python compatibility view of the x basis factors.

        Provenance
        ----------
        MATLAB source : @chebfun3t/chebfun3t.m
        Chebfun commit: 7574c77
        """
        return self.to_chebfun3().cols

    @property
    def rows(self):
        """Python compatibility view of the y basis factors.

        Provenance
        ----------
        MATLAB source : @chebfun3t/chebfun3t.m
        Chebfun commit: 7574c77
        """
        return self.to_chebfun3().rows

    @property
    def tubes(self):
        """Python compatibility view of the z basis factors.

        Provenance
        ----------
        MATLAB source : @chebfun3t/chebfun3t.m
        Chebfun commit: 7574c77
        """
        return self.to_chebfun3().tubes

    @property
    def rank(self):
        """Factor counts of the compatibility representation, not minimal rank.

        Provenance
        ----------
        MATLAB source : @chebfun3t/chebfun3t.m
        Chebfun commit: 7574c77
        """
        return self._tucker.rank if self._tucker is not None else self.coeffs.shape

    def core_numpy(self):
        """Return the compatibility core as a host array.

        Provenance
        ----------
        MATLAB source : @chebfun3t/chebfun3t.m
        Chebfun commit: 7574c77
        """
        return jax.device_get(self.core)

    def col_coeffs(self):
        """Return host coefficient arrays for x compatibility factors.

        Provenance
        ----------
        MATLAB source : @chebfun3t/chebfun3t.m
        Chebfun commit: 7574c77
        """
        return [jax.device_get(f.coeffs) for f in self.cols]

    def row_coeffs(self):
        """Return host coefficient arrays for y compatibility factors.

        Provenance
        ----------
        MATLAB source : @chebfun3t/chebfun3t.m
        Chebfun commit: 7574c77
        """
        return [jax.device_get(f.coeffs) for f in self.rows]

    def tube_coeffs(self):
        """Return host coefficient arrays for z compatibility factors.

        Provenance
        ----------
        MATLAB source : @chebfun3t/chebfun3t.m
        Chebfun commit: 7574c77
        """
        return [jax.device_get(f.coeffs) for f in self.tubes]

    @eqx.filter_jit
    def __call__(self, x, y, z):
        """Evaluate the dense polynomial on broadcast query arrays.

        Provenance
        ----------
        MATLAB source : @chebfun3t/feval.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return jnp.array([])
        return _evaluate(self.coeffs, self.domain, x, y, z)

    def sum3(self):
        """Integrate over the physical box.

        Provenance
        ----------
        MATLAB source : @chebfun3t/sum3.m
        Chebfun commit: 7574c77
        """
        if self.isempty():
            return jnp.array([])
        weights = [2 / (1 - jnp.arange(0, n, 2, dtype=float) ** 2) for n in self.coeffs.shape]
        even = self.coeffs[:, :, ::2]
        out = even.reshape((-1, even.shape[2]), order="F") @ weights[2]
        out = out.reshape(even.shape[:2], order="F")[:, ::2] @ weights[1]
        out = out[::2] @ weights[0]
        scale = 0.5 * (self.domain[1] - self.domain[0])
        scale = scale * 0.5 * (self.domain[3] - self.domain[2])
        scale = scale * 0.5 * (self.domain[5] - self.domain[4])
        return out * scale

    def __repr__(self):
        """Describe dense coefficient dimensions and domain.

        Provenance
        ----------
        MATLAB source : @chebfun3t/disp.m
        Chebfun commit: 7574c77
        """
        return f"Chebfun3T(shape={self.coeffs.shape}, domain={self.domain})"


def _halton(domain):
    columns = []
    for base, a, b in zip((2, 3, 5), domain[::2], domain[1::2]):
        values = [0.0]
        digits = list(range(base))
        while len(values) <= 30:
            digits = [v / base for v in digits[: max(2, min(base, math.ceil(31 / len(values))))]]
            values = [v + d for d in digits for v in values]
        columns.append((b - a) * jnp.asarray(values[1:31]) + a)
    return columns


def _happy(coeffs, pref):
    happy, cutoffs = [], []
    for axis in range(3):
        c = jnp.sum(
            jnp.moveaxis(jnp.abs(coeffs), axis, 0).reshape((coeffs.shape[axis], -1), order="F"),
            axis=1,
        )
        h, k = Chebtech2.happiness_check(
            c,
            _coeffs2vals_jax(c),
            tol=pref.chebfuneps,
            check=pref.happinessCheck,
            sample_test=False,
        )
        happy.append(h)
        cutoffs.append(k)
    return happy, cutoffs


def _sample_bound(points, values, domain, eps):
    derivatives = []
    for axis in range(3):
        slices = tuple(slice(None) if k == axis else slice(None, -1) for k in range(3))
        spacing = jnp.diff(points[axis]).reshape(tuple(-1 if k == axis else 1 for k in range(3)))
        derivatives.append(jnp.max(jnp.abs(jnp.diff(values[slices], axis=axis) / spacing)))
    return (
        max(values.shape) ** 0.8
        * max(map(abs, domain))
        * jnp.maximum(max(derivatives), jnp.max(jnp.abs(values)))
        * eps
    )


def chebfun3t(
    f_or_chebfun3=None,
    domain=_UNIT,
    *,
    tol=None,
    min_samples=None,
    sample_test=None,
    pref=None,
    **kwargs,
):
    """Construct a dense adaptive tensor using native two-phase refinement.

    ``tol`` and ``min_samples`` override native chebfuneps and minSamples.
    No low-rank approximation or rank cutoff participates in this constructor.

    Provenance
    ----------
    MATLAB source : @chebfun3t/constructor.m, sampleTest.m
    Chebfun commit: 7574c77
    """
    op = f_or_chebfun3
    if op is None:
        return Chebfun3T.empty()
    if isinstance(op, Chebfun3T):
        return op
    if isinstance(op, Chebfun3):
        return Chebfun3T.from_chebfun3(op)
    # Native front door returns empty before parsing options or domain.
    # Inspect numeric shape metadata only; never call a supplied operator.
    if not callable(op):
        empty = (isinstance(op, str) and not op) or (isinstance(op, (list, tuple)) and len(op) == 0)
        if not isinstance(op, str) and hasattr(op, "shape"):
            empty = empty or math.prod(op.shape) == 0
        if empty:
            return Chebfun3T.empty()
    if kwargs:
        raise TypeError(f"Unsupported dense constructor options: {sorted(kwargs)}")
    if isinstance(op, str):
        from chebfunjax.utils.matlab_expr import matlab_expression

        op = matlab_expression(op, ("x", "y", "z"))
    if not callable(op):
        raise TypeError("Chebfun3T expects a callable, expression, or Chebfun3")
    domain = tuple(float(v) for v in domain)
    if (
        len(domain) != 6
        or not all(map(math.isfinite, domain))
        or any(a >= b for a, b in zip(domain[::2], domain[1::2]))
    ):
        raise ValueError("Chebfun3T requires three increasing finite intervals")
    pref = ChebfunPref() if pref is None else ChebfunPref(pref)
    if str(pref.tech).lower().lstrip("@") != "chebtech2":
        raise NotImplementedError("Dense Chebfun3T currently supports chebtech2")
    if tol is not None:
        pref.chebfuneps = tol
    grid = int(pref.minSamples if min_samples is None else min_samples)
    sample_test = pref.cheb2Prefs.sampleTest if sample_test is None else sample_test
    pseudo = pref.cheb3Prefs.chebfun3eps
    dimensions = [grid] * 3
    result = None
    for iteration in range(10):
        shape = list(dimensions)
        first = True
        while True:
            points = [chebpts_ab(n, a, b) for n, a, b in zip(shape, domain[::2], domain[1::2])]
            x, y, z = jnp.meshgrid(*points, indexing="ij")
            values = jnp.broadcast_to(jnp.asarray(op(x, y, z)), tuple(shape))
            if first:
                if bool(jnp.any(jnp.isinf(values))):
                    raise ValueError("Function returned INF when evaluated")
                if bool(jnp.any(jnp.isnan(values))):
                    raise ValueError("Function returned NaN when evaluated")
            coeffs = _vals2coeffs(values)
            happy, cutoff = _happy(coeffs, pref)
            if all(happy) or not any(happy):
                break
            shape = [n if h else 2 * n - 1 for n, h in zip(shape, happy)]
            first = False
        if all(happy):
            cropped = coeffs[tuple(slice(0, k) for k in cutoff)]
            result = Chebfun3T.from_coeffs(cropped, domain, jnp.max(jnp.abs(values)))
            if sample_test:
                tolerance = _sample_bound(points, values, domain, pseudo)
                q = _halton(domain)
                if bool(jnp.max(jnp.abs(op(*q) - result(*q))) <= 100 * tolerance):
                    return result
        dimensions = [
            math.floor(math.sqrt(2) ** (math.floor(2 * math.log2(n)) + 1)) + 1 for n in dimensions
        ]
    if not all(happy):
        warnings.warn(
            "Unresolved with maximum CHEBFUN3T length: 363.", RuntimeWarning, stacklevel=2
        )
        result = Chebfun3T.from_coeffs(coeffs, domain, jnp.max(jnp.abs(values)))
    return result
