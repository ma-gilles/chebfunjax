"""Numeric-matrix Chebfun2 construction using source-ordered JAX arithmetic.

Rank and representation metadata are selected eagerly; this is not an outer-jit
constructor. The callable adaptive constructor shares the completeACA kernel.

Provenance
----------
MATLAB source : @chebfun2/constructor.m (constructFromDouble/getTol/completeACA)
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
import jax.numpy as jnp

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech
from chebfunjax.utils._trigpts import global_trigpts_nodes, map_global_nodes
from chebfunjax.utils.misc import standard_chop
from chebfunjax.utils.quadrature import chebpts
from chebfunjax.utils.transforms import _vals2coeffs_jax


def _source_inf_magnitude(value):
    """MATLAB magnitude infinity precedence for complex Inf/NaN components.

    Provenance
    ----------
    MATLAB source : @separableApprox/cdr.m, @chebfun2/constructor.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    Source tests abs(d)==Inf; either infinite component dominates NaN in
    MATLAB magnitude. JAX complex abs may instead propagate NaN.
    """
    return (jnp.isinf(jnp.abs(value)) | jnp.isinf(jnp.real(value))
            | jnp.isinf(jnp.imag(value)))


def numeric_points(n, interval, tech):
    """Use global source grids for numeric factor data.

    Provenance
    ----------
    MATLAB source : @chebfun2/constructor.m (myPoints), trigpts.m
    Chebfun commit: 7574c77
    """
    a, b = interval
    if tech == "trig":
        return map_global_nodes(global_trigpts_nodes(n), a, b)
    nodes = chebpts(n, kind=1 if tech == "cheb1" else 2)
    if a == -1 and b == 1:
        return nodes
    return b*(nodes + 1)/2 + a*(1 - nodes)/2


def numeric_tolerances(x, y, values, domain, pseudo_level):
    """Return source relative and absolute gradient-aware tolerances.

    Provenance
    ----------
    MATLAB source : @chebfun2/constructor.m (getTol)
    Chebfun commit: 7574c77
    """
    m, n = values.shape
    dx = dy = jnp.zeros((1,), dtype=jnp.float64)
    if m > 1 and n > 1:
        dx = jnp.diff(values[:-1, :], axis=1)/jnp.diff(x)[None, :]
        dy = jnp.diff(values[:, :-1], axis=0)/jnp.diff(y)[:, None]
    elif m > 1:
        dy = jnp.diff(values, axis=0)/jnp.diff(y)[:, None]
    elif n > 1:
        dx = jnp.diff(values, axis=1)/jnp.diff(x)[None, :]
    # MATLAB max omits NaNs; binary max likewise keeps the finite operand.
    gradient = jnp.nanmax(jnp.fmax(jnp.abs(dx).reshape(-1),
                                    jnp.abs(dy).reshape(-1)))
    scale = jnp.nanmax(jnp.abs(values))
    relative = jnp.asarray(max(m, n), dtype=jnp.float64)**(2/3)*pseudo_level
    absolute = jnp.max(jnp.abs(jnp.asarray(domain)))*jnp.fmax(gradient, scale)*relative
    return relative, absolute


def _stack_in_blocks(items, axis=0):
    """Assemble homogeneous ACA arrays with bounded JAX stack input arity.

    Only grouping changes: source array order, dtype, axes, and all pivot
    arithmetic are unchanged. Blocks contain at most32 inputs to avoid a
    single high-arity backend compilation for full-rank numeric matrices.
    """
    blocks = [jnp.stack(items[start:start + 32], axis=axis)
              for start in range(0, len(items), 32)]
    return blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks, axis=axis)


def numeric_aca(values, abs_tol, factor=0):
    """Source completeACA, optionally with its adaptive pivot budget.

    Provenance
    ----------
    MATLAB source : @chebfun2/constructor.m (completeACA/myind2sub)
    Chebfun commit: 7574c77
    Selected diagonal norm controls the next loop, matching the source.
    """
    residual = jnp.asarray(values)
    ny, nx = residual.shape

    def choose(a):
        magnitudes = jnp.abs(a)
        selection = jnp.where(jnp.isnan(magnitudes), -jnp.inf, magnitudes)
        index = int(jnp.argmax(selection.T.reshape(-1)))
        row, col = index % ny, index // ny
        norm = magnitudes[row, col]
        if ny == nx:
            diagonal = jnp.diag(magnitudes)
            diagonal_index = int(jnp.argmax(jnp.where(jnp.isnan(diagonal),
                                                       -jnp.inf, diagonal)))
            if bool(diagonal[diagonal_index] - norm > -abs_tol):
                row = col = diagonal_index
                norm = diagonal[diagonal_index]
        return row, col, norm

    row, col, norm = choose(residual)
    failed = not bool(norm == 0)
    pivots, positions, rows, cols = [], [], [], []
    budget = min(ny, nx)/factor if factor else float("inf")
    while (bool(norm > abs_tol) and len(pivots) < budget
           and len(pivots) < min(ny, nx)):
        r, c = residual[row, :], residual[:, col]
        pivot = residual[row, col]
        rows.append(r)
        cols.append(c)
        pivots.append(pivot)
        positions.append((row, col))
        # Source divides the row before forming the nonconjugating outer product.
        residual = residual - c[:, None]*(r/pivot)[None, :]
        row, col, norm = choose(residual)
    if bool(norm <= abs_tol):
        failed = False
    if len(pivots) >= budget:
        failed = True
    if not pivots:
        return (jnp.zeros((1,), dtype=jnp.float64), ((0, 0),),
                jnp.zeros((1, nx), dtype=jnp.float64),
                jnp.zeros((ny, 1), dtype=jnp.float64), failed)
    return (_stack_in_blocks(pivots), tuple(positions), _stack_in_blocks(rows),
            _stack_in_blocks(cols, axis=1), failed)


def numeric_factors(values, tech, tolerance, chop=False):
    """Transform whole nonadaptive factor matrices before splitting columns.

    Provenance
    ----------
    MATLAB source : @chebfun2/constructor.m (constructFromDouble),
        @trigtech/populate.m, @chebtech/populate.m
    Chebfun commit: 7574c77
    Finite data retains sample length; nonfinite Cheb rows are extrapolated.
    """
    if tech == "trig":
        joint = Trigtech.from_values(values)
        scale = jnp.max(jnp.abs(values), axis=0)
        flags = jnp.max(jnp.abs(jnp.imag(values)), axis=0) <= (
            3*jnp.finfo(jnp.float64).eps*scale)
        return [Trigtech(coeffs=joint.coeffs[:, k], is_real=bool(flags[k]),
                         ishappy=True) for k in range(values.shape[1])]
    if tech == "cheb1":
        values = Chebtech1.extrapolate(values)
        coefficients = Chebtech1.vals2coeffs(values)
        return [Chebtech1.from_coeffs(
            coefficients[:standard_chop(coefficients[:, k], tolerance), k]
            if chop and bool(jnp.max(jnp.abs(values[:, k])) > 0)
            else coefficients[:, k]) for k in range(values.shape[1])]
    values = extrapolate_cheb_values(values)
    coefficients = _vals2coeffs_jax(values)
    result = []
    for k in range(values.shape[1]):
        c = coefficients[:, k]
        if chop and bool(jnp.max(jnp.abs(values[:, k])) > 0):
            c = c[:standard_chop(c, tolerance)]
        result.append(Chebtech2.from_coeffs(c))
    return result


def numeric_cdr(values, domain, tolerance, techs, chop=False, *,
                equi=(False, False), preferences=None):
    """Prepare ordinary SeparableApprox factors from numeric values.

    Provenance
    ----------
    MATLAB source : @chebfun2/constructor.m (constructFromDouble),
        @separableApprox/cdr.m
    Chebfun commit: 7574c77
    Python stores inverse CDR weights and retains the original raw pivot values.
    """
    equi = tuple(bool(flag and tech != "trig") for flag, tech in zip(equi, techs))
    def points(n, interval, tech, is_equi):
        if not is_equi:
            return numeric_points(n, interval, tech)
        if n == 1:
            return jnp.asarray([interval[1]], dtype=jnp.float64)
        from chebfunjax.utils._matlab_linspace import source_grid
        return source_grid(interval[0], interval[1], n)
    x = points(values.shape[1], domain[:2], techs[0], equi[0])
    y = points(values.shape[0], domain[2:], techs[1], equi[1])
    relative, absolute = numeric_tolerances(x, y, values, domain, tolerance)
    pivots, _, rows, cols, _ = numeric_aca(values, absolute)
    inverse = 1/pivots
    inverse = jnp.where(_source_inf_magnitude(inverse), 0, inverse)
    def factors(data, axis):
        if not equi[axis]:
            return numeric_factors(data, techs[axis], tolerance, chop)
        return _equi_factors(data, techs[axis], relative,
                             domain[2*axis:2*axis+2], preferences[axis])
    return dict(cols=factors(cols, 1), rows=factors(rows.T, 0), pivots=inverse,
                domain=domain, techs=techs, pivot_locations=(),
                pivot_values=pivots)


def extrapolate_cheb_values(values):
    """Replace missing numeric Chebyshev rows by source barycentric extrapolation.

    Provenance
    ----------
    MATLAB source : @chebtech/{populate,extrapolate}.m, @chebtech2/barywts.m
    Chebfun commit: 7574c77
    A missing entry masks the whole row across all factor columns.
    Reuses the qualified shared JAX compensated barycentric extrapolator.
    """
    if bool(jnp.all(jnp.isnan(values))):
        return values
    return Chebtech2.extrapolate(values)


def scalar_cdr(value, domain):
    """Source constant-callable GE and final degree-zero factor representation.

    Provenance
    ----------
    MATLAB source : @chebfun2/constructor.m (scalar constructFromDouble recursion)
    Chebfun commit: 7574c77
    This is the algebraic constant specialization of that recursion: both
    selected slices equal the scalar. Source recursion resets tech/prefs;
    numeric scalar trig input therefore returns default Chebtech2 factors.
    """
    value = jnp.asarray(value).reshape(())
    if bool(_source_inf_magnitude(value)):
        raise ValueError("CHEBFUN:CHEBFUN2:constructor:inf")
    if bool(jnp.isnan(value)):
        raise ValueError("CHEBFUN:CHEBFUN2:constructor:nan")
    factor = Chebtech2.from_coeffs(value[None])
    inverse = 1/value
    inverse = jnp.where(_source_inf_magnitude(inverse), 0, inverse)
    # Scalar numeric input recurses through the callable constructor. Its
    # zero-factor branch sets pivotValue=Inf, unlike zero numeric-matrix ACA.
    raw = jnp.asarray([jnp.inf]) if bool(value == 0) else value[None]
    # Source recursion uses its first diagonal pivot for a nonzero constant;
    # its exact-zero branch explicitly replaces the location by [0, 0].
    locations = ((0.0, 0.0),) if bool(value == 0) else ((domain[0], domain[2]),)
    return dict(cols=[factor], rows=[factor], pivots=inverse[None],
                pivot_values=raw, domain=domain, techs=("cheb", "cheb"),
                pivot_locations=locations)


def _equi_factors(values, tech, tolerance, interval, preference):
    """Literal numeric EQUI array-valued FUNQUI followed by selected tech.

    Source: @chebfun2/constructor.m constructFromDouble and
    @smoothfun/smoothfun.m funqui, pin7574c77. Existing rational/interpolation
    and adaptive-tech kernels are reused; their provider limits remain.
    """
    from chebfunjax.utils.interpolation import funqui
    prefs = preference.techPrefs
    tech_class = Chebtech1 if tech == "cheb1" else Chebtech2
    kwargs = dict(
        tol=float(tolerance), hscale=max(abs(float(x)) for x in interval),
        max_length=int(prefs.maxLength), min_samples=int(prefs.minSamples),
        sample_test=bool(prefs.sampleTest), check=prefs.happinessCheck,
        refinement_function=prefs.refinementFunction)
    fixed = prefs.get("fixedLength", float("nan"))
    if fixed is not None and bool(jnp.isfinite(fixed)):
        kwargs["n"] = int(fixed)
    joint = tech_class.from_function(funqui(values), **kwargs)
    coeffs = joint.coeffs
    if coeffs.ndim == 1:
        return [joint]
    return [tech_class(coeffs=coeffs[:, j], ishappy=joint.ishappy)
            for j in range(coeffs.shape[1])]
