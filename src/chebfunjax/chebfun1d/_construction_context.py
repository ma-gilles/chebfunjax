"""Private public-constructor ownership context, Chebfun7574c77.

Draft public wiring primitive. Source @chebfun/chebfun.m parseInputs/parseOp
and outer constructor. One normalized operator is retained for reconstruction
and breakpoint metadata; no global preference mutation or hidden reprobe.
"""
from __future__ import annotations

import copy
import inspect
import math
import warnings
from dataclasses import dataclass

import jax.numpy as jnp

from chebfunjax.chebfun1d._construction import (
    OMITTED,
    prepare_preferences,
    vector_check,
)
from chebfunjax.chebpref import ChebfunPref


@dataclass(frozen=True)
class ConstructionContext:
    op: object
    domain: tuple[float, ...]
    pref: ChebfunPref
    data: dict
    explicit_periodic: bool
    is_cell: bool
    double_length: bool
    truncate: object

    def interval(self, index):
        """Select a normalized cell/interval without repeating parseOp."""
        op = self.op[index] if self.is_cell else self.op
        return ConstructionContext(
            op, self.domain[index:index+2], ChebfunPref(self.pref),
            copy.deepcopy(self.data), self.explicit_periodic, False,
            False, None)

    def fixed_length(self, length):
        """Second doubleLength construction reuses op and fresh source data."""
        pref = ChebfunPref(self.pref)
        pref.fixedLength = length
        return ConstructionContext(
            self.op, self.domain, pref, copy.deepcopy(self.data),
            self.explicit_periodic, self.is_cell, False, None)

    @property
    def metadata_operator(self):
        # Source cells/numeric data use endpoint limits, including cells of
        # already-normalized callbacks. One callable is evaluated at all ends.
        return self.op if callable(self.op) and not self.is_cell else None


def prepare_context(op, domain=OMITTED, pref=None, *, keywords=None,
                    is_cell=None):
    """Parse private preferences and normalize each source operator once.

    Empty input and cell-of-FUN fast paths precede this helper at the public
    boundary. Python cell spelling is resolved explicitly by the caller.
    Numeric/coeff modifier dispatch remains owned by the construction layer.
    """
    from chebfunjax.chebfun1d.chebfun import Chebfun, _string_op

    options = dict(keywords or {})
    if options.get('resampling') and options.get('refinement_function') not in (None, 'resampling'):
        raise ValueError('resampling=True conflicts with refinement_function')
    # Source checks explicit equi's keyword fixedLength before pref merging;
    # a fixedLength existing only in pref does not satisfy this condition.
    numeric = not callable(op) and not isinstance(op, (str, Chebfun)) and not is_cell
    if options.get('equi') and not numeric and options.get('n') is None:
        raise ValueError("CHEBFUN:CHEBFUN:parseInputs:equi: 'equi' flag requires the number of points to be specified.")
    if options.get('coeffs') and (options.get('tech') is not None or options.get('chebkind') is not None):
        raise ValueError("CHEBFUN:CHEBFUN:parseInputs:coeffschebkind: 'coeffs' and 'chebkind' should not be specified simultaneously.")
    operand_domain = None
    if isinstance(op, Chebfun):
        operand_domain = (op.domain.a, op.domain.b)
    private, points, data, periodic = prepare_preferences(
        domain, pref, operand_domain=operand_domain, keywords=options)
    if is_cell is None:
        is_cell = isinstance(op, (list, tuple)) and bool(op) and (
            any(callable(value) or isinstance(value, str) for value in op)
            or (len(points) > 2 and len(op) == len(points) - 1
                and all(isinstance(value, (int, float)) or hasattr(value, '__len__')
                        for value in op)))
    if is_cell and len(op) != len(points) - 1:
        raise ValueError('CHEBFUN:CHEBFUN:constructor:cellInput: '
                         'Number of cell elements must match domain intervals.')
    for source, target in (('split_length', 'splitLength'),
                           ('split_max_length', 'splitMaxLength')):
        if options.get(source) is not None:
            private.splitPrefs[target] = options[source]
    data['coefficients'] = bool(options.get('coeffs'))

    def normalize(value):
        if isinstance(value, str):
            value = _string_op(value)
        if isinstance(value, Chebfun):
            # Native parseOp converts existing objects AFTER the handle's
            # vectorCheck branch; it does not infer output Tech from input.
            # Native parseOp uses conjugate transpose (op'), not op.'.
            original = value.conj().T if value.is_transposed else value

            def value(x):
                return original(x)
        elif callable(value) and options.get('vector_check', True):
            value = vector_check(value, points, bool(options.get('vectorize')))
        if callable(value) and private.enableFunqui:
            raw_length = private.fixedLength
            length = jnp.asarray([]) if raw_length is None else jnp.asarray(raw_length)
            if length.size and not bool(jnp.isnan(length.reshape(()))):
                value = value(jnp.linspace(points[0], points[-1], int(length.reshape(()))))
                private.fixedLength = jnp.nan
        return value

    if is_cell:
        normalized = tuple(normalize(value) for value in op)
    else:
        normalized = normalize(op)
    return ConstructionContext(normalized, points, private, data, periodic,
                               is_cell, bool(options.get('doubleLength')),
                               options.get('trunc'))


def selected_tech(pref):
    """Resolve supported public Tech names without an unknown-C2 fallback."""
    from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
    from chebfunjax.tech.trigtech import Trigtech

    selected = pref.tech
    if selected in (Chebtech1, Chebtech2, Trigtech):
        return selected
    name = str(selected).lower().lstrip('@')
    classes = {'chebtech': Chebtech2, 'chebtech1': Chebtech1,
               'chebtech2': Chebtech2, 'trigtech': Trigtech}
    if name not in classes:
        raise ValueError(f'Unsupported public construction Tech: {selected!r}')
    return classes[name]


def fixed_length(pref):
    """Native empty/NaN adaptive marker; Python size-one scalar adapter."""
    value = pref.fixedLength
    if value is None:
        return None
    value = jnp.asarray(value)
    if not value.size:
        return None
    if value.size != 1:
        raise ValueError('fixedLength must contain one scalar.')
    value = value.reshape(())
    if bool(jnp.isnan(value)):
        return None
    numeric = float(value)
    if not bool(jnp.isfinite(value)) or numeric < 0 or not numeric.is_integer():
        raise ValueError('fixedLength must be a nonnegative integer.')
    return int(numeric)


def bounded_get_fun(op, interval, data, pref):
    """Native constructor/getFun → bndfun → smoothfun → selected Tech.

    Chebfun7574c77: @chebfun/constructor.m264–290, @bndfun/bndfun.m93–104,
    @smoothfun/smoothfun.m43–63. This eager adapter owns shape/preference
    translation; existing Tech kernels retain their numerical algorithms.
    Singular and unbounded FUN dispatch is owned by the outer builder.
    Returns the represented piece, happiness and source updated global scale.
    """
    from chebfunjax.chebfun1d.chebfun import _Piece
    from chebfunjax.tech.chebtech import Chebtech2, _extrapolate_values
    from chebfunjax.tech.trigtech import Trigtech
    from chebfunjax.utils.interpolation import funqui
    from chebfunjax.utils.quadrature import chebpts

    left, right = interval
    cls = selected_tech(pref)
    local_data = copy.deepcopy(data)
    local_data['domain'] = interval
    if right - left < 4e-14 * data['hscale'] and callable(op):
        # Source getFun converts a narrow interval to numeric data before FUN.
        op = jnp.asarray(op((left + right) / 2))
        if op.ndim == 1:
            op = op[None, :]
    local_data['hscale'] = data['hscale'] / (right - left)
    if callable(op) and (left, right) != (-1., 1.):
        original = op

        def op(t):
            return original(right * (t + 1) / 2 + left * (1 - t) / 2)

    if pref.enableFunqui:
        # Native FUNQUI chooses one blending degree from a linear combination
        # of all columns. The existing matrix-capable helper preserves this.
        op = funqui(jnp.asarray(op))
    length = fixed_length(pref)
    if data.get('coefficients'):
        coefficients = jnp.atleast_1d(op)
        if cls is Trigtech:
            tech = cls.from_coeffs(coefficients, data=local_data, pref=pref.techPrefs)
        else:
            tech = cls.from_coeffs(coefficients)
            if length is not None:
                tech = tech.prolong(length)
    elif cls is Trigtech:
        # Preserve the full raw preference provenance at the accepted R2 core.
        if callable(op):
            tech = cls.from_function(op, data=local_data, pref=pref.techPrefs)
        else:
            tech = cls.from_values(jnp.atleast_1d(op), data=local_data,
                                   pref=pref.techPrefs)
    elif callable(op):
        options = dict(n=length, tol=pref.chebfuneps,
                       turbo=pref.techPrefs.get('useTurbo', False),
                       check=pref.happinessCheck, sample_test=pref.sampleTest,
                       refinement_function=pref.refinementFunction,
                       max_length=pref.maxLength, min_samples=pref.minSamples,
                       vscale=data['vscale'], hscale=local_data['hscale'])
        if cls is Chebtech2:
            options['extrapolate'] = pref.extrapolate
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=r'Chebtech[12]\.from_function: function did not converge with ')
            tech = cls.from_function(op, **options)
    else:
        values = jnp.atleast_1d(op)
        values = values.astype(jnp.complex128 if jnp.iscomplexobj(values) else jnp.float64)
        if not bool(jnp.all(jnp.isnan(values))) and not bool(jnp.all(jnp.isfinite(values))):
            kind = 2 if cls is Chebtech2 else 1
            values = _extrapolate_values(values, chebpts(values.shape[0], kind=kind),
                                         cls.barywts(values.shape[0]))[0]
        tech = cls.from_values(values)
        if length is not None:
            tech = tech.prolong(length)
    piece = _Piece(tech=tech, interval=interval)
    scale = data['vscale']
    if piece.ishappy:
        scale = jnp.max(jnp.concatenate((jnp.ravel(jnp.asarray(scale)),
                                         jnp.ravel(jnp.asarray(piece.vscale)))))
    return piece, piece.ishappy, scale


def construct_bounded(context):
    """Build bounded smooth FUNs; caller owns metadata, merge and truncation.

    Native @chebfun/constructor.m, pin7574c77. A private copy carries the
    split cap/extrapolation overrides; shared scale advances only after a
    happy fit. Existing split queue supplies the source widest-sad ordering.
    """
    from chebfunjax.chebfun1d.chebfun import Chebfun, _construct_with_splitting
    from chebfunjax.domain import Domain
    from chebfunjax.tech.trigtech import Trigtech

    if context.pref.splitting:
        return _construct_with_splitting(
            context.op, context.domain[0], context.domain[-1], 16,
            breakpoints=context.domain, _construction_context=context)
    funs = []
    data = copy.deepcopy(context.data)
    warning_thrown = False
    for index, interval in enumerate(zip(context.domain[:-1], context.domain[1:])):
        op = context.op[index] if context.is_cell else context.op
        # Native Tech constructors return happiness; the CHEBFUN layer emits
        # one unresolved warning for the complete no-split interval loop.
        piece, happy, data['vscale'] = bounded_get_fun(op, interval, data, context.pref)
        funs.append(piece)
        if not happy and not warning_thrown:
            remedy = ('a non-trig representation' if selected_tech(context.pref) is Trigtech
                      else "'splitting on'")
            warnings.warn(f'Function not resolved using {context.pref.maxLength} pts. '
                          f'Have you tried {remedy}?', stacklevel=2)
            warning_thrown = True
    return Chebfun(funs=funs, domain=Domain(context.domain))


def is_bounded_smooth(context, options):
    """Dispatch predicate only; excluded modifiers retain their own route."""
    return (all(math.isfinite(point) for point in context.domain)
            and not context.pref.blowup and options.get('exps') is None
            and options.get('singType') is None
            and not (options.get('coeffs') and context.pref.enableFunqui))


def public_construct(op, domain, pref, options):
    """Public source parser → constructor → metadata → merge → truncation.

    Native @chebfun/chebfun.m177–251, pin7574c77. Legacy singular/unbounded
    algorithms are retained by _chebfun_build, receiving the same normalized
    context rather than probing the operator again.
    """
    from chebfunjax.chebfun1d._construction import values_at_breakpoints
    from chebfunjax.chebfun1d.chebfun import Chebfun, _chebfun_build
    from chebfunjax.domain import Domain

    def store_metadata(result, operator=None):
        values = values_at_breakpoints(result.funs, result.domain.breakpoints, operator)
        # Existing Python scalar representations use rank-one breakpoint
        # metadata; array-valued representations retain the native matrix.
        if result.funs[0].tech.coeffs.ndim == 1 and values.shape[1:] == (1,):
            values = values[:, 0]
        object.__setattr__(result, '_point_values', values)

    # Native empty operand and cells of FUNs precede parseInputs/parseOp.
    if (op is None or (isinstance(op, Chebfun) and op.isempty())
            or (not callable(op) and getattr(op, 'size', None) == 0)
            or (not callable(op) and hasattr(op, '__len__')
                and getattr(op, 'ndim', 1) != 0 and len(op) == 0)):
        return Chebfun.empty()
    if isinstance(op, (list, tuple)) and op and all(
            hasattr(piece, 'tech') and hasattr(piece, 'interval') for piece in op):
        if domain is not OMITTED or pref is not None or options:
            raise ValueError('CHEBFUN:CHEBFUN:chebfun:nargin: '
                             'Only one input is allowed with an array of FUNs.')
        # Native unique([dom{:}]) sorts all endpoints, while FUN order stays
        # exactly as supplied. Do not invent adjacency validation here.
        ends = tuple(sorted({float(endpoint) for piece in op
                             for endpoint in piece.interval}))
        result = Chebfun(funs=list(op), domain=Domain(ends))
        store_metadata(result)
        return result
    # Parser validation follows the native empty and FUN-cell fast paths.
    inspect.signature(_chebfun_build).bind(op, domain=domain, **options)
    context = prepare_context(op, domain, pref, keywords=options)
    Domain(context.domain)  # Validate before constructing any FUNs.

    def build(current):
        if is_bounded_smooth(current, options):
            return construct_bounded(current)
        # This route retains inherited singular/unbounded/coefficient behavior.
        # The context suppresses parseOp and carries the private source prefs.
        legacy = dict(options)
        legacy.pop('doubleLength', None)
        legacy.pop('trunc', None)
        return _chebfun_build(current.op, domain=current.domain,
                             _construction_context=current, **legacy)

    result = build(context)
    if context.double_length:
        context_second = context.fixed_length(2 * result.funs[0].n - 1)
        result = build(context_second)
    if not result.funs:
        # Retain the existing canonical-empty adapter returned by legacy
        # singular/unbounded paths; retained-FUN parity there is separate.
        return result
    if getattr(result, '_point_values', None) is None:
        store_metadata(result, context.metadata_operator)
    if result.isempty():
        return result
    introduced = [point for point in result.domain.breakpoints[1:-1]
                  if point not in context.domain]
    if introduced:
        result = result.merge(index=introduced, _construction_pref=context.pref)
    if context.truncate:
        result = result.truncate(int(context.truncate))
    return result
