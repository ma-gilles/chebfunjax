"""Structural extraction for a narrow first-order coupled native IVP route.

Provenance: Chebfun 7574c77680d7e82b79626300bf255498271a72df,
@chebop/{mldivide,solveivp}.m and @treeVar/sortConditions.m. This explicit
expression adapter covers unit diagonal derivatives and reordered unit endpoint
conditions, not the complete native treeVar grammar. Marching delegates to the
accepted vector Adams provider (R2025b source), not a new 2017 solver claim.
"""
from dataclasses import dataclass
from numbers import Real

import jax.numpy as jnp


class UnsupportedStructure(NotImplementedError):
    """Expression cannot be established structurally in the supported grammar."""


@dataclass(frozen=True)
class _Expr:
    value: object
    derivatives: tuple = ()
    variable: int | None = None
    constant: object = None
    breaks: tuple = ()
    state: bool = False

    @staticmethod
    def coerce(other):
        from chebfunjax.chebfun1d.chebfun import Chebfun
        if isinstance(other, _Expr):
            return other
        if isinstance(other, Real):
            return _Expr(lambda t, y: jnp.asarray(other), constant=other)
        if isinstance(other, Chebfun):
            if other.n_columns != 1 or other.is_transposed or not other.isreal():
                raise UnsupportedStructure('forcing must be a real scalar column')
            return _Expr(lambda t, y: jnp.asarray(other(t)).reshape(()),
                         breaks=tuple(float(v) for v in other.domain.breakpoints))
        raise UnsupportedStructure('unsupported expression operand')

    def __bool__(self):
        raise UnsupportedStructure('state-dependent Python branching is unsupported')

    def __call__(self, *args):
        raise UnsupportedStructure('nonlocal evaluation of an unknown is unsupported')

    def diff(self, order=1):
        if order == 0:
            return self
        if order != 1 or self.variable is None:
            raise UnsupportedStructure('only first derivatives of individual unknowns')
        return _Expr(lambda t, y: jnp.asarray(0.0), ((self.variable, 1),))

    def __add__(self, other):
        other = self.coerce(other)
        terms = dict(self.derivatives)
        for key, value in other.derivatives:
            terms[key] = terms.get(key, 0) + value
        constant = (self.constant + other.constant
                    if self.constant is not None and other.constant is not None else None)
        return _Expr(lambda t, y: self.value(t, y) + other.value(t, y),
                     tuple((k, v) for k, v in sorted(terms.items()) if v != 0),
                     constant=constant, breaks=self.breaks+other.breaks,
                     state=self.state or other.state)

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -self.coerce(other)

    def __rsub__(self, other):
        return self.coerce(other) + -self

    def __mul__(self, other):
        other = self.coerce(other)
        if self.derivatives or other.derivatives:
            if self.derivatives and other.constant is not None:
                terms = tuple((k, v*other.constant) for k, v in self.derivatives)
            elif other.derivatives and self.constant is not None:
                terms = tuple((k, v*self.constant) for k, v in other.derivatives)
            else:
                raise UnsupportedStructure('nonconstant or nonlinear derivative coefficient')
        else:
            terms = ()
        constant = (self.constant*other.constant
                    if self.constant is not None and other.constant is not None else None)
        return _Expr(lambda t, y: self.value(t, y)*other.value(t, y), terms,
                     constant=constant, breaks=self.breaks+other.breaks,
                     state=self.state or other.state)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.coerce(other)
        if other.derivatives or (self.derivatives and other.constant is None):
            raise UnsupportedStructure('derivative quotient is unsupported')
        terms = tuple((k, v/other.constant) for k, v in self.derivatives)
        return _Expr(lambda t, y: self.value(t, y)/other.value(t, y), terms,
                     breaks=self.breaks+other.breaks, state=self.state or other.state)

    def __rtruediv__(self, other):
        return self.coerce(other) / self

    def __pow__(self, other):
        other = self.coerce(other)
        if self.derivatives or other.derivatives:
            raise UnsupportedStructure('powers of derivatives are unsupported')
        return _Expr(lambda t, y: self.value(t, y)**other.value(t, y),
                     breaks=self.breaks+other.breaks, state=self.state or other.state)

    def _unary(self, fun):
        if self.derivatives:
            raise UnsupportedStructure('nonlinear derivative expression')
        return _Expr(lambda t, y: fun(self.value(t, y)), breaks=self.breaks,
                     state=self.state)

    def exp(self):
        return self._unary(jnp.exp)

    def sin(self):
        return self._unary(jnp.sin)

    def cos(self):
        return self._unary(jnp.cos)


def _entries(value):
    from chebfunjax.operators.chebmatrix import ChebMatrix
    if isinstance(value, ChebMatrix):
        if value.ncols != 1:
            raise UnsupportedStructure('operator must return a column')
        return [row[0] for row in value.blocks]
    return list(value) if isinstance(value, (tuple, list)) else [value]


@dataclass(frozen=True)
class Extracted:
    rhs: object
    initial: object
    span: tuple


def extract(op, forcing=0):
    """Build an RHS expression once; no finite-probe classification."""
    count = op._n_vars()
    if count < 2 or op._bc_general is not None or op._periodic:
        raise UnsupportedStructure('requires a nonperiodic coupled endpoint IVP')
    if (op._lbc_raw is None) == (op._rbc_raw is None):
        raise UnsupportedStructure('requires conditions at exactly one endpoint')
    dom = tuple(float(x) for x in op.domain)
    if not all(bool(jnp.isfinite(v)) for v in dom):
        raise UnsupportedStructure('finite domain required')
    variables = [_Expr(lambda t, y, k=k: y[k], variable=k, state=True)
                 for k in range(count)]
    time = _Expr(lambda t, y: t)
    rows = [_Expr.coerce(v) for v in _entries(op._call_op(time, variables))]
    if len(rows) != count or any(row.derivatives != ((k, 1),)
                                 for k, row in enumerate(rows)):
        raise UnsupportedStructure('requires one unit diagonal derivative per equation')
    forcing_rows = _entries(forcing)
    if len(forcing_rows) == 1:
        forcing_rows *= count
    if len(forcing_rows) != count:
        raise UnsupportedStructure('forcing must have one entry per equation')
    forces = [_Expr.coerce(v) for v in forcing_rows]
    forward = op._lbc_raw is not None
    point = dom[0] if forward else dom[-1]
    boundary = op._lbc_raw if forward else op._rbc_raw
    if callable(boundary):
        # State symbols represented as affine indeterminates, so nonlinear
        # or mixed conditions cannot masquerade as a reorderable unit row.
        symbols = [_Expr(lambda t, y: jnp.asarray(0.0), ((k, 1),))
                   for k in range(count)]
        conditions = [_Expr.coerce(v) for v in _entries(boundary(*symbols))]
        if len(conditions) != count:
            raise UnsupportedStructure('one endpoint condition per unknown required')
        initial = [None]*count
        for row in conditions:
            if len(row.derivatives) != 1 or row.derivatives[0][1] != 1:
                raise UnsupportedStructure('only reordered unit endpoint conditions')
            index = row.derivatives[0][0]
            if initial[index] is not None:
                raise UnsupportedStructure('duplicate endpoint condition')
            initial[index] = -row.value(point, jnp.zeros(count))
    else:
        initial = jnp.asarray(boundary).reshape(-1)
        if initial.size != count:
            raise UnsupportedStructure('one numeric endpoint value per unknown required')
    initial = jnp.asarray(initial)
    if jnp.iscomplexobj(initial):
        raise UnsupportedStructure('complex initial state remains on existing route')
    breaks = set(dom)
    for row in rows+forces:
        breaks.update(v for v in row.breaks if dom[0] < v < dom[-1])
    span = tuple(sorted(breaks, reverse=not forward))

    def rhs(t, y):
        return jnp.stack([force.value(t, y)-row.value(t, y)
                          for row, force in zip(rows, forces)])
    return Extracted(rhs, initial, span)


def prepare(op, forcing=0, *, selected=None):
    """Return a structural native plan, or None BEFORE selecting native mode.

    Default unsupported grammars keep their prior legacy route. An explicit
    ode113 request rejects unsupported grammar rather than silently using SciPy.
    Once this returns a plan, all native execution errors propagate.
    """
    from chebfunjax.chebpref import ChebopPref
    explicit = selected is not None or getattr(op, 'ivp_method', None) is not None
    method = selected if selected is not None else getattr(op, 'ivp_method', None)
    method = ChebopPref().ivpSolver if method is None else method
    name = str(method).strip().lower().lstrip('@').split('.')[-1]
    if name != 'ode113':
        return None
    if getattr(op, 'maxnorm', None) is not None:
        if explicit:
            raise UnsupportedStructure('native coupled ode113 events remain unsupported')
        return None
    try:
        return extract(op, forcing)
    except (UnsupportedStructure, TypeError, AttributeError):
        if explicit:
            raise
        return None


def solve(op, plan):
    from chebfunjax.chebfun1d.chebfun import ode113
    from chebfunjax.chebpref import ChebopPref
    from chebfunjax.operators.chebop import SystemSolution
    pref = ChebopPref()
    op._ivp_backend_used = 'native_ode113'
    options = {'RelTol': getattr(op, 'ivp_reltol', pref.ivpRelTol),
               'AbsTol': getattr(op, 'ivp_abstol', pref.ivpAbsTol),
               'restartSolver': getattr(op, 'ivp_restart_solver',
                                        getattr(pref, 'ivpRestartSolver', True)),
               'happinessCheck': pref.happinessCheck}
    array = ode113(plan.rhs, plan.span, plan.initial, options, backend='native')
    return SystemSolution([array.extract_columns(k) for k in range(op._n_vars())])
