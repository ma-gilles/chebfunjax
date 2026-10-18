"""Polynomial pdeSolve spatial restart driver, JAX arithmetic.
Chebfun 7574c77 pdeSolve/chebdouble, MATLAB R2025b ode15s/daeic12.
Constant full mass, increasing time, scalar or real multicomponent handle BCs.
Fixed N follows pdeset. Real scalar numeric endpoint targets repeat per component.
No-BC unconstrained evolution is a Python compatibility extension.

Provenance
----------
Chebfun @chebfun/pdeSolve.m, @chebdouble/chebdouble.m and pdeset.m;
MATLAB R2025b ode15s.m and private/daeic12.m.
Chebfun native commit: 7574c77680d7e82b79626300bf255498271a72df.
Native ODE runtime/source release: MATLAB R2025b.
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.utils.diffmat import diffmat
from chebfunjax.utils.interpolation import barymat
from chebfunjax.utils.quadrature import chebpts_ab

from .adaptive import scalar_happiness, system_happiness
from .arity import _call_flexible as invoke
from .arity import _positional_arity
from .dae_init import dense_numjac, initialize_full_mass
from .linear_boundary import _linear_boundary_matrix, _split_endpoint_specs
from .ndf import startup
from .ndf_segment import segment
from .source_arithmetic import binary, matvec


class Values:
    __array_priority__ = 1000

    def __init__(self, vals, context, order=0, dependent=True):
        self.vals = jnp.asarray(vals)
        self.context = context
        self.order = order
        self.dependent = dependent

    def diff(self, k=1):
        if k not in self.context["D"]:
            self.context["D"][k] = diffmat(self.vals.size, p=k)
        if self.dependent:
            self.context["order"] = max(self.context["order"], self.order + k)
        return Values(
            self.context["scale"] ** k * matvec(self.context["D"][k], self.vals),
            self.context,
            self.order + k if self.dependent else 0,
            self.dependent,
        )

    def __call__(self, x):
        values = barymat(jnp.atleast_1d(x), self.context["x"]) @ self.vals
        return values[0] if jnp.ndim(x) == 0 else values

    def _bin(self, other, fn, operation=None):
        values = other.vals if isinstance(other, Values) else other
        order = max(self.order, other.order) if isinstance(other, Values) else self.order
        dependent = self.dependent or (isinstance(other, Values) and other.dependent)
        result = (
            fn(self.vals, values) if operation is None else binary(self.vals, values, operation)
        )
        return Values(result, self.context, order, dependent)

    def __add__(self, v):
        return self._bin(v, lambda a, b: a + b, 1)

    __radd__ = __add__

    def __sub__(self, v):
        return self._bin(v, lambda a, b: a - b, 2)

    def __rsub__(self, v):
        return self._bin(v, lambda a, b: b - a)

    def __mul__(self, v):
        return self._bin(v, lambda a, b: a * b, 0)

    __rmul__ = __mul__

    def __truediv__(self, v):
        return self._bin(v, lambda a, b: a / b, 3)

    def __rtruediv__(self, v):
        return self._bin(v, lambda a, b: b / a)

    def __pow__(self, v):
        return self._bin(v, lambda a, b: a**b)

    def __neg__(self):
        return Values(-self.vals, self.context, self.order, self.dependent)


def _invoke_system(fun, t, x, columns, *, role="pde", expected_count=None):
    """pdeSolve parseFun: independent input follows the callback's role."""
    arity = _positional_arity(fun)
    if arity == len(columns):
        result = fun(*columns)
    elif arity == len(columns) + 1:
        result = fun(x if role == "pde" else t, *columns)
    elif arity == len(columns) + 2:
        result = fun(t, x, *columns)
    else:
        raise NotImplementedError("System callbacks require N, N+1 or N+2 positional inputs")
    if role == "pde":
        expected_count = len(columns)
    if not isinstance(result, (tuple, list)):
        raise ValueError("System callbacks must return a tuple or list of expressions")
    if expected_count is not None and len(result) != expected_count:
        raise ValueError("System callback expression count changed or does not match its role")
    return result


def solve(
    pdefun,
    times,
    u0,
    *,
    lbc,
    rbc,
    rtol=1e-6,
    atol=1e-6,
    spatial_tol=1e-6,
    max_attempts=None,
    fixed_n=None,
    adjust_bcs=True,
    record_trace=True,
):
    times = jnp.asarray(times, dtype=jnp.float64)
    if times.ndim != 1 or times.size < 2 or not bool(jnp.all(jnp.diff(times) > 0)):
        raise ValueError("Increasing output times required")
    coefficients = u0.funs[0].tech.coeffs
    system = coefficients.ndim == 2
    size = coefficients.shape[1] if system else 1
    if system and jnp.iscomplexobj(coefficients):
        raise NotImplementedError("System NDF currently requires real components")
    if len(u0.funs) != 1:
        # pdeSolve first attempts merge(all, 1025, tol), then rejects pieces.
        u0 = u0.merge("all", max_length=1025, tol=spatial_tol)
        if len(u0.funs) != 1:
            raise ValueError(
                "CHEBFUN:CHEBFUN:pde15s:piecewise: Piecewise initial conditions are not supported."
            )
    domain = (float(u0.domain.a), float(u0.domain.b))
    if fixed_n is None:
        current = u0.simplify()
        length = max(current.funs[0].tech.coeffs.shape[0], 9)
    else:
        length = int(fixed_n)
        if length != fixed_n:
            raise ValueError("Fixed spatial size must be an integer")
        # pdeSolve fixed-N mode prolongs the stored tech, not sampled values.
        current = eqx.tree_at(lambda f: f.funs[0].tech, u0, u0.funs[0].tech.prolong(length))
    if length < 2:
        raise ValueError("At least two spatial points are required")
    current_time = float(times[0])
    outputs = [current]
    traces = []
    throw_bc_warning = adjust_bcs
    linear, nonlinear = _split_endpoint_specs(lbc, rbc, size, system)
    ordered = linear + nonlinear
    while current_time < float(times[-1]):
        n = length
        x = chebpts_ab(n, *domain)
        context = {
            "D": {},
            "x": x,
            "scale": jnp.asarray(2.0) / jnp.asarray(domain[1] - domain[0]),
            "order": 0,
        }
        # Traced public evaluation stays in JAX and preserves the stored polynomial.
        initial_values = jax.jit(lambda z: current(z))(x)
        coordinate = Values(x, context, dependent=False)

        def components(y):
            matrix = jnp.reshape(y, (n, size), order="F")
            return [Values(matrix[:, k], context) for k in range(size)]

        if system:
            initial_values = jnp.reshape(initial_values, (-1,), order="F")
            prototype = _invoke_system(pdefun, current_time, coordinate, components(initial_values))
            orders = [v.order if isinstance(v, Values) else 0 for v in prototype]
            order = sum(orders)
            boundary_counts = [
                len(
                    _invoke_system(
                        spec, current_time, coordinate, components(initial_values), role="boundary"
                    )
                )
                for _, spec in nonlinear
            ]
            if order != len(linear) + sum(boundary_counts):
                raise ValueError("System differential orders must match boundary residual count")
        else:
            invoke(pdefun, current_time, coordinate, Values(initial_values, context))
            order = context["order"]
        if not system and ordered and order != len(ordered):
            raise ValueError("Scalar differential order must equal number of boundary constraints")
        if system:
            blocks = [barymat(chebpts_ab(n - k + 2, *domain)[1:-1], x) for k in orders]
            projection = jnp.concatenate(
                (jnp.zeros((order, n * size)), jl.block_diag(*blocks)), axis=0
            )
        elif ordered:
            interior = chebpts_ab(n - order + 2, *domain)[1:-1]
            projection = jnp.concatenate((jnp.zeros((order, n)), barymat(interior, x)), axis=0)
        else:
            # Python compatibility: no supplied BC means unconstrained nodal
            # evolution, not periodic Chebfun semantics. No algebraic rows.
            order = 0
            projection = jnp.eye(n)
        mass = projection
        boundary_matrix, linear_rhs = _linear_boundary_matrix(
            linear, n, size, domain, initial_values.dtype
        )

        def raw(t, y):
            if system:
                columns = components(y)
                answer = _invoke_system(pdefun, t, coordinate, columns)
                matrix = jnp.column_stack(
                    [jnp.broadcast_to(v.vals if isinstance(v, Values) else v, (n,)) for v in answer]
                )
                rhs = matvec(projection, jnp.reshape(matrix, (-1,), order="F"))
                rhs = rhs.at[: len(linear)].set(matvec(boundary_matrix, y) - linear_rhs)
                row = len(linear)
                for (side, spec), count in zip(nonlinear, boundary_counts):
                    for value in _invoke_system(
                        spec, t, coordinate, columns, role="boundary", expected_count=count
                    ):
                        value = value.vals if isinstance(value, Values) else jnp.asarray(value)
                        rhs = rhs.at[row].set(value if value.ndim == 0 else value[side])
                        row += 1
                return rhs
            u = Values(y, context)
            answer = invoke(pdefun, t, coordinate, u)
            values = answer.vals if isinstance(answer, Values) else jnp.asarray(answer)
            rhs = matvec(projection, jnp.broadcast_to(values, y.shape))
            rhs = rhs.at[: len(linear)].set(matvec(boundary_matrix, y) - linear_rhs)
            for i, (side, spec) in enumerate(nonlinear, start=len(linear)):
                value = invoke(spec, t, coordinate, u)
                value = value.vals if isinstance(value, Values) else jnp.asarray(value)
                value = value if value.ndim == 0 else value[side]
                rhs = rhs.at[i].set(value)
            return rhs

        # pdeSolve.m: BCVALOFFSET = F(BCrows) - q.  q is nonzero only
        # for numeric linear boundary conditions, repeated per system component.
        boundary_rhs = jnp.asarray(
            jnp.concatenate(
                (linear_rhs, jnp.zeros(order - len(linear), dtype=initial_values.dtype))
            ),
            dtype=initial_values.dtype,
        )
        initial_raw = raw(current_time, initial_values)
        if (
            ordered
            and throw_bc_warning
            and bool(jnp.linalg.norm(initial_raw[:order]) > 0.05 * jnp.linalg.norm(initial_raw))
        ):
            warnings.warn(
                "CHEBFUN:CHEBFUN:pde15s:BadIC: Initial state may not satisfy the boundary conditions.",
                RuntimeWarning,
                stacklevel=2,
            )
            throw_bc_warning = False
        offset = (
            initial_raw[:order] - boundary_rhs
            if adjust_bcs else jnp.zeros_like(initial_raw[:order])
        )

        def fun(t, y):
            return raw(t, y).at[:order].add(-offset)

        if system or ordered:
            init = initialize_full_mass(
                lambda y: fun(current_time, y), mass, initial_values, reltol=rtol, abstol=atol
            )
        else:
            initial_rhs = fun(current_time, initial_values)
            jac, fac, _ = dense_numjac(
                lambda y: fun(current_time, y), initial_values, initial_rhs, atol
            )
            init = dict(y=initial_values, yp=initial_rhs, jac=jac, fac=fac, reason="ordinary_ode")
        remaining = times[times >= current_time]
        s = startup(
            init["y"],
            init["yp"],
            init["jac"],
            mass,
            t=current_time,
            rtol=rtol,
            threshold=atol / rtol,
            userhmax=float(times[-1] - current_time) / 10,
            htspan=float(remaining[1] - remaining[0]),
            dae=bool(system or ordered),
            rhs=fun,
        )
        s["tfinal"] = times[-1]
        s["jac_fac"] = init["fac"]
        s["jac_threshold"] = jnp.asarray(atol)
        accepted = []
        decisions = []

        def callback(t, value):
            nonlocal current, current_time, length
            if system:
                value = jnp.reshape(value, (n, size), order="F")
            happiness = system_happiness if system else scalar_happiness
            happy, cutoff = (True, None) if fixed_n is not None else happiness(value, spatial_tol)
            decisions.append((float(t), happy, cutoff))
            if not happy and fixed_n is None:
                length = 2 * length - 1
                return True
            # pdeSolve.m578-580/130/176 explicitly constructs nonperiodic
            # outputs with Chebtech2, matching this second-kind nodal grid.
            current = chebfun(value, domain=domain, chebkind=2)
            if fixed_n is None:
                current = current.simplify()
            current_time = float(t)
            outputs.append(current)
            accepted.append(float(t))
            return False

        final, rows = segment(
            s, fun, max_attempts, remaining[1:], callback, record_trace=record_trace
        )
        traces.append(
            {
                "n": n,
                "attempts": final["attempt_count"],
                "decisions": decisions,
                "accepted_times": accepted,
                "restart_time": current_time,
                "next_length": length,
                "initial_reason": init["reason"],
            }
        )
        if final.get("terminal_reason") == "IntegrationTolNotMet":
            return outputs, traces
        if not final.get("callback_stopped", False) and current_time < float(times[-1]):
            raise RuntimeError("NDF attempt limit reached")
    return outputs, traces
