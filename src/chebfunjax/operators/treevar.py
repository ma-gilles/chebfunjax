"""TreeVar: syntax-tree analysis of ODEs for first-order reformulation.

The TreeVar class lets chebfunjax analyse the syntax trees of ODE
operators.  Evaluating a chebop's ``op`` function with TreeVar arguments
records the mathematical expression as a tree, which is then expanded
(distributing products over sums so the highest-order derivative stands
alone), split into derivative and non-derivative parts, and converted to
a first-order system ``u' = f(t, u)`` suitable for time-stepping IVP
solvers.  MATLAB builds an anonymous function through infix strings and
``eval``; here the tree is compiled to nested closures directly, which
is semantically identical.

Provenance
----------
MATLAB source : @treeVar/treeVar.m, bivariate.m, univariate.m,
    expandTree.m, splitTree.m, toFirstOrder.m, toRHS.m, tree2infix.m,
    sortConditions.m, printTree.m, plotTree.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.
"""

from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np  # uses-numpy: tree bookkeeping on concrete ID vectors


class TreeVarError(ValueError):
    """Error raised during syntax-tree analysis, carrying the MATLAB
    error identifier in ``identifier``."""

    def __init__(self, identifier: str, message: str):
        super().__init__(message)
        self.identifier = identifier


def _tree(method, numArgs, **kw):
    d = {"method": method, "numArgs": numArgs}
    d.update(kw)
    return d


def _is_tree(t) -> bool:
    return isinstance(t, dict) and "method" in t


_UNIVARIATE = [
    "abs", "acos", "acosd", "acot", "acoth", "acsc", "acscd", "acsch",
    "airy", "asec", "asecd", "asech", "asin", "asind", "asinh", "atan",
    "atand", "atanh", "conj", "cos", "cosd", "cosh", "cot", "cotd",
    "coth", "csc", "cscd", "csch", "exp", "expm1", "imag", "log",
    "log10", "log1p", "log2", "pow2", "real", "sec", "secd", "sech",
    "sin", "sind", "sinh", "sqrt", "tan", "tand", "tanh", "uminus",
    "uplus",
]

_DEG = np.pi / 180.0


def _airy(x):
    import scipy.special  # uses-numpy: concrete special-function eval
    return scipy.special.airy(x)[0]


_UNARY_FNS_RAW = {
    "abs": np.abs, "acos": np.arccos,
    "acosd": lambda x: np.arccos(x) / _DEG,
    "acot": lambda x: np.arctan(1.0 / x),
    "acoth": lambda x: np.arctanh(1.0 / x),
    "acsc": lambda x: np.arcsin(1.0 / x),
    "acscd": lambda x: np.arcsin(1.0 / x) / _DEG,
    "acsch": lambda x: np.arcsinh(1.0 / x),
    "airy": _airy,
    "asec": lambda x: np.arccos(1.0 / x),
    "asecd": lambda x: np.arccos(1.0 / x) / _DEG,
    "asech": lambda x: np.arccosh(1.0 / x),
    "asin": np.arcsin, "asind": lambda x: np.arcsin(x) / _DEG,
    "asinh": np.arcsinh, "atan": np.arctan,
    "atand": lambda x: np.arctan(x) / _DEG,
    "atanh": np.arctanh, "conj": np.conj, "cos": np.cos,
    "cosd": lambda x: np.cos(_DEG * x), "cosh": np.cosh,
    "cot": lambda x: 1.0 / np.tan(x),
    "cotd": lambda x: 1.0 / np.tan(_DEG * x),
    "coth": lambda x: 1.0 / np.tanh(x),
    "csc": lambda x: 1.0 / np.sin(x),
    "cscd": lambda x: 1.0 / np.sin(_DEG * x),
    "csch": lambda x: 1.0 / np.sinh(x),
    "exp": np.exp, "expm1": np.expm1, "imag": np.imag,
    "log": np.log, "log10": np.log10, "log1p": np.log1p,
    "log2": np.log2, "pow2": lambda x: np.exp2(x), "real": np.real,
    "sec": lambda x: 1.0 / np.cos(x),
    "secd": lambda x: 1.0 / np.cos(_DEG * x),
    "sech": lambda x: 1.0 / np.cosh(x),
    "sin": np.sin, "sind": lambda x: np.sin(_DEG * x),
    "sinh": np.sinh, "sqrt": np.sqrt, "tan": np.tan,
    "tand": lambda x: np.tan(_DEG * x), "tanh": np.tanh,
    "uminus": lambda x: -x, "uplus": lambda x: +x,
}


def _complex_fallback(fn):
    """MATLAB's elementary functions return complex values off the real
    domain (acoth(0.4), sqrt(-1), ...); numpy returns nan.  Retry with a
    complex argument so the recorded tree evaluates MATLAB-faithfully."""
    def wrapped(x):
        with np.errstate(all="ignore"):
            r = fn(x)
        if np.any(np.isnan(r)) and not np.any(np.isnan(x)):
            with np.errstate(all="ignore"):
                r = fn(np.asarray(x, dtype=complex))
        return r
    return wrapped


_UNARY_FNS = {name: _complex_fallback(fn)
              for name, fn in _UNARY_FNS_RAW.items()}

_BINARY_FNS = {
    "plus": lambda a, b: a + b,
    "minus": lambda a, b: a - b,
    "times": lambda a, b: a * b,
    "rdivide": lambda a, b: a / b,
    "power": lambda a, b: a ** b,
}


class TreeVar:
    """A variable recording the syntax tree of ODE expressions.

    Parameters
    ----------
    id_vec : sequence of bool/int, default (1,)
        Which base variable this TreeVar represents.
    domain : tuple of float, default (-1, 1)
        Problem domain.

    Provenance
    ----------
    MATLAB source : @treeVar/treeVar.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """

    def __init__(self, id_vec=(1,), domain=(-1.0, 1.0)):
        self.domain = tuple(float(v) for v in domain)
        idv = np.asarray(id_vec, dtype=bool).ravel()
        self.tree = _tree("constr", 0, diffOrder=np.zeros(idv.size),
                          height=0, ID=idv, hasTerms=False)

    # -- helpers -------------------------------------------------------

    @staticmethod
    def univariate(tree_in, method):
        """Syntax tree for a univariate operation.

        Provenance
        ----------
        MATLAB source : @treeVar/univariate.m
        Chebfun commit: 7574c77
        """
        return _tree(method, 1, center=tree_in,
                     diffOrder=tree_in["diffOrder"],
                     height=tree_in["height"] + 1,
                     ID=tree_in["ID"], hasTerms=tree_in["hasTerms"])

    @staticmethod
    def bivariate(left, right, method, type_):
        """Syntax tree for a bivariate operation.  ``type_`` is 0 when
        only the left argument is a TreeVar tree, 1 when only the right
        is, and 2 when both are.

        Provenance
        ----------
        MATLAB source : @treeVar/bivariate.m
        Chebfun commit: 7574c77
        """
        is_pm = method in ("plus", "minus")
        if type_ == 2:
            return _tree(method, 2, left=left, right=right,
                         diffOrder=np.maximum(left["diffOrder"],
                                              right["diffOrder"]),
                         ID=left["ID"] | right["ID"],
                         height=max(left["height"],
                                    right["height"]) + 1,
                         hasTerms=is_pm or left["hasTerms"]
                         or right["hasTerms"])
        if type_ == 1:
            return _tree(method, 2, left=left, right=right,
                         diffOrder=right["diffOrder"],
                         height=right["height"] + 1, ID=right["ID"],
                         hasTerms=is_pm or right["hasTerms"])
        return _tree(method, 2, left=left, right=right,
                     diffOrder=left["diffOrder"],
                     height=left["height"] + 1, ID=left["ID"],
                     hasTerms=is_pm or left["hasTerms"])

    def _wrap(self, tree):
        out = TreeVar.__new__(TreeVar)
        out.domain = self.domain
        out.tree = tree
        return out

    def _update_domain(self, other):
        if isinstance(other, TreeVar):
            merged = sorted(set(self.domain) | set(other.domain))
            return tuple(merged)
        odom = getattr(other, "domain", None)
        if odom is not None:
            bps = (odom.breakpoints if hasattr(odom, "breakpoints")
                   else odom)
            merged = sorted(set(self.domain)
                            | {float(v) for v in bps})
            return tuple(merged)
        return self.domain

    def _bin(self, other, method, reflected=False):
        if isinstance(other, TreeVar):
            t = TreeVar.bivariate(self.tree, other.tree, method, 2)
        elif reflected:
            t = TreeVar.bivariate(other, self.tree, method, 1)
        else:
            t = TreeVar.bivariate(self.tree, other, method, 0)
        out = self._wrap(t)
        out.domain = self._update_domain(other)
        return out

    # -- arithmetic ----------------------------------------------------

    def __add__(self, other):
        return self._bin(other, "plus")

    def __radd__(self, other):
        return self._bin(other, "plus", reflected=True)

    def __sub__(self, other):
        return self._bin(other, "minus")

    def __rsub__(self, other):
        return self._bin(other, "minus", reflected=True)

    def __mul__(self, other):
        return self._bin(other, "times")

    def __rmul__(self, other):
        return self._bin(other, "times", reflected=True)

    def __truediv__(self, other):
        return self._bin(other, "rdivide")

    def __rtruediv__(self, other):
        return self._bin(other, "rdivide", reflected=True)

    def __pow__(self, other):
        return self._bin(other, "power")

    def __rpow__(self, other):
        return self._bin(other, "power", reflected=True)

    def __neg__(self):
        return self._wrap(TreeVar.univariate(self.tree, "uminus"))

    def __pos__(self):
        return self._wrap(TreeVar.univariate(self.tree, "uplus"))

    def diff(self, k: int = 1):
        """Derivative node.  Only bare base variables may be
        differentiated (``diff(-u)`` etc. is unsupported, matching
        MATLAB's first-order reformulation).

        Provenance
        ----------
        MATLAB source : @treeVar/treeVar.m (diff)
        Chebfun commit: 7574c77
        """
        if not (self.tree["method"] == "constr"
                and self.tree["height"] == 0):
            raise TreeVarError(
                "CHEBFUN:TREEVAR:diff:diffArguments",
                "For first order formulation, the diff method does "
                "currently not support arguments other than simply "
                "the unknown functions.")
        t = _tree("diff", 2, left=self.tree, right=int(k),
                  diffOrder=self.tree["diffOrder"]
                  + k * self.tree["ID"],
                  height=self.tree["height"] + 1,
                  ID=self.tree["ID"],
                  hasTerms=self.tree["hasTerms"])
        return self._wrap(t)

    def cumsum(self, *a, **k):
        raise TreeVarError(
            "CHEBFUN:TREEVAR:cumsum:notSupported",
            "First order reformulation does not support integral "
            "equations.")

    def sum(self, *a, **k):
        raise TreeVarError(
            "CHEBFUN:TREEVAR:cumsum:notSupported",
            "First order reformulation does not support integral "
            "equations.")

    def __call__(self, *a, **k):
        raise TreeVarError(
            "CHEBFUN:TREEVAR:SUBSREF:notSupported",
            "t() is not supported in treeVar.")

    def print(self, *var_names):
        """Text rendering of the syntax tree (MATLAB ``print``)."""
        return print_tree(self.tree, list(var_names) or None)

    def plot(self, ax=None):
        """Plot the syntax tree (MATLAB ``plot``)."""
        return plot_tree(self.tree, ax=ax)

    def __repr__(self):
        return ("treeVar with tree:\n" + print_tree(self.tree)
                + f"and the domain:\n    {self.domain}")


def _add_univariate_methods():
    def make(name):
        def method(self):
            return self._wrap(TreeVar.univariate(self.tree, name))
        method.__name__ = name
        method.__doc__ = (f"Record ``{name}`` in the syntax tree "
                          "(MATLAB @treeVar univariate method).")
        return method
    for name in _UNIVARIATE:
        if name in ("uminus", "uplus"):
            continue
        setattr(TreeVar, name, make(name))


_add_univariate_methods()


# ----------------------------------------------------------------------
# Tree analysis
# ----------------------------------------------------------------------

def _get(tree, field, default=0):
    if _is_tree(tree):
        return tree[field]
    return default


def expand_tree(tree, max_order):
    """Distribute products over sums so the highest-order derivative
    stands alone (``5*(diff(u)+u)`` -> ``5*diff(u) + 5*u``).

    Provenance
    ----------
    MATLAB source : @treeVar/expandTree.m
    Chebfun commit: 7574c77
    """
    max_order = np.asarray(max_order, dtype=float)
    if not _is_tree(tree):
        return tree
    tdo = np.broadcast_to(
        np.atleast_1d(np.asarray(tree["diffOrder"], dtype=float)),
        max_order.shape)
    if (tree["height"] <= 1 or np.all(tdo < max_order)
            or (not tree["hasTerms"] and np.sum(tree["ID"]) <= 1)):
        return tree
    if tree["numArgs"] == 1:
        return expand_tree(tree["center"], max_order)
    if tree["method"] in ("plus", "minus"):
        out = dict(tree)
        out["left"] = expand_tree(tree["left"], max_order)
        out["right"] = expand_tree(tree["right"], max_order)
        return out

    # Must be at .*, ./ or .^.
    if (_is_tree(tree["left"]) and _is_tree(tree["right"])
            and np.any(tdo == max_order)):
        raise TreeVarError(
            "CHEBFUN:TREEVAR:expandTree:nonlinearity",
            "Nonlinearity in highest order derivative detected. "
            "Unable to convert to first order format.")

    left, right = tree["left"], tree["right"]
    left_args = left["numArgs"] if _is_tree(left) else 0
    right_args = right["numArgs"] if _is_tree(right) else 0

    split_left = split_right = False
    if left_args == 0 or (_is_tree(left) and left["method"] == "diff"
                          and left["height"] <= 1):
        left_tree = left
    elif left_args == 1:
        left_tree = expand_tree(left, max_order)
    else:
        left_left = expand_tree(left["left"], max_order)
        left_right = expand_tree(left["right"], max_order)
        split_left = True

    if right_args == 0 or (_is_tree(right)
                           and right["method"] == "diff"
                           and right["height"] <= 1):
        right_tree = right
    elif right_args == 1:
        right_tree = expand_tree(right, max_order)
    else:
        right_left = expand_tree(right["left"], max_order)
        right_right = expand_tree(right["right"], max_order)
        split_right = True

    def times(a, b):
        return _tree(
            "times", 2, left=a, right=b,
            diffOrder=np.maximum(
                np.asarray(_get(a, "diffOrder", 0), dtype=float),
                np.asarray(_get(b, "diffOrder", 0), dtype=float)),
            height=max(_get(a, "height", 0), _get(b, "height", 0)) + 1,
            ID=(np.asarray(_get(a, "ID", False))
                | np.asarray(_get(b, "ID", False)))
            if (_is_tree(a) or _is_tree(b)) else np.asarray([False]),
            hasTerms=bool(_get(a, "hasTerms", 0))
            or bool(_get(b, "hasTerms", 0)))

    if not split_left and not split_right:
        new_left, new_right = left_tree, right_tree
    elif split_left:
        new_left = times(left_left, right_tree)
        new_right = times(left_right, right_tree)
    else:
        new_left = times(left_tree, right_left)
        new_right = times(left_tree, right_right)

    if _is_tree(new_left) and new_left.get("hasTerms"):
        new_left = expand_tree(new_left, max_order)
    if _is_tree(new_right) and new_right.get("hasTerms"):
        new_right = expand_tree(new_right, max_order)

    return _tree(
        "plus", 2, left=new_left, right=new_right,
        diffOrder=np.maximum(
            np.asarray(_get(new_left, "diffOrder", 0), dtype=float),
            np.asarray(_get(new_right, "diffOrder", 0), dtype=float)),
        height=max(_get(new_left, "height", 0),
                   _get(new_right, "height", 0)) + 1,
        ID=(np.asarray(_get(new_left, "ID", False))
            | np.asarray(_get(new_right, "ID", False))),
        hasTerms=bool(_get(new_left, "hasTerms", 0))
        or bool(_get(new_right, "hasTerms", 0)))


def split_tree(tree, max_order):
    """Split a tree into (non-derivative part, derivative part).

    Provenance
    ----------
    MATLAB source : @treeVar/splitTree.m
    Chebfun commit: 7574c77
    """
    max_order = np.asarray(max_order, dtype=float)
    diff_var = max_order > 0
    if not _is_tree(tree):
        return tree, None
    tdo = np.broadcast_to(
        np.atleast_1d(np.asarray(tree["diffOrder"], dtype=float)),
        max_order.shape)
    if np.all(tdo[diff_var] < max_order[diff_var]):
        return tree, None

    if tree["numArgs"] == 1:
        if tree["method"] == "uminus":
            temp = _tree("times", 2, left=-1.0, right=tree["center"],
                         diffOrder=tree["diffOrder"],
                         height=tree["height"], ID=tree["ID"],
                         hasTerms=tree["hasTerms"])
            return split_tree(temp, max_order)
        return split_tree(tree["center"], max_order)

    if tree["method"] in ("diff", "times", "rdivide"):
        return None, tree

    new_l, der_l = split_tree(tree["left"], max_order)
    new_r, der_r = split_tree(tree["right"], max_order)

    def from_right(t, op):
        if op != "minus":
            return t
        if _is_tree(t):
            return _tree("uminus", 1, center=t,
                         diffOrder=t["diffOrder"], ID=t["ID"],
                         height=t["height"] + 1)
        return -t

    if new_l is None:
        new_tree = from_right(new_r, tree["method"])
    elif new_r is None:
        new_tree = new_l
    elif not _is_tree(new_l) and not _is_tree(new_r):
        new_tree = _BINARY_FNS[tree["method"]](new_l, new_r)
    else:
        if not _is_tree(new_l):
            ndo, nh = new_r["diffOrder"], new_r["height"]
        elif not _is_tree(new_r):
            ndo, nh = new_l["diffOrder"], new_l["height"]
        else:
            ndo = np.maximum(new_l["diffOrder"], new_r["diffOrder"])
            nh = max(new_l["height"], new_r["height"])
        new_tree = _tree(tree["method"], 2, left=new_l, right=new_r,
                         diffOrder=ndo, height=nh)

    if der_l is None:
        der_tree = from_right(der_r, tree["method"])
    elif der_r is None:
        der_tree = der_l
    else:
        der_tree = _tree(tree["method"], 2, left=der_l, right=der_r,
                         diffOrder=np.maximum(der_l["diffOrder"],
                                              der_r["diffOrder"]),
                         height=max(der_l["height"], der_r["height"]))
    return new_tree, der_tree


def _leaf_eval(leaf, t):
    """Evaluate a scalar/Chebfun leaf at time ``t``."""
    if isinstance(leaf, (int, float, complex)):
        return leaf
    return float(leaf(jnp.asarray(float(t))))


def _compile(tree, index_start, is_coeff):
    """Compile a syntax tree to a closure ``f(t, u)``.

    This replaces MATLAB's tree2infix + toAnon string round-trip.

    Provenance
    ----------
    MATLAB source : @treeVar/tree2infix.m, @treeVar/toAnon.m
    Chebfun commit: 7574c77
    """
    if tree is None:
        return lambda t, u: 0.0
    if not _is_tree(tree):
        leaf = tree
        return lambda t, u: _leaf_eval(leaf, t)
    n_args = tree["numArgs"]
    if n_args == 0:
        idx = int(index_start[int(np.argmax(tree["ID"]))]) - 1
        return lambda t, u: u[idx]
    if n_args == 1:
        inner = _compile(tree["center"], index_start, is_coeff)
        fn = _UNARY_FNS[tree["method"]]
        return lambda t, u: fn(inner(t, u))
    if tree["method"] == "diff":
        idx = (int(tree["right"])
               + int(index_start[int(np.argmax(tree["ID"]))])) - 1
        return lambda t, u: u[idx]
    left = _compile(tree["left"], index_start, is_coeff)
    right = _compile(tree["right"], index_start, is_coeff)
    fn = _BINARY_FNS[tree["method"]]
    return lambda t, u: fn(left(t, u), right(t, u))


def to_first_order(fun_in, rhs, domain, num_args=None, cell_arg=False):
    """Convert a higher-order ODE operator to a first-order system.

    Returns ``(fun_out, index_start, problem_dom, coeffs,
    total_diff_orders)`` exactly as MATLAB does: ``fun_out(t, u)``
    evaluates the first-order right-hand side, ``index_start`` gives the
    first index of each variable in the state vector (1-based, matching
    the MATLAB test suite), ``problem_dom`` includes breakpoints from
    piecewise coefficients, ``coeffs`` are the leading-coefficient
    scalars/Chebfuns, and ``total_diff_orders`` the per-variable orders.

    Provenance
    ----------
    MATLAB source : @treeVar/toFirstOrder.m, @treeVar/toRHS.m
    Chebfun commit: 7574c77
    Original authors: Copyright 2017 by The University of Oxford
        and The Chebfun Developers.
    """
    from chebfunjax.chebfun1d.chebfun import chebfun

    domain = tuple(float(v) for v in domain)
    t_cheb = chebfun(lambda t: t, domain=domain)

    if num_args is None:
        try:
            n_in = len(inspect.signature(fun_in).parameters)
        except (TypeError, ValueError):
            n_in = 1
        num_args = max(1, n_in - 1) if n_in > 1 else 1
        takes_t = n_in > 1
    else:
        takes_t = True

    args = []
    for j in range(num_args):
        idv = np.zeros(num_args)
        idv[j] = 1
        args.append(TreeVar(idv, domain))

    if cell_arg:
        result = fun_in(t_cheb, args)
    elif not takes_t:
        result = fun_in(*args)
    else:
        result = fun_in(t_cheb, *args)
    if isinstance(result, TreeVar):
        result = [result]
    else:
        result = list(result)

    if isinstance(rhs, (int, float, complex)):
        rhs = [rhs] * len(result)
    elif not isinstance(rhs, (list, tuple)):
        rhs = [rhs]
    rhs = list(rhs)

    # Union of breakpoints from coefficients and the RHS, collapsing
    # floating-point near-duplicates as MATLAB's domain union does.
    problem_dom = set(domain)
    for res in result:
        problem_dom.update(res.domain)
    for entry in rhs:
        edom = getattr(entry, "domain", None)
        if edom is not None:
            bps = (edom.breakpoints if hasattr(edom, "breakpoints")
                   else edom)
            problem_dom.update(float(v) for v in bps)
    pts = sorted(problem_dom)
    htol = 1e-10 * max(1.0, abs(domain[0]), abs(domain[-1]))
    merged = [pts[0]]
    for p in pts[1:]:
        if p - merged[-1] > htol:
            merged.append(p)
        elif abs(p) < abs(merged[-1]):
            # Prefer the rounder representative (e.g. 0.0 over 1e-16).
            merged[-1] = p
    problem_dom = tuple(merged)

    total_orders = np.zeros(num_args)
    for res in result:
        total_orders = np.maximum(total_orders,
                                  res.tree["diffOrder"])
    total_orders_int = total_orders.astype(int)

    index_start = np.concatenate(
        [[1], np.cumsum(total_orders_int[:-1]) + 1]).astype(int)
    index_start_der = index_start + np.arange(num_args)

    coeffs = [None] * num_args
    eq_funs = [None] * num_args
    n_state_der = int(index_start_der[-1] + total_orders_int[-1])

    for res, rhs_k in zip(result, rhs):
        diff_orders = np.asarray(res.tree["diffOrder"], dtype=float)
        if np.sum(total_orders == diff_orders) > 1:
            raise TreeVarError(
                "CHEBFUN:TREEVAR:toFirstOrder:diffOrders",
                "The highest order derivative of more than one "
                "variable appears to be present in the same equation. "
                "Unable to convert to first order format.")
        exp_tree = expand_tree(res.tree, total_orders)
        new_tree, der_tree = split_tree(exp_tree, total_orders)
        exp_do = np.broadcast_to(
            np.atleast_1d(np.asarray(exp_tree["diffOrder"],
                                     dtype=float)),
            total_orders.shape)
        max_der_loc = int(np.argmax(exp_do == total_orders))

        # Coefficient of the highest-order derivative: evaluate the
        # derivative part with the highest-derivative slot set to 1.
        coeff_closure = _compile(der_tree, index_start_der, True)
        if max_der_loc == num_args - 1:
            one_idx = n_state_der - 1
        else:
            one_idx = int(index_start_der[max_der_loc + 1]) - 2
        coeff_arg = np.zeros(n_state_der)
        coeff_arg[one_idx] = 1.0

        def coeff_fun(tv, _cc=coeff_closure, _ca=coeff_arg):
            return _cc(tv, _ca)

        # Represent the coefficient as a scalar when it is constant,
        # a Chebfun otherwise (probe like MATLAB's compose(t, .)).
        probes = np.linspace(domain[0], domain[-1], 7)[1:-1]
        vals = np.asarray([coeff_fun(p) for p in probes])
        if np.allclose(vals, vals[0], rtol=1e-14, atol=1e-14):
            coeffs[max_der_loc] = float(vals[0])
        else:
            coeffs[max_der_loc] = chebfun(
                lambda x, _cf=coeff_fun: jnp.asarray(
                    np.asarray([_cf(float(v))
                                for v in np.atleast_1d(
                                    np.asarray(x))])).reshape(
                    jnp.shape(x)),
                domain=problem_dom)

        rhs_minus = _tree("minus", 2, left=rhs_k,
                          right=0.0 if new_tree is None else new_tree,
                          diffOrder=total_orders, height=1,
                          ID=np.ones(num_args, dtype=bool),
                          hasTerms=True)
        eq_funs[max_der_loc] = _compile(rhs_minus, index_start, False)

    coeff_vals = list(coeffs)

    def fun_out(t, u):
        u = np.asarray(u).ravel()
        out = []
        for j in range(num_args):
            for m in range(total_orders_int[j] - 1):
                out.append(u[int(index_start[j]) + m])
            c = coeff_vals[j]
            cv = c if isinstance(c, float) else float(
                c(jnp.asarray(float(t))))
            out.append(eq_funs[j](t, u) / cv)
        return np.asarray(out).reshape(-1, 1)

    return (fun_out, index_start, problem_dom, coeffs,
            total_orders_int)


def sort_conditions(fun_in, domain, max_diff_orders):
    """How to sort the results of evaluating N.lbc/rbc so they match
    the state ordering ``u, u', v, v', ...`` (1-based indices).

    Provenance
    ----------
    MATLAB source : @treeVar/sortConditions.m
    Chebfun commit: 7574c77
    """
    max_diff_orders = np.atleast_1d(
        np.asarray(max_diff_orders, dtype=int))
    try:
        n_in = len(inspect.signature(fun_in).parameters)
    except (TypeError, ValueError):
        n_in = -1
    num_args = max(n_in, max_diff_orders.size)
    args = []
    for j in range(num_args):
        idv = np.zeros(num_args)
        idv[j] = 1
        args.append(TreeVar(idv, domain))
    if n_in == 1 and num_args > 1:
        results = fun_in(args)
    else:
        results = fun_in(*args)
    if isinstance(results, TreeVar):
        results = [results]
    else:
        results = list(results)

    var_list = [[] for _ in range(num_args)]
    order_list = [[] for _ in range(num_args)]
    for count, res in enumerate(results):
        tree = res.tree
        if int(np.sum(tree["ID"])) > 1:
            raise TreeVarError(
                "CHEBFUN:TREEVAR:sortConditions:nonSeparated",
                "For initial value problems, only separated "
                "conditions are supported.")
        if not _accepted_condition(tree):
            raise TreeVarError(
                "CHEBFUN:TREEVAR:sortConditions:unsupportedCondition",
                "Initial/final condition not supported.")
        active = int(np.argmax(tree["ID"]))
        order = float(np.asarray(tree["diffOrder"]).ravel()[active])
        var_list[active].append(count + 1)
        order_list[active].append(order)

    idx = []
    for j in range(num_args):
        orders = np.asarray(order_list[j])
        perm = np.argsort(orders, kind="stable")
        sorted_orders = orders[perm]
        if sorted_orders.size and sorted_orders[-1] >= \
                max_diff_orders[j]:
            raise TreeVarError(
                "CHEBFUN:TREEVAR:sortConditions:tooHighOrderCondition",
                "The value of a derivative of a too high order was "
                "specified.")
        if sorted_orders.size > 1 and np.any(
                np.diff(sorted_orders) == 0):
            raise TreeVarError(
                "CHEBFUN:TREEVAR:sortConditions:"
                "multipleConditionsSameVariable",
                "Multiple initial conditions on the same "
                "variable/derivative specified.")
        if not (sorted_orders.size == max_diff_orders[j]
                and (sorted_orders.size == 0
                     or (sorted_orders[0] == 0
                         and np.all(np.diff(sorted_orders) == 1)))):
            raise TreeVarError(
                "CHEBFUN:TREEVAR:sortConditions:missingConditions",
                "Solving an nth order IVP/FVP requires specifying "
                "values of the solution and all the (n-1)st "
                "derivatives at the endpoint.")
        idx.extend(int(var_list[j][p]) for p in perm)
    return np.asarray(idx, dtype=int)


def _accepted_condition(tree) -> bool:
    """Reject conditions like ``5*u - 1`` or ``u + diff(u)``.

    Provenance
    ----------
    MATLAB source : @treeVar/sortConditions.m (acceptedCondition)
    Chebfun commit: 7574c77
    """
    if not _is_tree(tree) or tree["height"] == 0:
        return True
    if tree["method"] in ("plus", "minus"):
        if tree["height"] == 1:
            return True
        left, right = tree["left"], tree["right"]
        if not _is_tree(left):
            return _accepted_condition(right)
        if not _is_tree(right):
            return _accepted_condition(left)
        if np.any(left["ID"]) and np.any(right["ID"]):
            return False
        return _accepted_condition(right)
    if tree["method"] == "diff" and tree["height"] == 1:
        return True
    return False


def print_tree(tree, var_names=None, indent="") -> str:
    """Text rendering of a syntax tree.

    Provenance
    ----------
    MATLAB source : @treeVar/printTree.m
    Chebfun commit: 7574c77
    """
    if tree is None:
        return "(empty tree)\n"
    num_vars = int(np.asarray(tree["ID"]).size)
    if var_names is None:
        var_names = (["t", "u"] if num_vars == 1 else
                     ["t"] + [f"u{j + 1}" for j in range(num_vars)])

    label = tree["method"]
    if label == "constr":
        ids = np.flatnonzero(np.asarray(tree["ID"]))
        label = var_names[0] if ids.size == 0 else \
            var_names[int(ids[0]) + 1]

    do_str = " ".join(str(int(v))
                      for v in np.atleast_1d(tree["diffOrder"]))
    if indent == "":
        s = f"{label}\tdiffOrder: [{do_str}]\n"
    else:
        ind = indent if indent.endswith("|") else indent[:-1] + "|"
        s = f"{ind}--{label}\tdiffOrder: [{do_str}]\n"

    pad = indent + "  " if indent else "  "
    if tree["numArgs"] == 1:
        s += print_tree(tree["center"], var_names, pad + " ")
    elif tree["numArgs"] == 2:
        for child, mark in ((tree["left"], "|"),
                            (tree["right"], " ")):
            if _is_tree(child):
                s += print_tree(child, var_names, pad + mark)
            elif isinstance(child, (int, float, complex)):
                s += f"{pad}|--numerical \tValue: {child:2.2f}\n"
            else:
                s += f"{pad}|--chebfun\n"
    return s


def plot_tree(tree, ax=None):
    """Plot a syntax tree with matplotlib.

    Provenance
    ----------
    MATLAB source : @treeVar/plotTree.m
    Chebfun commit: 7574c77
    """
    import matplotlib
    if ax is None:
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    if ax is None:
        _fig, ax = plt.subplots()

    def draw(node, x, y, dx):
        label = node["method"] if _is_tree(node) else (
            f"{node:2.2f}" if isinstance(node, (int, float, complex))
            else "chebfun")
        ax.text(x, y, str(label), ha="center", va="center",
                bbox={"boxstyle": "round", "fc": "w"})
        if not _is_tree(node):
            return
        children = ([node["center"]] if node["numArgs"] == 1 else
                    [node["left"], node["right"]]
                    if node["numArgs"] == 2 else [])
        if node.get("method") == "diff":
            children = [node["left"]]
        offs = ([0.0] if len(children) == 1 else [-dx, dx])
        for child, off in zip(children, offs):
            ax.plot([x, x + off], [y - 0.12, y - 1 + 0.12], "k-",
                    lw=0.8)
            draw(child, x + off, y - 1, dx / 2)

    draw(tree, 0.0, 0.0, 1.0)
    ax.set_axis_off()
    return ax
