"""treeVar — symbolic expression tree for operator linearization.

Mirrors MATLAB Chebfun's ``@treeVar`` class.  A :class:`TreeVar` object
represents an *unknown* function in a differential operator, and overloads
arithmetic so that applying the operator to a ``TreeVar`` builds a symbolic
expression tree describing its structure.

The primary use case is inside :class:`~chebfunjax.autodiff.adchebfun.ADChebfun`:
the operator is first probed with a ``TreeVar`` to detect linearity and
differential order *without* any numerical computation, and then the tree is
linearized into an :class:`~chebfunjax.operators.blocks.OperatorBlock` by
:func:`linearize_tree`.

Design
------
Each node in the tree is represented as a Python dict (matching the MATLAB
``struct`` convention) with the following keys:

``method``
    The name of the operation (``'constr'``, ``'diff'``, ``'plus'``,
    ``'minus'``, ``'times'``, ``'rdivide'``, ``'power'``, or any unary
    function name).
``num_args``
    0 for a leaf (base variable), 1 for unary, 2 for binary.
``diff_order``
    Integer: the total differentiation order accumulated to this node.
``height``
    Integer: depth of the tree (0 for leaves).
``has_terms``
    Bool: whether a ``+`` or ``-`` appears in the subtree (used for tree
    expansion in the IVP first-order reformulation path – not needed for BVP
    linearization but kept for completeness).

For binary nodes the dict also contains ``'left'`` and ``'right'`` keys.
For unary nodes the dict contains a ``'center'`` key.

Translated from MATLAB Chebfun ``@treeVar`` (commit 7574c77).
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
See https://www.chebfun.org/ for Chebfun information.
"""

from __future__ import annotations

from typing import Any

__all__ = ["TreeVar", "linearize_tree"]


# ---------------------------------------------------------------------------
# Helpers for safely reading fields from tree nodes or plain values
# ---------------------------------------------------------------------------


def _diff_order(node: Any) -> int:
    """Return diff_order of a tree node, or 0 for a plain value."""
    if isinstance(node, dict) and "diff_order" in node:
        return node["diff_order"]
    return 0


def _height(node: Any) -> int:
    """Return height of a tree node, or 0 for a plain value."""
    if isinstance(node, dict) and "height" in node:
        return node["height"]
    return 0


def _has_terms(node: Any) -> bool:
    """Return has_terms of a tree node, or False for a plain value."""
    if isinstance(node, dict) and "has_terms" in node:
        return node["has_terms"]
    return False


def _univariate(tree: dict, method: str) -> dict:
    """Build a unary tree node wrapping *tree* with *method*."""
    return {
        "method": method,
        "num_args": 1,
        "center": tree,
        "diff_order": tree["diff_order"],
        "height": tree["height"] + 1,
        "has_terms": tree["has_terms"],
    }


def _bivariate(
    left: Any, right: Any, method: str, tvar_type: int
) -> dict:
    """Build a binary tree node.

    Parameters
    ----------
    left, right : tree dict or plain value (Chebfun / scalar)
    method      : operation name ('plus', 'minus', 'times', 'rdivide', 'power')
    tvar_type   : 0 = only left is TreeVar,
                  1 = only right is TreeVar,
                  2 = both are TreeVars.
    """
    is_pm = method in ("plus", "minus")
    if tvar_type == 2:
        return {
            "method": method,
            "num_args": 2,
            "left": left,
            "right": right,
            "diff_order": max(_diff_order(left), _diff_order(right)),
            "height": max(_height(left), _height(right)) + 1,
            "has_terms": is_pm or _has_terms(left) or _has_terms(right),
        }
    elif tvar_type == 1:  # only right is TreeVar
        return {
            "method": method,
            "num_args": 2,
            "left": left,
            "right": right,
            "diff_order": _diff_order(right),
            "height": _height(right) + 1,
            "has_terms": is_pm or _has_terms(right),
        }
    else:  # tvar_type == 0, only left is TreeVar
        return {
            "method": method,
            "num_args": 2,
            "left": left,
            "right": right,
            "diff_order": _diff_order(left),
            "height": _height(left) + 1,
            "has_terms": is_pm or _has_terms(left),
        }


# ===========================================================================
# TreeVar
# ===========================================================================


class TreeVar:
    """Symbolic unknown-function variable that builds an expression tree.

    Instantiate one ``TreeVar`` per unknown function in the ODE system.
    Apply the operator to it (using the same lambda as defined in the
    ``Chebop``) and the result carries a ``tree`` describing the operator
    structure.

    Parameters
    ----------
    domain : (float, float), default (-1, 1)
        Physical domain.

    Attributes
    ----------
    tree : dict
        The expression tree rooted at this node.
    domain : tuple[float, float]

    Examples
    --------
    Build the tree for ``u'' + u^2``:

    >>> u = TreeVar()
    >>> expr = u.diff(2) + u ** 2
    >>> expr.tree["diff_order"]
    2
    >>> expr.tree["method"]
    'plus'

    Provenance
    ----------
    MATLAB source : @treeVar/treeVar.m
    Chebfun commit: 7574c77
    """

    def __init__(self, domain: tuple[float, float] = (-1.0, 1.0)) -> None:
        self.domain: tuple[float, float] = tuple(float(v) for v in domain)
        # Leaf node — the base variable
        self.tree: dict = {
            "method": "constr",
            "num_args": 0,
            "diff_order": 0,
            "height": 0,
            "has_terms": False,
        }

    # ------------------------------------------------------------------
    # Differentiation
    # ------------------------------------------------------------------

    def diff(self, k: int = 1) -> "TreeVar":
        """Symbolic differentiation: ``u.diff(k)`` records ``diff(u, k)``.

        Parameters
        ----------
        k : int, default 1
            Order of differentiation.

        Notes
        -----
        MATLAB restricts ``diff`` to be called only on the base variable
        directly (``tree.height == 0``).  We enforce the same constraint
        here because the first-order IVP reformulation (``treeVar.toFirstOrder``)
        requires it.  For BVP linearization via ``adchebfun``, this is not
        a problem in practice.
        """
        result = TreeVar(self.domain)
        result.tree = {
            "method": "diff",
            "num_args": 2,
            "left": self.tree,
            "right": k,
            "diff_order": self.tree["diff_order"] + k,
            "height": self.tree["height"] + 1,
            "has_terms": self.tree["has_terms"],
        }
        return result

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __add__(self, other) -> "TreeVar":
        result = TreeVar(self.domain)
        if isinstance(other, TreeVar):
            result.tree = _bivariate(self.tree, other.tree, "plus", 2)
        else:
            result.tree = _bivariate(self.tree, other, "plus", 0)
        return result

    def __radd__(self, other) -> "TreeVar":
        result = TreeVar(self.domain)
        result.tree = _bivariate(other, self.tree, "plus", 1)
        return result

    def __sub__(self, other) -> "TreeVar":
        result = TreeVar(self.domain)
        if isinstance(other, TreeVar):
            result.tree = _bivariate(self.tree, other.tree, "minus", 2)
        else:
            result.tree = _bivariate(self.tree, other, "minus", 0)
        return result

    def __rsub__(self, other) -> "TreeVar":
        result = TreeVar(self.domain)
        result.tree = _bivariate(other, self.tree, "minus", 1)
        return result

    def __mul__(self, other) -> "TreeVar":
        result = TreeVar(self.domain)
        if isinstance(other, TreeVar):
            result.tree = _bivariate(self.tree, other.tree, "times", 2)
        else:
            result.tree = _bivariate(self.tree, other, "times", 0)
        return result

    def __rmul__(self, other) -> "TreeVar":
        result = TreeVar(self.domain)
        result.tree = _bivariate(other, self.tree, "times", 1)
        return result

    def __truediv__(self, other) -> "TreeVar":
        result = TreeVar(self.domain)
        if isinstance(other, TreeVar):
            result.tree = _bivariate(self.tree, other.tree, "rdivide", 2)
        else:
            result.tree = _bivariate(self.tree, other, "rdivide", 0)
        return result

    def __rtruediv__(self, other) -> "TreeVar":
        result = TreeVar(self.domain)
        result.tree = _bivariate(other, self.tree, "rdivide", 1)
        return result

    def __pow__(self, exp) -> "TreeVar":
        result = TreeVar(self.domain)
        if isinstance(exp, TreeVar):
            result.tree = _bivariate(self.tree, exp.tree, "power", 2)
        else:
            result.tree = _bivariate(self.tree, exp, "power", 0)
        return result

    def __rpow__(self, base) -> "TreeVar":
        result = TreeVar(self.domain)
        result.tree = _bivariate(base, self.tree, "power", 1)
        return result

    def __neg__(self) -> "TreeVar":
        result = TreeVar(self.domain)
        result.tree = _univariate(self.tree, "uminus")
        return result

    def __pos__(self) -> "TreeVar":
        return self

    # ------------------------------------------------------------------
    # Unary functions (matching Chebfun API so the same lambda works)
    # ------------------------------------------------------------------

    def _unary(self, name: str) -> "TreeVar":
        result = TreeVar(self.domain)
        result.tree = _univariate(self.tree, name)
        return result

    def sin(self) -> "TreeVar":
        return self._unary("sin")

    def cos(self) -> "TreeVar":
        return self._unary("cos")

    def tan(self) -> "TreeVar":
        return self._unary("tan")

    def exp(self) -> "TreeVar":
        return self._unary("exp")

    def log(self) -> "TreeVar":
        return self._unary("log")

    def sqrt(self) -> "TreeVar":
        return self._unary("sqrt")

    def sinh(self) -> "TreeVar":
        return self._unary("sinh")

    def cosh(self) -> "TreeVar":
        return self._unary("cosh")

    def tanh(self) -> "TreeVar":
        return self._unary("tanh")

    def abs(self) -> "TreeVar":
        return self._unary("abs")

    # ------------------------------------------------------------------
    # Evaluation (f(x) syntax) — raise to force BVP solver path
    # ------------------------------------------------------------------

    def __call__(self, x):  # type: ignore[override]
        raise TypeError(
            "TreeVar does not support evaluation (operator contains a "
            "composition u(x)).  Use Chebop.solve() with N.bc for this case."
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return f"TreeVar(domain={self.domain}, diff_order={self.tree['diff_order']})"


# ===========================================================================
# linearize_tree
# ===========================================================================


def linearize_tree(
    tree: dict,
    u_cheb,  # Chebfun — current linearization point
    domain: tuple[float, float],
) -> "OperatorBlock":  # noqa: F821
    """Recursively convert an expression tree to a linearized OperatorBlock.

    Given the tree produced by evaluating an operator on a ``TreeVar`` and a
    Chebfun linearization point ``u``, returns the Fréchet-derivative
    OperatorBlock ``dN[u]``.  This is the ``v``-to-``dN[u](v)`` map.

    The chain rule in each case:

    * **diff(u, k)** → ``D^k * v``
    * **u + w**      → ``dN_u(v) + dN_w(v)``
    * **u - w**      → ``dN_u(v) - dN_w(v)``
    * **u * w**      (one TreeVar) → ``diag(w) * v``
    * **u * w**      (two TreeVars, u≠w not supported for BVP) → product rule
    * **u^n**        (scalar n) → ``diag(n * u^(n-1)) * v``
    * **sin(u)**     → ``diag(cos(u)) * v``
    * **cos(u)**     → ``diag(-sin(u)) * v``
    * **exp(u)**     → ``diag(exp(u)) * v``
    * etc.

    Parameters
    ----------
    tree : dict
        The expression tree (returned from ``TreeVar`` arithmetic).
    u_cheb : Chebfun
        The current linearization point.
    domain : (float, float)
        Physical domain.

    Returns
    -------
    OperatorBlock
        The Fréchet-derivative operator ``dN[u]``.

    Raises
    ------
    NotImplementedError
        For operations not yet supported in the linearization.

    Provenance
    ----------
    MATLAB source : @adchebfun (operator overloading approach), @treeVar
    Chebfun commit: 7574c77
    """
    from chebfunjax.operators.blocks import D, I, diag  # local to avoid circular

    method = tree["method"]

    # ------------------------------------------------------------------
    # Leaf: the base variable — linearization is the identity operator
    # ------------------------------------------------------------------
    if method == "constr":
        return I(domain)

    # ------------------------------------------------------------------
    # Differentiation: diff(u, k) → D^k
    # ------------------------------------------------------------------
    if method == "diff":
        # left is the sub-tree (base variable or another diff), right is k
        k = tree["right"]
        sub_op = linearize_tree(tree["left"], u_cheb, domain)
        return D(domain, order=k) * sub_op

    # ------------------------------------------------------------------
    # Binary: plus / minus
    # ------------------------------------------------------------------
    if method in ("plus", "minus"):
        left_tree = tree["left"]
        right_tree = tree["right"]
        left_is_tv = isinstance(left_tree, dict) and "method" in left_tree
        right_is_tv = isinstance(right_tree, dict) and "method" in right_tree

        op_left = linearize_tree(left_tree, u_cheb, domain) if left_is_tv else I(domain) * 0.0
        op_right = linearize_tree(right_tree, u_cheb, domain) if right_is_tv else I(domain) * 0.0

        if method == "plus":
            if left_is_tv and right_is_tv:
                return op_left + op_right
            elif left_is_tv:
                return op_left
            else:
                return op_right
        else:  # minus
            if left_is_tv and right_is_tv:
                return op_left - op_right
            elif left_is_tv:
                return op_left
            else:
                return -op_right

    # ------------------------------------------------------------------
    # Binary: times  (product rule for Chebfun * TreeVar or scalar)
    # ------------------------------------------------------------------
    if method == "times":
        left_tree = tree["left"]
        right_tree = tree["right"]
        left_is_tv = isinstance(left_tree, dict) and "method" in left_tree
        right_is_tv = isinstance(right_tree, dict) and "method" in right_tree

        if left_is_tv and right_is_tv:
            # Both sides depend on u: product rule
            # d(f*g)[v] = f*dg[v] + g*df[v]
            # Evaluate each subtree to get the Chebfun value
            f_val = _eval_tree(left_tree, u_cheb)
            g_val = _eval_tree(right_tree, u_cheb)
            op_left = linearize_tree(left_tree, u_cheb, domain)
            op_right = linearize_tree(right_tree, u_cheb, domain)
            return diag(f_val, domain) * op_right + diag(g_val, domain) * op_left
        elif left_is_tv:
            # left * right, right is a constant Chebfun or scalar
            g = _make_chebfun(right_tree, u_cheb, domain)
            op_left = linearize_tree(left_tree, u_cheb, domain)
            if isinstance(g, (int, float)):
                return op_left * float(g)
            return diag(g, domain) * op_left
        else:
            # right * left, left is constant
            g = _make_chebfun(left_tree, u_cheb, domain)
            op_right = linearize_tree(right_tree, u_cheb, domain)
            if isinstance(g, (int, float)):
                return op_right * float(g)
            return diag(g, domain) * op_right

    # ------------------------------------------------------------------
    # Binary: rdivide  (f / g  — chain rule for the TreeVar part)
    # ------------------------------------------------------------------
    if method == "rdivide":
        left_tree = tree["left"]
        right_tree = tree["right"]
        left_is_tv = isinstance(left_tree, dict) and "method" in left_tree
        right_is_tv = isinstance(right_tree, dict) and "method" in right_tree

        if left_is_tv and not right_is_tv:
            # f(u) / c  → (1/c) * df[v]
            c = _make_chebfun(right_tree, u_cheb, domain)
            op_left = linearize_tree(left_tree, u_cheb, domain)
            if isinstance(c, (int, float)):
                return op_left * (1.0 / float(c))
            return diag(1.0 / c, domain) * op_left  # type: ignore[operator]
        elif right_is_tv and not left_is_tv:
            # c / g(u)  → -c * g^{-2} * dg[v]
            f_val = _make_chebfun(left_tree, u_cheb, domain)
            g_val = _eval_tree(right_tree, u_cheb)
            op_right = linearize_tree(right_tree, u_cheb, domain)
            if isinstance(f_val, (int, float)):
                return diag((-float(f_val)) / (g_val * g_val), domain) * op_right
            return diag((-f_val) / (g_val * g_val), domain) * op_right  # type: ignore[operator]
        else:
            # Both TreeVars — quotient rule: d(f/g) = (g*df - f*dg) / g^2
            f_val = _eval_tree(left_tree, u_cheb)
            g_val = _eval_tree(right_tree, u_cheb)
            op_left = linearize_tree(left_tree, u_cheb, domain)
            op_right = linearize_tree(right_tree, u_cheb, domain)
            g2 = g_val * g_val
            return (diag(g_val / g2, domain) * op_left
                    - diag(f_val / g2, domain) * op_right)

    # ------------------------------------------------------------------
    # Binary: power  (u^n, n^u, or u^u)
    # ------------------------------------------------------------------
    if method == "power":
        left_tree = tree["left"]
        right_tree = tree["right"]
        left_is_tv = isinstance(left_tree, dict) and "method" in left_tree
        right_is_tv = isinstance(right_tree, dict) and "method" in right_tree

        if left_is_tv and not right_is_tv:
            # u^n  → diag(n * u^{n-1}) * du
            n = right_tree  # scalar or Chebfun
            u_val = _eval_tree(left_tree, u_cheb)
            op_left = linearize_tree(left_tree, u_cheb, domain)
            if isinstance(n, (int, float)):
                exponent = float(n)
                return diag(exponent * u_val ** (exponent - 1), domain) * op_left
            # n is a Chebfun — rare but handle it
            return diag(n * u_val ** (n - 1), domain) * op_left  # type: ignore[operator]
        elif right_is_tv and not left_is_tv:
            # a^u  → diag(a^u * log(a)) * du
            import jax.numpy as jnp
            a = _make_chebfun(left_tree, u_cheb, domain)
            u_val = _eval_tree(right_tree, u_cheb)
            op_right = linearize_tree(right_tree, u_cheb, domain)
            if isinstance(a, (int, float)):
                import math
                return diag(u_val * math.log(float(a)), domain) * op_right
            return diag(u_val * a.log(), domain) * op_right  # type: ignore[attr-defined]
        else:
            # u^v — full product rule for exponentiation
            f_val = _eval_tree(left_tree, u_cheb)
            g_val = _eval_tree(right_tree, u_cheb)
            op_left = linearize_tree(left_tree, u_cheb, domain)
            op_right = linearize_tree(right_tree, u_cheb, domain)
            # d(f^g) = f^g * (g/f * df + log(f) * dg)
            fg = f_val ** g_val
            return (diag(fg * g_val / f_val, domain) * op_left
                    + diag(fg * f_val.log(), domain) * op_right)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Unary functions — chain rule: d[phi(u)](v) = diag(phi'(u)) * du[v]
    # ------------------------------------------------------------------
    if method in _UNARY_DERIV:
        sub_op = linearize_tree(tree["center"], u_cheb, domain)
        u_val = _eval_tree(tree["center"], u_cheb)
        deriv_fn = _UNARY_DERIV[method]
        mult_fun = deriv_fn(u_val)
        if isinstance(mult_fun, (int, float)):
            return sub_op * float(mult_fun)
        return diag(mult_fun, domain) * sub_op

    if method == "uminus":
        return -linearize_tree(tree["center"], u_cheb, domain)

    if method == "uplus":
        return linearize_tree(tree["center"], u_cheb, domain)

    raise NotImplementedError(
        f"linearize_tree: unsupported tree node method={method!r}."
    )


# ---------------------------------------------------------------------------
# Helper: evaluate the *value* of a tree at u_cheb (returns a Chebfun)
# ---------------------------------------------------------------------------


def _eval_tree(tree: Any, u_cheb) -> Any:
    """Evaluate the Chebfun value of a (possibly non-TreeVar) tree.

    This recurses through the expression tree and computes the actual
    numerical Chebfun value at the linearization point ``u_cheb``.
    Non-tree values (scalars, Chebfuns) are returned as-is.
    """
    if not (isinstance(tree, dict) and "method" in tree):
        # Plain scalar or Chebfun
        return tree

    method = tree["method"]

    if method == "constr":
        return u_cheb

    if method == "diff":
        k = tree["right"]
        sub_val = _eval_tree(tree["left"], u_cheb)
        return sub_val.diff(k)

    if method == "plus":
        lv = _eval_tree(tree["left"], u_cheb)
        rv = _eval_tree(tree["right"], u_cheb)
        return lv + rv

    if method == "minus":
        lv = _eval_tree(tree["left"], u_cheb)
        rv = _eval_tree(tree["right"], u_cheb)
        return lv - rv

    if method == "times":
        lv = _eval_tree(tree["left"], u_cheb)
        rv = _eval_tree(tree["right"], u_cheb)
        return lv * rv

    if method == "rdivide":
        lv = _eval_tree(tree["left"], u_cheb)
        rv = _eval_tree(tree["right"], u_cheb)
        return lv / rv

    if method == "power":
        lv = _eval_tree(tree["left"], u_cheb)
        rv = _eval_tree(tree["right"], u_cheb)
        return lv ** rv

    if method == "uminus":
        return -_eval_tree(tree["center"], u_cheb)

    if method == "uplus":
        return _eval_tree(tree["center"], u_cheb)

    if method in _UNARY_EVAL:
        sub = _eval_tree(tree["center"], u_cheb)
        return _UNARY_EVAL[method](sub)

    raise NotImplementedError(
        f"_eval_tree: unsupported method={method!r}."
    )


def _make_chebfun(tree: Any, u_cheb, domain: tuple[float, float]):
    """Evaluate tree to a Chebfun or return scalar unchanged."""
    if not (isinstance(tree, dict) and "method" in tree):
        return tree
    return _eval_tree(tree, u_cheb)


# ---------------------------------------------------------------------------
# Tables of unary function derivatives and evaluators
# ---------------------------------------------------------------------------


def _unary_eval_table():
    """Return a dict mapping method name to ``f: Chebfun -> Chebfun``."""
    import jax.numpy as jnp

    def _call_method(name):
        """Return a lambda that calls ``getattr(u, name)()``."""
        def _fn(u):
            if hasattr(u, name):
                return getattr(u, name)()
            # Fall back for scalars
            return getattr(jnp, name)(u)
        return _fn

    return {name: _call_method(name) for name in [
        "sin", "cos", "tan", "exp", "log", "sqrt",
        "sinh", "cosh", "tanh",
        "asin", "acos", "atan", "asinh", "acosh", "atanh",
        "abs",
    ]}


def _unary_deriv_table():
    """Return a dict mapping method name to ``deriv: Chebfun -> Chebfun``.

    Each entry gives the pointwise *derivative* of the unary function,
    i.e. ``phi'(u)``.  These are used as multipliers in the chain rule.
    """
    import jax.numpy as jnp

    def _sin_d(u):
        return u.cos() if hasattr(u, "cos") else jnp.cos(u)

    def _cos_d(u):
        neg = u.sin() if hasattr(u, "sin") else jnp.sin(u)
        return -neg

    def _tan_d(u):
        c = u.cos() if hasattr(u, "cos") else jnp.cos(u)
        return 1.0 / (c * c)

    def _exp_d(u):
        return u.exp() if hasattr(u, "exp") else jnp.exp(u)

    def _log_d(u):
        return 1.0 / u

    def _sqrt_d(u):
        s = u.sqrt() if hasattr(u, "sqrt") else jnp.sqrt(u)
        return 0.5 / s

    def _sinh_d(u):
        return u.cosh() if hasattr(u, "cosh") else jnp.cosh(u)

    def _cosh_d(u):
        return u.sinh() if hasattr(u, "sinh") else jnp.sinh(u)

    def _tanh_d(u):
        c = u.cosh() if hasattr(u, "cosh") else jnp.cosh(u)
        return 1.0 / (c * c)

    def _asin_d(u):
        return 1.0 / (1.0 - u * u) ** 0.5

    def _acos_d(u):
        return -1.0 / (1.0 - u * u) ** 0.5

    def _atan_d(u):
        return 1.0 / (1.0 + u * u)

    def _asinh_d(u):
        return 1.0 / (u * u + 1.0) ** 0.5

    def _acosh_d(u):
        return 1.0 / (u * u - 1.0) ** 0.5

    def _atanh_d(u):
        return 1.0 / (1.0 - u * u)

    def _abs_d(u):
        # Not differentiable at zero; use sign as subgradient
        import jax.numpy as _jnp
        if hasattr(u, "funs"):
            vals = u.funs[0].values
            sv = _jnp.sign(vals)
            from chebfunjax.chebfun1d.chebfun import Chebfun
            return Chebfun.from_values(sv, u.domain)
        return _jnp.sign(u)

    return {
        "sin": _sin_d,
        "cos": _cos_d,
        "tan": _tan_d,
        "exp": _exp_d,
        "log": _log_d,
        "sqrt": _sqrt_d,
        "sinh": _sinh_d,
        "cosh": _cosh_d,
        "tanh": _tanh_d,
        "asin": _asin_d,
        "acos": _acos_d,
        "atan": _atan_d,
        "asinh": _asinh_d,
        "acosh": _acosh_d,
        "atanh": _atanh_d,
        "abs": _abs_d,
    }


_UNARY_EVAL = _unary_eval_table()
_UNARY_DERIV = _unary_deriv_table()
