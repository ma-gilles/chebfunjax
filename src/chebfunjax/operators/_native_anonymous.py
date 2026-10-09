"""Explicit native anonymous syntax for assignment-time multiplication rewriting.

Provenance: @chebop/vectorizeOp.m, @chebop/chebop.m setters and parseBC.m,
Chebfun 7574c77680d7e82b79626300bf255498271a72df.
This deliberately limited expression adapter is not a general MATLAB parser.
"""
from __future__ import annotations

import ast
import copy
import inspect
import re
from types import MappingProxyType


def _mtimes(left, right):
    # Match the native AD class's precedence over Chebfun/numeric operands.
    from chebfunjax.autodiff.adchebfun import ADChebfun
    if isinstance(left, ADChebfun):
        return left.mtimes(right)
    if isinstance(right, ADChebfun):
        return right.mtimes(left)
    method = getattr(left, "mtimes", None)
    if method is not None:
        return method(right)
    method = getattr(right, "__rmatmul__", None)
    if method is not None:
        return method(left)
    return left * right


class NativeAnonymous:
    """Tagged expression with captured bindings and an explicit argument list.

    Supported: numeric literals, names, +/-, unary signs, * and .*, calls to
    diff or captured/argument functions, and comma/semicolon scalar lists.
    Bindings are snapshotted at creation, before later workspace rebinding.
    Division, powers, attributes, indexing, strings and statements are rejected.
    """

    def __init__(self, source, workspace=None):
        match = re.fullmatch(r"\s*@\(([^()]*)\)\s*(.+)", source, re.S)
        if match is None:
            raise ValueError("nativeAnonymous requires @(arguments) expression")
        names = [name.strip() for name in match[1].split(",") if name.strip()]
        if (len(set(names)) != len(names)
                or any(not re.fullmatch(r"[A-Za-z]\w*", n) for n in names)):
            raise ValueError("nativeAnonymous arguments must be unique names")
        self.__signature__ = inspect.Signature([
            inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for n in names])
        self.source = source
        self._workspace = MappingProxyType(copy.deepcopy(dict(workspace or {})))
        # Python @ has precisely multiplication's precedence and associativity.
        if any(token in match[2] for token in ("@", "~", "**")):
            raise ValueError("unsupported nativeAnonymous grammar")
        expression = match[2].replace(".*", "~").replace("*", "@")
        expression = expression.replace("~", "*").replace(";", ",")
        try:
            self._tree = ast.parse(expression, mode="eval").body
        except SyntaxError as exc:
            raise ValueError("unsupported nativeAnonymous grammar") from exc
        self._validate(self._tree, set(names) | set(self._workspace) | {"diff"})

    @classmethod
    def _validate(cls, node, names):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, complex):
            return
        if isinstance(node, ast.Name) and node.id in names:
            return
        if isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.Add, ast.Sub, ast.Mult, ast.MatMult)):
            cls._validate(node.left, names)
            cls._validate(node.right, names)
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            cls._validate(node.operand, names)
            return
        if isinstance(node, ast.Call) and not node.keywords:
            cls._validate(node.func, names)
            for arg in node.args:
                cls._validate(arg, names)
            return
        if isinstance(node, ast.List):
            for item in node.elts:
                cls._validate(item, names)
            return
        raise ValueError("unsupported nativeAnonymous grammar or unbound name")

    def compile(self, vectorize):
        """Freeze the assignment's flag without rewriting arbitrary callables."""
        def evaluate(node, bindings):
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Name):
                return bindings[node.id]
            if isinstance(node, ast.List):
                return [evaluate(item, bindings) for item in node.elts]
            if isinstance(node, ast.Call):
                return evaluate(node.func, bindings)(
                    *(evaluate(arg, bindings) for arg in node.args))
            if isinstance(node, ast.UnaryOp):
                val = evaluate(node.operand, bindings)
                return -val if isinstance(node.op, ast.USub) else +val
            left, right = evaluate(node.left, bindings), evaluate(node.right, bindings)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.MatMult) and not vectorize:
                return _mtimes(left, right)
            return left * right

        def compiled(*args, **kwargs):
            bound = self.__signature__.bind(*args, **kwargs)
            bindings = {"diff": lambda u, order=1: u.diff(order),
                        **self._workspace, **bound.arguments}
            return evaluate(self._tree, bindings)
        compiled.__signature__ = self.__signature__
        return compiled


def compile_assignment(value, vectorize):
    return value.compile(vectorize) if isinstance(value, NativeAnonymous) else value
