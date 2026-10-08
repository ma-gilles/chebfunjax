"""Source-faithful basic simplifier for Chebfun expression strings.

This ports the parenthesis/sign cleanup stage used by Chebfun's stringParser.
The transformation operates on strings without evaluation.

Provenance
----------
MATLAB source : @stringParser/parSimp.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""

from __future__ import annotations

import re
from typing import Literal, NamedTuple


class _ParSimpError(ValueError):
    """Preserve the source diagnostic for unbalanced parentheses."""

    identifier = "CHEBFUN:STRINGPARSER:parSimp:parenthSimplify"

    def __init__(self):
        super().__init__(f"{self.identifier}: Incorrect number of parenthesis.")


def par_simp(text: str) -> str:
    """Remove redundant parentheses and adjacent signs as MATLAB ``parSimp``.

    Parentheses are removed only when the minimum operator precedence inside
    is no lower than either adjacent operator. Function-call parentheses and
    parentheses protecting comparisons/division remain. Parenthesized negation
    distributes over a sum, except that exponent signs such as ``1e-2`` are
    preserved.

    Provenance
    ----------
    MATLAB source : @stringParser/parSimp.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    if not isinstance(text, str):
        raise TypeError("par_simp expects a string")
    if len(text) < 2:
        return text
    old_length = 0
    while len(text) != old_length:
        old_length = len(text)
        text = _par_simp_main(text)
    return text


def _par_simp_main(text: str) -> str:
    chars = list(text)
    left_locs = [i for i, c in enumerate(chars) if c == "("]
    right_locs = [i for i, c in enumerate(chars) if c == ")"]
    op_vec = [_precedence(c) for c in chars]
    op_locs = [i for i, weight in enumerate(op_vec) if weight]
    char_locs = [i for i, c in enumerate(chars) if re.fullmatch(r"[A-Za-z0-9@]", c)]
    ltgt_locs = [i for i, c in enumerate(chars) if c in "<>"]

    if len(left_locs) != len(right_locs):
        raise _ParSimpError()

    # Pair in close-order (the source's stack traversal order), so inner
    # parentheses are considered before their enclosing pair.
    pairs: list[list[int]] = []
    stack: list[int] = []
    for i, char in enumerate(chars):
        if char == "(":
            stack.append(i)
        elif char == ")":
            left = stack.pop()
            pairs.append([left, i])
    for loc in ltgt_locs:
        enclosing = [(i, pair) for i, pair in enumerate(pairs) if pair[0] < loc < pair[1]]
        if enclosing:
            pair_index, pair = max(enclosing, key=lambda entry: entry[1][0])
            char_locs.extend(pair)
            pairs.pop(pair_index)
    pairs.sort(key=lambda pair: pair[1])
    char_locs.sort()
    left_pars = [pair[0] for pair in pairs]
    right_pars = [pair[1] for pair in pairs]
    num_pairs = len(pairs)

    def update_locs(values: list[int | None], left: int, right: int):
        for index, value in enumerate(values):
            if value is not None:
                values[index] = value - (value > left) - (value > right)

    def remove_extras(recursive: bool = False):
        nonlocal chars, op_vec, left_pars, right_pars, op_locs, char_locs
        if chars and chars[0] == "+":
            chars.pop(0)
            op_vec.pop(0)
            left_pars = [None if x is None else x - 1 for x in left_pars]
            right_pars = [None if x is None else x - 1 for x in right_pars]
            op_locs = [x - 1 for x in op_locs]
            char_locs = [x - 1 for x in char_locs]

        signs = [weight == 1 for weight in op_vec]
        pair_at = next(
            (i for i in range(max(0, len(signs) - 1)) if signs[i] and signs[i] == signs[i + 1]),
            None,
        )
        if pair_at is not None:
            chars[pair_at] = "+" if chars[pair_at] == chars[pair_at + 1] else "-"
            chars.pop(pair_at + 1)
            op_vec.pop(pair_at + 1)
            left_pars = [None if x is None else x - (x > pair_at) for x in left_pars]
            right_pars = [None if x is None else x - (x > pair_at) for x in right_pars]
            op_locs = [x - (x > pair_at) for x in op_locs]
            char_locs = [x - (x > pair_at) for x in char_locs]
            remove_extras(True)
        if recursive:
            return

    remove_extras()

    for p_index in range(num_pairs):
        remove_extras()
        p_left, p_right = left_pars[p_index], right_pars[p_index]
        if p_left is None or p_right is None:
            continue

        prev_ops = [loc for loc in op_locs if loc < p_left]
        next_ops = [loc for loc in op_locs if loc > p_right]
        next_left = max(prev_ops) if prev_ops else None
        next_right = min(next_ops) if next_ops else None

        if p_left - 1 in char_locs:
            continue

        mask = [False] * len(chars)
        for i in range(p_left + 1, p_right):
            mask[i] = True
        for left, right in zip(left_pars, right_pars):
            if left is not None and right is not None and p_left < left and right < p_right:
                for i in range(left, right + 1):
                    mask[i] = False
        inside = [_precedence(char) if mask[i] else 0 for i, char in enumerate(chars)]
        inside = [float("inf") if value == 0 else value for value in inside]
        min_inside = min(inside[p_left : p_right + 1], default=float("inf"))

        if (next_left is not None and min_inside < op_vec[next_left]) or (
            next_right is not None and min_inside < op_vec[next_right]
        ):
            continue

        if min_inside == 1 and next_left is not None and chars[p_left - 1] == "-":
            interior_pm = [inside[i] == 1 for i in range(len(chars))]
            interior_pm[p_left] = False
            minus_positions = [i for i, c in enumerate(chars) if c == "-" and interior_pm[i]]
            plus_positions = [i for i, c in enumerate(chars) if c == "+" and interior_pm[i]]
            for i in minus_positions:
                if i > 0 and chars[i - 1] == "e":
                    interior_pm[i] = False
            for i in plus_positions:
                if i > 0 and chars[i - 1] == "e":
                    interior_pm[i] = False
            for i in range(len(chars)):
                if not interior_pm[i]:
                    continue
                if chars[i] == "-":
                    chars[i] = "+"
                elif chars[i] == "+":
                    chars[i] = "-"

        if min_inside == 2 and next_left is not None and chars[p_left - 1] == "/":
            continue

        chars.pop(p_left)
        chars.pop(p_right - 1)
        op_vec.pop(p_left)
        op_vec.pop(p_right - 1)
        left_pars[p_index] = None
        right_pars[p_index] = None
        update_locs(left_pars, p_left, p_right)
        update_locs(right_pars, p_left, p_right)
        update_locs(op_locs, p_left, p_right)
        update_locs(char_locs, p_left, p_right)

    if chars and chars[0] == "+":
        chars.pop(0)
    return "".join(chars)


def _precedence(char: str) -> int:
    if char in "+-":
        return 1
    if char in "*/":
        return 2
    if char == "^":
        return 3
    return 0


class _StringParserError(ValueError):
    """Carry a source diagnostic identifier through Python exceptions."""

    def __init__(self, identifier: str, message: str):
        self.identifier = identifier
        super().__init__(message)


_FUNC1 = set(
    "sin cos tan cot sec csc sinh cosh tanh coth sech csch asin acos atan acot asec acsc asinh acosh atanh acoth asech acsch hypot asind acosd atand acotd asecd acscd sind cosd tand cotd secd cscd sqrt exp expm1 heaviside log log10 log2 log1p realsqrt reallog abs sign var std erf erfc erfcx erfinv erfcinv".split()
)
_FUNC2 = set("airy besselj cumsum diff power mean eq ne ge gt le lt jump".split())
_FUNC3 = set("feval fred volt sum integral".split())
_FLAGS = {"left", "right", "onevar", "start", "end"}
_ONE_OR_TWO = {"diff", "cumsum", "airy", "mean"}
_ONE_OR_THREE = {"sum", "integral"}
_TWO_OR_THREE = {"feval", "fred", "volt"}

# AST nodes are tuples: (kind, token, child...); keeping the node tags parallel
# to MATLAB's lexer labels makes the lowering steps auditable.
Node = tuple


class ParseResult(NamedTuple):
    """Six outputs from MATLAB natural-syntax parsing, in source order.

    Provenance
    ----------
    MATLAB source : @stringParser/str2anon.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """

    an_fun: str | tuple[str, ...]
    independent_vars: tuple[str, str]
    variable_names: tuple[str, ...]
    pde_variable_names: tuple[str, ...]
    eigenvalue_names: tuple[str, ...]
    comma_separated: bool


def str2anon(
    text: str,
    problem_type: Literal["bvp", "ivp", "eig", "pde"],
    field_type: str | None = None,
    *,
    outputs: Literal[1, 2, 3, 4, 5, 6] = 1,
) -> str | tuple[str, ...] | ParseResult:
    """Convert Chebfun natural syntax into callback text or six metadata outputs.

    The default returns the one-output MATLAB callback representation. Set
    ``outputs=6`` to receive ``ParseResult`` containing MATLAB's six-output
    values, in the same order; outputs 2 through 5 return that prefix. EIG callbacks and expressions are two-tuples.

    Provenance
    ----------
    MATLAB source : @stringParser/str2anon.m, @stringParser/lexer.m,
        @stringParser/parser.m, @stringParser/splitTree.m,
        @stringParser/splitTreeEIG.m, @stringParser/splitTreePDE.m,
        @stringParser/tree2prefix.m, @stringParser/pref2inf.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df

    Python represents MATLAB cell arrays as tuples. The six-output result
    supports unpacking as well as named field access. One-output INITSCALAR
    calls preserve the source failure: MATLAB does not assign anFunComplete.
    Lexer identifier discovery uses the source token alphabet; MATLAB symvar
    edge cases remain unqualified.
    """
    if not isinstance(text, str):
        raise TypeError("str2anon expects a string")
    kind = problem_type
    if kind.lower() not in {"bvp", "ivp", "eig", "pde"}:
        raise ValueError("problem_type must be bvp, ivp, eig, or pde")
    if outputs not in range(1, 7):
        raise ValueError("outputs must be an integer from 1 to 6")
    lex = _Lexer(text, kind)
    variables = list(lex.variable_names)
    pde_vars = list(lex.pde_names)
    eig_vars = list(lex.eigen_names)
    ind = lex.independent_vars

    if not variables and field_type != "INITSCALAR":
        if field_type == "INIT":
            message = (
                "No dependent variables detected in the initial field. Input "
                'must be of the form "u = x, v = 2*x, ...'
            )
        elif ind[1]:
            message = (
                f"No dependent variables detected. Variables '{ind[0]}' and "
                f"'{ind[1]}' treated as independent."
            )
        else:
            message = (
                f"No dependent variables detected. Variable '{ind[0]}' treated as independent."
            )
        raise _StringParserError("CHEBFUN:STRINGPARSER:str2anon:depvars", message)
    if len(pde_vars) > 1:
        raise _StringParserError(
            "CHEBFUN:STRINGPARSER:str2anon:pdeVariables",
            "Only one time derivative per line is allowed",
        )

    tree = _Parser(lex.tokens).parse()

    if kind.lower() in {"bvp", "ivp"}:
        residual = _split_equation(tree)
        metadata_tree = residual
        raw = par_simp(_render(residual))
        exprs: tuple[str, ...] | str = raw
        comma = _contains(tree, "COMMA")
    elif kind == "pde":
        pde_tree, pde_sign = _split_pde(tree)
        metadata_tree = _split_equation(pde_tree)
        # Source str2anon prepends UN- to the prefix tree before pref2inf.
        # Rendering first and wrapping an already simplified string changes
        # parSimp semantics for a leading negative term (u_t = v).
        if pde_sign == 1:
            metadata_tree = ("UN-", "-", metadata_tree)
        raw = par_simp(_render(metadata_tree))
        exprs = raw
        comma = _contains(pde_tree, "COMMA")
    elif kind == "eig":
        spatial_tree, eig_tree, eig_sign = _split_eig(tree)
        metadata_tree = spatial_tree
        spatial = par_simp(_render(_split_equation(spatial_tree)))
        coefficient = ""
        if eig_tree is not None:
            if eig_sign == -1:
                eig_tree = ("UN-", "-", eig_tree)
            eig_tree = _replace_lambda(eig_tree)
            coefficient = par_simp(_render(eig_tree))
        exprs = (spatial, coefficient) if coefficient else spatial
        comma = _contains(spatial_tree, "COMMA")

    else:
        raise UnboundLocalError("MATLAB str2anon leaves prefixOut undefined for this problemType")

    # Source pref2inf emits kernel anonymous callbacks and reports their bound
    # variable names; those names must not leak into dependent variable metadata.
    kernel_vars: set[str] = set()
    _collect_kernel_vars(metadata_tree, kernel_vars)
    variables = [name for name in variables if name not in kernel_vars]

    if outputs > 1:
        result = ParseResult(
            an_fun=exprs,
            independent_vars=ind,
            variable_names=tuple(variables),
            pde_variable_names=tuple(pde_vars),
            eigenvalue_names=tuple(eig_vars),
            comma_separated=comma,
        )
        return result if outputs == 6 else result[:outputs]

    if field_type == "INITSCALAR":
        # str2anon.m:176-183 never assigns this one-output value for INITSCALAR.
        raise UnboundLocalError("MATLAB str2anon leaves anFunComplete undefined for INITSCALAR")
    # Source callback assembly includes slot2 only when slot1 is nonempty.
    independent_args = [ind[0]] + ([ind[1]] if ind[1] else []) if ind[0] else []
    names = independent_args + variables
    arglist = ",".join(names)
    if isinstance(exprs, tuple):
        callbacks = (_callback(arglist, exprs[0], len(variables)), _callback(arglist, exprs[1], 1))
        return callbacks if len(callbacks) > 1 else callbacks[0]
    return _callback(arglist, exprs, len(variables))


def _callback(arglist: str, expression: str, nvars: int) -> str:
    if nvars > 1:
        return f"@({arglist}) [{expression}]"
    return f"@({arglist}) {expression}"


def _normalize_lexer_signs(text: str) -> str:
    chars = list(text)
    k = 0
    while k < len(chars) - 2:
        if chars[k] in "+-" and chars[k + 1] in "+-":
            chars[k] = "+" if chars[k] == chars[k + 1] else "-"
            chars.pop(k + 1)
        else:
            k += 1
    return "".join(chars)


class _Lexer:
    def __init__(self, text: str, problem_type: str):
        self.problem_type = problem_type
        text = text.replace(" ", "").replace('"', "''").replace("`", "'")
        if text.endswith(","):
            text = text[:-1]
        text = _normalize_lexer_signs(text)
        self.text = text
        discovery = re.sub(
            r"(?<![A-Za-z_0-9])(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[ij]?", "", text
        )
        name_occurrences = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", discovery)
        raw_names = set(name_occurrences)
        excluded = _FUNC1 | _FUNC2 | _FUNC3 | _FLAGS | {"true", "false", "pi"}
        variables = sorted(
            name for name in raw_names if name not in excluded and name not in {"x", "r", "t"}
        )
        eigen_names = [
            name
            for name in name_occurrences
            if problem_type == "eig" and name in {"l", "lam", "lambda"}
        ]
        variables = [name for name in variables if name not in eigen_names]
        pde_names = list(
            dict.fromkeys(name for name in name_occurrences if name in variables and "_" in name)
        )
        variables = [name for name in variables if name not in pde_names]
        coords = [name for name in ("r", "t", "x") if name in raw_names]
        first = "r" if "r" in coords else "t" if "t" in coords else "x" if "x" in coords else ""
        second = ""
        if problem_type == "pde" and pde_names:
            second = pde_names[-1].split("_", 1)[1]
        self.independent_vars = (first, second)
        self.variable_names = tuple(variables)
        self.pde_names = tuple(pde_names)
        self.eigen_names = tuple(eigen_names)
        self.tokens = self._scan(text)
        # Literal lexer.m:359-365 also rejects two explicit coordinates in PDE
        # mode because its second condition is not restricted to BVP/EIG.
        if len(coords) > 1:
            raise _StringParserError(
                "CHEBFUN:STRINGPARSER:lexer:tooManyIndVars",
                "Too many independent variables in input.",
            )

    def _scan(self, text: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        i = 0
        previous = "operator"
        seen_pde: set[str] = set()
        while i < len(text):
            c = text[i]
            if c in "+-" and previous in {"operator", "unary"}:
                out.append((c, "UN" + c))
                i += 1
                previous = "unary"
                continue
            number = re.match(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[ij]?", text[i:])
            if number:
                token = number.group(0)
                # A terminal dot in 2.* is an operator, not part of 2.
                if (
                    token.endswith(".")
                    and i + len(token) < len(text)
                    and text[i + len(token)] in "+-*/^"
                ):
                    token = token[:-1]
                out.append((token, "NUM"))
                i += len(token)
                previous = "num"
                continue
            if c == "'":
                match = re.match(r"'+", text[i:])
                token = match.group(0)
                out.append((token, "DER" + str(len(token))))
                i += len(token)
                previous = "deriv"
                continue
            ident = re.match(r"[A-Za-z_][A-Za-z_0-9]*", text[i:])
            if ident:
                token = ident.group(0)
                i += len(token)
                if token == "pi":
                    label = "NUM"
                elif self.problem_type == "eig" and token in {"l", "lam", "lambda"}:
                    label = "LAMBDA"
                elif token in self.pde_names:
                    # The source removes a name from varNames on its first
                    # occurrence; later occurrences fall through to INDVAR.
                    label = "INDVAR" if token in seen_pde else "PDEVAR"
                    seen_pde.add(token)
                elif token in self.variable_names:
                    label = "VAR"
                elif token in _FUNC1:
                    label = "FUNC1"
                elif token in _FUNC2:
                    label = "FUNC2"
                elif token in _FUNC3:
                    label = "FUNC3"
                elif token in _FLAGS:
                    token, label = "'" + token + "'", "STR"
                else:
                    label = "INDVAR"
                out.append((token, label))
                previous = "char"
                continue
            if c == "." and i + 1 < len(text) and text[i + 1] in "+-*/^":
                op = text[i + 1]
                labels = {"+": "OP+", "-": "OP-", "*": "OP*", "/": "OP/", "^": "OP^"}
                out.append(("." + op, labels[op]))
                i += 2
                previous = "operator"
                continue
            if c == "." and i + 1 < len(text) and text[i + 1].isdigit():
                number = re.match(r"\.\d+(?:[eE][+-]?\d+)?[ij]?", text[i:])
                out.append((number.group(0), "NUM"))
                i += len(number.group(0))
                previous = "num"
                continue
            if c == ",":
                out.append((c, "COMMA"))
                i += 1
                previous = "comma"
                continue
            if c in "()":
                out.append((c, "LPAR" if c == "(" else "RPAR"))
                i += 1
                previous = "operator" if c == "(" else "char"
                continue
            pair = text[i : i + 2]
            if pair in {"<=", ">=", "==", "~="}:
                labels = {"<=": "OP<=", ">=": "OP>=", "==": "OP==", "~=": "OP~="}
                out.append((pair, labels[pair]))
                i += 2
                previous = "operator"
                continue
            if c == "~":
                raise _StringParserError(
                    "CHEBFUN:STRINGPARSER:lexer:UnsupportedOperator", "Unsupported operator ~."
                )
            if c in "+-*/^=<>~":
                labels = {
                    "+": "OP+",
                    "-": "OP-",
                    "*": "OP*",
                    "/": "OP/",
                    "^": "OP^",
                    "=": "OP=",
                    ">": "OP>",
                    "<": "OP<",
                    "~": "OP~",
                }
                out.append((c, labels[c]))
                i += 1
                previous = "operator"
                continue
            raise _StringParserError(
                "CHEBFUN:STRINGPARSER:lexer:unknownType",
                f"Invalid token '{c}' in input.\nChebgui does not support "
                f"'{c}' in its input fields.",
            )
        out.append(("", "$"))
        return out


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]):
        self.tokens = tokens
        self.i = 0

    @property
    def label(self) -> str:
        return self.tokens[self.i][1]

    @property
    def token(self) -> str:
        return self.tokens[self.i][0]

    def take(self, label: str | None = None) -> tuple[str, str]:
        tok = self.tokens[self.i]
        if label is not None and tok[1] != label:
            raise _StringParserError(
                "CHEBFUN:Parse:parenths", "Parenthesis imbalance in input fields."
            )
        self.i += 1
        return tok

    def parse(self) -> Node:
        if self.label not in {
            "NUM",
            "VAR",
            "INDVAR",
            "PDEVAR",
            "LAMBDA",
            "FUNC1",
            "FUNC2",
            "FUNC3",
            "UN-",
            "UN+",
            "LPAR",
        }:
            raise _StringParserError(
                "CHEBFUN:Parse:start", "Input field started with unaccepted symbol."
            )
        tree = self.comma()
        if self.label != "$":
            raise _StringParserError(
                "CHEBFUN:Parse:end", "Input expression ended in unexpected manner."
            )
        return tree

    def comma(self) -> Node:
        node = self.equality()
        if self.label == "COMMA":
            self.take()
            node = ("COMMA", ",", node, self.comma())
        return node

    def equality(self) -> Node:
        node = self.relational()
        if self.label == "OP=":
            op, kind = self.take()
            node = (kind, op, node, self.relational())
        return node

    def relational(self) -> Node:
        node = self.add()
        if self.label in {"OP>", "OP>=", "OP<", "OP<="}:
            op, kind = self.take()
            node = (kind, op, node, self.add())
        return node

    def add(self) -> Node:
        node = self.multiply()
        while self.label in {"OP+", "OP-"}:
            op, kind = self.take()
            node = (kind, op, node, self.multiply())
        return node

    def multiply(self) -> Node:
        node = self.power()
        while self.label in {"OP*", "OP/"}:
            op, kind = self.take()
            node = (kind, op, node, self.power())
            _reject_pde(
                node,
                "Cannot multiply time derivative"
                if kind == "OP*"
                else "Cannot divide with time derivatives",
            )
        return node

    def power(self) -> Node:
        node = self.unary()
        while self.label == "OP^" or self.label.startswith("DER"):
            if self.label == "OP^":
                op, kind = self.take()
                node = (kind, op, node, self.unary())
                _reject_pde(node, "Cannot take powers with time derivative")
            else:
                _reject_pde(node, "Cannot differentiate time derivative")
                order = self.take()[1][3:]
                node = ("DER", order, node)
        return node

    def unary(self) -> Node:
        if self.label in {"UN+", "UN-", "OP+", "OP-"}:
            op, kind = self.take()
            return ("UN" + op[-1], op, self.unary())
        return self.primary()

    def primary(self) -> Node:
        tok, label = self.tokens[self.i]
        if label in {"NUM", "VAR", "INDVAR", "PDEVAR", "LAMBDA", "STR"}:
            self.take()
            node: Node = (label, tok)
            if label == "VAR" and self.label.startswith("DER"):
                order = self.take()[1][3:]
                node = ("DER", order, node)
            if label == "VAR" and self.label == "LPAR":
                self.take("LPAR")
                argument = self.add()
                if self.label == "RPAR":
                    self.take()
                    node = ("FUNC2", "feval", node, argument)
                elif self.label == "COMMA":
                    self.take()
                    if self.label != "STR":
                        raise _StringParserError(
                            "CHEBFUN:Parse:secondArg",
                            "Invalid second argument to u(0,...) type of expression.",
                        )
                    flag, label = self.take()
                    self.take("RPAR")
                    node = ("FUNC3", "feval", node, argument, (label, flag))
                else:
                    self.take("RPAR")
            return node
        if label.startswith("FUNC"):
            self.take()
            name = tok
            if self.label != "LPAR":
                raise _StringParserError(
                    "CHEBFUN:Parse:parenths",
                    "Need parenthesis when using functions in input fields.",
                )
            self.take("LPAR")
            first = self.add()
            if label == "FUNC1":
                if self.label == "COMMA":
                    raise _StringParserError(
                        "CHEBFUN:Parse:func1", f"Method '{name}' only takes one input argument."
                    )
                self.take("RPAR")
                _reject_pde(first, "Cannot use time derivative as function arguments.")
                return ("FUNC1", name, first)
            _reject_pde(first, "Cannot use time derivative as function arguments.")
            if self.label == "RPAR":
                self.take()
                allowed = _ONE_OR_TWO if label == "FUNC2" else _ONE_OR_THREE
                if name in allowed:
                    return ("FUNC1", name, first)
                raise _StringParserError(
                    "CHEBFUN:Parse:func2", f"Method '{name}' requires two input arguments."
                )
            self.take("COMMA")
            second = self.add()
            if label == "FUNC2":
                self.take("RPAR")
                _reject_pde(second, "Cannot use time derivative as function arguments.")
                return ("FUNC2", name, first, second)
            _reject_pde(second, "Cannot use time derivative as function arguments.")
            if self.label == "COMMA":
                self.take()
                third = self.add()
                self.take("RPAR")
                # Source parseFunction3 does not check its third arg's pdeflag.
                return ("FUNC3", name, first, second, third)
            self.take("RPAR")
            if name in _TWO_OR_THREE:
                return ("FUNC2", name, first, second)
            raise _StringParserError(
                "CHEBFUN:Parse:parenths", "Parenthesis imbalance in input fields."
            )
        if label == "LPAR":
            self.take()
            node = self.relational()
            self.take("RPAR")
            return node
        raise _StringParserError(
            "CHEBFUN:Parse:terminal", f"Unrecognized character in input field:{label}"
        )


def _has_pde(node: Node) -> bool:
    """Propagate the source parser's pdeflag (function results reset it)."""
    if node[0] == "PDEVAR":
        return True
    if node[0].startswith("FUNC"):
        return False
    return any(_has_pde(child) for child in node[2:])


def _reject_pde(node: Node, message: str) -> None:
    if _has_pde(node):
        raise _StringParserError("CHEBFUN:STRINGPARSER:parser:PDE", message)


def _split_equation(node: Node) -> Node:
    if node[0] == "COMMA":
        return ("COMMA", ",", _split_equation(node[2]), _split_equation(node[3]))
    if node[0] == "OP=":
        return ("OP-", "-", node[2], node[3])
    return node


def _contains(node: Node, kind: str) -> bool:
    if node[0] == kind:
        return True
    return any(_contains(child, kind) for child in node[2:] if isinstance(child, tuple))


def _split_pde(node: Node) -> tuple[Node, int]:
    new_tree, _, sign = _find_pde(node, 1)
    return new_tree, sign


def _find_pde(node: Node, sign: int) -> tuple[Node, Node | None, int]:
    """Port findPDE's left/right traversal, including its printed diagnostic."""
    if len(node) == 2:
        if node[0] == "PDEVAR":
            return ("NUM", "0"), node, sign
        return node, None, sign
    children = list(node[2:])
    left_tree = right_tree = None
    if len(children) == 1:
        children[0], right_tree, sign = _find_pde(children[0], sign)
    else:
        children[0], left_tree, sign = _find_pde(children[0], sign)
        children[1], right_tree, sign = _find_pde(children[1], sign)
    result = left_tree
    if left_tree is not None and node[0] == "OP*":
        result = node
    if right_tree is not None:
        result = right_tree
        if node[0] == "OP=":
            print("PDE on right")
            sign = -sign
        elif node[0] in {"OP-", "UN-"}:
            sign = -sign
    return (node[0], node[1], *children), result, sign


def _split_eig(node: Node) -> tuple[Node, Node | None, int]:
    """Port findLambda's traversal, including division and sign conventions."""
    new_tree, lambda_tree, sign, factors = _find_lambda(node, 1)
    if factors >= 2:
        raise _StringParserError(
            "CHEBFUN:STRINGPARSER:splitTreeEIG:factors",
            "Only one factor where lambda appears is supported.\nFor example,\n"
            "       lambda*u*x    and    lambda*(u+ u')*2\n"
            "are not supported and should be rewritten as\n"
            "       lambda*(u*x)  and    lambda*((u+ u')*2),\n"
            "respectively.",
        )
    return _split_equation(new_tree), lambda_tree, sign


def _find_lambda(node: Node, sign: int) -> tuple[Node, Node | None, int, int]:
    if len(node) == 2:
        if node[0] == "LAMBDA":
            return ("NUM", "0"), node, sign, 0
        return node, None, sign, 0
    children = list(node[2:])
    # MATLAB's struct traversal uses left/right, not a third function argument.
    left_tree = right_tree = None
    factors = 0
    if len(children) == 1:
        children[0], right_tree, sign, factors = _find_lambda(children[0], sign)
    else:
        children[0], left_tree, sign, left_factors = _find_lambda(children[0], sign)
        children[1], right_tree, sign, right_factors = _find_lambda(children[1], sign)
        factors = left_factors + right_factors
    result = None
    if left_tree is not None:
        if node[0] == "OP*":
            result = node
        else:
            result = left_tree
            if node[0] == "OP=":
                sign = -sign
        factors += node[0] in {"OP*", "OP/"}
    if right_tree is not None:
        if node[0] == "OP*":
            result = node
        elif node[0] in {"UN-", "OP-"}:
            result = ("UN-", "-", right_tree)
        else:
            result = right_tree
        factors += node[0] in {"OP*", "OP/"}
    return (node[0], node[1], *children), result, sign, factors


def _replace_lambda(node: Node) -> Node:
    if node[0] == "LAMBDA":
        return ("NUM", "1")
    return tuple(_replace_lambda(child) if isinstance(child, tuple) else child for child in node)


def _collect_kernel_vars(node: Node, output: set[str]) -> None:
    if node[0] == "FUNC2" and node[1] in {"fred", "volt"}:
        # pref2inf relexes the rendered kernel in BVP mode, without defaults.
        output.update(_Lexer(_render(node[2]), "bvp").variable_names)
    for child in node[2:]:
        _collect_kernel_vars(child, output)


def _render(node: Node) -> str:
    kind = node[0]
    if kind in {"NUM", "VAR", "INDVAR", "PDEVAR", "LAMBDA", "STR"}:
        return node[1]
    if kind in {"OP+", "OP-", "OP*", "OP/", "OP^", "OP=", "OP>", "OP>=", "OP<", "OP<=", "COMMA"}:
        a, b = _render(node[2]), _render(node[3])
        if kind == "OP=":
            return a + "=" + b
        if kind == "OP+":
            if a == b == "0":
                return ""
            if a == "0":
                return b
            if b == "0":
                return a
            return f"({a}+{b})"
        if kind == "OP-":
            if a == b == "0":
                return ""
            if a == "0":
                return "-" + b
            if b == "0":
                return a
            return f"({a}-{b})"
        if kind == "OP*":
            if a == b == "1":
                return "1"
            if a == "1":
                return b
            if b == "1":
                return a
            if a == "-1":
                return "-" + b
            if b == "-1":
                return "-" + a
            if a in {"0", "-0"} or b in {"0", "-0"}:
                return "0"
            return f"({a}.*{b})"
        if kind == "OP/":
            return a if b == "1" else f"({a}./{b})"
        if kind == "OP^":
            return f"{a}.^({b})"
        if kind == "COMMA":
            return f"({a};{b})"
        return f"({a}{node[1]}{b})"
    if kind in {"UN-", "UN+"}:
        arg = _render(node[2])
        return "-" + arg if kind == "UN-" else arg
    if kind == "DER":
        arg = _render(node[2])
        return f"diff({arg})" if node[1] == "1" else f"diff({arg},{node[1]})"
    if kind == "FUNC1":
        return f"{node[1]}({_render(node[2])})"
    if kind == "FUNC2":
        name, first, second = node[1], node[2], node[3]
        if name in {"diff", "cumsum"} and _render(second) == "1":
            return f"{name}({_render(first)})"
        if name in {"fred", "volt"}:
            kernel = _render(first)
            kernel_lex = _Lexer(kernel, "bvp")
            coord = kernel_lex.independent_vars[0]
            kval = kernel_lex.variable_names[0]
            return f"{name}(@({coord},{kval}){kernel},{_render(second)})"
        return f"{name}({_render(first)},{_render(second)})"
    if kind == "FUNC3":
        return f"{node[1]}({_render(node[2])},{_render(node[3])},{_render(node[4])})"
    raise ValueError(f"Unsupported AST node {kind}")
