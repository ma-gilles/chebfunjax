"""Chebgui field assembly and safe numeric metadata parsing.

Source: @chebgui/setupFields.m, isIVPorFVP.m and bcReform.m at
Chebfun7574c77680d7e82b79626300bf255498271a72df.
No differential equation is evaluated or solved by this string adapter.
"""
import ast
import math
import operator
import re
from typing import NamedTuple

from chebfunjax.utils.string_parser import ParseResult, str2anon


def numeric_scalar(text):
    """Evaluate bounded scalar metadata syntax, never arbitrary MATLAB code.

    Provenance
    ----------
    MATLAB source: @chebguiExporterPDE/exportInfo.m (bounded str2num metadata adapter).
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    """
    tree = ast.parse(text.replace('^', '**').strip(), mode='eval')
    def visit(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name) and node.id in ('pi', 'inf', 'Inf', 'eps'):
            return {'pi': math.pi, 'inf': math.inf, 'Inf': math.inf, 'eps': 2**-52}[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow):
            return {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                    ast.Div: operator.truediv, ast.Pow: operator.pow}[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError('Unsupported numeric metadata expression')
    return visit(tree.body)


def numeric_vector(text):
    """Numeric vector.

    Provenance
    ----------
    MATLAB source: @chebguiExporterPDE/{exportInfo,printDescription}.m (bounded str2num/eval metadata adapter).
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    """
    text = text.strip()
    if text.startswith('linspace(') and text.endswith(')'):
        args = [numeric_scalar(v) for v in text[9:-1].split(',')]
        a, b = args[:2]
        count = int(args[2]) if len(args) > 2 else 100
        if count > 100000 or count < 0:
            raise ValueError('Numeric metadata vector exceeds bounded parser scope')
        return [b] if count == 1 else [a + (b-a)*k/(count-1) for k in range(count)]
    if text.startswith('[') and text.endswith(']'):
        body = text[1:-1].strip()
        if not body:
            return []
        result = []
        for value in re.split(r'[,;\s]+', body):
            result.extend(numeric_vector(value))
        return result
    if ':' in text:
        values = [numeric_scalar(v) for v in text.split(':')]
        if len(values) == 2:
            a, b = values
            step = 1
        elif len(values) == 3:
            a, step, b = values
        else:
            raise ValueError('Invalid colon metadata expression')
        count = max(0, math.floor((b-a)/step + 1e-12)+1)
        if count > 100000:
            raise ValueError('Numeric metadata vector exceeds bounded parser scope')
        return [a+step*k for k in range(count)]
    return [numeric_scalar(text)]


def cellstr(value):
    """Cellstr.

    Provenance
    ----------
    MATLAB source: @chebgui/setupFields.m (cell-array representation adapter).
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    """
    return list(value) if isinstance(value, (tuple, list)) else [value]


def vectorize(text):
    """Vectorize.

    Provenance
    ----------
    MATLAB source: @chebguiExporterBVP/printSetup.m (vectorize calls).
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    """
    return re.sub(r'(?<!\.)([*/^])', r'.\1', text)


def pretty_print_feval(text, varnames):
    """Pretty print feval.

    Provenance
    ----------
    MATLAB source: @chebguiExporter/chebguiExporter.m (prettyPrintFevalString).
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    """
    if isinstance(text, (tuple, list)):
        return tuple(pretty_print_feval(v, varnames) for v in text)
    for name in varnames:
        text = text.replace('feval('+name+',', name+'(')
        for old, new in [('end', 'end'), ('right', 'end'),
                         ('start', name+'.ends(1)'), ('left', name+'.ends(1)')]:
            text = text.replace(name+"('"+old+"'", name+'('+new)
    return text


class FieldResult(NamedTuple):
    field: str | tuple
    all_var_string: str
    independent_vars: tuple
    pde_variable_names: tuple
    pde_flags: tuple
    eigenvalue_names: tuple
    variable_names: tuple


def _line(gui, text, kind):
    if '@' in text:
        return str2anon(text[text.index(')')+1:], gui.type, kind, outputs=6)
    problem = gui.type
    if kind in ('BC', 'BCnew'):
        try:
            numeric = numeric_vector(text)
        except (ValueError, SyntaxError, ZeroDivisionError):
            numeric = []
        if numeric or text.lower() in ('dirichlet', 'neumann', 'periodic'):
            field = text if numeric else "'"+text+"'"
            return ParseResult(field, (), (), (), (), False)
        problem = 'bvp'
    return str2anon(text, problem, kind, outputs=6)


def setup_fields(gui, value, kind, all_var_string=''):
    """Setup fields.

    Provenance
    ----------
    MATLAB source: @chebgui/setupFields.m.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    """
    rows = cellstr(value)
    parsed = [_line(gui, line, kind) for line in rows]
    if len(rows) == 1:
        item = parsed[0]
        expr = item.an_fun
        names = item.variable_names
        pdes = item.pde_variable_names
        flags = (bool(pdes),)
        if pdes:
            if gui.type.lower() != 'pde':
                raise ValueError('CHEBFUN:CHEBGUI:setupField:incorrectPDEmode')
            if len(names) > 1 or len(pdes) > 1:
                raise ValueError('CHEBFUN:CHEBGUI:setupFields:numberOfVariables')
            if names[0] != pdes[0].split('_')[0]:
                raise ValueError('CHEBFUN:CHEBGUI:setupFields:variableNames')
        if kind == 'DE':
            all_var_string = names[0]
        start = '@(DUMMYSPACE,'+all_var_string+') '
        if gui.type == 'eig' and isinstance(expr, tuple):
            field = tuple(start+v for v in expr)
        elif gui.type != 'pde' and kind == 'DE':
            field = start+expr
        elif names:
            if item.comma_separated:
                expr = '['+expr+']'
            field = (start if kind == 'BCnew' else '@('+all_var_string+') ')+expr
        else:
            field = expr
        return FieldResult(field, all_var_string, item.independent_vars, pdes,
                           flags, item.eigenvalue_names, names)
    names = tuple(sorted({name for item in parsed for name in item.variable_names}))
    ind = parsed[0].independent_vars
    for item in parsed[1:]:
        if not ind:
            ind = item.independent_vars
            continue
        if not item.independent_vars:
            continue
        if any(a and b and a != b for a, b in zip(ind, item.independent_vars, strict=True)):
            raise ValueError('CHEBFUN:CHEBGUI:setupFields:differentIndepVars')
    if kind == 'DE':
        all_var_string = ','.join(names)
    flags = [bool(item.pde_variable_names) for item in parsed]
    pdes = tuple(name for item in parsed for name in item.pde_variable_names)
    order = list(range(len(rows)))
    if any(flags):
        if gui.type.lower() != 'pde':
            raise ValueError('CHEBFUN:CHEBGUI:setupField:incorrectPDEmode')
        for k, item in enumerate(parsed):
            if not item.pde_variable_names:
                continue
            idx = names.index(item.pde_variable_names[0].split('_')[0])
            position = order.index(idx)
            order[position] = order[k]
            order[k] = idx
        order = sorted(range(len(order)), key=order.__getitem__)
        flags = [flags[k] for k in order]
    start = ('@(DUMMYSPACE,' if (gui.type != 'pde' and kind == 'DE') or kind == 'BCnew' else '@(')+all_var_string+') '
    expressions = [item.an_fun for item in parsed]
    if gui.type == 'eig' and isinstance(expressions[0], tuple):
        field = tuple(start+'['+';'.join(expressions[k][j] for k in order)+']' for j in range(2))
    else:
        expressions = [e[1:-1] if e.startswith('[') and e.endswith(']') else e for e in expressions]
        field = start+'['+'; '.join(expressions[k] for k in order)+']'
    return FieldResult(field, all_var_string, ind, pdes, tuple(flags),
                       parsed[-1].eigenvalue_names, names)


def is_ivp_or_fvp(gui, names):
    """Is ivp or fvp.

    Provenance
    ----------
    MATLAB source: @chebgui/isIVPorFVP.m.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    """
    dom = gui.domain.replace(',', ' ')
    spaces = [i for i, c in enumerate(dom) if c == ' ']
    left, right = dom[1:min(spaces)], dom[max(spaces)+1:-1]
    bc = cellstr(gui.BC)
    if bc == ['periodic']:
        return 0
    num = r'[+-]?[0-9]*\.?[0-9]*(e[0-9]+)*'
    locations = [set(), set(), set(), set()]
    for name in names:
        patterns = [name+r"'*\("+left+r'\)', name+r"'*\("+right+r'\)',
                    name+r"'*\("+num+r'\)', name+"[^'*\\("+num+r'\)]']
        for index, pattern in enumerate(patterns):
            for row, value in enumerate(bc):
                locations[index].update((row, match.start()) for match in re.finditer(pattern, value))
    l, r, point, glob = locations
    if point-(l|r) or glob or (l and r):
        return 0
    return 1 if l else 2


def bc_reform(domain, conditions, direction):
    """Bc reform.

    Provenance
    ----------
    MATLAB source: @chebgui/bcReform.m.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    """
    locs = [k for k, value in enumerate(domain) if value in ' ,']
    endpoint = domain[1:min(locs)] if direction == 1 else domain[max(locs)+1:-1]
    return [line.replace('('+endpoint+')', '') for line in conditions]
