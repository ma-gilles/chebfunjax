"""Chebgui problem data and literal demo loading, without opening a window.

Provenance
----------
MATLAB source: @chebgui/{chebgui,set,demo2chebgui}.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
The loader accepts the literal string/cell assignments used by all bundled
demos; arbitrary MATLAB statements are not executed.
"""
import re
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path


def _defaults():
    return dict(damping='1', plotting='0.5', grid=1,
                discretization='values', ivpSolver='ode113', pdeSolver='pde15s',
                pdeholdplot=0, fixYaxisLower='', fixYaxisUpper='', fixN='', numeigs='')


@dataclass
class ChebguiData:
    """Source problem fields; ``set`` returns a new value like MATLAB SET."""

    type: str = ''
    domain: str = ''
    DE: str | list = ''
    LBC: str | list = ''
    RBC: str | list = ''
    BC: str | list = ''
    timedomain: str = ''
    sigma: str = ''
    init: str | list = ''
    tol: str = '5e-13'
    options: dict = field(default_factory=_defaults)

    def set(self, name, value):
        """Case-insensitive source field/option assignment and cell collapse.

        Provenance
        ----------
        MATLAB source: @chebgui/set.m.
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        if isinstance(value, (tuple, list)) and value and value[0] == '':
            value = ''
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = format(value, '.5g')
        key = name.lower()
        fields = {n.lower(): n for n in self.__dataclass_fields__}
        options = {n.lower(): n for n in _defaults()}
        result = replace(self, options=dict(self.options))
        if key == 'type' and (not isinstance(value, str)
                              or value.lower() not in ('bvp', 'ivp', 'pde', 'eig')):
            raise ValueError('CHEBFUN:CHEBGUI:set:type')
        if key in fields:
            setattr(result, fields[key], value)
        elif key in options:
            if key == 'grid':
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = float('nan')
            result.options[options[key]] = value
        else:
            raise ValueError('CHEBFUN:CHEBGUI:set:propName')
        return result


def _literal(text):
    """Decode MATLAB single-quoted strings and one-dimensional string cells."""
    text = text.strip().removesuffix(';').rstrip()
    cell = text.startswith('{') and text.endswith('}')
    body = text[1:-1] if cell else text
    values, pos = [], 0
    while pos < len(body):
        while pos < len(body) and (body[pos].isspace() or body[pos] in ',;'):
            pos += 1
        if pos == len(body):
            break
        if body[pos] != "'":
            raise ValueError('Demo assignment is not a literal string/cell')
        pos += 1
        value = []
        while pos < len(body):
            ch = body[pos]
            pos += 1
            if ch == "'":
                if pos < len(body) and body[pos] == "'":
                    value.append("'")
                    pos += 1
                else:
                    break
            else:
                value.append(ch)
        else:
            raise ValueError('Unterminated demo string')
        values.append(''.join(value))
    if not cell and len(values) != 1:
        raise ValueError('Expected one demo string')
    return values if cell else values[0]


def _statements(line):
    """Split assignment statements outside quoted strings and cell braces."""
    start, pos, depth, quoted = 0, 0, 0, False
    while pos < len(line):
        ch = line[pos]
        if ch == "'":
            if quoted and pos + 1 < len(line) and line[pos + 1] == "'":
                pos += 2
                continue
            quoted = not quoted
        elif not quoted:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
            elif ch == ';' and depth == 0:
                yield line[start:pos]
                start = pos + 1
        pos += 1
    if line[start:].strip():
        yield line[start:]


def demo2chebgui(path):
    """Load source literal demos, rename t, and apply sorted workspace fields.

    Provenance
    ----------
    MATLAB source: @chebgui/demo2chebgui.m.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    """
    try:
        lines = Path(path).read_text().splitlines()
    except OSError as exc:
        raise ValueError('CHEBFUN:CHEBGUI:demo2chebgui:noload') from exc
    workspace = {}
    for line in lines:
        if not line or '=' not in line or line.startswith('#'):
            continue
        for statement in _statements(line):
            match = re.fullmatch(r'\s*(\w+)\s*=\s*(.*?)\s*', statement)
            if match is None:
                raise ValueError('Unsupported demo assignment')
            workspace[match[1]] = _literal(match[2])
    if 't' in workspace:
        workspace['timedomain'] = workspace.pop('t')
    result = ChebguiData(type='bvp')
    for name in sorted(workspace):
        try:
            result = result.set(name, workspace[name])
        except ValueError:
            warnings.warn('CHEBFUN:CHEBGUI:loaddemos:unknown: ' + name, stacklevel=2)
    return result
