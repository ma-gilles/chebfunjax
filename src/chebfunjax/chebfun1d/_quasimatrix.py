"""Public representation conversion and construction from Chebfun 7574c77.

Provenance
----------
MATLAB source: quasimatrix.m, @chebfun/cheb2quasi.m, and the positional
constructor grammar in @chebfun/chebfun.m (parseInputs).
Copyright 2017 by The University of Oxford and The Chebfun Developers.
"""

import jax.numpy as jnp

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.domain import Domain


def cheb2quasi(f):
    """Extract independent scalar columns without resampling their data.

    A scalar or empty Chebfun stays scalar; an existing multi-column
    Quasimatrix stays unchanged. Transposed columns retain their orientation.

    Provenance
    ----------
    MATLAB source: @chebfun/cheb2quasi.m, @chebfun/num2cell.m.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    """
    from .chebfun import Chebfun
    from .linalg import Quasimatrix

    if isinstance(f, Quasimatrix):
        return f if f.n_cols > 1 else f.cols[0]
    if not isinstance(f, Chebfun):
        raise TypeError("cheb2quasi requires a Chebfun or Quasimatrix")
    columns = f.num2cell()
    return columns[0] if len(columns) == 1 else Quasimatrix(columns, f.domain)


def _constructor_options(args, op):
    """Translate source positional arguments to the existing Python boundary."""
    args = list(args)
    options = {}
    preference_fields = {}
    try:
        numeric_op = not callable(op) and not isinstance(op, str) and jnp.issubdtype(
            jnp.asarray(op).dtype, jnp.number)
    except (TypeError, ValueError):
        numeric_op = False
    if args and isinstance(args[0], Domain):
        options['domain'] = args.pop(0).breakpoints
    elif args and not isinstance(args[0], (str, dict, ChebfunPref)):
        value = jnp.asarray(args[0])
        if value.ndim and (not value.size or max(value.shape) >= 2):
            options['domain'] = args.pop(0)

    aliases = {
        'minsamples': 'min_samples', 'eps': 'eps', 'maxdegree': 'max_length',
        'splitlength': 'split_length', 'splitdegree': 'split_length',
        'splitmaxlength': 'split_max_length', 'exps': 'exps',
        'exponents': 'exps', 'singtype': 'singType', 'chebkind': 'chebkind',
        'vscale': 'vscale', 'hscale': 'hscale',
    }
    bare = {'equi': 'equi', 'vectorize': 'vectorize',
            'vectorise': 'vectorize', 'doublelength': 'doubleLength',
            'turbo': 'turbo', 'coeffs': 'coeffs',
            'periodic': 'periodic', 'trig': 'trig'}
    while args:
        item = args.pop(0)
        if isinstance(item, (dict, ChebfunPref)):
            if 'pref' in options:
                raise ValueError('CHEBFUN:CHEBFUN:parseInputs:twoPrefs')
            options['pref'] = item
            continue
        if not isinstance(item, str):
            value = jnp.asarray(item)
            if value.ndim == 0 and jnp.issubdtype(value.dtype, jnp.number):
                options['n'] = item
                continue
            raise ValueError('CHEBFUN:CHEBFUN:parseInputs:badInputNumeric')
        key = item.lower()
        if key == 'coeffs' and not numeric_op and isinstance(op, (list, tuple)):
            raise ValueError('CHEBFUN:CHEBFUN:parseInputs:coeffcell')
        if key in bare and (key != 'coeffs' or numeric_op):
            options[bare[key]] = True
            continue
        if key == 'novectorcheck':
            options['vector_check'] = False
            continue
        if not args:
            if key in aliases or key in ('splitting', 'resampling', 'vectorcheck', 'blowup') or key.startswith('trunc'):
                raise IndexError('MATLAB:badsubscript: Missing constructor option value')
            raise ValueError('CHEBFUN:CHEBFUN:parseInputs:noPrefValue: ' + item)
        value = args.pop(0)
        if key in ('splitting', 'resampling'):
            if not isinstance(value, str) or value.lower() not in ('on', 'off'):
                identifier = 'badSplitting' if key == 'splitting' else 'badResampling'
                raise ValueError('CHEBFUN:CHEBFUN:parseInputs:' + identifier)
            options[key] = value.lower() == 'on'
        elif key == 'vectorcheck':
            options['vector_check'] = isinstance(value, str) and value.lower() == 'on'
        elif key.startswith('trunc'):
            options['trunc'] = value
        elif key == 'blowup':
            if isinstance(value, str):
                value = {'off': 0, 'on': 1}.get(value.lower(), value)
            if value not in (0, 1, 2):
                raise ValueError('CHEBFUN:CHEBFUN:parseInputs:badBlowupOption')
            options['blowup'] = bool(value)
            if value:
                options['singType'] = ['pole' if value == 1 else 'sing']
        elif key in aliases:
            options[aliases[key]] = value
        else:
            # Native arbitrary name/value pairs update preferences. Retain
            # that merge rather than inventing constructor keyword names.
            preference_fields[item] = value
    if preference_fields:
        options['pref'] = ChebfunPref(ChebfunPref(options.get('pref')),
                                     preference_fields)
    return options


def quasimatrix(f=None, *args, **kwargs):
    """Construct independent scalar columns through the public Chebfun builder.

    Lists/tuples containing Chebfuns use ``cell2quasi``. Existing Chebfuns
    are converted directly, so constructor options are ignored on these two
    source fast paths. Other operands use the existing ``chebfun`` constructor.
    Native positional domain, preference, length and flag arguments are
    accepted; Python constructor keywords are final explicit overrides.
    Constructor algorithm limitations are inherited rather than bypassed.

    Provenance
    ----------
    MATLAB source: quasimatrix.m, @chebfun/cheb2quasi.m.
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
    """
    from .chebfun import Chebfun, cell2quasi, chebfun
    from .linalg import Quasimatrix

    if isinstance(f, (list, tuple)) and any(
            isinstance(x, (Chebfun, Quasimatrix)) for x in f):
        cells = []
        for value in f:
            if isinstance(value, Quasimatrix):
                if value.n_cols != 1:
                    raise ValueError('CHEBFUN:CHEBFUN:cell2quasi:notascalar')
                value = value.cols[0]
            cells.append(value)
        return cell2quasi(cells)
    if not isinstance(f, (Chebfun, Quasimatrix)):
        # CHEBFUN returns empty operands before parsing any supplied options.
        empty = f is None or (not callable(f) and (
            getattr(f, 'size', None) == 0 or (
                hasattr(f, '__len__') and getattr(f, 'ndim', 1) != 0
                and len(f) == 0)))
        options = {} if empty else _constructor_options(args, f)
        options.update(kwargs)
        f = chebfun(f, **options)
    return cheb2quasi(f)
