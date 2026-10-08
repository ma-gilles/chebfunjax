"""Emit executable MATLAB source from Chebgui problem descriptions.

Source: @chebguiExporter and the BVP/IVP/EIG/PDE concrete classes,
Chebfun7574c77680d7e82b79626300bf255498271a72df. No solver is called.
"""
import os
from datetime import datetime
from pathlib import Path

from chebfunjax.utils.chebgui_fields import (
    bc_reform,
    cellstr,
    is_ivp_or_fvp,
    numeric_vector,
    pretty_print_feval,
    setup_fields,
    vectorize,
)


def _num2str(value):
    return format(value, '.5g')


def _replace(text, old, new):
    if isinstance(text, tuple):
        return tuple(v.replace(old, new) for v in text)
    return text.replace(old, new)


def _space_name(de, init, default, error):
    a, b = de[0], init[0]
    if a and b and a != b:
        raise ValueError(error)
    return a or b or default


def _generalized_rhs(rhs, names, domain):
    """Pinned source matrix-column decision, including its identity-RHS quirk.

    exportInfo passes empty initial data to linearize with paramReshape=false.
    Each unknown is seeded as a function (linearize.m65-72,131-133). Thus its
    discrete column block has10 points per subinterval; chebmatrix.size counts
    blocks, not collocation columns. The first source size comparison is true
    even for RHS u: Bdisc is10-by10 while size(B,2) is1. See
    @operatorBlock/operatorBlock.m328-341 and @valsDiscretization/eye.m.
    No equation is solved to determine these dimensions.
    """
    if not rhs:
        return False
    subintervals = len(numeric_vector(domain))-1
    discrete_columns = 10*subintervals*len(names)
    return discrete_columns != len(names)


class ChebguiExporter:
    """Base source-to-file adapter; construction selects a concrete exporter."""
    kind = ''
    default_file_name = ''
    description = ''

    @staticmethod
    def constructor(kind):
        """Constructor.

        Provenance
        ----------
        MATLAB source: @chebguiExporter/chebguiExporter.m (constructor).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        classes = {'bvp': ChebguiExporterBVP, 'ivp': ChebguiExporterIVP,
                   'eig': ChebguiExporterEIG, 'pde': ChebguiExporterPDE}
        if kind not in classes:
            raise ValueError('CHEBFUN:CHEBGUIEXPORTER:chebguiExporter:constructor')
        return classes[kind]()

    @staticmethod
    def disc_option(periodic, domain, option):
        """Disc option.

        Provenance
        ----------
        MATLAB source: @chebguiExporter/chebguiExporter.m (discOption).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        return 'periodic' if periodic and option == 'collocation' and len(numeric_vector(domain)) == 2 else option

    def to_file(self, gui, filename, pathname, *, user=None, created=None):
        """Write a real MATLAB script; close the stream if parsing raises.

        Provenance
        ----------
        MATLAB source: @chebguiExporter/chebguiExporter.m (toFile).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        path = Path(pathname) / filename
        with path.open('w') as stream:
            stream.write(self.render(gui, filename, user=user, created=created))
        return path

    def render(self, gui, filename=None, *, user=None, created=None):
        """Render.

        Provenance
        ----------
        MATLAB source: @chebguiExporter/chebguiExporter.m (toFile and writeHeader).
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        info = self.export_info(gui)
        filename = self.default_file_name if filename is None else filename
        user = os.environ.get('USER', '') if user is None else user
        created = datetime.now() if created is None else created
        header = (f'%% {filename} -- an executable m-file for solving {self.description}\n'
                  f'% Automatically created in CHEBGUI by user {user}.\n'
                  '% Created on '+created.strftime('%B %d, %Y at %H:%M.\n\n')+
                  '%% Problem description.\n')
        return (header+self.print_description(info)+self.print_setup(info, gui)+
                self.print_options(info)+self.print_solver(info)+self.print_post_solver(info))

    def export_info(self, gui):
        """Export info.

        Provenance
        ----------
        MATLAB source: @chebguiExporter{BVP,IVP,EIG,PDE}/exportInfo.m.
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        if self.kind == 'pde':
            return self._pde_info(gui)
        de, bc, initial = map(cellstr, (gui.DE, gui.BC, gui.init))
        parsed = setup_fields(gui, de, 'DE')
        names = parsed.variable_names
        allvars = parsed.all_var_string
        if self.kind in ('bvp', 'ivp'):
            if any(name in names for name in ('lambda', 'lam', 'l')):
                raise ValueError('CHEBFUN:CHEBGUIEXPORTER:exportInfo:eig')
            latest = initial[0].lower() == 'using latest solution'
            indinit = setup_fields(gui, initial, 'INIT', allvars).independent_vars if initial[0] and not latest else ('', '')
            space = _space_name(parsed.independent_vars, indinit,
                                't' if self.kind == 'ivp' else 'x',
                                'CHEBFUN:CHEBGUIEXPORTERBVP:exportInfo:SolveGUIbvp')
            de_string = pretty_print_feval(_replace(parsed.field, 'DUMMYSPACE', space), names)
            periodic = self.kind == 'bvp' and bc[0].lower() == 'periodic'
            if periodic:
                bc[0] = ''
            info = dict(dom=gui.domain.replace(',', ', '), deInput=de, bcInput=bc,
                        initInput=initial, deString=de_string, allVarString=allvars.replace(',', ', '),
                        allVarNames=names, indVarNameSpace=space, periodic=periodic,
                        useLatest=latest, numVars=len(names), tol=gui.tol,
                        dampingOn=gui.options['damping'], plotting=gui.options['plotting'],
                        discretization=self.disc_option(periodic, gui.domain, gui.options['discretization']))
            if self.kind == 'ivp':
                info.update(ivpSolver=gui.options['ivpSolver'],
                            timeSteppingSolver='ode' in gui.options['ivpSolver'])
            return info
        lname = next((name for name in ('lambda', 'lam', 'l') if name in de[0]), '')
        if not lname:
            raise ValueError('CHEBFUN:CHEBGUIEXPORTEREIG:exportInfo:lname')
        space = parsed.independent_vars[0] or 'x'
        strings = pretty_print_feval(_replace(parsed.field, 'DUMMYSPACE', space), names)
        lhs, rhs = strings if isinstance(strings, tuple) else (strings, '')
        periodic = bc[0].lower() == 'periodic'
        bc_string = ''
        if not bc[0]:
            raise UnboundLocalError('MATLAB EIG exportInfo leaves bcString undefined')
        if periodic:
            bc[0] = ''
        elif bc[0]:
            bc_string = _replace(setup_fields(gui, bc, 'BCnew', allvars).field, 'DUMMYSPACE', space)
        return dict(dom=gui.domain, deInput=de, bcInput=bc,
                    K=gui.options['numeigs'] or '6', sigma=gui.sigma,
                    generalized=_generalized_rhs(rhs, names, gui.domain), lname=lname,
                    lhsString=lhs, rhsString=rhs, allStrings=strings,
                    allVarString=allvars, allVarNames=names,
                    indVarName=(space, parsed.independent_vars[1]), periodic=periodic,
                    bcString=bc_string,
                    discretization=self.disc_option(periodic, gui.domain, gui.options['discretization']))

    def _pde_info(self, gui):
        domain = numeric_vector(gui.domain)
        de, left, right, initial = map(cellstr, (gui.DE, gui.LBC, gui.RBC, gui.init))
        parsed = setup_fields(gui, de, 'DE')
        if not any(parsed.pde_flags):
            raise ValueError('CHEBFUN:CHEBGUIEXPORTERPDE:exportInfo:notPDE')
        if not initial[0]:
            raise UnboundLocalError('MATLAB PDE exportInfo leaves initString undefined')
        init = setup_fields(gui, initial, 'BC', parsed.all_var_string)
        xname = _space_name(parsed.independent_vars, init.independent_vars, 'x',
                           'CHEBFUN:CHEBGUIEXPORTERPDE:exportInfo:solveGUIpde')
        tname = parsed.independent_vars[1] or 't'
        if xname == tname or not parsed.independent_vars[1]:
            raise ValueError('CHEBFUN:CHEBGUIEXPORTERPDE:exportInfo:solveGUIpde')
        periodic = left[0].lower() == 'periodic' or right[0].lower() == 'periodic'
        if periodic:
            left[0] = right[0] = ''
        def boundary(value):
            if not value[0]:
                return ''
            text = setup_fields(gui, value, 'BC', parsed.all_var_string).field
            return text[:2]+tname+','+text[2:] if ')' in text else text
        names = parsed.variable_names
        sol = names[0] if len(de) == 1 else 'sol'
        return dict(dom=gui.domain, deInput=de, a=_num2str(domain[0]), b=_num2str(domain[-1]),
                    xName=xname, tName=tname, tt=gui.timedomain,
                    pdeflag=parsed.pde_flags, initInput=initial, s=names,
                    sol=sol, sol0=sol+'0', deString='@('+tname+','+xname+','+parsed.field[2:],
                    initString=init.field, pdeVarName=parsed.pde_variable_names,
                    lbcInput=left, rbcInput=right, allVarString=parsed.all_var_string,
                    allVarNames=names, indVarName=(xname, tname), periodic=periodic,
                    lbcString=boundary(left), rbcString=boundary(right), tol=gui.tol,
                    doplot=gui.options['plotting'], dohold=gui.options['pdeholdplot'],
                    ylim1=gui.options['fixYaxisLower'], ylim2=gui.options['fixYaxisUpper'],
                    fixN=gui.options['fixN'], pdeSolver=gui.options['pdeSolver'])

    def print_description(self, info):
        """Print description.

        Provenance
        ----------
        MATLAB source: @chebguiExporter{BVP,IVP,EIG,PDE}/printDescription.m.
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        out = '% Solving\n'
        if self.kind == 'eig' and len(info['deInput']) == 1 and '=' not in info['deInput'][0]:
            out += f"%   {info['deInput'][0]} = {info['lname']}*{info['allVarString']}\n"
        else:
            out += ''.join('%   '+line+',\n' for line in info['deInput'])
        if self.kind == 'pde':
            tt = numeric_vector(info['tt'])
            out += f"% for {info['xName']} in [{info['a']},{info['b']}] and {info['tName']} in [{_num2str(tt[0])},{_num2str(tt[-1])}]"
            left, right = list(info['lbcInput']), list(info['rbcInput'])
            if left[0] or right[0]:
                out += ', subject to\n%'
                for side, endpoint in [(left, info['a']), (right, info['b'])]:
                    if not side[0]:
                        continue
                    if side is right and left[0]:
                        out += '% and\n%'
                    if len(side) == 1 and '=' not in side[0] and side[0].lower() not in ('dirichlet', 'neumann'):
                        side[0] = info['allVarString']+' = '+side[0]
                    out += '   '+', '.join(side)+' at '+info['xName']+' = '+endpoint+'\n'
                return out+'\n'
        else:
            space = info['indVarName'][0] if self.kind == 'eig' else info['indVarNameSpace']
            out += f"% for {space} in {info['dom']}"
            if info['bcInput'][0]:
                return out+', subject to\n%   '+',\n%   '.join(info['bcInput'])+'.\n'
        return out+(', subject to periodic boundary conditions.\n\n' if info.get('periodic') else '.\n')

    def print_setup(self, info, gui):
        """Print setup.

        Provenance
        ----------
        MATLAB source: @chebguiExporter{BVP,IVP,EIG,PDE}/printSetup.m.
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        if self.kind == 'pde':
            return self._pde_setup(info)
        if self.kind == 'eig':
            out = "\n%% Define the domain we're working on.\n"+f"dom = {info['dom']};\n"
            if info['generalized']:
                out += f"\n% Assign the equation to two chebops N and B such that N(u) = {info['lname']}*B(u).\n"
                out += f"N = chebop({info['lhsString']}, dom);\nB = chebop({info['rhsString']}, dom);\n"
            else:
                out += f"\n% Assign the equation to a chebop N such that N(u) = {info['lname']}*u.\n"
                out += f"N = chebop({info['lhsString']}, dom);\n"
            out += '\n% Assign boundary conditions to the chebop.\n'
            if info['bcInput'][0]:
                out += 'N.bc = '+pretty_print_feval(info['bcString'], info['allVarNames'])+';\n'
            if info['periodic']:
                out += "N.bc = 'periodic';\n"
            return out
        out = '\n%% Problem set-up.\n% Define the domain.\n'+f"dom = {info['dom']};\n"
        out += '\n% Assign the differential equation to a chebop on that domain.\n'
        out += f"N = chebop({info['deString']}, dom);\n"
        out += f"\n% Set up the rhs of the differential equation so that N({info['allVarString']}) = rhs.\n"
        sep = ',' if self.kind == 'ivp' else ';'
        rhs = '['+sep.join('0' for _ in info['deInput'])+']' if len(info['deInput']) > 1 else '0'
        out += 'rhs = '+rhs+';\n\n% Assign boundary conditions to the chebop.\n'
        if self.kind == 'ivp':
            direction = is_ivp_or_fvp(gui, info['allVarNames'])
            bc = bc_reform(info['dom'], info['bcInput'], direction)
            bc_string = setup_fields(gui, bc, 'BC', info['allVarString']).field
            out += ('N.lbc = ' if direction == 1 else 'N.rbc = ')+bc_string+';\n'
        else:
            if info['bcInput'][0]:
                bc = setup_fields(gui, info['bcInput'], 'BCnew', info['allVarString']).field
                bc = pretty_print_feval(_replace(bc, 'DUMMYSPACE', info['indVarNameSpace']), info['allVarNames'])
                out += 'N.bc = '+bc+';\n'
            if info['periodic']:
                out += "N.bc = 'periodic';\n"
        if info['useLatest']:
            return out+'\n% Note that it is not possible to use the "Use latest" option \n% when exporting to .m files. \n'
        if not info['initInput'][0]:
            return out
        space = info['indVarNameSpace']
        out += f'\n% Construct a linear chebfun on the domain, \n{space} = chebfun(@({space}) {space}, dom);\n'
        out += '% and assign an initial guess to the chebop.\n'
        initial = info['initInput']
        if len(initial) == 1:
            guess = vectorize(initial[0].strip()).rsplit('=', 1)[-1]
            return out+'N.init = '+guess+';\n'
        assignments = self._initial_assignments(initial, info['allVarNames'])
        out += ''.join(name+'_init = '+value+';\n' for name, value in assignments)
        return out+'N.init = ['+'; '.join(name+'_init' for name, _ in assignments)+'];\n'

    @staticmethod
    def _initial_assignments(initial, names):
        result = []
        for value in initial:
            name, guess = value.split('=', 1)
            result.append((name.strip(), vectorize(guess.strip())))
        return sorted(result, key=lambda item: names.index(item[0]))

    def _pde_setup(self, info):
        out = '%% Problem set-up\n% Create an interval of the space domain...\n'
        out += f"dom = {info['dom']};\n%...and specify a sampling of the time domain:\n{info['tName']} = {info['tt']};\n"
        out += f"\n% Make the right-hand side of the PDE.\npdefun = {info['deString']};\n"
        if not all(info['pdeflag']):
            out += 'pdeflag = ['+'  '.join(str(int(flag)) for flag in info['pdeflag'])+']; % Zero when a variable is indep of time.\n'
        out += '\n% Assign boundary conditions.\n'
        for side in ('left', 'right'):
            value = info['lbcString' if side == 'left' else 'rbcString']
            if value:
                out += 'bc.'+side+' = '+value+';\n'
        if info['periodic']:
            out += "bc = 'periodic';\n"
        xname = info['xName']
        initial = info['initInput']
        out += f'\n% Construct a chebfun of the space variable on the domain,\n{xname} = chebfun(@({xname}) {xname}, dom);\n'
        out += '% and of the initial condition'+('s' if len(initial) > 1 else '')+'.\n'
        if len(info['deInput']) == 1:
            guess = vectorize(initial[0])
            guess = guess.rsplit('=', 1)[-1].strip() if '=' in guess else guess
            if xname not in initial[0]:
                guess = 'chebfun('+guess+',dom)'
            return out+info['sol0']+' = '+guess+';\n'
        assignments = self._initial_assignments(initial, info['allVarNames'])
        if not any('x' in value for value in initial):
            assignments = [(name, 'chebfun('+value+',dom)') for name, value in assignments]
        out += ''.join(name+'0 = '+value+';\n' for name, value in assignments)
        return out+info['sol0']+' = ['+', '.join(name+'0' for name, _ in assignments)+'];\n'

    def print_options(self, info):
        """Print options.

        Provenance
        ----------
        MATLAB source: @chebguiExporter{BVP,IVP,EIG,PDE}/printOptions.m.
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        if self.kind == 'pde':
            opts = "'Eps', "+info['tol']
            if not all(info['pdeflag']):
                opts += ", 'PDEflag', pdeflag"
            if info['doplot'].lower() == 'off':
                opts += ", 'Plot', 'off'"
            else:
                if info['dohold']:
                    opts += ", 'HoldPlot', 'on'"
                if info['ylim1'] and info['ylim2']:
                    opts += ", 'Ylim', ["+info['ylim1']+','+info['ylim2']+']'
            if info['fixN']:
                # Source concatenates numeric N into a char vector, not num2str(N).
                opts += ",'N',"+chr(int(float(info['fixN'])))
            return '\n%% Setup preferences for solving the problem.\nopts = pdeset('+opts+');\n'
        out = ('\n%% Setup preferences for solving the problem.\n'
               '% Create a CHEBOPPREF object for passing preferences.\n'
               "% (See 'help cheboppref' for more possible options.)\n"
               'options = cheboppref();\n')
        if self.kind == 'eig':
            out += self._discretization(info['discretization'])
            return out+f"\n% Number of eigenvalue and eigenmodes to compute.\nk = {info['K']};\n"
        out += '\n'
        if self.kind == 'ivp':
            out += ("% Specify the IVP solver to use. Possible options are:\n"
                    "%   Time-stepping solvers:\n"
                    "%     'ode113' (default), 'ode15s' or 'ode45'.\n"
                    "%   Global methods:\n"
                    "%     'values' or 'coefficients'.\n")
            out += f"options.ivpSolver = '{info['ivpSolver']}';\n"
            if info['timeSteppingSolver']:
                return out
        else:
            out += '% Print information to the command window while solving:\n'
        out += "options.display = 'iter';\n"
        if info['tol']:
            out += '\n% Option for tolerance.\noptions.bvpTol = '+info['tol']+';\n'
        out += '\n% Option for damping.\noptions.damping = '+('true' if info['dampingOn'] == '1' else 'false')+';\n'
        if self.kind == 'ivp':
            out += "\n% Option for discretization (either 'values' or 'coeffs').\n"
            out += f"options.discretization = '{info['ivpSolver']}';\n"
        else:
            out += self._discretization(info['discretization'])
        plotting = "'off'" if info['plotting'] == 'off' else info['plotting']
        return out+'\n% Option for determining how long each Newton step is shown.\noptions.plotting = '+plotting+';\n'

    @staticmethod
    def _discretization(value):
        return ("\n% Specify the discretization to use. Possible options are:\n"
                "%  'values' (default)\n%  'coeffs'\n"
                "%  A function handle (see 'help cheboppref' for details).\n"
                f"options.discretization = '{value}';\n")

    def print_solver(self, info):
        """Print solver.

        Provenance
        ----------
        MATLAB source: @chebguiExporter{BVP,IVP,EIG,PDE}/printSolver.m.
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        if self.kind == 'eig':
            args = 'N, B, k' if info['generalized'] else 'N, k'
            if info['sigma']:
                args += ", '"+info['sigma']+"'"
            return '\n%% Solve the eigenvalue problem.\n[V, D] = eigs('+args+', options);\n'
        if self.kind == 'pde':
            solver, time = info['pdeSolver'], info['tName']
            return f"\n%% Call {solver} to solve the problem.\n[{time}, {info['allVarString'].replace(',', ', ')}] = {solver}(pdefun, {time}, {info['sol0']}, bc, opts);\n"
        solver = 'solveivp' if self.kind == 'ivp' else 'solvebvp'
        parens = '()' if self.kind == 'ivp' else ''
        out = '\n%% Solve!\n% Call '+solver+parens+' to solve the problem.\n'
        out += '% (With the default options, this is equivalent to u = N\\rhs.)\n'
        names = info['allVarString'] if info['numVars'] == 1 else '['+info['allVarString']+']'
        return out+names+' = '+solver+'(N, rhs, options);\n'

    def print_post_solver(self, info):
        """Print post solver.

        Provenance
        ----------
        MATLAB source: @chebguiExporter{BVP,IVP,EIG,PDE}/printPostSolver.m.
        Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
        """
        if self.kind == 'eig':
            out = ('\n%% Plot the results.\n% Plot the eigenvalues.\nD = diag(D);\nfigure\n'
                   "plot(real(D), imag(D), '.', 'markersize', 25)\n"
                   "title('Eigenvalues'); xlabel('real'); ylabel('imag');\n")
            if len(info['allVarNames']) == 1:
                out += ('\n% Plot the eigenmodes.\nfigure\n'
                        "plot(real(V), 'linewidth', 2);\n"
                        f"title('Eigenmodes'); xlabel('{info['indVarName'][0]}'); ylabel('{info['allVarString']}');")
            return out
        if self.kind == 'pde':
            x, t = info['indVarName']
            if len(info['deInput']) == 1:
                return f"\n%% Plot the solution.\nwaterfall({info['sol']}, {t})\nxlabel('{x}'), ylabel('{t}')"
            out = '\n%% Plot the solution components.'
            for name in info['s']:
                out += f"\nfigure\nwaterfall({name}, {t})\nxlabel('{x}'), ylabel('{t}'), title('{name}')"
            return out
        names = info['allVarString'] if info['numVars'] == 1 else '['+info['allVarString']+']'
        out = f"\n%% Plot the solution.\nfigure\nplot({names}, 'LineWidth', 2)\n"
        title = 'IVP solution' if self.kind == 'ivp' else 'Final solution'
        out += f"title('{title}'), xlabel('{info['indVarNameSpace']}')"
        if info['numVars'] == 1:
            return out+f", ylabel('{info['allVarNames'][0]}')"
        return out+', legend('+','.join("'"+name+"'" for name in info['allVarNames'])+')\n'


class ChebguiExporterBVP(ChebguiExporter):
    kind = 'bvp'
    default_file_name = 'chebbvp.m'
    description = 'a boundary-value problem'


class ChebguiExporterIVP(ChebguiExporter):
    kind = 'ivp'
    default_file_name = 'chebivp.m'
    description = 'an initial-value problem'


class ChebguiExporterEIG(ChebguiExporter):
    kind = 'eig'
    default_file_name = 'chebeig.m'
    description = 'an eigenvalue problem'


class ChebguiExporterPDE(ChebguiExporter):
    kind = 'pde'
    default_file_name = 'chebpde.m'
    description = 'a partial differential equation'
