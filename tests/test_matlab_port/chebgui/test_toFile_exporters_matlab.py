"""Port all four chebgui test_toFile cohorts with source-stage controls.

MATLAB source: tests/chebgui/test_toFile{BVP,IVP,EIG,PDE}.m,
@chebguiExporter*/print*.m and @chebgui/setupFields.m.
Chebfun commit7574c77680d7e82b79626300bf255498271a72df.
Exports only: no generated script is executed and no differential problem solved.
"""
import hashlib
import json
import warnings
from datetime import datetime
from pathlib import Path

import pytest

from chebfunjax.utils.chebgui_data import ChebguiData, demo2chebgui
from chebfunjax.utils.chebgui_exporters import ChebguiExporter
from chebfunjax.utils.chebgui_fields import (
    bc_reform,
    is_ivp_or_fvp,
    pretty_print_feval,
    setup_fields,
)

FIXTURES = Path(__file__).parent/'fixtures/chebgui_demos'
CREATED = datetime(2026, 10, 8, 12)


def _source_cohort(kind, expected_count, tmp_path):
    demos = sorted((FIXTURES/(kind+'demos')).glob('*.guifile'))
    assert len(demos) == expected_count
    exporter = ChebguiExporter.constructor(kind)
    for demo in demos:
        with warnings.catch_warnings(record=True) as captured:
            gui = demo2chebgui(demo)
        assert not captured
        target = exporter.to_file(gui, demo.stem+'.m', tmp_path,
                                  user='source-test', created=CREATED)
        text = target.read_text()
        assert text.startswith('%% '+demo.stem+'.m -- an executable m-file')
        assert '% Created on October 08, 2026 at 12:00.\n\n' in text
        assert '%% Problem description.' in text and '% Solving' in text
        assert 'dom = '+gui.domain.replace(',', ', ' if kind in ('bvp', 'ivp') else ',')+';' in text
        assert 'DUMMYSPACE' not in text
        assert len(text) > 600
        if kind in ('bvp', 'ivp'):
            solver = 'solve'+kind
            assert f'{solver}(N, rhs, options);' in text
            assert 'N = chebop(@(' in text
            assert 'options = cheboppref();' in text
            assert ('N.bc =' if kind == 'bvp' else 'N.lbc =') in text or 'N.rbc =' in text
        elif kind == 'eig':
            assert '[V, D] = eigs(N, B, k' in text
            assert 'B = chebop(@(' in text
            assert "title('Eigenvalues')" in text
        else:
            assert 'pdefun = @(' in text
            assert 'opts = pdeset(' in text
            assert gui.options['pdeSolver']+'(pdefun,' in text
            assert 'waterfall(' in text


def test_to_file_bvp_matlab(tmp_path):
    _source_cohort('bvp', 25, tmp_path)


def test_to_file_ivp_matlab(tmp_path):
    _source_cohort('ivp', 11, tmp_path)


def test_to_file_eig_matlab(tmp_path):
    _source_cohort('eig', 12, tmp_path)


def test_to_file_pde_matlab(tmp_path):
    _source_cohort('pde', 16, tmp_path)


def test_original_demo_bytes_match_pinned_source_manifest():
    provenance = json.loads((FIXTURES/'provenance.json').read_text())
    assert len(provenance['files']) == 64
    for item in provenance['files']:
        assert hashlib.sha256((FIXTURES/item['path']).read_bytes()).hexdigest() == item['sha256']


def test_bvp_natural_callbacks_and_initial_guess_order():
    gui = ChebguiData(type='bvp', domain='[0,1]', DE=["u''=v", "v''=u"],
                      BC=['u(0)=1', 'v(1)=2'], init=['v = x^2', 'u = x'])
    exporter = ChebguiExporter.constructor('bvp')
    info = exporter.export_info(gui)
    text = exporter.render(gui, user='source-test', created=CREATED)
    assert info['allVarNames'] == ('u', 'v')
    assert 'N.init = [u_init; v_init];' in text
    assert text.index('u_init = x;') < text.index('v_init = x.^2;')
    assert 'rhs = [0;0];' in text
    assert 'N.bc = @(x,u, v) [u(0)-1; v(1)-2];' in text


def test_pde_reorders_time_derivative_rows_and_keeps_algebraic_flags():
    gui = ChebguiData(type='pde')
    result = setup_fields(gui, ['v_t = u', 'u_t = v'], 'DE')
    assert result.variable_names == ('u', 'v')
    assert result.field == '@(u,v) [v; u]'
    assert result.pde_flags == (True, True)
    result = setup_fields(gui, ['v_t = u', "u'' = v"], 'DE')
    assert result.pde_flags == (False, True)
    assert result.field == '@(u,v) [-diff(u,2)+v; u]'


def test_ivp_and_fvp_boundary_reform_source():
    gui = ChebguiData(type='ivp', domain='[0 2]', DE="u''=-u", BC=['u(0)=1', "u'(0)=0"])
    assert is_ivp_or_fvp(gui, ('u',)) == 1
    assert bc_reform(gui.domain, gui.BC, 1) == ['u=1', "u'=0"]
    text = ChebguiExporter.constructor('ivp').render(gui, created=CREATED)
    assert 'N.lbc = @(u) [u-1; diff(u)];' in text
    gui = gui.set('BC', ['u(2)=1', "u'(2)=0"])
    assert is_ivp_or_fvp(gui, ('u',)) == 2
    assert 'N.rbc =' in ChebguiExporter.constructor('ivp').render(gui, created=CREATED)


def test_pretty_feval_endpoint_spellings_source():
    text = "feval(u,'left')+feval(u,'right')+feval(v,'start')+feval(v,'end')"
    assert pretty_print_feval(text, ('u', 'v')) == 'u(u.ends(1))+u(end)+v(v.ends(1))+v(end)'


def test_source_periodic_discretization_ignores_breakpoint_domains():
    exporter = ChebguiExporter.constructor('bvp')
    assert exporter.disc_option(True, '[-1 1]', 'collocation') == 'periodic'
    assert exporter.disc_option(True, '[-1 0 1]', 'collocation') == 'collocation'
    assert exporter.disc_option(True, '[-1 1]', 'values') == 'values'


def test_eig_identity_rhs_source_matrix_dimension_quirk():
    gui = ChebguiData(type='eig', domain='[-1 1]', DE="u''=lambda*u", BC='dirichlet')
    exporter = ChebguiExporter.constructor('eig')
    info = exporter.export_info(gui)
    assert info['generalized'] is True
    text = exporter.render(gui, created=CREATED)
    assert 'B = chebop(@(x,u) u, dom);' in text
    assert '[V, D] = eigs(N, B, k, options);' in text


def test_pde_fixn_numeric_character_concatenation_is_not_silently_corrected():
    gui = ChebguiData(type='pde', domain='[0 1]', timedomain='0:.1:1',
                      DE='u_t = u"', LBC='0', RBC='0', init='u=1').set('fixN', '64')
    text = ChebguiExporter.constructor('pde').render(gui, created=CREATED)
    assert "'N',@);" in text
    assert "'N',64" not in text


def test_ivp_time_solver_and_global_solver_options():
    gui = ChebguiData(type='ivp', domain='[0 1]', DE="u'=-u", BC='u(0)=1')
    exporter = ChebguiExporter.constructor('ivp')
    text = exporter.render(gui, created=CREATED)
    assert "options.ivpSolver = 'ode113';" in text
    assert 'options.bvpTol' not in text
    gui = gui.set('ivpSolver', 'coefficients').set('plotting', 'off')
    text = exporter.render(gui, created=CREATED)
    assert "options.discretization = 'coefficients';" in text
    assert "options.plotting = 'off';" in text


def test_errors_close_created_file_and_preserve_source_identifiers(tmp_path):
    with pytest.raises(ValueError, match='CHEBFUN:CHEBGUIEXPORTER:chebguiExporter:constructor'):
        ChebguiExporter.constructor('BVP')
    gui = ChebguiData(type='eig', domain='[-1 1]', DE="u''=u", BC='dirichlet')
    with pytest.raises(ValueError, match='CHEBFUN:CHEBGUIEXPORTEREIG:exportInfo:lname'):
        ChebguiExporter.constructor('eig').to_file(gui, 'failed.m', tmp_path)
    # A subsequent write demonstrates that the source-like error path closed its stream.
    (tmp_path/'failed.m').write_text('closed')


@pytest.mark.parametrize('text, expected', [
    ('u_t=v', 'v'), ('u_t=-v', '-v'), ('u_t=u+v', 'u+v'),
    ('u_t=u-v', 'u-v'), ('u_t+u=v', '-u+v'), ('v=u_t', 'v'),
])
def test_pde_prefix_negation_precedes_source_infix_simplification(text, expected):
    from chebfunjax.utils.string_parser import str2anon
    assert str2anon(text, 'pde', outputs=6).an_fun == expected
