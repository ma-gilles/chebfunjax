"""Actual stdout must survive generation of a page's printed output blocks."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def generator(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[2] / 'scripts/gen_example_pages.py'
    spec = importlib.util.spec_from_file_location('page_generator_warning_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'PROJECT', tmp_path)
    for name, directory in [('EXAMPLES', 'examples'), ('DOCS', 'docs/examples'),
                            ('IMAGES', 'docs/images')]:
        monkeypatch.setattr(module, name, tmp_path / directory)
    return module


@pytest.mark.parametrize('output', [
    "Warning: The syntax 'splitting on' is deprecated.\n"
    "Please see CHEBFUNPREF documentation for further details.",
    'Warming up =\n13',
], ids=['native-multiline-warning', 'ordinary-output-prefix'])
def test_generate_preserves_captured_output(generator, tmp_path, output):
    """Exercise stdout loading, HTML alignment, and the resulting Markdown."""
    category, stem = 'stats', 'WarningOutput'
    url = f'https://www.chebfun.org/examples/{category}/{stem}.html'
    script = generator.EXAMPLES / category / 'fixture.py'
    script.parent.mkdir(parents=True)
    script.write_text(f'"""{url}"""\n')
    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / f'{category}_{stem}.html').write_text(
        '<div id="content"><h1>Output preservation fixture</h1>'
        f'<pre class="mcode-output">{output}</pre>'
        '<pre class="mcode-output">ans =\n1</pre></div>'
    )
    stdout = tmp_path / 'stdout'
    stdout.mkdir()
    # More computed values than the reference must remain visible, too.
    (stdout / 'stats_fixture.txt').write_text(output + '\nans =\n1\n2\n3\n')
    present, missing, images, found = generator.generate(category, stem, stdout, cache)
    assert present and missing == [] and images == 0
    assert found == 'examples/stats/fixture.py'
    markdown = (generator.DOCS / category / f'{stem}.md').read_text()
    assert f'```text\n{output}\n```' in markdown
    assert '```text\nans =\n1\n2\n3\n```' in markdown


@pytest.mark.parametrize('warning', ['', 'Warning: Python wording.\n'])
def test_warning_led_timing_stays_with_its_source_block(generator, warning):
    reference = [['Elapsed time is 1 seconds.'],
                 ['Warning: MATLAB wording', 'continuation text',
                  'Elapsed time is 2 seconds.'],
                 ['Elapsed time is 3 seconds.']]
    actual = ('Elapsed time is 11 seconds.\n' + warning +
              'Elapsed time is 22 seconds.\nElapsed time is 33 seconds.').splitlines()
    chunks = generator.align_outputs(reference, actual)
    assert chunks == [['Elapsed time is 11 seconds.'],
                      warning.splitlines() + ['Elapsed time is 22 seconds.'],
                      ['Elapsed time is 33 seconds.']]
    assert sum(chunks, []) == actual


def test_warning_led_repeated_label_keeps_all_values(generator):
    reference = [['f =', 'reference first'],
                 ['Warning: MATLAB wording', 'f =', 'reference second'],
                 ['f =', 'reference third']]
    actual = ['f =', 'computed first', 'Warning: Python wording',
              'f =', 'computed second', 'f =', 'computed third']
    chunks = generator.align_outputs(reference, actual)
    assert chunks == [actual[:2], actual[2:5], actual[5:]]
    assert sum(chunks, []) == actual


def test_different_computed_scalars_keep_separate_source_cells(generator):
    reference = [['Integral of pdf 1.000000000000000'],
                 ['Error of marginal = 1.8e-15'],
                 ['Error in conditional pdf is 1.7e-15'],
                 ['final J[y]: 2.3']]
    actual = ['Integral of pdf 0.9999999999999983',
              'Error of marginal = 1.671e-15',
              'Error in conditional pdf is 1.38119e-15',
              'final J[y]: -2.1e+02']
    assert generator.align_outputs(reference, actual) == [[line] for line in actual]


@pytest.mark.parametrize('line', ['1.7e-15', '1 2 3', 'Columns 1 through 5',
                                  'Column 2', 'x2', 'Warning: iteration failed'])
def test_numeric_values_and_matrix_headers_are_not_scalar_labels(generator, line):
    assert generator._key(line) == line


def test_repeated_scalar_labels_remain_in_order(generator):
    actual = ['Residual is 1e-6', 'Residual is 2e-8', 'Residual is 3e-10']
    reference = [['Residual is 4e-6'], ['Residual is 5e-8'], ['Residual is 6e-10']]
    assert generator.align_outputs(reference, actual) == [[line] for line in actual]


def test_leading_blank_fprintf_cell_keeps_its_own_values(generator):
    reference = [['ans =', '7.5e-6'], ['', '  final J[y]: 2.8134302039411909',
                                            'optimal J[y]: 2.8134302039235082']]
    actual = ['ans =', '7.515155780849027e-6', '',
              '  final J[y]: 2.8134302039413677', 'optimal J[y]: 2.8134302039235086']
    assert generator.align_outputs(reference, actual) == [actual[:2], actual[3:]]
