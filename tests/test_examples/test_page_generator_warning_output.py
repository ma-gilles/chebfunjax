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
