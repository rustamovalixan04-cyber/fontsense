"""Independent source provenance, split isolation and real reference matching."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

import pytest

from api.extended_match import catalog, render_reference
from api.font_match import identify
from scripts.independent_font_assets import member, prepare
from scripts import independent_font_assets

ROOT = Path(__file__).resolve().parents[1]


def test_independent_catalog_is_licensed_deduplicated_and_unknowns_are_not_enrolled():
    data = json.loads((ROOT / 'artifacts/web/independent_catalog.json').read_text())
    contract = json.loads((ROOT / 'reports/independent_fonts/data_contract.json').read_text())
    families = {row['family'] for row in data['fonts']}
    assert len(families) == data['families'] == 81
    assert len(data['fonts']) == data['styles'] == 154
    assert len({row['id'] for row in catalog()}) == len(catalog())
    assert {'Mister Brush', 'Leporid', 'Bluu Next', 'Necto Mono'} <= families
    assert {'Auseklis', 'Naganoshi'}.isdisjoint(families)
    unknowns = {name for name, split in contract['new_family_splits'].items() if split.startswith('unknown_')}
    assert len(unknowns) == 7 and unknowns.isdisjoint(families)
    assert unknowns.isdisjoint(contract['active_training_families'])
    for group in contract['new_groups']:
        assert len({contract['new_family_splits'][name] for name in group}) == 1
    for source in data['sources']:
        assert source['license'] in ('OFL-1.1', 'CC0-1.0')
        assert source['source_sha256'] and source['license_sha256']
        for key, digest in [('file', 'sha256'), ('license_file', 'license_sha256')]:
            assert hashlib.sha256((ROOT / 'artifacts/web' / source[key]).read_bytes()).hexdigest() == source[digest]
    assert any(source['publisher'].startswith('FontSpace') for source in data['sources'])
    assert all(row['category'] == 'serif' for row in data['fonts'] if row['family'] in ('Bluu Next', 'Mazius Display', 'Sinistre'))


@pytest.mark.parametrize('family', ['Mister Brush', 'Bluu Next'])
def test_independent_training_families_can_be_compared_without_false_other_names(family):
    face = next(row for row in catalog() if row['family'] == family)
    result = identify(render_reference(face['id'], 'Clear new letters', 73).convert('RGB'), 'Clear new letters')
    named = result.get('identified_font')
    assert not named or named['family'] == family
    names = ([named['family']] if named else []) + [row['family'] for row in result['matches']]
    assert family in names[:3]


def test_independent_import_rejects_paths_outside_asset_directory(tmp_path):
    with pytest.raises(ValueError, match='asset path'):
        prepare({'file': '../outside.ttf', 'license_file': 'independent-fonts/license.txt'}, tmp_path)
    assert not list(tmp_path.iterdir())


def test_archive_member_is_read_without_extracting_its_path(tmp_path, monkeypatch):
    stream = BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive: archive.writestr('../outside.txt', b'in-memory only')
    monkeypatch.chdir(tmp_path)
    assert member(stream.getvalue(), '../outside.txt') == b'in-memory only'
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('changed_member', [None, 'font.ttf', 'LICENSE.txt'])
def test_archive_wrapper_may_change_but_font_and_license_must_match(tmp_path, monkeypatch, changed_member):
    original = {'font.ttf': b'exact-font-program', 'LICENSE.txt': b'exact-license'}
    def package(values, year):
        stream = BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            for name, data in values.items(): archive.writestr(zipfile.ZipInfo(name, (year, 1, 1, 0, 0, 0)), data)
        return stream.getvalue()
    sha = lambda data: hashlib.sha256(data).hexdigest()
    old_archive = package(original, 2020)
    current = dict(original)
    if changed_member: current[changed_member] += b'-modified'
    new_archive = package(current, 2026)
    assert sha(new_archive) != sha(old_archive)
    source = {'id': 'example', 'kind': 'archive', 'file': 'independent-fonts/example.ttf',
              'license_file': 'independent-fonts/example-LICENSE.txt', 'download_url': 'https://get.fontspace.co/example.zip',
              'archive_sha256': sha(old_archive), 'member': 'font.ttf', 'license_member': 'LICENSE.txt',
              'source_sha256': sha(original['font.ttf']), 'license_sha256': sha(original['LICENSE.txt']),
              'sha256': sha(b'renamed-subset')}
    monkeypatch.setattr(independent_font_assets, 'payload', lambda url, digest: new_archive)
    converted = []
    monkeypatch.setattr(independent_font_assets, 'make_subset', lambda raw, name: converted.append(raw) or b'renamed-subset')
    if changed_member:
        with pytest.raises(ValueError, match='source or license checksum'): prepare(source, tmp_path)
        assert not converted and not list(tmp_path.iterdir())
    else:
        prepare(source, tmp_path)
        assert (tmp_path / source['file']).read_bytes() == b'renamed-subset'
        assert (tmp_path / source['license_file']).read_bytes() == original['LICENSE.txt']
