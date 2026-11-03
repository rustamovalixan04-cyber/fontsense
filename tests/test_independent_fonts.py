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
