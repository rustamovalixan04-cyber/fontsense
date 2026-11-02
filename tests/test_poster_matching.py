"""Regressions for mixed artwork/text, kept separate from accuracy evidence."""
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from api import extended_match, retrieval_match
from api.font_match import identify, prepare_query
from api.index import app

POSTER = Path(__file__).parent / 'fixtures/allover_poster_crop.png'


@pytest.mark.parametrize('filename', ['ocr_nabla_browser.png','ocr_pixelify_sans_browser.png'])
def test_secondary_ocr_uses_shapes_to_fix_or_preserve_actual_browser_text(filename):
    from api.compact import text_crop
    from api.font_match import read_text, straighten_line
    image = Image.open(POSTER.parent / filename)
    text, confidence = read_text(straighten_line(text_crop(image)))
    assert text == 'Quantum Archive'
    assert confidence >= .75


def test_reported_poster_reads_headline_and_offers_serif_alternatives():
    result = identify(Image.open(POSTER))
    assert result['text'] == 'ALLOVER'
    assert result['category'] == 'serif'
    assert result['identified_font'] is None  # This commercial face is not enrolled.
    assert len(result['matches']) >= 3
    assert sum(font['category'] == 'serif' for font in result['matches'][:3]) >= 2
    assert 'MODERN SERIF FONT' in {line['text'] for line in result['text_regions']}
    assert result['selected_region'] is not None


def test_line_selection_and_manual_correction_use_the_same_pixels():
    image = Image.open(POSTER)
    _, _, regions, selected, _ = prepare_query(image, '', None)
    subtitle = next(line for line in regions if line['text'] == 'MODERN SERIF FONT')
    _, _, _, chosen, reading = prepare_query(image, '', subtitle['index'])
    assert chosen == subtitle['index'] != selected
    assert reading[0] == 'MODERN SERIF FONT'
    manual_image, _, manual_regions, chosen, reading = prepare_query(image, 'MODERN SERIF', subtitle['index'])
    assert chosen == subtitle['index']
    assert reading == ('MODERN SERIF', None)
    assert next(row['box'] for row in manual_regions if row['index'] == chosen) == subtitle['box']
    assert manual_image.height < image.height / 3


def test_separated_words_on_the_same_headline_stay_together():
    image = Image.open(POSTER.parent / 'spaced_headline_poster.png')
    _, _, regions, selected, reading = prepare_query(image, '', None)
    assert reading[0] == 'Cedar Valley'
    assert next(row['text'] for row in regions if row['index'] == selected) == 'Cedar Valley'


@pytest.mark.parametrize('region', [-2, 99])
def test_invalid_region_is_rejected_by_the_api(region):
    with TestClient(app) as client:
        response = client.post('/identify?region=' + str(region), content=POSTER.read_bytes(), headers={'content-type':'image/png'})
    assert response.status_code == 400


def test_appearance_only_does_not_assert_identity_even_for_a_high_neural_score():
    rows = [{'family':'A plausible alternative', 'score':.999, 'method':'trained-appearance-alternatives'},
            {'family':'Another alternative', 'score':.7, 'method':'trained-appearance-alternatives'}]
    assert extended_match.likely_match(rows, 'Distinct letters') is None


def test_weak_direct_match_keeps_neural_alternatives_without_exhaustive_cost(monkeypatch):
    alternatives = [{'family':'A visual alternative', 'score':.8, 'method':'trained-appearance-alternatives'}]
    monkeypatch.setattr(extended_match, 'learned_enabled', lambda: True)
    monkeypatch.setattr(retrieval_match, 'rank', lambda *args: [{'pixel_score':.6, 'appearance_candidates':alternatives}])
    monkeypatch.setattr(extended_match, 'exhaustive_rank', lambda *args: pytest.fail('Appearance alternatives should not rebuild every text reference'))
    assert extended_match.rank(None, 'Distinct letters') == alternatives
