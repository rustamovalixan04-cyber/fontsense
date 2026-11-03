import json
from pathlib import Path

import numpy as np

from api.retrieval_index import session, index

ROOT = Path(__file__).resolve().parents[1]


def test_portable_retrieval_model_matches_real_pytorch_export():
    with np.load(ROOT / "tests/fixtures/retrieval_encoder_parity.npz", allow_pickle=False) as archive:
        pixels, expected = archive["pixels"], archive["vectors"]
    tensor = (pixels[:, None].astype(np.float32) / 255 - .5) / .5
    actual = session().run(["vectors"], {"images": tensor})[0]
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1., atol=3e-5)


def test_reference_index_contains_verified_unit_vectors_and_native_statistics():
    metadata, arrays = index()
    from api.extended_match import primary_styles
    assert metadata['reference_families'] == len(primary_styles()) + 22
    assert arrays["vectors"].shape == (metadata["reference_vectors"], 96)
    np.testing.assert_allclose(np.linalg.norm(arrays["vectors"], axis=1), 1., atol=3e-5)
    assert len({row["family"] for row in metadata["sources"] if row["native"]}) == 22
    assert all("local_file" not in row and "file" not in row for row in metadata["sources"])


def test_training_families_and_shared_latin_groups_do_not_cross_query_splits():
    contract = json.loads((ROOT / "artifacts/web/retrieval_encoder.json").read_text())
    data = json.loads((ROOT / 'reports/independent_fonts/data_contract.json').read_text())
    assert contract['training_families'] == data['training_families']
    assert contract['training_images'] == data['training_images']
    assert len(contract['active_training_families']) == contract['training_families']
    assert all(contract["family_splits"][name] == "train" for name in contract["active_training_families"])
    for group in contract["duplicate_groups"]:
        assert len({contract["family_splits"][name] for name in group}) == 1


def test_confidence_filter_excludes_final_native_query_families():
    gate = json.loads((ROOT / "artifacts/web/retrieval_confidence.json").read_text())
    assert not set(gate["fitting_families"]) & set(gate["selection_families"])
    final_native = set(gate["native"]["final_families_excluded"])
    assert not final_native & (set(gate["fitting_families"]) | set(gate["selection_families"]))


def test_native_candidates_survive_exhaustive_fallback(monkeypatch):
    from api import extended_match, retrieval_match
    sentinel = [{"family": "A native reference", "embedding_similarity": .97}]
    monkeypatch.setattr(extended_match, "learned_enabled", lambda: True)
    monkeypatch.setattr(retrieval_match, "rank", lambda *args: [{"pixel_score": .6, "native_candidates": sentinel}])
    monkeypatch.setattr(extended_match, "exhaustive_rank", lambda *args: [{"family": "An open alternative", "score": .7}])
    result = extended_match.rank(None, "Sample text")
    assert result[0]["native_candidates"] == sentinel
    assert result[0]["family"] == "An open alternative"
