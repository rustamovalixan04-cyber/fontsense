"""Validate the portable learner and its recorded family separation."""
import json

import numpy as np
import pytest

from api.category_ensemble import ROOT, predict
from api.compact import CLASSES


def test_portable_combiner_matches_saved_sklearn_outputs():
    report = json.loads((ROOT / "category_ensemble.json").read_text())
    for fixture in report["portable_export_checks"]:
        category, confidence = predict(np.asarray(fixture["features"], dtype=np.float32))
        expected = np.asarray(fixture["probabilities"])
        assert category == CLASSES[int(expected.argmax())]
        assert confidence == pytest.approx(float(expected.max()), abs=1e-5)


def test_training_provenance_keeps_families_in_one_split():
    report = json.loads((ROOT / "category_ensemble.json").read_text())
    splits = report["family_splits"]
    train = {family for family, split in splits.items() if split == "train"}
    validation = {family for family, split in splits.items() if split == "validation"}
    assert train and validation and train.isdisjoint(validation)
    assert train | validation == set(splits)
    assert "Own source family excluded" in report["feature_rule"]
    assert report["classes"] == list(CLASSES)
