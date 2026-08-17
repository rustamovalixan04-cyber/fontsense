from __future__ import annotations

import json
from pathlib import Path
import sys

import app
import windows_launcher


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HASH = "c98cf0d1a02503a02b8f8242fec462ea2a0c455380238ec54fc4f62fdb13bb2f"
EXPECTED_CLASSES = [
    "display",
    "handwriting",
    "monospace",
    "sans_serif",
    "serif",
]


def test_source_runtime_root_is_repository_root():
    assert app.get_runtime_root() == ROOT
    assert app.ROOT == ROOT


def test_runtime_root_uses_pyinstaller_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    assert app.get_runtime_root() == tmp_path.resolve()


def test_launcher_self_test_preserves_frozen_contract():
    report = windows_launcher.frozen_runtime_report()

    assert report["status"] == "PASS"
    assert report["checkpoint_sha256"] == EXPECTED_HASH
    assert report["class_order"] == EXPECTED_CLASSES
    assert report["threshold"] == 0.60
    assert report["preprocessing"] == {
        "image_size": [112, 48],
        "grayscale": True,
        "normalize_mean": [0.5],
        "normalize_std": [0.5],
    }


def test_launcher_passes_browser_choice_to_gradio(monkeypatch):
    calls = []

    class DemoStub:
        def launch(self, **kwargs):
            calls.append(kwargs)

    class AppStub:
        APP_CSS = "css"

        @staticmethod
        def build_demo():
            return DemoStub()

    monkeypatch.setattr(windows_launcher, "load_application", lambda: AppStub())

    windows_launcher.launch_application(inbrowser=True, port=7999)
    windows_launcher.launch_application(inbrowser=False, port=7998)

    assert calls[0]["inbrowser"] is True
    assert calls[1]["inbrowser"] is False
    assert all(call["server_name"] == "127.0.0.1" for call in calls)
    assert all(call["share"] is False for call in calls)


def test_equivalence_contract_was_predeclared_with_five_classes_and_images():
    contract = json.loads(
        (ROOT / "packaging" / "windows" / "equivalence_contract.json").read_text(
            encoding="utf-8"
        )
    )

    assert contract["status"] == "predeclared_before_first_packaged_comparison"
    assert contract["checkpoint_sha256"] == EXPECTED_HASH
    assert contract["class_order"] == EXPECTED_CLASSES
    assert contract["threshold"] == 0.60
    assert contract["maximum_absolute_probability_difference"] == 1e-5
    assert len(contract["comparison_images"]) == 5
    assert {item["category"] for item in contract["comparison_images"]} == set(
        EXPECTED_CLASSES
    )


def test_pyinstaller_spec_includes_only_required_assessed_runtime_resources():
    source = (ROOT / "packaging" / "windows" / "FontSense.spec").read_text(
        encoding="utf-8"
    )

    assert "artifacts\" / \"cnn\" / \"cnn_model.pt" in source
    assert "pre_test_freeze.json" in source
    assert "hog_pipeline.joblib" not in source
    assert "full_manifest.csv" not in source
    assert "google_fonts_final_family_split.csv" not in source
