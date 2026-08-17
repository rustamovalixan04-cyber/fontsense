from __future__ import annotations

import json
from pathlib import Path
import socket
import sys

import app
import pytest
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


def test_launcher_chooses_a_free_local_port():
    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for candidate in range(20_000, 60_000):
        try:
            occupied.bind((windows_launcher.LOCAL_HOST, candidate))
            break
        except OSError:
            continue
    else:
        occupied.close()
        pytest.fail("Could not reserve a test port.")

    occupied.listen()
    original_default = windows_launcher.DEFAULT_PORT
    try:
        windows_launcher.DEFAULT_PORT = candidate
        selected = windows_launcher.find_available_port()
    finally:
        windows_launcher.DEFAULT_PORT = original_default
        occupied.close()

    assert selected != candidate
    assert candidate < selected < candidate + windows_launcher.PORT_SEARCH_LIMIT


def test_normal_launch_prints_startup_and_browser_fallback(monkeypatch, capsys):
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

    windows_launcher.launch_application(inbrowser=True, port=7997)

    output = capsys.readouterr().out
    assert "FontSense is starting" in output
    assert "Loading the final CNN" in output
    assert "Opening FontSense in your browser" in output
    assert "Open this address in your browser" in output
    assert "http://127.0.0.1:7997" in output
    assert calls[0]["server_port"] == 7997


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
