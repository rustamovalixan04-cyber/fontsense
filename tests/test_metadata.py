from contextlib import contextmanager
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
from matplotlib import get_data_path

from fontsense.font_audit import (
    AUDIT_COLUMNS,
    audit_system_fonts,
    font_search_directories,
    parse_metadata_pb,
    resolve_system_font,
)
from fontsense.generate_dataset import generate_dataset
from fontsense.google_fonts import (
    CATEGORY_ORDER,
    GOOGLE_FONTS_AUDIT_COLUMNS,
    download_catalog,
    summarise_audit,
    validate_font_file,
)
from fontsense.mlflow_utils import optional_mlflow_run


def packaged_font(filename: str) -> Path:
    return Path(get_data_path()) / "fonts" / "ttf" / filename


def test_parse_google_fonts_metadata():
    text = """name: \"Example Sans\"
license: \"OFL\"
category: \"SANS_SERIF\"
fonts { filename: \"ExampleSans-Regular.ttf\" }
subsets: \"latin\"
"""
    result = parse_metadata_pb(text)
    assert result["family"] == "Example Sans"
    assert result["category"] == "SANS_SERIF"
    assert result["license"] == "OFL"
    assert result["filenames"] == ["ExampleSans-Regular.ttf"]
    assert "latin" in result["subsets"]


def test_google_fonts_audit_uses_official_metadata_and_validates_rendering(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    output_dir = tmp_path / "fonts"
    manifest_path = tmp_path / "manifest.csv"
    catalog_path.write_text(json.dumps({"serif": ["example-serif"]}), encoding="utf-8")
    metadata = {
        "family": "Example Serif",
        "category": "SERIF",
        "license": "OFL",
        "filenames": ["ExampleSerif-Regular.ttf"],
        "subsets": ["latin", "latin-ext"],
    }
    monkeypatch.setattr(
        "fontsense.google_fonts.locate_metadata",
        lambda slug: ("ofl", 'name: "Example Serif"', metadata),
    )
    font_bytes = packaged_font("DejaVuSerif.ttf").read_bytes()

    def fake_request(url, *, binary=False, retries=3):
        return font_bytes if binary else "Official licence text"

    monkeypatch.setattr("fontsense.google_fonts.request_with_retry", fake_request)

    frame = download_catalog(catalog_path, output_dir, manifest_path, max_per_category=1)
    saved = pd.read_csv(manifest_path)
    row = frame.iloc[0]

    assert list(frame.columns) == GOOGLE_FONTS_AUDIT_COLUMNS
    assert list(saved.columns) == GOOGLE_FONTS_AUDIT_COLUMNS
    assert row["family"] == "Example Serif"
    assert row["category"] == "serif"
    assert row["source"] == "Google Fonts official repository"
    assert row["license"] == "OFL"
    assert bool(row["latin_support"])
    assert row["validation_status"] == "passed"
    assert row["failure_reason"] == ""
    assert bool(row["usable"])
    assert Path(row["path"]).is_file()
    assert validate_font_file(row["path"]) == (True, "")


def test_google_fonts_audit_keeps_failed_rows_in_the_same_schema(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    manifest_path = tmp_path / "manifest.csv"
    catalog_path.write_text(json.dumps({"display": ["missing-font"]}), encoding="utf-8")
    monkeypatch.setattr("fontsense.google_fonts.locate_metadata", lambda slug: None)

    frame = download_catalog(catalog_path, tmp_path / "fonts", manifest_path, max_per_category=1)
    row = frame.iloc[0]

    assert list(frame.columns) == GOOGLE_FONTS_AUDIT_COLUMNS
    assert row["family"] == ""
    assert row["category"] == ""
    assert row["catalog_category"] == "display"
    assert not bool(row["usable"])
    assert row["validation_status"] == "failed"
    assert row["failure_reason"] == "official METADATA.pb not found"


def test_google_fonts_audit_rejects_catalog_label_that_disagrees_with_official_metadata(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    manifest_path = tmp_path / "manifest.csv"
    catalog_path.write_text(json.dumps({"display": ["example-font"]}), encoding="utf-8")
    metadata = {
        "family": "Example Font",
        "category": "SANS_SERIF",
        "license": "OFL",
        "filenames": ["ExampleFont-Regular.ttf"],
        "subsets": ["latin"],
    }
    monkeypatch.setattr(
        "fontsense.google_fonts.locate_metadata",
        lambda slug: ("ofl", 'name: "Example Font"', metadata),
    )

    frame = download_catalog(catalog_path, tmp_path / "fonts", manifest_path, max_per_category=1)
    row = frame.iloc[0]

    assert row["catalog_category"] == "display"
    assert row["metadata_category"] == "SANS_SERIF"
    assert row["category"] == "sans_serif"
    assert not bool(row["usable"])
    assert row["failure_reason"] == "official metadata category does not match catalog category"


def test_google_fonts_audit_rejects_missing_licence_file(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    manifest_path = tmp_path / "manifest.csv"
    catalog_path.write_text(json.dumps({"monospace": ["example-mono"]}), encoding="utf-8")
    metadata = {
        "family": "Example Mono",
        "category": "MONOSPACE",
        "license": "OFL",
        "filenames": ["ExampleMono-Regular.ttf"],
        "subsets": ["latin"],
    }
    monkeypatch.setattr(
        "fontsense.google_fonts.locate_metadata",
        lambda slug: ("ofl", 'name: "Example Mono"', metadata),
    )

    def missing_request(url, **kwargs):
        raise FileNotFoundError(url)

    monkeypatch.setattr("fontsense.google_fonts.request_with_retry", missing_request)

    frame = download_catalog(catalog_path, tmp_path / "fonts", manifest_path, max_per_category=1)
    row = frame.iloc[0]

    assert not bool(row["usable"])
    assert row["path"] == ""
    assert "FileNotFoundError" in row["failure_reason"]


def test_google_fonts_summary_counts_independent_usable_families():
    frame = pd.DataFrame(
        [
            {"family": "A", "category": "serif", "catalog_category": "serif", "usable": True},
            {"family": "A", "category": "serif", "catalog_category": "serif", "usable": True},
            {"family": "B", "category": "serif", "catalog_category": "serif", "usable": False},
        ]
    )

    summary = summarise_audit(frame)

    assert summary["category"].tolist() == list(CATEGORY_ORDER)
    serif = summary.loc[summary["category"] == "serif"].iloc[0]
    assert serif["usable_families"] == 1
    assert serif["failed_entries"] == 1
    assert serif["audited_entries"] == 3


def test_windows_font_discovery_checks_system_and_user_directories(tmp_path, monkeypatch):
    windows_root = tmp_path / "Windows"
    system_fonts = windows_root / "Fonts"
    local_app_data = tmp_path / "LocalAppData"
    user_fonts = local_app_data / "Microsoft" / "Windows" / "Fonts"
    system_fonts.mkdir(parents=True)
    user_fonts.mkdir(parents=True)
    shutil.copy2(packaged_font("DejaVuSans.ttf"), system_fonts / "system-font.ttf")
    shutil.copy2(packaged_font("DejaVuSerif.ttf"), user_fonts / "user-font.ttf")
    monkeypatch.setenv("WINDIR", str(windows_root))
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))

    search_directories = font_search_directories("Windows")
    assert system_fonts in search_directories
    assert user_fonts in search_directories
    assert resolve_system_font("DejaVu Sans", system_name="Windows") == str((system_fonts / "system-font.ttf").resolve())
    assert resolve_system_font("DejaVu Serif", system_name="Windows") == str((user_fonts / "user-font.ttf").resolve())


def test_missing_font_returns_none(tmp_path):
    assert resolve_system_font(
        "Definitely Missing Font",
        system_name="Windows",
        search_dirs=[tmp_path],
    ) is None


def test_audit_uses_consistent_columns_and_keeps_missing_family(tmp_path):
    config_path = tmp_path / "fonts.json"
    output_path = tmp_path / "audit.csv"
    valid_path = packaged_font("DejaVuSerif.ttf")
    config_path.write_text(json.dumps({"serif": ["DejaVu Serif", "Missing Serif"]}), encoding="utf-8")
    resolver = lambda family: str(valid_path) if family == "DejaVu Serif" else None

    frame = audit_system_fonts(config_path, output_path, resolver=resolver)
    saved = pd.read_csv(output_path)

    assert list(frame.columns) == AUDIT_COLUMNS
    assert list(saved.columns) == AUDIT_COLUMNS
    missing = frame.loc[frame["family"] == "Missing Serif"].iloc[0]
    assert missing["category"] == "serif"
    assert missing["path"] == ""
    assert not bool(missing["usable"])
    assert missing["reason"] == "not found"


def test_zero_usable_fonts_stop_before_audit_manifest(tmp_path):
    config_path = tmp_path / "fonts.json"
    output_path = tmp_path / "audit.csv"
    config_path.write_text(json.dumps({"serif": ["Missing Serif"]}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="No usable fonts were found"):
        audit_system_fonts(config_path, output_path, resolver=lambda family: None)

    assert not output_path.exists()


def test_zero_usable_manifest_stops_before_dataset_generation(tmp_path):
    manifest_path = tmp_path / "font_manifest.csv"
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "dataset"
    pd.DataFrame([
        {"family": "Missing Serif", "category": "serif", "path": "", "usable": False, "reason": "not found"}
    ]).to_csv(manifest_path, index=False)
    config_path.write_text(json.dumps({
        "image_width": 224,
        "image_height": 96,
        "images_per_family": 1,
        "train_ratio": 0.7,
        "validation_ratio": 0.15,
        "test_ratio": 0.15,
        "phrases": ["FontSense"],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="zero usable fonts"):
        generate_dataset(manifest_path, output_dir, config_path, images_per_family=1)

    assert not output_dir.exists()


def test_mlflow_uses_ignored_sqlite_store(tmp_path, monkeypatch):
    calls: dict[str, str] = {}

    class FakeMlflow:
        def set_tracking_uri(self, uri: str) -> None:
            calls["tracking_uri"] = uri

        def set_experiment(self, name: str) -> None:
            calls["experiment"] = name

        @contextmanager
        def start_run(self, run_name: str):
            calls["run_name"] = run_name
            yield

    fake_mlflow = FakeMlflow()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    with optional_mlflow_run("FontSense-Test", "sqlite-smoke") as active_mlflow:
        assert active_mlflow is fake_mlflow

    expected_database = (tmp_path / "mlruns" / "mlflow.db").as_posix()
    assert calls == {
        "tracking_uri": f"sqlite:///{expected_database}",
        "experiment": "FontSense-Test",
        "run_name": "sqlite-smoke",
    }
