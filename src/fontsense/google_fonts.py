from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from .font_audit import parse_metadata_pb
from .utils import project_root

CATEGORY_ORDER = ("serif", "sans_serif", "display", "handwriting", "monospace")
CATEGORY_FROM_METADATA = {
    "SERIF": "serif",
    "SANS_SERIF": "sans_serif",
    "DISPLAY": "display",
    "HANDWRITING": "handwriting",
    "MONOSPACE": "monospace",
}
BASE_FOLDERS = ("ofl", "apache", "ufl")
LICENSE_FILE_BY_CODE = {
    "OFL": "OFL.txt",
    "APACHE2": "LICENSE.txt",
    "UFL": "UFL.txt",
}
LICENSE_CODE_BY_FOLDER = {
    "ofl": "OFL",
    "apache": "APACHE2",
    "ufl": "UFL",
}
GOOGLE_FONTS_AUDIT_COLUMNS = [
    "family",
    "category",
    "source",
    "license",
    "path",
    "latin_support",
    "validation_status",
    "failure_reason",
    "usable",
    "slug",
    "catalog_category",
    "metadata_category",
    "metadata_url",
    "source_url",
    "license_url",
    "license_path",
    "subsets",
]
OFFICIAL_SOURCE = "Google Fonts official repository"
LATIN_VALIDATION_TEXT = "FontSense Typography 123"
RAW_ROOT = "https://raw.githubusercontent.com/google/fonts/main"


def request_with_retry(url: str, *, binary: bool = False, retries: int = 3) -> bytes | str:
    headers = {"User-Agent": "FontSense-Capstone/0.1"}
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=40)
            if response.status_code == 404:
                raise FileNotFoundError(url)
            response.raise_for_status()
            return response.content if binary else response.text
        except FileNotFoundError:
            raise
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def choose_font_file(filenames: list[str]) -> str | None:
    candidates = [name for name in filenames if Path(name).suffix.lower() in {".ttf", ".otf"}]
    if not candidates:
        return None
    preferred_patterns = ("Regular.ttf", "Regular.otf", "[wght].ttf", "-VariableFont_wght.ttf")
    for pattern in preferred_patterns:
        for name in candidates:
            if name.endswith(pattern):
                return name
    non_italic = [name for name in candidates if "italic" not in name.lower()]
    return (non_italic or candidates)[0]


def validate_font_file(font_path: str | Path, text: str = LATIN_VALIDATION_TEXT) -> tuple[bool, str]:
    """Open a font with Pillow and prove that it paints visible Latin text."""
    try:
        font = ImageFont.truetype(str(font_path), 48)
        bbox = font.getbbox(text)
        if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return False, "font produced an empty glyph box"
        width = bbox[2] - bbox[0] + 20
        height = bbox[3] - bbox[1] + 20
        image = Image.new("L", (width, height), color=0)
        draw = ImageDraw.Draw(image)
        draw.text((10 - bbox[0], 10 - bbox[1]), text, fill=255, font=font)
        if image.getbbox() is None:
            return False, "font produced no visible pixels"
        return True, ""
    except Exception as exc:  # font parsing errors vary by Pillow and font format
        return False, str(exc)


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _audit_row(slug: str, catalog_category: str) -> dict:
    return {
        "family": "",
        "category": "",
        "source": OFFICIAL_SOURCE,
        "license": "",
        "path": "",
        "latin_support": None,
        "validation_status": "failed",
        "failure_reason": "",
        "usable": False,
        "slug": slug,
        "catalog_category": catalog_category,
        "metadata_category": "",
        "metadata_url": "",
        "source_url": "",
        "license_url": "",
        "license_path": "",
        "subsets": "",
    }


def _failed_row(row: dict, reason: str) -> dict:
    row.update(
        path="",
        validation_status="failed",
        failure_reason=reason,
        usable=False,
    )
    return row


def locate_metadata(slug: str) -> tuple[str, str, dict] | None:
    for base in BASE_FOLDERS:
        url = f"{RAW_ROOT}/{base}/{slug}/METADATA.pb"
        try:
            text = request_with_retry(url)
        except FileNotFoundError:
            continue
        metadata = parse_metadata_pb(str(text))
        if metadata.get("family"):
            return base, str(text), metadata
    return None


def summarise_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Count independent usable families and failures in every target category."""
    rows: list[dict] = []
    usable_mask = frame["usable"].astype(str).str.lower().isin({"true", "1"})
    for category in CATEGORY_ORDER:
        catalog_rows = frame.loc[frame["catalog_category"] == category]
        usable_rows = frame.loc[usable_mask & (frame["category"] == category)]
        catalog_usable = catalog_rows["usable"].astype(str).str.lower().isin({"true", "1"})
        rows.append(
            {
                "category": category,
                "usable_families": int(usable_rows["family"].replace("", pd.NA).nunique()),
                "failed_entries": int((~catalog_usable).sum()),
                "audited_entries": int(len(catalog_rows)),
            }
        )
    return pd.DataFrame(rows)


def download_catalog(
    catalog_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    max_per_category: int = 35,
) -> pd.DataFrame:
    if max_per_category < 1:
        raise ValueError("max_per_category must be positive")
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    if not isinstance(catalog, dict):
        raise ValueError("Google Fonts catalog must map categories to font slug lists")
    unsupported = sorted(set(catalog) - set(CATEGORY_ORDER))
    if unsupported:
        raise ValueError(f"Unsupported catalog categories: {unsupported}")
    all_slugs = [
        str(slug).strip().lower()
        for slugs in catalog.values()
        if isinstance(slugs, list)
        for slug in slugs
    ]
    if any(not isinstance(slugs, list) for slugs in catalog.values()):
        raise ValueError("Every Google Fonts category must contain a list of slugs")
    if any(not slug for slug in all_slugs):
        raise ValueError("Google Fonts catalog contains a blank slug")
    if len(all_slugs) != len(set(all_slugs)):
        raise ValueError("Google Fonts catalog contains duplicate slugs")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    accepted_families: set[str] = set()

    for expected_category, slugs in catalog.items():
        accepted = 0
        for slug in tqdm(slugs, desc=f"Downloading {expected_category}"):
            if accepted >= max_per_category:
                break
            slug = str(slug).strip().lower()
            row = _audit_row(slug, expected_category)
            destination: Path | None = None
            try:
                located = locate_metadata(slug)
                if located is None:
                    rows.append(_failed_row(row, "official METADATA.pb not found"))
                    continue
                base, metadata_text, metadata = located
                metadata_category = str(metadata.get("category", "")).strip()
                actual_category = CATEGORY_FROM_METADATA.get(metadata_category, "")
                subsets = [str(subset).strip() for subset in metadata.get("subsets", [])]
                license_code = str(metadata.get("license", "")).strip()
                metadata_url = f"{RAW_ROOT}/{base}/{slug}/METADATA.pb"
                row.update(
                    family=str(metadata.get("family", "")).strip(),
                    category=actual_category,
                    license=license_code,
                    latin_support="latin" in subsets,
                    metadata_category=metadata_category,
                    metadata_url=metadata_url,
                    subsets=",".join(subsets),
                )
                if not row["family"]:
                    rows.append(_failed_row(row, "official metadata has no family name"))
                    continue
                if not actual_category:
                    rows.append(_failed_row(row, f"unsupported official category: {metadata_category or 'missing'}"))
                    continue
                if actual_category != expected_category:
                    rows.append(_failed_row(row, "official metadata category does not match catalog category"))
                    continue
                if not row["latin_support"]:
                    rows.append(_failed_row(row, "official metadata has no latin subset"))
                    continue
                filename = choose_font_file(metadata.get("filenames", []))
                if filename is None:
                    rows.append(_failed_row(row, "official metadata has no TTF/OTF file"))
                    continue
                license_filename = LICENSE_FILE_BY_CODE.get(license_code)
                if license_filename is None:
                    rows.append(_failed_row(row, f"unsupported or missing licence code: {license_code or 'missing'}"))
                    continue
                expected_license = LICENSE_CODE_BY_FOLDER.get(base)
                if license_code != expected_license:
                    rows.append(_failed_row(row, "metadata licence does not match repository folder"))
                    continue

                category_dir = output_dir / expected_category
                category_dir.mkdir(parents=True, exist_ok=True)
                safe_family = re.sub(r"[^A-Za-z0-9._-]+", "_", metadata["family"]).strip("_")
                destination = category_dir / f"{safe_family}__{Path(filename).name}"
                font_url = f"{RAW_ROOT}/{base}/{slug}/{quote(filename)}"
                license_url = f"{RAW_ROOT}/{base}/{slug}/{license_filename}"
                row.update(source_url=font_url, license_url=license_url)
                license_text = str(request_with_retry(license_url))
                if not license_text.strip():
                    rows.append(_failed_row(row, "official licence file is empty"))
                    continue

                destination.write_bytes(request_with_retry(font_url, binary=True))
                valid, validation_reason = validate_font_file(destination)
                if not valid:
                    destination.unlink(missing_ok=True)
                    rows.append(_failed_row(row, f"font validation failed: {validation_reason}"))
                    continue

                family_key = row["family"].casefold()
                if family_key in accepted_families:
                    destination.unlink(missing_ok=True)
                    rows.append(_failed_row(row, "duplicate family already accepted"))
                    continue

                metadata_path = category_dir / f"{safe_family}__METADATA.pb"
                license_path = category_dir / f"{safe_family}__{license_filename}"
                metadata_path.write_text(metadata_text, encoding="utf-8")
                license_path.write_text(license_text, encoding="utf-8")
                row.update(
                    path=_portable_path(destination),
                    validation_status="passed",
                    failure_reason="",
                    usable=True,
                    source_url=font_url,
                    license_url=license_url,
                    license_path=_portable_path(license_path),
                )
                rows.append(row)
                accepted_families.add(family_key)
                accepted += 1
            except Exception as exc:
                if destination is not None:
                    destination.unlink(missing_ok=True)
                rows.append(_failed_row(row, f"{type(exc).__name__}: {exc}"))

    frame = pd.DataFrame(rows, columns=GOOGLE_FONTS_AUDIT_COLUMNS)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, GOOGLE_FONTS_AUDIT_COLUMNS].to_csv(manifest_path, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a balanced, licensed subset of Google Fonts.")
    parser.add_argument("--catalog", default=str(project_root() / "config/google_fonts_catalog.json"))
    parser.add_argument("--output-dir", default=str(project_root() / "data/raw/fonts/google_fonts"))
    parser.add_argument("--manifest", default=str(project_root() / "data/interim/google_fonts_manifest.csv"))
    parser.add_argument("--summary", default=str(project_root() / "data/interim/google_fonts_audit_summary.csv"))
    parser.add_argument("--max-per-category", type=int, default=35)
    args = parser.parse_args()
    frame = download_catalog(args.catalog, args.output_dir, args.manifest, args.max_per_category)
    summary = summarise_audit(frame)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Saved manifest: {args.manifest}")
    print(f"Saved summary: {args.summary}")
    missing_categories = summary.loc[summary["usable_families"] == 0, "category"].tolist()
    if missing_categories:
        raise RuntimeError(f"No usable Google Fonts families were found for: {missing_categories}")


if __name__ == "__main__":
    main()
