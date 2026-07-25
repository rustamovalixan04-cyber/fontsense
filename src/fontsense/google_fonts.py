from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from PIL import ImageFont
from tqdm import tqdm

from .font_audit import parse_metadata_pb
from .utils import project_root

CATEGORY_FROM_METADATA = {
    "SERIF": "serif",
    "SANS_SERIF": "sans_serif",
    "DISPLAY": "display",
    "HANDWRITING": "handwriting",
    "MONOSPACE": "monospace",
}
BASE_FOLDERS = ("ofl", "apache", "ufl")
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


def download_catalog(
    catalog_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    max_per_category: int = 35,
) -> pd.DataFrame:
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for expected_category, slugs in catalog.items():
        accepted = 0
        for slug in tqdm(slugs, desc=f"Downloading {expected_category}"):
            if accepted >= max_per_category:
                break
            try:
                located = locate_metadata(slug.lower())
                if located is None:
                    rows.append({"slug": slug, "expected_category": expected_category, "usable": False, "reason": "metadata not found"})
                    continue
                base, metadata_text, metadata = located
                actual_category = CATEGORY_FROM_METADATA.get(metadata.get("category", ""), "")
                if actual_category != expected_category:
                    rows.append({
                        "slug": slug,
                        "family": metadata.get("family", ""),
                        "expected_category": expected_category,
                        "actual_category": actual_category,
                        "usable": False,
                        "reason": "metadata category mismatch",
                    })
                    continue
                if "latin" not in metadata.get("subsets", []):
                    rows.append({
                        "slug": slug,
                        "family": metadata.get("family", ""),
                        "expected_category": expected_category,
                        "actual_category": actual_category,
                        "usable": False,
                        "reason": "no latin subset",
                    })
                    continue
                filename = choose_font_file(metadata.get("filenames", []))
                if filename is None:
                    rows.append({"slug": slug, "expected_category": expected_category, "usable": False, "reason": "no TTF/OTF file"})
                    continue
                category_dir = output_dir / expected_category
                category_dir.mkdir(parents=True, exist_ok=True)
                safe_family = re.sub(r"[^A-Za-z0-9._-]+", "_", metadata["family"]).strip("_")
                destination = category_dir / f"{safe_family}__{Path(filename).name}"
                font_url = f"{RAW_ROOT}/{base}/{slug}/{quote(filename)}"
                destination.write_bytes(request_with_retry(font_url, binary=True))
                # Confirm Pillow can load and render Latin text.
                font = ImageFont.truetype(str(destination), 48)
                bbox = font.getbbox("Typography 123")
                if bbox is None or bbox[2] <= bbox[0]:
                    destination.unlink(missing_ok=True)
                    raise ValueError("font produced an empty glyph box")
                metadata_url = f"{RAW_ROOT}/{base}/{slug}/METADATA.pb"
                (category_dir / f"{safe_family}__METADATA.pb").write_text(metadata_text, encoding="utf-8")
                license_code = metadata.get("license", "")
                license_candidates = {"OFL": "OFL.txt", "APACHE2": "LICENSE.txt", "UFL": "UFL.txt"}
                license_filename = license_candidates.get(license_code, "")
                license_url = ""
                if license_filename:
                    candidate_url = f"{RAW_ROOT}/{base}/{slug}/{license_filename}"
                    try:
                        license_text = request_with_retry(candidate_url)
                        (category_dir / f"{safe_family}__{license_filename}").write_text(str(license_text), encoding="utf-8")
                        license_url = candidate_url
                    except Exception:
                        license_url = candidate_url
                rows.append({
                    "family": metadata["family"],
                    "category": expected_category,
                    "path": str(destination),
                    "slug": slug,
                    "license": license_code,
                    "license_url": license_url,
                    "metadata_url": metadata_url,
                    "subsets": ",".join(metadata.get("subsets", [])),
                    "source_url": font_url,
                    "usable": True,
                    "reason": "ok",
                })
                accepted += 1
            except Exception as exc:
                rows.append({"slug": slug, "expected_category": expected_category, "usable": False, "reason": str(exc)})

    frame = pd.DataFrame(rows)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(manifest_path, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a balanced, licensed subset of Google Fonts.")
    parser.add_argument("--catalog", default=str(project_root() / "config/google_fonts_catalog.json"))
    parser.add_argument("--output-dir", default=str(project_root() / "data/raw/fonts/google_fonts"))
    parser.add_argument("--manifest", default=str(project_root() / "data/interim/google_fonts_manifest.csv"))
    parser.add_argument("--max-per-category", type=int, default=35)
    args = parser.parse_args()
    frame = download_catalog(args.catalog, args.output_dir, args.manifest, args.max_per_category)
    if "usable" in frame:
        usable = frame[frame["usable"] == True]  # noqa: E712
        print(usable.groupby("category").size())
    print(f"Saved manifest: {args.manifest}")


if __name__ == "__main__":
    main()
