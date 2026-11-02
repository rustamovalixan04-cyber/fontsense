"""Discover every eligible OFL Latin family in the pinned Google Fonts tree.

Writes provenance, exclusions and checksums; downloaded font programs stay
ignored. The original 56-family category-combiner catalog is never modified.
"""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
from urllib.parse import quote

from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_font_catalog import REVISION, fetch, CATEGORY_AUDITS
from scripts.extended_font_assets import make_subset

logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
DESTINATION = ROOT / "artifacts/web"


def audit(row):
    path = row["path"]
    metadata = row.get("metadata", "")
    slug = path.split("/")[1]
    if 'license: "OFL"' not in metadata or 'subsets: "latin"' not in metadata:
        return [], [], {"family_id": slug, "reason": "No declared OFL Latin subset"}
    family = re.search(r'^name: "([^"]+)"', metadata, re.M).group(1)
    declared = [value.lower() for value in re.findall(r'^category: "([^"]+)"', metadata, re.M)]
    category = next((value for value in declared if value in ("serif", "sans_serif", "display", "handwriting", "monospace")), "display")
    if slug in CATEGORY_AUDITS:
        category = "handwriting"
    base = f"https://raw.githubusercontent.com/google/fonts/{REVISION}/ofl/{slug}"
    license_url = base + "/OFL.txt"
    if slug == "tinos":
        license_url = "https://raw.githubusercontent.com/googlefonts/tinos/3b4482a99b80ea5fc75f187b1be3120a3f5905b3/OFL.txt"
    try:
        license_data = fetch(license_url)
        if b"SIL OPEN FONT LICENSE" not in license_data.upper():
            raise ValueError("License text is not OFL")
        blocks = re.findall(r'fonts \{(.*?)\n\}', metadata, re.S)
        sources, records, seen_styles = {}, [], set()
        for italic, weight, suffix in ((False, 400, ""), (False, 700, "-bold"), (True, 400, "-italic"), (True, 700, "-bolditalic")):
            candidates = [block for block in blocks if f'style: "{"italic" if italic else "normal"}"' in block]
            if not candidates:
                continue
            block = min(candidates, key=lambda value: abs(int(re.search(r'weight: (\d+)', value).group(1)) - weight))
            filename = re.search(r'filename: "([^"]+)"', block).group(1)
            actual_weight = int(re.search(r'weight: (\d+)', block).group(1))
            source_id = "x-" + slug + ("-italic" if italic else "") + "-" + hashlib.sha256(filename.encode()).hexdigest()[:8]
            if filename not in sources:
                data = fetch(base + "/" + quote(filename))
                relative = "extended-fonts/" + source_id + ".ttf"
                target = DESTINATION / relative
                # Explicit interrupted-build resume. Ordinary/fresh builds
                # always derive bytes from the pinned original again.
                converted = target.read_bytes() if "--resume" in sys.argv and target.exists() else make_subset(data, source_id)
                (DESTINATION / relative).parent.mkdir(parents=True, exist_ok=True)
                (DESTINATION / relative).write_bytes(converted)
                (DESTINATION / "extended-fonts" / (slug + "-OFL.txt")).write_bytes(license_data)
                sources[filename] = {"id": source_id, "file": relative,
                                     "download_url": base + "/" + quote(filename),
                                     "source_sha256": hashlib.sha256(data).hexdigest(),
                                     "sha256": hashlib.sha256(converted).hexdigest(),
                                     "license_file": "extended-fonts/" + slug + "-OFL.txt",
                                     "license_url": license_url, "license_sha256": hashlib.sha256(license_data).hexdigest()}
            source = sources[filename]
            font = ImageFont.truetype(str(DESTINATION / source["file"]), 64)
            try:
                axes = font.get_variation_axes()
            except OSError:
                axes = []
            axis = next((item for item in axes if item["name"] == b"Weight"), None)
            if axis and not axis["minimum"] <= weight <= axis["maximum"]:
                continue
            if not axis and abs(actual_weight - weight) > 150:
                continue
            if axis:
                font.set_variation_by_axes([weight if item["name"] == b"Weight" else item["default"] for item in axes])
            advances = [font.getlength(char) for char in "iWm012"]
            audited_category = "monospace" if max(advances) - min(advances) <= .1 else category
            key = (source["file"], weight if axis else actual_weight)
            if key in seen_styles:
                continue
            seen_styles.add(key)
            records.append({"id": "x-" + slug + suffix, "family_id": slug, "family": family,
                            "category": audited_category, "style": ("Bold Italic" if italic else "Bold") if weight == 700 else ("Italic" if italic else "Regular"),
                            "weight": weight if axis else actual_weight, "variable_weight": bool(axis),
                            "file": source["file"], "italic": italic,
                            "url": "https://fonts.google.com/specimen/" + quote(family)})
        if not records:
            raise ValueError("No supported Regular/Bold/Italic reference styles")
        return records, list(sources.values()), None
    except Exception as error:
        return [], [], {"family_id": slug, "reason": str(error)}


def main():
    cached = ROOT.parent / "broad-source-metadata.json"
    if cached.exists():
        rows = json.loads(cached.read_text())
    else:
        tree = json.loads(fetch(f"https://api.github.com/repos/google/fonts/git/trees/{REVISION}?recursive=1"))
        if tree.get("truncated"):
            raise RuntimeError("Incomplete source tree")
        paths = [row["path"] for row in tree["tree"] if row["path"].startswith("ofl/") and row["path"].endswith("/METADATA.pb")]
        def metadata(path):
            return {"path": path, "metadata": fetch(f"https://raw.githubusercontent.com/google/fonts/{REVISION}/{path}").decode()}
        with ThreadPoolExecutor(max_workers=12) as pool:
            rows = list(pool.map(metadata, paths))
        cached.write_text(json.dumps(rows), encoding="utf-8")
    records, sources, exclusions = [], [], []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i, (fonts, assets, excluded) in enumerate(pool.map(audit, rows)):
            records.extend(fonts); sources.extend(assets)
            if excluded:
                exclusions.append(excluded)
            if i % 100 == 0:
                print("Audited", i, "/", len(rows), "families; included", len({font['family'] for font in records}), flush=True)
    manifest = {"version": "extended-latin-1", "revision": REVISION,
                "scope": "All eligible OFL Latin families in this pinned tree, not all existing fonts or scripts.",
                "transform": "Renamed Latin U+0020..00FF subset; copyright/license retained; fonttools==4.61.1",
                "families": len({font["family"] for font in records}), "styles": len(records),
                "sources": sources, "fonts": records, "exclusions": exclusions}
    (DESTINATION / "extended_catalog.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("RESULT", manifest["families"], "families", len(records), "styles", len(sources), "assets", flush=True)


if __name__ == "__main__":
    main()
