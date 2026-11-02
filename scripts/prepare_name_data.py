"""Prepare licensed source records for the separate font-name retrieval task.

Windows font programs stay on the local Windows machine. Only complete rendered
words/phrases are used to derive reference statistics; no glyph atlas is made.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_font_catalog import build_family

EXTRA = {
    "display": "lobster pressstart2p luckiestguy bowlbyonesc frijole nosifer specialelite fascinate limelight gravitasone modak rampartone chango",
    "handwriting": "greatvibes satisfy cookie permanentmarker yellowtail alexbrush badscript coveredbyyourgrace gloriahallelujah homemadeapple kaushanscript nanumpenscript nothingyoucoulddo",
}
NATIVE = [
    ("Impact", "display", ["impact.ttf"]),
    ("Times New Roman", "serif", ["times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"]),
    ("Arial", "sans_serif", ["arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"]),
    ("Verdana", "sans_serif", ["verdana.ttf", "verdanab.ttf", "verdanai.ttf", "verdanaz.ttf"]),
    ("Georgia", "serif", ["georgia.ttf", "georgiab.ttf", "georgiai.ttf", "georgiaz.ttf"]),
    ("Consolas", "monospace", ["consola.ttf", "consolab.ttf", "consolai.ttf", "consolaz.ttf"]),
    ("Courier New", "monospace", ["cour.ttf", "courbd.ttf", "couri.ttf", "courbi.ttf"]),
    ("Tahoma", "sans_serif", ["tahoma.ttf", "tahomabd.ttf"]),
    ("Trebuchet MS", "sans_serif", ["trebuc.ttf", "trebucbd.ttf", "trebucit.ttf", "trebucbi.ttf"]),
    ("Comic Sans MS", "handwriting", ["comic.ttf", "comicbd.ttf", "comici.ttf", "comicz.ttf"]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT.parent / "name-data")
    parser.add_argument("--windows-fonts", type=Path, default=Path("C:/Windows/Fonts"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "local-paths.json").write_text(json.dumps({"windows_fonts": str(args.windows_fonts.resolve())}), encoding="utf-8")
    main_catalog = json.loads((ROOT / "artifacts/web/font_catalog.json").read_text())
    records = [{**font, "asset_root": "main", "native": False} for font in main_catalog["fonts"]]
    jobs = [(category, slug) for category, names in EXTRA.items() for slug in names.split()]
    failures = []

    def audit(job):
        try:
            return [{**font, "asset_root": "extra", "native": False} for font in build_family(job, args.output)]
        except (OSError, ValueError) as exc:
            failures.append({"slug": job[1], "reason": str(exc)})
            return []

    with ThreadPoolExecutor(max_workers=6) as pool:
        records.extend(record for group in pool.map(audit, jobs) for record in group)
    for family, category, files in NATIVE:
        for index, filename in enumerate(files):
            path = args.windows_fonts / filename
            if not path.exists():
                continue
            style = ("Regular", "Bold", "Italic", "Bold Italic")[index]
            slug = "native-" + family.lower().replace(" ", "-") + "-" + str(index)
            records.append({"id": slug, "family": family, "family_id": "native-" + family.lower().replace(" ", "-"),
                            "category": category, "style": style, "native": True, "asset_root": "native",
                            "file": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "source": "https://learn.microsoft.com/en-us/typography/font-list/" + family.lower().replace(" ", "-"),
                            "usage": "Whole-word raster examples only; local font program is not copied or redistributed.",
                            "usage_source": "https://learn.microsoft.com/en-us/typography/fonts/font-faq"})
    report = {"revision": main_catalog["revision"], "fonts": records, "excluded_sources": failures}
    (args.output / "catalog.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Sources:", len({f["family"] for f in records}), "families /", len(records), "styles")
    print("Excluded sources:", json.dumps(failures), flush=True)


if __name__ == "__main__":
    main()
