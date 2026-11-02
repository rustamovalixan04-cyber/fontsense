"""Compare a matching candidate with the previous deployment on fixed images.

Validation is used to choose the implementation. The independent family holdout
is rendered/evaluated only after selection. Neither reads capstone test images.
"""
import argparse
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.font_match import identify

BASELINE = "7e899e98bf0322d0d3537d6bc6c3f91a03c2264b"


def previous_version():
    def old(path):
        return subprocess.check_output(["git", "show", BASELINE + ":" + path], cwd=ROOT, text=True, encoding="utf-8")
    compact = types.ModuleType("previous_compact")
    compact.__file__ = str(ROOT / "api/compact.py")
    exec(compile(old("api/compact.py"), compact.__file__, "exec"), compact.__dict__)
    matching = types.ModuleType("previous_matching")
    matching.__file__ = str(ROOT / "api/font_match.py")
    exec(compile(old("api/font_match.py"), matching.__file__, "exec"), matching.__dict__)
    matching.catalog = lambda: tuple(json.loads(old_catalog)["fonts"])
    matching.text_crop = compact.text_crop
    old_catalog = old("artifacts/web/font_catalog.json")
    return matching


def specimen(path, text, weight=None):
    font = ImageFont.truetype(str(path), 64)
    if weight is not None:
        try:
            axes = font.get_variation_axes()
            font.set_variation_by_axes([weight if a["name"] == b"Weight" else a["default"] for a in axes])
        except OSError:
            pass
    bounds = font.getbbox(text)
    image = Image.new("RGB", (bounds[2] - bounds[0] + 160, bounds[3] - bounds[1] + 110), "white")
    ImageDraw.Draw(image).text((80 - bounds[0], 55 - bounds[1]), text, font=font, fill="black")
    return image


def variants(image):
    yield "clean", image
    stream = BytesIO()
    image.resize((round(image.width * .55), round(image.height * .55)), Image.Resampling.LANCZOS).save(stream, format="JPEG", quality=55)
    yield "small_jpeg", Image.open(BytesIO(stream.getvalue())).convert("RGB")
    yield "dark", ImageOps.invert(image)
    yield "rotated", image.rotate(2.5, Image.Resampling.BICUBIC, expand=True, fillcolor="white")


def summarize(rows, field):
    count = len(rows)
    categories = sorted({r["expected"] for r in rows})
    f1 = []
    for category in categories:
        tp = sum(r["expected"] == category and r[field]["category"] == category for r in rows)
        fp = sum(r["expected"] != category and r[field]["category"] == category for r in rows)
        fn = sum(r["expected"] == category and r[field]["category"] != category for r in rows)
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {"images": count, "category_correct": sum(r["expected"] == r[field]["category"] for r in rows),
            "macro_f1": round(float(np.mean(f1)), 4),
            "median_ms": round(float(np.median([r[field]["ms"] for r in rows])), 1),
            "p95_ms": round(float(np.percentile([r[field]["ms"] for r in rows], 95)), 1),
            "needs_text": sum(r[field]["needs_text"] for r in rows),
            "per_category": {category: {"images": sum(r["expected"] == category for r in rows), "correct": sum(r["expected"] == category and r[field]["category"] == category for r in rows)} for category in categories}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("validation", "external_validation", "holdout", "regression"), required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = previous_version()
    if args.phase == "validation":
        sources = [(file, category, Path("C:/Windows/Fonts") / file, None) for category, files in {
            "serif": "times.ttf timesbd.ttf timesi.ttf timesbi.ttf georgia.ttf georgiab.ttf GARA.TTF pala.ttf constan.ttf ANTQUAB.TTF",
            "sans_serif": "arial.ttf arialbd.ttf ariali.ttf arialbi.ttf calibri.ttf calibriz.ttf segoeui.ttf segoeuib.ttf tahoma.ttf GOTHIC.TTF",
            "monospace": "consola.ttf consolab.ttf cour.ttf courbi.ttf lucon.ttf",
            "handwriting": "comic.ttf comicbd.ttf MTCORSVA.TTF",
            "display": "impact.ttf",
        }.items() for file in files.split()]
        words = ("Sphinx 27", "Letters and lines")
    elif args.phase in ("holdout", "external_validation"):
        assert args.assets, "Provide an independent holdout manifest"
        holdout = json.loads((args.assets / "font_catalog.json").read_text())
        library = json.loads((ROOT / "artifacts/web/font_catalog.json").read_text())
        assert not ({f["family"] for f in holdout["fonts"]} & {f["family"] for f in library["fonts"]})
        # At most two real styles per family; all 20 families are held out.
        selected = []
        families = {}
        for record in holdout["fonts"]:
            families.setdefault(record["family"], []).append(record)
        for records in families.values():
            selected.append(records[0])
            if len(records) > 1:
                selected.append(records[-1])
        sources = [(f["id"], f["category"], args.assets / f["file"], f["weight"] if f["variable_weight"] else None) for f in selected]
        words = ("Memory Lane", "Vector Studio")
    else:
        old = baseline.catalog()
        sources = [(f["id"], f["category"], ROOT / "artifacts/web" / f["file"], None) for f in old]
        words = ("Quiet forms", "Design 42", "Pixel craft")
    rows = []
    for name, category, path, weight in sources:
        for text in words:
            for condition, image in variants(specimen(path, text, weight)):
                if args.phase == "regression" and condition == "rotated":
                    continue
                row = {"source": name, "expected": category, "text": text, "condition": condition}
                for field, function in (("before", baseline.identify), ("after", identify)):
                    start = time.perf_counter()
                    result = function(image)
                    row[field] = {"category": result["category"], "text": result["text"],
                                  "matches": [f["id"] for f in result["matches"][:3]],
                                  "needs_text": result["needs_text"], "ms": round((time.perf_counter() - start) * 1000, 1)}
                rows.append(row)
        print(f"Compared {name}: {len(rows)} images", flush=True)
    report = {"phase": args.phase, "baseline_commit": BASELINE,
              "method": "Controlled synthetic product evidence, not a real-world accuracy estimate. No historical capstone training/test images were used.",
              "before": summarize(rows, "before"), "after": summarize(rows, "after"), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({field: report[field] for field in ("phase", "before", "after")}, indent=2))


if __name__ == "__main__":
    main()
