"""Reproducible synthetic product checks, separate from capstone test evidence."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.compact import get_model
from api.font_match import catalog, identify

WORDS = ("Quiet forms", "Design 42", "Pixel craft")
EXTERNAL = {
    "serif": ("ANTQUAB.TTF", "GARA.TTF", "pala.ttf", "constan.ttf"),
    "sans_serif": ("calibri.ttf", "segoeui.ttf", "tahoma.ttf", "GOTHIC.TTF"),
    "monospace": ("lucon.ttf", "monos.ttf", "OCR-B 10 BT.ttf", "unispace bd.ttf"),
    "handwriting": ("comic.ttf", "MTCORSVA.TTF", "Embassy BT.ttf", "Freehand521 BT.ttf"),
}


def specimen(path, text):
    font = ImageFont.truetype(str(path), 64)
    bounds = font.getbbox(text)
    image = Image.new("RGB", (bounds[2] - bounds[0] + 160, bounds[3] - bounds[1] + 110), "white")
    ImageDraw.Draw(image).text((80 - bounds[0], 55 - bounds[1]), text, font=font, fill="black")
    return image


def variants(image):
    yield "clean", image
    small = image.resize((round(image.width * .55), round(image.height * .55)), Image.Resampling.LANCZOS)
    stream = BytesIO()
    small.save(stream, format="JPEG", quality=55)
    yield "small_jpeg", Image.open(BytesIO(stream.getvalue())).convert("RGB")
    yield "dark", ImageOps.invert(image)


def summarize(rows):
    n = len(rows)
    return {
        "images": n,
        "baseline_category_correct": sum(r["baseline_category"] == r["expected_category"] for r in rows),
        "matching_category_correct": sum(r["category"] == r["expected_category"] for r in rows),
        "needs_manual_text": sum(r["needs_text"] for r in rows),
        "ocr_text_correct_ignoring_spaces_case": sum(r["text"].replace(" ", "").casefold() == r["expected_text"].replace(" ", "").casefold() for r in rows),
        "median_ms": round(float(np.median([r["ms"] for r in rows])), 1),
        "p95_ms": round(float(np.percentile([r["ms"] for r in rows], 95)), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows-fonts", type=Path)
    args = parser.parse_args()
    sources = [("in_library", f["family"], f["id"], f["category"], ROOT / "artifacts/web" / f["file"]) for f in catalog()]
    if args.windows_fonts:
        sources.extend(("outside_library", ImageFont.truetype(str(args.windows_fonts / filename), 64).getname()[0], None, category, args.windows_fonts / filename) for category, files in EXTERNAL.items() for filename in files)
    rows = []
    for group, family, font_id, category, path in sources:
        for text in WORDS:
            for condition, image in variants(specimen(path, text)):
                start = time.perf_counter()
                result = identify(image)
                rows.append({"group": group, "family": family, "font_id": font_id, "expected_category": category,
                             "expected_text": text, "condition": condition, "category": result["category"],
                             "baseline_category": result["classifier"]["predicted_category"], "text": result["text"],
                             "needs_text": result["needs_text"], "matches": [f["id"] for f in result["matches"]],
                             "ms": round((time.perf_counter() - start) * 1000, 1)})
        print(f"Checked {family}: {len(rows)} images", flush=True)
    summary = {group: summarize([r for r in rows if r["group"] == group]) for group in sorted({r["group"] for r in rows})}
    known = [r for r in rows if r["group"] == "in_library"]
    summary["in_library"].update({"family_top1": sum(bool(r["matches"]) and r["matches"][0] == r["font_id"] for r in known),
                                   "family_top3": sum(r["font_id"] in r["matches"][:3] for r in known)})
    result = {"method": "Synthetic product checks, not a held-out accuracy estimate. In-library families are rendered references; out-of-library examples use Windows fonts. Three new phrases, clean/55%-size JPEG quality55/dark variants. No historical training or final test data was read.",
              "summary": summary, "first_request_ms": rows[0]["ms"], "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
