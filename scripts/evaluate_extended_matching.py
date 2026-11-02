"""Synthetic product retrieval audit, separate from assessed category results.

Queries use different size/rasterization and phrases from name-encoder training.
Catalog identities are known retrieval references, not held-out ML classes.
"""
import argparse
from io import BytesIO
import json
from pathlib import Path
import random
import sys
import time

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.extended_match import catalog, render_reference
from api.font_match import identify

PHRASES = ("Quiet Harbor", "Silver Workshop")


def summary(rows):
    known = [row for row in rows if row["expected"] is not None]
    returned = [row for row in rows if row["named"] is not None]
    unknown = [row for row in rows if row["expected"] is None]
    correct = sum(row["named"] == row["expected"] for row in returned)
    return {"images": len(rows), "known_images": len(known), "unknown_images": len(unknown),
            "names_returned": len(returned), "correct_returned": correct,
            "returned_precision": correct / max(len(returned), 1),
            "known_name_coverage": sum(row["named"] is not None for row in known) / max(len(known), 1),
            "known_top3": sum(row["expected"] in row["top3"] for row in known) / max(len(known), 1),
            "unknown_names_withheld": sum(row["named"] is None for row in unknown)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7211)
    parser.add_argument("--fresh", action="store_true", help="Use a different family sample and fresh phrases after development")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    grouped = {}
    for font in catalog():
        grouped.setdefault(font["family"], []).append(font)
    names = sorted(grouped)
    if args.fresh:
        development_names = set(random.Random(7211).sample(names, min(100, len(names))))
        names = [name for name in names if name not in development_names]
    selected = rng.sample(names, min(args.families, len(names)))
    records = [rng.choice(grouped[name]) for name in selected]
    rows = []
    # Rendering all queries for a phrase together allows the bounded reference
    # cache to be used, while cold cache latency is still recorded separately.
    phrases = ("Velvet Lantern", "Ocean Stories") if args.fresh else PHRASES
    for phrase in phrases:
        queries = [(font["family"], font["style"], render_reference(font["id"], phrase, 73)) for font in records]
        for filename, family in (("calibri.ttf", "Calibri"), ("cambria.ttc", "Cambria"),
                                 ("segoeui.ttf", "Segoe UI"), ("candara.ttf", "Candara"),
                                 ("bahnschrift.ttf", "Bahnschrift")):
            path = Path("C:/Windows/Fonts") / filename
            if not path.exists():
                continue
            font = ImageFont.truetype(str(path), 73)
            box = font.getbbox(phrase)
            image = Image.new("RGB", (box[2] - box[0] + 60, box[3] - box[1] + 40), "white")
            ImageDraw.Draw(image).text((30 - box[0], 20 - box[1]), phrase, font=font, fill="black")
            queries.append((None, "Unknown " + family, image))
        for expected, style, image in queries:
            for condition in ("clean", "jpeg_dark", "rotated"):
                current = ImageOps.expand(image.convert("RGB"), (30, 20), "white")
                if condition == "jpeg_dark":
                    current.thumbnail((480, 100), Image.Resampling.LANCZOS)
                    stream = BytesIO()
                    ImageOps.invert(current).save(stream, format="JPEG", quality=65)
                    current = Image.open(BytesIO(stream.getvalue())).convert("RGB")
                elif condition == "rotated":
                    current = current.rotate(2, Image.Resampling.BICUBIC, expand=True, fillcolor="white")
                started = time.perf_counter()
                result = identify(current, phrase)
                top3 = []
                if result["identified_font"]:
                    top3.append(result["identified_font"]["family"])
                for match in result["matches"]:
                    if match["family"] not in top3:
                        top3.append(match["family"])
                row = {"expected": expected, "style": style, "phrase": phrase, "condition": condition,
                       "named": result["identified_font"]["family"] if result["identified_font"] else None,
                       "top3": top3[:3], "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}
                rows.append(row)
            if len(rows) % 30 == 0:
                print("Evaluated", len(rows), "images", flush=True)
    report = {"method": "Synthetic known-catalog retrieval with supplied visible text, 73px source against 48/96px references. Not end-to-end OCR accuracy, not universal-font accuracy, not the assessed category test.",
              "seed": args.seed, "fresh_families_exclude_development": args.fresh,
              "catalog_families": len(grouped), "sampled_families": len(selected), "phrases": phrases,
              "summary": summary(rows), "by_condition": {condition: summary([row for row in rows if row["condition"] == condition]) for condition in ("clean", "jpeg_dark", "rotated")},
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
