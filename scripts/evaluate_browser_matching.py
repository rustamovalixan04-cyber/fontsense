"""Evaluate captured browser crops; validation and final test stay separate."""
import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.font_match import identify
from scripts.retrieval_data import DATA


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--split", choices=("validation", "test", "test2", "native_test", "native_test2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=("legacy", "candidate"), default="legacy"); args = parser.parse_args()
    os.environ["FONTSENSE_RETRIEVAL_ENGINE"] = args.engine
    samples = [row for row in json.loads((DATA / "browser/samples.json").read_text()) if row["split"] == args.split]
    rows = []
    # Group by the visible phrase to avoid measuring pointless cache eviction.
    samples.sort(key=lambda row: (row["phrase"], row["case_id"]))
    for number, sample in enumerate(samples):
        image = Image.open(DATA / "browser" / sample["file"]).convert("RGB")
        for condition in ("browser", "compressed"):
            current = image
            if condition == "compressed":
                current = image.resize((round(image.width * .65), round(image.height * .65)), Image.Resampling.LANCZOS)
                stream = BytesIO(); current.save(stream, format="JPEG", quality=55)
                current = Image.open(BytesIO(stream.getvalue())).convert("RGB")
            started = time.perf_counter(); result = identify(current)
            names = ([result["identified_font"]["family"]] if result.get("identified_font") else [])
            for font in result["matches"]:
                if font["family"] not in names: names.append(font["family"])
            identity_candidates = ([result["identified_font"]["family"]] if result.get("identified_font") else [font["family"] for font in result.get("font_candidates", [])])
            rows.append({"family": sample["family"], "font_id": sample["font_id"], "condition": condition,
                         "known_reference": sample.get("known_reference", True),
                         "phrase": sample["phrase"], "ocr_text": result["text"], "needs_text": result["needs_text"],
                         "named": result["identified_font"]["family"] if result.get("identified_font") else None,
                         "top3": names[:3], "identity_candidates": identity_candidates,
                         "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)})
        if number % 10 == 0: print("Evaluated browser specimens", number, flush=True)
    accepted = [row for row in rows if row["named"]]
    known = [row for row in rows if row["known_reference"]]
    unknown = [row for row in rows if not row["known_reference"]]
    summary = {"images": len(rows), "names_returned": len(accepted),
               "correct_returned": sum(row["known_reference"] and row["named"] == row["family"] for row in accepted),
               "returned_precision": sum(row["known_reference"] and row["named"] == row["family"] for row in accepted) / max(len(accepted), 1),
               "known_images": len(known), "unknown_images": len(unknown),
               "coverage": sum(row["named"] is not None for row in known) / max(len(known), 1),
               "top1": sum(bool(row["top3"]) and row["top3"][0] == row["family"] for row in known) / max(len(known), 1),
               "top3": sum(row["family"] in row["top3"] for row in known) / max(len(known), 1),
               "identity_candidate_top3": sum(row["family"] in row["identity_candidates"][:3] for row in known) / max(len(known), 1),
               "unknown_names_withheld": sum(row["named"] is None for row in unknown),
               "ocr_needs_text": sum(row["needs_text"] for row in rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"split": args.split, "engine": args.engine, "method": "Actual browser screenshot crops plus separately reported JPEG/downsampling damage; complete image-to-OCR-to-font pipeline.", "summary": summary, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__": main()
