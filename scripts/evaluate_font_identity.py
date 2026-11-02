"""Evaluate accepted names and abstention on disjoint complete query phrases.

Gallery identities are known by definition. This is selective font-name
retrieval, not the capstone's unseen-family broad-category benchmark.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.font_identity import case_id, gallery, reference_scores
from scripts.build_name_gallery import extract, load_models, distort
from scripts.name_data import DATA, REFERENCE_WORDS, VALIDATION_WORDS, render

TEST_WORDS = ("Typography", "Rocket Science", "Silver Letters")


def summary(rows):
    accepted = [row for row in rows if row["predicted"] is not None]
    known = [row for row in rows if row["expected"] is not None]
    unknown = [row for row in rows if row["expected"] is None]
    return {"images": len(rows), "known_images": len(known), "unknown_images": len(unknown),
            "accepted": len(accepted), "correct_accepted": sum(row["predicted"] == row["expected"] for row in accepted),
            "accepted_precision": sum(row["predicted"] == row["expected"] for row in accepted) / max(len(accepted), 1),
            "known_coverage": sum(row["predicted"] is not None for row in known) / max(len(known), 1),
            "unknown_rejected": sum(row["predicted"] is None for row in unknown),
            "unknown_rejection_rate": sum(row["predicted"] is None for row in unknown) / max(len(unknown), 1)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    assert not set(TEST_WORDS) & set(REFERENCE_WORDS + VALIDATION_WORDS)
    torch.set_num_threads(4)
    metadata, arrays = gallery(); model, old = load_models()
    records = json.loads((DATA / "catalog.json").read_text())["fonts"]
    names = [row["family"] for row in metadata["families"]]
    images, texts, rows = [], [], []

    def add(image, text, family, style, native):
        for condition in ("clean", "small_jpeg", "dark", "rotated"):
            if condition == "small_jpeg":
                current = distort(image.resize((round(image.width * .6), round(image.height * .6)), Image.Resampling.LANCZOS), "jpeg")
            elif condition == "dark": current = ImageOps.invert(image)
            elif condition == "rotated": current = image.rotate(2, Image.Resampling.BICUBIC, expand=True, fillcolor="white")
            else: current = image
            images.append(current); texts.append(text)
            rows.append({"expected": family, "style": style, "native": native, "text": text, "condition": condition})

    for record in records:
        for text in TEST_WORDS: add(render(record, text), text, record["family"], record["style"], record["native"])
        print("Prepared evaluation", record["id"], flush=True)
    unknown_root = ROOT.parent / "font-final-holdout"
    unknown_records = [row for row in json.loads((unknown_root / "font_catalog.json").read_text())["fonts"] if row["style"] == "Regular"]
    assert not set(names) & {row["family"] for row in unknown_records}
    for record in unknown_records:
        font = ImageFont.truetype(str(unknown_root / record["file"]), 64)
        if record.get("variable_weight"):
            font.set_variation_by_axes([record["weight"] if axis["name"] == b"Weight" else axis["default"] for axis in font.get_variation_axes()])
        for text in TEST_WORDS:
            box = font.getbbox(text); image = Image.new("RGB", (box[2] - box[0] + 80, box[3] - box[1] + 60), "white")
            ImageDraw.Draw(image).text((40 - box[0], 30 - box[1]), text, font=font, fill="black")
            add(image, text, None, "Unknown " + record["family"], False)
    new, raw = extract(model, old, images, texts)
    old_vectors = (raw - arrays["mean"]) / arrays["scale"]
    old_vectors /= np.maximum(np.linalg.norm(old_vectors, axis=1, keepdims=True), 1e-6)
    scores = reference_scores(new, old_vectors, [case_id(text) for text in texts], arrays, metadata["neural_weight"], metadata["case_penalty"])
    for row, values in zip(rows, scores):
        order = np.argsort(-values); best = float(values[order[0]]); margin = float(values[order[0]] - values[order[1]])
        row["predicted"] = names[int(order[0])] if best >= metadata["minimum_similarity"] and margin >= metadata["minimum_margin"] else None
        row["top_candidate"] = names[int(order[0])]; row["similarity"] = round(best, 4); row["margin"] = round(margin, 4)
    # Typography is the reported regression word. Report it separately from the
    # two fresh complete phrases, without presenting it as unseen evidence.
    fresh = [row for row in rows if row["text"] != "Typography"]
    report = {"method": "Selective known-gallery font-name retrieval, disjoint query phrases. Typography is a reported regression example and is separated from fresh-query metrics. Synthetic renderings are not a real-world accuracy guarantee.",
              "fresh_queries": summary(fresh), "regression_word": summary([row for row in rows if row["text"] == "Typography"]),
              "impact_fresh": summary([row for row in fresh if row["expected"] == "Impact"]),
              "per_family": {family: summary([row for row in fresh if row["expected"] == family]) for family in names},
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("fresh_queries", "regression_word", "impact_fresh")}, indent=2), flush=True)


if __name__ == "__main__": main()
