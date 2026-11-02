"""Build font-name reference statistics and select conservative confidence gates.

The gallery stores learned/statistical vectors, not font programs or per-letter
bitmap fonts. Native Windows source files never leave their original directory.
"""
from io import BytesIO
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from api.font_identity import case_id, geometry, reference_scores
from api.name_encoder import patches
from fontsense.cnn_model import FontSenseCNN
from scripts.name_data import DATA, REFERENCE_WORDS, VALIDATION_WORDS, render
from scripts.train_name_encoder import StyleModel


def extract(new_model, old_model, images, texts):
    new, old = [], []
    with torch.inference_mode():
        for start in range(0, len(images), 32):
            block = images[start:start + 32]
            pixels = np.concatenate([patches(image) for image in block])
            tensor = torch.from_numpy((pixels[:, None].astype(np.float32) / 255 - .5) / .5)
            new_values = F.normalize(new_model.embed(tensor).reshape(-1, 3, 64).mean(1), dim=1).numpy()
            old_values = torch.log1p(old_model.features(tensor).flatten(1).reshape(-1, 3, 64).mean(1)).numpy()
            new.extend(new_values)
            old.extend(np.concatenate((values, geometry(image, text))) for values, image, text in zip(old_values, block, texts[start:start + 32]))
    return np.asarray(new, dtype=np.float32), np.asarray(old, dtype=np.float32)


def distort(image, mode):
    if mode == "dark": return ImageOps.invert(image)
    if mode == "jpeg":
        stream = BytesIO(); image.save(stream, format="JPEG", quality=55)
        return Image.open(BytesIO(stream.getvalue())).convert("RGB")
    return image


def load_models():
    state = torch.load(ROOT.parent / "name-training/selected-encoder.pt", map_location="cpu", weights_only=True)
    model = StyleModel(state["head"].shape[0]); model.load_state_dict(state); model.eval()
    original = torch.load(ROOT / "artifacts/cnn/cnn_model.pt", map_location="cpu", weights_only=True)
    old = FontSenseCNN(width=16); old.load_state_dict(original["state_dict"]); old.eval()
    return model, old


def main():
    torch.set_num_threads(4)
    model, old = load_models()
    source = json.loads((DATA / "catalog.json").read_text())
    encoder_meta = json.loads((ROOT / "artifacts/web/font_name_encoder.json").read_text())
    records = source["fonts"]
    names = sorted({record["family"] for record in records})
    name_index = {name: index for index, name in enumerate(names)}
    new_vectors, old_vectors, family_indices, cases, source_ids = [], [], [], [], []
    for record in records:
        for case_index, transform in enumerate((str.lower, str.title, str.upper)):
            texts = [transform(text) for text in REFERENCE_WORDS]
            images = [render(record, text) for text in texts]
            new, old_values = extract(model, old, images, texts)
            vector = new.mean(0); vector /= max(float(np.linalg.norm(vector)), 1e-6)
            new_vectors.append(vector); old_vectors.append(old_values.mean(0))
            family_indices.append(name_index[record["family"]]); cases.append(case_index); source_ids.append(record["id"])
        print("Referenced", record["id"], flush=True)
    new_vectors = np.asarray(new_vectors, dtype=np.float32)
    raw = np.asarray(old_vectors, dtype=np.float32)
    fitting_rows = [encoder_meta["family_splits"].get(names[index]) == "train" for index in family_indices]
    mean = raw[fitting_rows].mean(0); scale = raw[fitting_rows].std(0) + .025
    normalized = (raw - mean) / scale
    normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-6)
    arrays = {"new_vectors": new_vectors, "old_vectors": normalized, "mean": mean, "scale": scale,
              "families": np.asarray(family_indices), "cases": np.asarray(cases)}
    # Validation combines family-held-out OFL queries and gallery-only Windows
    # queries. Complete phrases differ from gallery/training words. Nothing
    # here is reported as the original capstone final test set.
    queries, texts, labels = [], [], []
    for record in records:
        is_validation = encoder_meta["family_splits"].get(record["family"]) == "validation"
        if not (record["native"] or is_validation): continue
        for phrase in VALIDATION_WORDS[:2]:
            for transform in (str.lower, str.title, str.upper):
                text = transform(phrase)
                image = render(record, text)
                for mode in ("clean", "jpeg"):
                    queries.append(distort(image, mode)); texts.append(text); labels.append(name_index[record["family"]])
    external_root = ROOT.parent / "font-family-holdout"
    unknown = json.loads((external_root / "font_catalog.json").read_text())["fonts"]
    unknown = [font for font in unknown if font["style"] == "Regular"]
    assert not ({font["family"] for font in unknown} & set(names))
    for record in unknown:
        font = ImageFont.truetype(str(external_root / record["file"]), 64)
        if record["variable_weight"]:
            font.set_variation_by_axes([record["weight"] if a["name"] == b"Weight" else a["default"] for a in font.get_variation_axes()])
        for text in VALIDATION_WORDS[:2]:
            box = font.getbbox(text); image = Image.new("RGB", (box[2] - box[0] + 80, box[3] - box[1] + 60), "white")
            ImageDraw.Draw(image).text((40 - box[0], 30 - box[1]), text, font=font, fill="black")
            queries.append(image); texts.append(text); labels.append(-1)
    qnew, qraw = extract(model, old, queries, texts)
    qold = (qraw - mean) / scale; qold /= np.maximum(np.linalg.norm(qold, axis=1, keepdims=True), 1e-6)
    labels = np.asarray(labels); query_cases = [case_id(text) for text in texts]
    best = None
    for weight in (0., .25, .5, .75, 1.):
        for penalty in (0., .025):
            scores = reference_scores(qnew, qold, query_cases, arrays, weight, penalty)
            order = np.argsort(-scores, axis=1)
            first = scores[np.arange(len(scores)), order[:, 0]]
            margins = first - scores[np.arange(len(scores)), order[:, 1]]
            correct = order[:, 0] == labels
            for minimum in (.8, .85, .9, .93, .95, .97, .985):
                for margin in (.01, .02, .03, .05, .08, .12, .18):
                    accepted = (first >= minimum) & (margins >= margin)
                    count = int(accepted.sum())
                    if count < 20: continue
                    unknown_accepted = int(((labels == -1) & accepted).sum())
                    if unknown_accepted > int((labels == -1).sum()) * .05: continue
                    precision = float(correct[accepted].mean())
                    if precision < .97: continue
                    candidate = {"accepted": count, "correct": int(correct[accepted].sum()), "precision": precision,
                                 "queries": len(labels), "unknown_queries": int((labels == -1).sum()),
                                 "unknown_accepted": unknown_accepted,
                                 "neural_weight": weight, "case_penalty": penalty,
                                 "minimum_similarity": minimum, "minimum_margin": margin}
                    if best is None or (count, precision) > (best["accepted"], best["precision"]): best = candidate
    if best is None: raise RuntimeError("No confidence gate reached 97% validation precision")
    families = []
    for name in names:
        record = next(record for record in records if record["family"] == name)
        families.append({"family": name, "category": record["category"], "native": record["native"],
                         "url": record["source"] if record["native"] else "https://fonts.google.com/specimen/" + quote(name)})
    destination = ROOT / "artifacts/web/font_name_gallery.npz"
    np.savez_compressed(ROOT.parent / "name-training/gallery-fit-cache.npz", **arrays)
    deployed = {key: value for key, value in arrays.items() if best["neural_weight"] < 1 or key not in ("old_vectors", "mean", "scale")}
    np.savez_compressed(destination, **deployed)
    metadata = {"version": "font-name-gallery-1", "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "encoder_sha256": encoder_meta["sha256"], "families": families,
                "legacy_checkpoint_sha256": hashlib.sha256((ROOT / "artifacts/cnn/cnn_model.pt").read_bytes()).hexdigest(),
                "reference_styles": len(records), "reference_word_images": len(records) * 3 * len(REFERENCE_WORDS),
                "reference_words": REFERENCE_WORDS, "validation": best,
                **{key: best[key] for key in ("neural_weight", "case_penalty", "minimum_similarity", "minimum_margin")},
                "source_records": [{key: value for key, value in record.items() if key not in ("asset_root", "file")} for record in records],
                "excluded_sources": source["excluded_sources"],
                "limits": "Selective font-name retrieval, not proof of identity. Similar or metrically compatible fonts may be indistinguishable. Uncertain names are withheld. Font binaries and per-letter bitmaps are not included."}
    metadata["validation_policy"] = "At least 97% precision among accepted names and at least 95% rejection of the unknown-font calibration examples; maximize accepted names within these bounds. These are empirical development thresholds, not guarantees."
    path = ROOT / "artifacts/web/font_name_gallery.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("SELECTED", json.dumps(best), flush=True)


if __name__ == "__main__": main()
