"""Train a small product combiner without touching the assessed CNN or data.

Each source family belongs to one learner split. Its own references are removed
while creating features, so a family cannot simply match its own database entry.
No source family name/path/ground-truth category is passed to the learner.
"""
import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.compact import CLASSES, get_model, text_crop
from api.category_ensemble import feature_vector
from api.font_match import alignment_scores, catalog, straighten_line

SEED = 20261102
TEXTS = ("Vector type", "Forest road", "Craft 42", "Bright ideas", "Home studio", "Modern letters")


def specimen(record, text, rng):
    font = ImageFont.truetype(str(ROOT / "artifacts/web" / record["file"]), int(rng.integers(48, 81)))
    if record.get("variable_weight"):
        font.set_variation_by_axes([record["weight"] if a["name"] == b"Weight" else a["default"] for a in font.get_variation_axes()])
    box = font.getbbox(text)
    image = Image.new("RGB", (box[2] - box[0] + 160, box[3] - box[1] + 110), "white")
    ImageDraw.Draw(image).text((80 - box[0], 55 - box[1]), text, font=font, fill="black")
    return image


def variants(image, rng):
    yield image
    yield ImageOps.invert(image)
    factor = float(rng.uniform(.45, .75))
    small = image.resize((round(image.width * factor), round(image.height * factor)), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    small.save(buffer, format="JPEG", quality=int(rng.integers(45, 80)))
    yield Image.open(BytesIO(buffer.getvalue())).convert("RGB")
    yield image.rotate(float(rng.choice((-3, -2, 2, 3))), Image.Resampling.BICUBIC, expand=True, fillcolor="white")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=ROOT.parent / "category-training")
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    records = catalog()
    splits = {}
    for category in CLASSES:
        families = sorted({f["family"] for f in records if f["category"] == category})
        rng.shuffle(families)
        cut = max(1, round(len(families) * .25))
        splits.update({family: "validation" if index < cut else "train" for index, family in enumerate(families)})
    cache = args.workdir / "features.npz"
    provenance = {"seed": SEED, "texts": list(TEXTS), "family_splits": splits,
                  "catalog_sha256": hashlib.sha256((ROOT / "artifacts/web/font_catalog.json").read_bytes()).hexdigest(),
                  "feature_rule": "Own source family excluded; image/text/scores/classifier outputs only; no historical datasets.",
                  "variants": ["clean", "inverted", "small_JPEG", "rotation_2_to_3_degrees"]}
    provenance["feature_code_sha256"] = hashlib.sha256(b"".join((ROOT / path).read_bytes() for path in ("api/font_match.py", "api/category_ensemble.py", "api/compact.py"))).hexdigest()
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    signature_file = args.workdir / "signature.txt"
    if cache.exists() and signature_file.exists() and signature_file.read_text() == signature:
        with np.load(cache, allow_pickle=False) as data:
            x, y, is_train = data["x"], data["y"], data["is_train"]
    else:
        features, labels, masks = [], [], []
        for text in TEXTS:
            for record in records:
                for image in variants(specimen(record, text, rng), rng):
                    normalized = straighten_line(text_crop(image))
                    recognized = text if rng.random() > .2 else text.replace(" ", "")
                    classifier = get_model().predict(image)
                    scores = alignment_scores(normalized, recognized)
                    features.append(feature_vector(scores, records, classifier, normalized, recognized, excluded_family=record["family"]))
                    labels.append(CLASSES.index(record["category"]))
                    masks.append(splits[record["family"]] == "train")
            print(f"Generated features for {text}: {len(features)} images", flush=True)
        x, y, is_train = np.asarray(features), np.asarray(labels), np.asarray(masks)
        np.savez_compressed(cache, x=x, y=y, is_train=is_train)
        signature_file.write_text(signature)
    scaler = StandardScaler().fit(x[is_train])
    learner = LogisticRegression(C=1., max_iter=1500, class_weight="balanced", random_state=SEED)
    started = time.perf_counter()
    learner.fit(scaler.transform(x[is_train]), y[is_train])
    predicted = learner.predict(scaler.transform(x[~is_train]))
    destination = ROOT / "artifacts/web/category_ensemble.npz"
    np.savez_compressed(destination, coefs=learner.coef_.astype(np.float32), intercept=learner.intercept_.astype(np.float32),
                        mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32))
    # Check exact inference behavior of the portable export before enabling it.
    logits = ((x[~is_train] - scaler.mean_) / scaler.scale_) @ learner.coef_.T + learner.intercept_
    assert np.array_equal(logits.argmax(axis=1), predicted)
    report = {**provenance, "method": "StandardScaler plus balanced multinomial LogisticRegression, C=1.0, max_iter=1500; one real training run.",
              "classes": list(CLASSES), "features": x.shape[1], "training_images": int(is_train.sum()), "validation_images": int((~is_train).sum()),
              "validation_accuracy": float(accuracy_score(y[~is_train], predicted)),
              "validation_macro_f1": float(f1_score(y[~is_train], predicted, average="macro")),
              "classification_report": classification_report(y[~is_train], predicted, labels=list(range(5)), target_names=list(CLASSES), output_dict=True, zero_division=0),
              "portable_export_checks": [{"features": values.tolist(), "probabilities": probabilities.tolist()} for values, probabilities in zip(x[~is_train][:5], learner.predict_proba(scaler.transform(x[~is_train][:5])))],
              "fit_seconds": round(time.perf_counter() - started, 3),
              "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
              "limitations": "Synthetic development data with known image text (20% spaces removed). Original CNN remains pretrained; these are not capstone final test metrics. Probabilities are not calibrated."}
    (ROOT / "artifacts/web/category_ensemble.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.workdir.joinpath("training-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    import mlflow
    mlflow.set_tracking_uri((args.workdir / "mlruns").resolve().as_uri())
    mlflow.set_experiment("FontSense product category combiner")
    with mlflow.start_run(run_name="family-excluded similarity and frozen CNN") as run:
        mlflow.log_params({"seed": SEED, "C": 1., "learner": "balanced logistic regression", "features": x.shape[1], "catalog_sha256": provenance["catalog_sha256"]})
        mlflow.log_metrics({"validation_accuracy": report["validation_accuracy"], "validation_macro_f1": report["validation_macro_f1"]})
        report["mlflow_run_id"] = run.info.run_id
        report["catalog_semantic_sha256"] = hashlib.sha256(json.dumps(json.loads((ROOT / "artifacts/web/font_catalog.json").read_text()), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        (ROOT / "artifacts/web/category_ensemble.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        args.workdir.joinpath("training-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(args.workdir / "training-report.json"))
        print("MLflow run:", run.info.run_id, flush=True)
    print(json.dumps({key: report[key] for key in ("training_images", "validation_images", "validation_accuracy", "validation_macro_f1", "features", "sha256")}, indent=2))


if __name__ == "__main__":
    main()
