"""Train font-style embeddings with family-separated validation and word images.

No Windows font participates in gradient training or epoch selection. Their
whole-word references are added later to the retrieval gallery.
"""
from io import BytesIO
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from fontsense.cnn_model import FontSenseCNN
from api.name_encoder import patches
from scripts.name_data import DATA, WORDS, REFERENCE_WORDS, VALIDATION_WORDS, render

SEED = 20261103
EPOCHS = 12
SAMPLES_PER_FAMILY = 128


class StyleModel(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.features = FontSenseCNN(width=16).features
        self.projection = nn.Linear(64, 64, bias=False)
        nn.init.eye_(self.projection.weight)
        self.head = nn.Parameter(torch.randn(classes, 64) * .1)

    def embed(self, x):
        return F.normalize(self.projection(self.features(x).flatten(1)), dim=1)

    def forward(self, x, y):
        cosine = self.embed(x) @ F.normalize(self.head, dim=1).T
        margin = torch.zeros_like(cosine).scatter_(1, y[:, None], .1)
        return 20 * (cosine - margin)


def image_vectors(model, images):
    output = []
    with torch.inference_mode():
        for start in range(0, len(images), 32):
            batch = np.concatenate([patches(image) for image in images[start:start + 32]])
            tensor = torch.from_numpy((batch[:, None].astype(np.float32) / 255 - .5) / .5)
            values = model.embed(tensor).reshape(-1, 3, 64).mean(1)
            output.extend(F.normalize(values, dim=1).numpy())
    return np.asarray(output)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(SEED); random.seed(SEED)
    rng = np.random.default_rng(SEED)
    report_root = ROOT.parent / "name-training"
    report_root.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((DATA / "catalog.json").read_text())
    records = [f for f in catalog["fonts"] if not f["native"]]
    families = {}
    for record in records:
        families.setdefault(record["family"], []).append(record)
    validation = set()
    for category in sorted({r["category"] for r in records}):
        group = sorted({r["family"] for r in records if r["category"] == category})
        rng.shuffle(group); validation.update(group[:max(1, round(len(group) * .25))])
    training = sorted(set(families) - validation)
    splits = {family: "validation" if family in validation else "train" for family in families}
    pixels, labels = [], []
    for index, family in enumerate(training):
        for number in range(SAMPLES_PER_FAMILY):
            record = families[family][int(rng.integers(len(families[family])))]
            word = WORDS[number % len(WORDS)]
            if number % 3 == 0: word += " " + WORDS[(number * 7 + 19) % len(WORDS)]
            word = (str.lower, str.title, str.upper)[number % 3](word)
            image = render(record, word, size=int(rng.choice((48, 64, 80))))
            if number % 4 == 0:
                image = image.rotate(float(rng.uniform(-2, 2)), Image.Resampling.BICUBIC, expand=True, fillcolor="white")
            if number % 4 == 1:
                buffer = BytesIO(); image.save(buffer, format="JPEG", quality=int(rng.integers(45, 90)))
                image = Image.open(BytesIO(buffer.getvalue())).convert("RGB")
            values = patches(image, count=5)
            pixels.append(values[int(rng.integers(len(values)))])
            labels.append(index)
        print(f"Prepared {family}: {len(pixels)} whole-word patches", flush=True)
    x = np.asarray(pixels, dtype=np.uint8); y = np.asarray(labels, dtype=np.int64)
    np.savez_compressed(report_root / "training-patches.npz", pixels=x, labels=y)
    gallery_images, gallery_labels = [], []
    query_images, query_labels = [], []
    for family, faces in families.items():
        record = faces[0]
        gallery_images.extend(render(record, text) for text in REFERENCE_WORDS[:4])
        gallery_labels.extend([family] * 4)
        if family in validation:
            query_images.extend(render(record, text) for text in VALIDATION_WORDS)
            query_labels.extend([family] * len(VALIDATION_WORDS))
    model = StyleModel(len(training))
    original = torch.load(ROOT / "artifacts/cnn/cnn_model.pt", map_location="cpu", weights_only=True)
    model.features.load_state_dict({k[len("features."):]: v for k, v in original["state_dict"].items() if k.startswith("features.")})
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    best, best_epoch, best_state = -1, 0, None
    history = []
    import mlflow
    mlflow.set_tracking_uri((report_root / "mlruns").resolve().as_uri())
    mlflow.set_experiment("FontSense font-name encoder")
    with mlflow.start_run(run_name="whole-word images with unseen-family validation") as run:
        mlflow.log_params({"seed": SEED, "epochs": EPOCHS, "training_families": len(training), "validation_families": len(validation), "patches": len(x), "embedding_dimensions": 64, "margin": .1, "scale": 20})
        for epoch in range(1, EPOCHS + 1):
            model.train(); order = rng.permutation(len(x)); losses = []
            started = time.perf_counter()
            for offset in range(0, len(order), 96):
                indices = order[offset:offset + 96]
                images = torch.from_numpy((x[indices, None].astype(np.float32) / 255 - .5) / .5)
                targets = torch.from_numpy(y[indices])
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(images, targets), targets)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step(); losses.append(float(loss.detach()))
            model.eval()
            references = image_vectors(model, gallery_images).reshape(len(families), 4, 64).mean(1)
            references /= np.maximum(np.linalg.norm(references, axis=1, keepdims=True), 1e-6)
            queries = image_vectors(model, query_images)
            predicted = np.asarray(list(families))[np.argmax(queries @ references.T, axis=1)]
            accuracy = float(np.mean(predicted == np.asarray(query_labels)))
            row = {"epoch": epoch, "loss": float(np.mean(losses)), "validation_name_top1": accuracy, "seconds": round(time.perf_counter() - started, 1)}
            history.append(row); print(json.dumps(row), flush=True)
            mlflow.log_metrics({"loss": row["loss"], "validation_name_top1": accuracy}, step=epoch)
            if accuracy > best:
                best, best_epoch, best_state = accuracy, epoch, copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state); model.eval()
        arrays = {name: value.detach().numpy() for name, value in model.state_dict().items() if name != "head"}
        # Portable export checks use OFL word images, never Windows glyph data.
        check_pixels = patches(query_images[0])
        with torch.inference_mode():
            check_vectors = model.embed(torch.from_numpy((check_pixels[:, None].astype(np.float32) / 255 - .5) / .5)).numpy()
        arrays.update(parity_pixels=check_pixels, parity_vectors=check_vectors)
        destination = ROOT / "artifacts/web/font_name_encoder.npz"
        np.savez_compressed(destination, **arrays)
        torch.save(best_state, report_root / "selected-encoder.pt")
        report = {"version": "font-name-encoder-1", "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "seed": SEED,
                  "family_splits": splits, "training_families": len(training), "validation_families": len(validation),
                  "training_patches": len(x), "validation_queries": len(query_images), "selected_epoch": best_epoch,
                  "validation_name_top1": best, "history": history, "mlflow_run_id": run.info.run_id,
                  "input_size": [112, 48], "embedding_dimensions": 64,
                  "source": "OFL whole-word images only; source-family names are training targets, never image inputs. Windows families are gallery-only.",
                  "warm_start_checkpoint_sha256": hashlib.sha256((ROOT / "artifacts/cnn/cnn_model.pt").read_bytes()).hexdigest(),
                  "validation_scope": "Validation families are excluded from this fine-tuning run. The starting category CNN was pretrained separately, so this is not a claim that all validation families were unseen in every pretraining source.",
                  "evaluation": "Family-held-out embedding retrieval: reference words and validation query words are disjoint. This is font-name retrieval, separate from frozen capstone category evaluation."}
        path = ROOT / "artifacts/web/font_name_encoder.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(path)); print("Selected encoder:", json.dumps({k: report[k] for k in ("selected_epoch", "validation_name_top1", "sha256", "mlflow_run_id")}), flush=True)


if __name__ == "__main__": main()
