"""Train a new product retrieval CNN from scratch on the expanded OFL data."""
import argparse
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
from api.extended_match import catalog, primary_styles, render_reference
from api.retrieval_features import text_patches
from scripts.retrieval_data import DATA, SEED, VALIDATION_TEXTS, augment


class RetrievalModel(nn.Module):
    def __init__(self, classes, width=16, dimensions=96):
        super().__init__()
        self.features = FontSenseCNN(width=width).features
        self.projection = nn.Linear(width * 4, dimensions, bias=False)
        self.head = nn.Parameter(torch.randn(classes, dimensions) * .05)

    def embed(self, images):
        return F.normalize(self.projection(self.features(images).flatten(1)), dim=1)

    def forward(self, images):
        return self.embed(images)


def vectors(model, pixels, chunks=192):
    result = []
    with torch.inference_mode():
        for start in range(0, len(pixels), chunks):
            x = torch.from_numpy((pixels[start:start + chunks, None].astype(np.float32) / 255 - .5) / .5)
            result.append(model.embed(x).numpy())
    return np.concatenate(result)


def validation_data(manifest):
    cache = DATA / "epoch-validation.npz"
    if cache.exists():
        with np.load(cache) as archive: return {key: archive[key].copy() for key in archive.files}
    names = [font["family"] for font in primary_styles()]
    candidates = [font for font in primary_styles() if manifest["families"][font["family"]] == "validation"]
    rng = random.Random(SEED + 51); rng.shuffle(candidates); candidates = candidates[:80]
    reference, queries, labels = [], [], []
    for record in primary_styles():
        for phrase in ("Cloud design", "Golden hour"):
            try:
                reference.extend(text_patches(render_reference(record["id"], phrase, 64)))
            except (ValueError, OSError):
                reference.extend(np.full((3, 64, 160), 255, dtype=np.uint8))
    for i, record in enumerate(candidates):
        for phrase in VALIDATION_TEXTS:
            try:
                image = render_reference(record["id"], phrase, 73)
                text_patches(image)
            except (ValueError, OSError): continue
            for number in range(2):
                current = image if number == 0 else augment(image, random.Random(SEED + i * 9 + number))
                queries.extend(text_patches(current)); labels.append(names.index(record["family"]))
    result = {"reference": np.asarray(reference, dtype=np.uint8), "queries": np.asarray(queries, dtype=np.uint8),
              "labels": np.asarray(labels), "family_names": np.asarray(names)}
    np.savez_compressed(cache, **result)
    return result


def evaluate(model, data):
    refs = vectors(model, data["reference"]).reshape(-1, 6, 96).mean(1)
    refs /= np.maximum(np.linalg.norm(refs, axis=1, keepdims=True), 1e-6)
    queries = vectors(model, data["queries"]).reshape(-1, 3, 96).mean(1)
    queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-6)
    order = np.argsort(-(queries @ refs.T), axis=1)
    labels = data["labels"]
    return {f"validation_top{k}": float((order[:, :k] == labels[:, None]).any(axis=1).mean()) for k in (1, 10, 100)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--epochs", type=int, default=16); args = parser.parse_args()
    torch.set_num_threads(6); torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    manifest = json.loads((DATA / "splits.json").read_text())
    with np.load(DATA / "training.npz") as archive:
        pixels, labels = archive["pixels"], archive["labels"]
    # Real browser images are supplementary training data only for train families.
    browser_path = DATA / "browser" / "samples.json"
    if browser_path.exists():
        more, targets = [], []
        training_names = manifest["training_families_in_label_order"]
        for sample in json.loads(browser_path.read_text()):
            if sample["split"] != "train": continue
            image = Image.open(DATA / "browser" / sample["file"]).convert("RGB")
            rand = random.Random(SEED + len(more))
            for j in range(4):
                more.extend(text_patches(image if j == 0 else augment(image, rand)))
                targets.extend([training_names.index(sample["family"])] * 3)
        if more:
            pixels = np.concatenate((pixels, np.asarray(more, dtype=np.uint8)))
            labels = np.concatenate((labels, np.asarray(targets)))
        print("Browser training patches", len(more), flush=True)
    observed, counts = np.unique(labels, return_counts=True)
    active_labels = observed[counts >= 2]
    keep = np.isin(labels, active_labels)
    pixels, labels = pixels[keep], np.searchsorted(active_labels, labels[keep])
    active_names = [manifest["training_families_in_label_order"][int(label)] for label in active_labels]
    positions = [np.flatnonzero(labels == index) for index in range(len(active_labels))]
    model = RetrievalModel(len(positions))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.0002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=.0001)
    validation = validation_data(manifest)
    import mlflow
    mlflow.set_tracking_uri((DATA / "mlruns").resolve().as_uri()); mlflow.set_experiment("FontSense broad retrieval encoder")
    best, best_state, best_epoch, history = (-1, -1), None, 0, []
    with mlflow.start_run(run_name="Grouped families with screenshot damage and paired style learning") as run:
        mlflow.log_params({"seed": SEED, "epochs": args.epochs, "training_images": len(pixels),
                           "training_families": len(positions), "input": "160x64", "dimensions": 96,
                           "initialization": "random; no earlier category/name weights", "pair_loss_weight": .15,
                           "family_splits_sha256": hashlib.sha256((DATA / "splits.json").read_bytes()).hexdigest()})
        for epoch in range(1, args.epochs + 1):
            model.train(); losses = []; correct, count = 0, 0; started = time.perf_counter()
            # Pair different examples of a family, with different text/style,
            # while interleaving families to keep the image model discriminative.
            family_order = rng.permutation(np.tile(np.arange(len(positions)), max(1, len(pixels) // len(positions) // 2)))
            for offset in range(0, len(family_order), 48):
                selected = family_order[offset:offset + 48]
                indices = np.concatenate([rng.choice(positions[int(label)], 2, replace=False) for label in selected])
                x = torch.from_numpy((pixels[indices, None].astype(np.float32) / 255 - .5) / .5)
                y = torch.from_numpy(np.repeat(selected, 2))
                optimizer.zero_grad(set_to_none=True)
                z = model.embed(x)
                cosine = z @ F.normalize(model.head, dim=1).T
                margin = torch.zeros_like(cosine).scatter_(1, y[:, None], .1)
                loss = F.cross_entropy(24 * (cosine - margin), y) + .15 * (1 - (z[::2] * z[1::2]).sum(1)).mean()
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
                losses.append(float(loss.detach())); correct += int((cosine.argmax(1) == y).sum()); count += len(y)
            scheduler.step(); model.eval()
            metrics = evaluate(model, validation)
            row = {"epoch": epoch, "loss": float(np.mean(losses)), "training_accuracy": correct / count,
                   "seconds": round(time.perf_counter() - started, 1), **metrics}
            history.append(row); print(json.dumps(row), flush=True); mlflow.log_metrics(row, step=epoch)
            key = (metrics["validation_top100"], metrics["validation_top10"])
            if key > best:
                best, best_epoch, best_state = key, epoch, copy.deepcopy(model.state_dict())
                torch.save(best_state, DATA / "selected-model.pt")
            (DATA / "training-progress.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        model.load_state_dict(best_state); model.eval()
        torch.save(best_state, DATA / "selected-model.pt")
        destination = ROOT / "artifacts/web/retrieval_encoder.onnx"
        torch.onnx.export(model, torch.zeros(3, 1, 64, 160), destination, input_names=["images"], output_names=["vectors"],
                          dynamic_axes={"images": {0: "batch"}, "vectors": {0: "batch"}}, opset_version=17, dynamo=False)
        metadata = {"version": "broad-retrieval-1", "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    "seed": SEED, "epochs": args.epochs, "selected_epoch": best_epoch,
                    "training_images": len(pixels), "training_families": len(positions),
                    "active_training_families": active_names,
                    "validation": evaluate(model, validation), "dimensions": 96, "input": [160, 64], "history": history,
                    "initialization": "Random. No assessed or previous name-model weights are used.",
                    "family_splits": manifest["families"], "duplicate_groups": manifest["duplicate_groups"],
                    "mlflow_run_id": run.info.run_id,
                    "limits": "Family-held-out image retrieval, gallery references are necessary enrollment data. Not worldwide accuracy or a capstone category benchmark."}
        contract = ROOT / "artifacts/web/retrieval_encoder.json"
        contract.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        checks = validation["queries"][:12]
        np.savez_compressed(DATA / "onnx-parity.npz", pixels=checks, vectors=vectors(model, checks))
        mlflow.log_artifact(str(contract)); print("SELECTED", best_epoch, metadata["validation"], run.info.run_id, flush=True)


if __name__ == "__main__": main()
