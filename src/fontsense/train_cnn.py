from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .cnn_model import FontSenseCNN
from .mlflow_utils import optional_mlflow_run
from .split import assert_no_family_leakage
from .utils import load_json, project_root, save_json, set_seed


class ManifestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, encoder: LabelEncoder, training: bool = False):
        self.frame = frame.reset_index(drop=True)
        self.encoder = encoder
        ops: list = [transforms.Grayscale(num_output_channels=1), transforms.Resize((96, 224))]
        if training:
            ops.extend([
                transforms.RandomAffine(degrees=3.0, translate=(0.03, 0.04), scale=(0.95, 1.05)),
                transforms.RandomAdjustSharpness(sharpness_factor=1.4, p=0.25),
            ])
        ops.extend([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        label = int(self.encoder.transform([row["category"]])[0])
        return tensor, label


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    losses: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.item()))
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
    }


def train_cnn(
    manifest_path: str | Path,
    experiment_config_path: str | Path,
    artifact_dir: str | Path,
    report_dir: str | Path,
    epochs: int = 15,
    batch_size: int = 64,
    seed: int = 42,
) -> dict:
    set_seed(seed)
    manifest = pd.read_csv(manifest_path)
    assert_no_family_leakage(manifest)
    encoder = LabelEncoder().fit(manifest["category"])
    train_frame = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_frame = manifest[manifest["split"] == "validation"].reset_index(drop=True)
    train_loader = DataLoader(ManifestDataset(train_frame, encoder, True), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ManifestDataset(val_frame, encoder, False), batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = load_json(experiment_config_path)["experiments"]
    artifact_dir = Path(artifact_dir)
    report_dir = Path(report_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    all_history: list[dict] = []
    best_global: dict | None = None
    best_state = None
    best_architecture = None

    for exp in configs:
        set_seed(seed)
        model = FontSenseCNN(len(encoder.classes_), width=int(exp["width"]), dropout=float(exp["dropout"])).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(exp["learning_rate"]), weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        patience = 4
        stale = 0
        best_run_f1 = -1.0
        started = time.perf_counter()

        with optional_mlflow_run("FontSense-CNN", exp["name"]) as mlflow:
            if mlflow is not None:
                mlflow.log_params(exp | {"epochs": epochs, "batch_size": batch_size, "seed": seed, "device": str(device)})
            for epoch in range(1, epochs + 1):
                train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, True)
                val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, False)
                row = {"run": exp["name"], "epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
                all_history.append(row)
                if mlflow is not None:
                    mlflow.log_metrics({k: float(v) for k, v in row.items() if isinstance(v, (float, int)) and k != "epoch"}, step=epoch)
                if val_metrics["macro_f1"] > best_run_f1 + 1e-4:
                    best_run_f1 = val_metrics["macro_f1"]
                    stale = 0
                    candidate_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    if best_global is None or best_run_f1 > best_global["macro_f1"]:
                        best_global = {"run": exp["name"], "macro_f1": best_run_f1, "epoch": epoch, **exp}
                        best_state = candidate_state
                        best_architecture = {"width": int(exp["width"]), "dropout": float(exp["dropout"])}
                else:
                    stale += 1
                if stale >= patience:
                    break
            if mlflow is not None:
                mlflow.log_metric("total_training_seconds", time.perf_counter() - started)

    if best_global is None or best_state is None or best_architecture is None:
        raise RuntimeError("No CNN model was trained")
    model_path = artifact_dir / "cnn_model.pt"
    torch.save({
        "state_dict": best_state,
        "classes": encoder.classes_.tolist(),
        "architecture": best_architecture,
        "selected_validation_run": best_global,
        "image_size": [224, 96],
        "seed": seed,
    }, model_path)
    pd.DataFrame(all_history).to_csv(report_dir / "cnn_training_history.csv", index=False)
    save_json({"artifact": str(model_path), "best_validation": best_global, "device": str(device)}, artifact_dir / "cnn_metadata.json")
    return {"artifact": str(model_path), "best_validation": best_global}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FontSense compact PyTorch CNN experiments.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default=str(project_root() / "config/cnn_experiments.json"))
    parser.add_argument("--artifact-dir", default=str(project_root() / "artifacts"))
    parser.add_argument("--report-dir", default=str(project_root() / "reports"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = train_cnn(args.manifest, args.config, args.artifact_dir, args.report_dir, args.epochs, args.batch_size, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
