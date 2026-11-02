"""Measure the frozen older encoder on the new development retrieval queries."""
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.extended_match import primary_styles, render_reference
from api.name_encoder import patches
from scripts.retrieval_data import DATA, SEED, VALIDATION_TEXTS, augment
from scripts.train_name_encoder import StyleModel


def main():
    torch.set_num_threads(2)
    manifest = json.loads((DATA / "splits.json").read_text())
    model = StyleModel(57)
    with np.load(ROOT / "artifacts/web/font_name_encoder.npz") as source:
        state = {key: torch.from_numpy(source[key].copy()) for key in source.files if key in model.state_dict()}
    model.load_state_dict(state, strict=False); model.eval()
    names = [font["family"] for font in primary_styles()]
    candidates = [font for font in primary_styles() if manifest["families"][font["family"]] == "validation"]
    random.Random(SEED + 51).shuffle(candidates); candidates = candidates[:80]
    refs, queries, labels = [], [], []
    for record in primary_styles():
        for phrase in ("Cloud design", "Golden hour"):
            try: refs.extend(patches(render_reference(record["id"], phrase, 64)))
            except (ValueError, OSError): refs.extend(np.full((3, 48, 112), 255, dtype=np.uint8))
    for i, record in enumerate(candidates):
        for phrase in VALIDATION_TEXTS:
            try:
                image = render_reference(record["id"], phrase, 73); patches(image)
            except (ValueError, OSError): continue
            for number in range(2):
                current = image if number == 0 else augment(image, random.Random(SEED + i * 9 + number))
                queries.extend(patches(current)); labels.append(names.index(record["family"]))
    def vectors(pixels, average):
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(pixels), 192):
                x = np.asarray(pixels[start:start + 192], dtype=np.float32)
                outputs.extend(model.embed(torch.from_numpy((x[:, None] / 255 - .5) / .5)).numpy())
        result = np.asarray(outputs).reshape(-1, average, 64).mean(1)
        return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-6)
    references, questions = vectors(refs, 6), vectors(queries, 3)
    order = np.argsort(-(questions @ references.T), axis=1)
    labels = np.asarray(labels)
    metrics = {f"validation_top{k}": float((order[:, :k] == labels[:, None]).any(axis=1).mean()) for k in (1, 10, 100)}
    report = {"encoder": "frozen earlier 57-family fine-tuned encoder", "queries": len(labels), "metrics": metrics,
              "scope": "Same development images/reference words as the new encoder. Earlier encoder had separate historical training, so this is not an unseen-from-all-prior-training claim for the baseline."}
    (DATA / "frozen-encoder-baseline.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__": main()
