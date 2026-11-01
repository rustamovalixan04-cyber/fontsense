"""Export the verified frozen checkpoint; PyTorch is needed only for this step."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "c98cf0d1a02503a02b8f8242fec462ea2a0c455380238ec54fc4f62fdb13bb2f"


def main():
    source = ROOT / "artifacts/cnn/cnn_model.pt"
    if hashlib.sha256(source.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError("Export requires the unchanged frozen V1 checkpoint.")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    destination = ROOT / "artifacts/web/cnn_v1.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **{
        name: value.detach().cpu().numpy() for name, value in checkpoint["state_dict"].items()
    })
    contract = {
        "source_checkpoint_sha256": EXPECTED,
        "weights_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "format": "float32 NumPy arrays, fused batch normalization at load",
        "class_order": checkpoint["classes"],
        "input_size": [112, 48],
    }
    destination.with_name("web_runtime.json").write_text(json.dumps(contract, indent=2) + "\n")
    print("Exported verified frozen CNN:", destination.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
