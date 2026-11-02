"""Verified ONNX font embeddings and a compact image-reference search index."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from api.retrieval_features import text_patches

ROOT = Path(__file__).resolve().parents[1] / "artifacts/web"


@lru_cache(maxsize=1)
def session():
    import onnxruntime as ort
    metadata = json.loads((ROOT / "retrieval_encoder.json").read_text(encoding="utf-8"))
    path = ROOT / "retrieval_encoder.onnx"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise RuntimeError("Retrieval encoder checksum mismatch")
    options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def encode(image):
    pixels = text_patches(image)
    tensor = (pixels[:, None].astype(np.float32) / 255 - .5) / .5
    values = session().run(["vectors"], {"images": tensor})[0].mean(0)
    return values / max(float(np.linalg.norm(values)), 1e-6)


@lru_cache(maxsize=1)
def index():
    metadata = json.loads((ROOT / "retrieval_index.json").read_text(encoding="utf-8"))
    path = ROOT / "retrieval_index.npz"
    encoder = json.loads((ROOT / "retrieval_encoder.json").read_text(encoding="utf-8"))
    if metadata["encoder_sha256"] != encoder["sha256"] or hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise RuntimeError("Retrieval index does not match the verified encoder")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return metadata, arrays


def candidates(image, text, limit=400):
    from api.font_identity import case_id
    metadata, arrays = index()
    scores = arrays["vectors"] @ encode(image)
    scores -= metadata["case_penalty"] * (arrays["cases"] != case_id(text))
    # Each real style is represented in three cases. Return each source once.
    rows, seen, family_ranks = [], set(), {}
    for position in np.argsort(-scores):
        source = int(arrays["sources"][position])
        if source in seen: continue
        seen.add(source)
        record = metadata["sources"][source]
        if record["family"] not in family_ranks: family_ranks[record["family"]] = len(family_ranks) + 1
        rows.append({**record, "embedding_similarity": float(scores[position]), "embedding_family_rank": family_ranks[record["family"]]})
        if len(rows) >= limit: break
    return rows


def native_candidates(rows, limit=5):
    result, seen = [], set()
    for row in rows:
        if not row.get("native") or row["family"] in seen: continue
        seen.add(row["family"])
        other = max((font["embedding_similarity"] for font in rows if font["family"] != row["family"]), default=row["embedding_similarity"])
        result.append({**row, "prototype_margin": row["embedding_similarity"] - other})
        if len(result) >= limit: break
    return result
