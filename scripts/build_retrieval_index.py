"""Build whole-word reference vectors without deploying native font programs."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.extended_match import catalog, render_reference
from api.retrieval_features import text_patches
from scripts.retrieval_data import DATA, REFERENCE_TEXTS
from scripts.train_retrieval_encoder import RetrievalModel, vectors
from scripts.prepare_name_data import NATIVE

EXTRA_NATIVE = [
    ("Calibri", "sans_serif", ["calibri.ttf", "calibrib.ttf", "calibrii.ttf", "calibriz.ttf"]),
    ("Cambria", "serif", ["cambria.ttc", "cambriab.ttf", "cambriai.ttf", "cambriaz.ttf"]),
    ("Segoe UI", "sans_serif", ["segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf", "segoeuiz.ttf"]),
    ("Arial Black", "sans_serif", ["ariblk.ttf"]),
    ("Palatino Linotype", "serif", ["pala.ttf", "palab.ttf", "palai.ttf", "palabi.ttf"]),
    ("Lucida Console", "monospace", ["lucon.ttf"]),
    ("Lucida Sans Unicode", "sans_serif", ["l_10646.ttf"]),
    ("Franklin Gothic Medium", "sans_serif", ["framd.ttf", "framdit.ttf"]),
    ("Gabriola", "handwriting", ["gabriola.ttf"]),
    ("Segoe Print", "handwriting", ["segoepr.ttf", "segoeprb.ttf"]),
    ("Segoe Script", "handwriting", ["segoesc.ttf", "segoescb.ttf"]),
    ("Sylfaen", "serif", ["sylfaen.ttf"]),
]


def native_sources():
    result = []
    for family, category, files in NATIVE + EXTRA_NATIVE:
        for number, filename in enumerate(files):
            path = Path("C:/Windows/Fonts") / filename
            if not path.exists(): continue
            reported_family, reported_style = ImageFont.truetype(str(path), 64).getname()
            result.append({"id": "native-" + family.lower().replace(" ", "-") + "-" + str(number),
                           "family": family, "category": category, "native": True,
                           "style": reported_style, "reported_family": reported_family,
                           "local_file": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "url": "https://learn.microsoft.com/en-us/typography/fonts/font-faq"})
    return result


def native_render(record, text, size=64):
    font = ImageFont.truetype(str(Path("C:/Windows/Fonts") / record["local_file"]), size)
    box = font.getbbox(text)
    image = Image.new("L", (box[2] - box[0] + 40, box[3] - box[1] + 40), 255)
    ImageDraw.Draw(image).text((20 - box[0], 20 - box[1]), text, font=font, fill=0)
    return image


def main():
    torch.set_num_threads(6)
    state = torch.load(DATA / "selected-model.pt", map_location="cpu", weights_only=True)
    model = RetrievalModel(state["head"].shape[0]); model.load_state_dict(state); model.eval()
    sources = [{**record, "native": False} for record in catalog()] + native_sources()
    values, cases, source_ids = [], [], []
    for i, record in enumerate(sources):
        for case, transform in enumerate((str.lower, str.title, str.upper)):
            pixels = []
            for text in REFERENCE_TEXTS:
                try:
                    image = native_render(record, transform(text)) if record["native"] else render_reference(record["id"], transform(text), 64)
                    pixels.extend(text_patches(image))
                except (OSError, ValueError): continue
            if not pixels: continue
            vector = vectors(model, np.asarray(pixels, dtype=np.uint8)).mean(0)
            vector /= max(float(np.linalg.norm(vector)), 1e-6)
            values.append(vector); cases.append(case); source_ids.append(i)
        if i % 100 == 0: print("Indexed styles", i, "/", len(sources), flush=True)
    destination = ROOT / "artifacts/web/retrieval_index.npz"
    np.savez_compressed(destination, vectors=np.asarray(values, dtype=np.float32), cases=np.asarray(cases, dtype=np.int8), sources=np.asarray(source_ids, dtype=np.int32))
    model_metadata = json.loads((ROOT / "artifacts/web/retrieval_encoder.json").read_text())
    metadata = {"version": "broad-reference-index-1", "encoder_sha256": model_metadata["sha256"],
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "sources": [{key: value for key, value in source.items() if key not in ("local_file", "file")} for source in sources],
                "reference_texts": REFERENCE_TEXTS, "reference_vectors": len(values),
                "reference_families": len({source["family"] for source in sources}),
                "case_penalty": .01,
                "native_usage": "Only non-rendering statistics of complete locally rendered phrases. No native font program, outline, or per-letter bitmap atlas is deployed."}
    (ROOT / "artifacts/web/retrieval_index.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("INDEXED", metadata["reference_families"], "families", len(values), "vectors", flush=True)


if __name__ == "__main__": main()
