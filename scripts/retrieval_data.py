"""Reproducible product data, isolated from all assessed capstone artifacts."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.extended_match import catalog, render_reference
from api.retrieval_features import text_patches
from scripts.name_data import WORDS

DATA = ROOT.parent / "retrieval-training"
SEED = 31057
REFERENCE_TEXTS = ("Bright river", "Cloud design", "Golden hour", "Modern paper")
VALIDATION_TEXTS = ("Zebra Festival", "Quantum Archive")
TEST_TEXTS = ("Jupiter Gazette", "Saffron Museum")


def augment(image, rng):
    image = image.convert("RGB")
    # Affine shear, unequal scaling, blur, downsampling and compression model
    # screenshot/photographic damage. They are simulations, not camera captures.
    if rng.random() < .35:
        shear = rng.uniform(-.12, .12)
        image = image.transform(image.size, Image.Transform.AFFINE, (1, shear, -shear * image.height / 2, 0, 1, 0), Image.Resampling.BICUBIC, fillcolor="white")
    if rng.random() < .4:
        image = image.resize((max(1, round(image.width * rng.uniform(.85, 1.15))), image.height), Image.Resampling.BICUBIC)
    if rng.random() < .45:
        scale = rng.uniform(.45, .85)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    if rng.random() < .4:
        image = image.rotate(rng.uniform(-3, 3), Image.Resampling.BICUBIC, expand=True, fillcolor="white")
    if rng.random() < .3:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(.15, .8)))
    values = np.asarray(image.convert("L"), dtype=np.float32) / 255
    foreground, background = rng.uniform(0, 85), rng.uniform(180, 255)
    values = foreground + values * (background - foreground)
    if rng.random() < .3:
        values += np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(1, 5), values.shape)
    image = Image.fromarray(np.uint8(np.clip(values, 0, 255))).convert("RGB")
    if rng.random() < .25:
        image = ImageOps.invert(image)
    if rng.random() < .55:
        stream = BytesIO(); image.save(stream, format="JPEG", quality=rng.randrange(40, 94))
        image = Image.open(BytesIO(stream.getvalue())).convert("RGB")
    return image


def prepare(samples=48):
    DATA.mkdir(parents=True, exist_ok=True)
    families = defaultdict(list)
    for font in catalog():
        families[font["family"]].append(font)
    # Group exact shared Latin renderings before splitting, so script-specific
    # families with copied Latin glyphs do not cross the family split.
    parent = {name: name for name in families}
    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]; name = parent[name]
        return name
    signatures = {}
    for i, (family, faces) in enumerate(sorted(families.items())):
        for record in faces:
            try:
                image = render_reference(record["id"], "Hamburgefonts 012", 48)
                signature = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                if signature in signatures:
                    first, second = find(family), find(signatures[signature])
                    if first != second: parent[first] = second
                else: signatures[signature] = family
            except (OSError, ValueError):
                continue
        if i % 300 == 0: print("Audited Latin duplicate groups", i, flush=True)
    groups = defaultdict(list)
    for family in families: groups[find(family)].append(family)
    rng = random.Random(SEED)
    items = sorted(groups.values(), key=lambda group: sorted(group)[0]); rng.shuffle(items)
    splits = {}
    train_end, validation_end = round(len(items) * .7), round(len(items) * .85)
    for index, group in enumerate(items):
        for family in group:
            splits[family] = "train" if index < train_end else "validation" if index < validation_end else "test"
    manifest = {"seed": SEED, "families": splits, "duplicate_groups": [sorted(group) for group in items if len(group) > 1],
                "counts": {split: sum(value == split for value in splits.values()) for split in ("train", "validation", "test")},
                "samples_per_training_family": samples,
                "scope": "New random-initialized product encoder. Family and exact Latin-rendering groups are disjoint. Assessed capstone data and weights are not used."}
    (DATA / "splits.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    training = sorted(family for family in families if splits[family] == "train")
    def one(job):
        index, family = job
        rand = random.Random(SEED + index * 19)
        output = []
        for number in range(samples):
            face = families[family][number % len(families[family])]
            words = rand.sample(WORDS, 2 if number % 3 else 1)
            text = (str.lower, str.title, str.upper)[number % 3](" ".join(words))
            try:
                image = render_reference(face["id"], text, rand.choice((40, 56, 73, 96)))
                patches = text_patches(augment(image, rand), count=5)
                output.append(patches[rand.randrange(5)])
            except (ValueError, OSError):
                continue
        return index, output
    pixels, labels = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (label, images) in enumerate(pool.map(one, enumerate(training))):
            pixels.extend(images); labels.extend([label] * len(images))
            if i % 100 == 0: print("Prepared training families", i, "images", len(pixels), flush=True)
    np.savez_compressed(DATA / "training.npz", pixels=np.asarray(pixels, dtype=np.uint8), labels=np.asarray(labels, dtype=np.int64))
    manifest["training_images"] = len(pixels)
    manifest["training_families_in_label_order"] = training
    (DATA / "splits.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("PREPARED", manifest["counts"], len(pixels), "patches", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--samples", type=int, default=48)
    prepare(parser.parse_args().samples)
