"""Crop genuine UI captures using observed DOM coordinates and provenance."""
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.retrieval_data import DATA


def main():
    folder = DATA / "browser"
    samples = []
    # Trusted local UI captures can be larger than Pillow's generic warning
    # limit at browser zoom. Keep a finite bound and validate every crop.
    Image.MAX_IMAGE_PIXELS = 150_000_000
    for metadata_path in sorted((folder / "captures").glob("*.json")):
        metadata = json.loads(metadata_path.read_text())
        screenshot_path = metadata_path.with_suffix(".jpg")
        source_hash = hashlib.sha256(screenshot_path.read_bytes()).hexdigest()
        with Image.open(screenshot_path) as screenshot:
            # Detect pixel mapping from the page's known specimen/background
            # registration colors. Some browser backends pad full-page images
            # at zoom, so neither DPR nor canvas dimensions describe the ink.
            scales = (1., metadata["dpr"], screenshot.width / metadata["cssWidth"])
            def registration_error(scale):
                errors = []
                for row in metadata["cases"]:
                    box = row["box"]
                    color = (32, 32, 38) if row["dark"] else (250, 249, 245)
                    for dx, dy, expected in ((5, 5, color), (box["width"] - 5, 5, color),
                                             (5, box["height"] - 5, color), (box["width"] + 8, 5, (204, 204, 204))):
                        point = (round((box["x"] + dx) * scale), round((box["y"] + dy) * scale))
                        if point[0] >= screenshot.width or point[1] >= screenshot.height: return 255
                        actual = screenshot.getpixel(point)
                        errors.extend(abs(a - b) for a, b in zip(actual, expected))
                return sum(errors) / len(errors)
            scale = min(scales, key=registration_error)
            assert registration_error(scale) < 8, "Capture coordinate registration failed"
            for row in metadata["cases"]:
                assert row["loaded"]
                box = row["box"]
                bounds = [round(value * scale) for value in (box["x"] + 3, box["y"] + 3, box["x"] + box["width"] - 3, box["y"] + box["height"] - 3)]
                assert 0 <= bounds[0] < bounds[2] <= screenshot.width
                assert 0 <= bounds[1] < bounds[3] <= screenshot.height
                relative = f"{row['split']}/{row['case_id']:03d}.png"
                destination = folder / relative; destination.parent.mkdir(parents=True, exist_ok=True)
                crop = screenshot.crop(bounds).convert("RGB")
                crop.thumbnail((560, 130), Image.Resampling.LANCZOS)
                crop.save(destination)
                samples.append({**{key: value for key, value in row.items() if key not in ("box", "loaded")},
                                "file": relative, "source_capture": screenshot_path.name,
                                "source_sha256": source_hash, "crop_pixels": bounds,
                                "observed_pixel_scale": scale,
                                "kind": "actual-browser-screenshot", "declared_face_loaded": True})
    (folder / "samples.json").write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
    training = [sample for sample in samples if sample["split"] == "train"][:16]
    sheet = Image.new("RGB", (800, len(training) // 2 * 125), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(training):
        image = Image.open(folder / row["file"]); image.thumbnail((390, 95))
        x, y = index % 2 * 400, index // 2 * 125
        sheet.paste(image, (x, y)); draw.text((x + 4, y + 100), row["family"] + " / " + row["style"], fill="black")
    sheet.save(DATA / "browser-training-contact-sheet.jpg")
    print("Imported", len(samples), "genuine browser images; only train examples are in the contact sheet.")


if __name__ == "__main__": main()
