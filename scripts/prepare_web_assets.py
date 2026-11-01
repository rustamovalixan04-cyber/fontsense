"""Download checksum-pinned font/OCR assets during deployment, never per request."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1] / "artifacts/web"


def download(job):
    relative, url, digest = job
    path = ROOT / relative
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
        return
    if urlparse(url).hostname not in {"raw.githubusercontent.com", "www.modelscope.cn"}:
        raise ValueError("Unexpected runtime asset source")
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "FontSense-build"}), timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError("Runtime asset checksum mismatch: " + relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1)


if __name__ == "__main__":
    catalog = json.loads((ROOT / "font_catalog.json").read_text())
    ocr = json.loads((ROOT / "ocr_source.json").read_text())
    jobs = [("text_recognition.onnx", ocr["url"], ocr["sha256"])]
    for font in catalog["fonts"]:
        jobs.extend([
            (font["file"], font["download_url"], font["sha256"]),
            ("fonts/" + font["id"] + "-OFL.txt", font["license_url"], font["license_sha256"]),
        ])
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download, jobs))
    print("Verified", len(catalog["fonts"]), "licensed reference fonts and the OCR model.")
