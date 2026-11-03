"""Download checksum-pinned font/OCR assets during deployment, never per request."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def extended_asset(source):
    from scripts.extended_font_assets import make_subset
    path = ROOT / source["file"]
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        url = source["download_url"]
        if not url.startswith("https://raw.githubusercontent.com/google/fonts/"):
            raise ValueError("Unexpected extended font source")
        for attempt in range(3):
            try:
                with urlopen(Request(url, headers={"User-Agent": "FontSense-build"}), timeout=60) as response:
                    data = response.read()
                if hashlib.sha256(data).hexdigest() != source["source_sha256"]:
                    raise ValueError("Extended source checksum mismatch")
                converted = make_subset(data, source["id"])
                if hashlib.sha256(converted).hexdigest() != source["sha256"]:
                    raise ValueError("Extended subset checksum mismatch")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(converted)
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(1)
    download((source["license_file"], source["license_url"], source["license_sha256"]))


if __name__ == "__main__":
    # Verify the independent download service before the much larger Google bank.
    independent_path = ROOT / 'independent_catalog.json'
    if independent_path.exists():
        from scripts.independent_font_assets import prepare
        independent = json.loads(independent_path.read_text(encoding='utf-8'))
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda source: prepare(source, ROOT), independent['sources']))
        print('Verified independent catalog:', independent['families'], 'families /', independent['styles'], 'styles.', flush=True)
    catalog = json.loads((ROOT / "font_catalog.json").read_text())
    ocr = json.loads((ROOT / "ocr_source.json").read_text())
    jobs = [("text_recognition.onnx", ocr["url"], ocr["sha256"])]
    layout_path = ROOT / 'ocr_layout_source.json'
    if layout_path.exists():
        layout = json.loads(layout_path.read_text())
        jobs.extend((source['file'], source['url'], source['sha256']) for source in layout.values() if isinstance(source, dict) and 'file' in source)
    for font in catalog["fonts"]:
        jobs.extend([
            (font["file"], font["download_url"], font["sha256"]),
            ("fonts/" + font["id"] + "-OFL.txt", font["license_url"], font["license_sha256"]),
        ])
    jobs = list(dict.fromkeys(jobs))  # Variable styles can share the same file.
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download, jobs))
    print("Verified", len({font["family"] for font in catalog["fonts"]}), "licensed families /", len(catalog["fonts"]), "styles and the OCR model.")
    extended_path = ROOT / "extended_catalog.json"
    if extended_path.exists():
        extended = json.loads(extended_path.read_text(encoding="utf-8"))
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(extended_asset, extended["sources"]))
        print("Verified extended catalog:", extended["families"], "families /", extended["styles"], "styles.")
