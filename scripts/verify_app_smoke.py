"""Check a running FontSense API using generated inputs, without dataset access."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import statistics
import time

import httpx
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def verify(base_url: str) -> dict:
    font = ImageFont.load_default(size=44)
    clear = Image.new("RGB", (600, 150), "white")
    ImageDraw.Draw(clear).text((24, 40), "FontSense upload check", font=font, fill="black")
    transparent = Image.new("RGBA", clear.size, (0, 0, 0, 0))
    ImageDraw.Draw(transparent).text((24, 40), "FontSense upload check", font=font, fill=(0, 0, 0, 255))
    dark = Image.new("RGB", clear.size, "#101010")
    ImageDraw.Draw(dark).text((24, 40), "FontSense upload check", font=font, fill="white")
    cases = [
        ("clear_text", clear, "PNG", "image/png", 200),
        ("transparent_text", transparent, "PNG", "image/png", 200),
        ("dark_background", dark, "PNG", "image/png", 200),
        ("mild_blur_jpeg", clear.filter(ImageFilter.GaussianBlur(0.65)), "JPEG", "image/jpeg", 200),
        ("blank", Image.new("RGB", clear.size, "white"), "PNG", "image/png", 400),
        ("disguised_gif", clear, "GIF", "image/png", 415),
    ]
    results = []
    with httpx.Client(base_url=base_url, timeout=90) as client:
        page = client.get("/")
        assert page.status_code == 200 and "FontSense" in page.text
        health = client.get("/healthz")
        health.raise_for_status()
        clear_payload = b""
        for name, image, format_name, content_type, expected_status in cases:
            buffer = BytesIO()
            image.save(buffer, format=format_name)
            payload = buffer.getvalue()
            started = time.perf_counter()
            response = client.post("/predict", content=payload, headers={"content-type": content_type})
            elapsed = (time.perf_counter() - started) * 1000
            assert response.status_code == expected_status, (name, response.status_code, response.text)
            result = response.json()
            if expected_status == 200:
                assert len(result["probabilities"]) == 5
                assert abs(sum(result["probabilities"].values()) - 1) < 1e-5
                assert result["accepted"] == (result["confidence"] >= 0.60)
            results.append({"case": name, "http_status": response.status_code, "request_ms": elapsed, "upload_bytes": len(payload), "response": result})
            if name == "clear_text":
                clear_payload = payload
        assert results[0]["response"]["probabilities"] == results[1]["response"]["probabilities"]
        for payload, content_type in [(b"broken image", "image/png"), (b"", "image/jpeg")]:
            assert client.post("/predict", content=payload, headers={"content-type": content_type}).status_code == 400
        durations = []
        for _ in range(10):
            started = time.perf_counter()
            client.post("/predict", content=clear_payload, headers={"content-type": "image/png"}).raise_for_status()
            durations.append((time.perf_counter() - started) * 1000)
        def concurrent_prediction(_: int) -> float:
            started = time.perf_counter()
            client.post("/predict", content=clear_payload, headers={"content-type": "image/png"}).raise_for_status()
            return (time.perf_counter() - started) * 1000
        with ThreadPoolExecutor(max_workers=8) as workers:
            concurrent_durations = list(workers.map(concurrent_prediction, range(8)))
    return {
        "status": "passed",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Technical HTTP smoke checks only. Generated inputs are not performance evidence.",
        "base_url": base_url,
        "checkpoint_sha256": health.json()["checkpoint_sha256"],
        "dataset_manifests_loaded": False,
        "v2_test_images_loaded": 0,
        "cases": results,
        "warm_request_repetitions": len(durations),
        "warm_request_median_ms": statistics.median(durations),
        "warm_request_max_ms": max(durations),
        "concurrent_requests": len(concurrent_durations),
        "concurrent_request_median_ms": statistics.median(concurrent_durations),
        "concurrent_request_max_ms": max(concurrent_durations),
        "timing_limitations": "Local warmed process; excludes public network latency and Vercel cold starts.",
        "visual_browser_check": "Blocked: Windows computer-use tool could not reliably determine the browser URL.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--output", type=Path, default=Path("reports/v2/product_verification.json"))
    args = parser.parse_args()
    result = verify(args.base_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
