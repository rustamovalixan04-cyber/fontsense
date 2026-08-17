from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "packaging" / "windows" / "equivalence_contract.json"
DEFAULT_EXE = ROOT / "dist" / "FontSense" / "FontSense.exe"
DEFAULT_REPORT = ROOT / "reports" / "windows_package_equivalence.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_relative(path: str | Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def run_json(command: list[str], timeout: int = 180) -> dict:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    for line in reversed(completed.stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"Command did not return JSON: {' '.join(command)}")


def audit_distribution(exe: Path, expected_hash: str) -> dict:
    distribution = exe.parent
    internal = distribution / "_internal"
    checkpoint = internal / "artifacts" / "cnn" / "cnn_model.pt"
    freeze = (
        internal
        / "reports"
        / "final_evaluation"
        / "pre_test_freeze.json"
    )
    required = [exe, checkpoint, freeze, distribution / "README_WINDOWS.txt"]
    missing = [str(path) for path in required if not path.is_file()]

    files = [path for path in distribution.rglob("*") if path.is_file()]
    relative = [path.relative_to(distribution).as_posix().lower() for path in files]
    forbidden_markers = (
        "data/train",
        "data/validation",
        "data/test",
        "full_manifest.csv",
        "google_fonts_final_family_split.csv",
        "hog_pipeline.joblib",
        "label_encoder.joblib",
        "mlruns/",
        "notebooks/",
        "feature_spec",
        ".env",
        "id_rsa",
        "id_ed25519",
    )
    forbidden = [
        name for name in relative if any(marker in name for marker in forbidden_markers)
    ]
    forbidden += [
        name
        for name in relative
        if Path(name).suffix == ".key"
        or (
            Path(name).suffix == ".pem"
            and "private" in Path(name).name
        )
    ]
    checkpoint_hash = sha256_file(checkpoint) if checkpoint.is_file() else None
    size_bytes = sum(path.stat().st_size for path in files)
    dll_names = {path.name.lower() for path in files if path.suffix.lower() == ".dll"}
    torch_runtime_present = "torch_cpu.dll" in dll_names and "c10.dll" in dll_names

    passed = (
        not missing
        and not forbidden
        and checkpoint_hash == expected_hash
        and torch_runtime_present
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "distribution": repository_relative(distribution),
        "size_bytes": size_bytes,
        "file_count": len(files),
        "missing_required_files": missing,
        "forbidden_files": forbidden,
        "bundled_checkpoint_path": repository_relative(checkpoint),
        "bundled_checkpoint_sha256": checkpoint_hash,
        "torch_cpu_runtime_present": torch_runtime_present,
    }


def compare_predictions(
    contract: dict,
    exe: Path,
    source_python: Path,
) -> tuple[list[dict], float]:
    tolerance = float(contract["maximum_absolute_probability_difference"])
    results = []
    global_maximum = 0.0

    for image_record in contract["comparison_images"]:
        image_path = (ROOT / image_record["path"]).resolve()
        actual_image_hash = sha256_file(image_path)
        if actual_image_hash != image_record["sha256"]:
            raise RuntimeError(f"Comparison image hash changed: {image_path}")

        source = run_json(
            [str(source_python), str(ROOT / "windows_launcher.py"), "--predict-json", str(image_path)]
        )
        packaged = run_json([str(exe), "--predict-json", str(image_path)])
        differences = {
            category: abs(
                float(source["probabilities"][category])
                - float(packaged["probabilities"][category])
            )
            for category in contract["class_order"]
        }
        maximum = max(differences.values())
        global_maximum = max(global_maximum, maximum)
        exact_fields = {
            "predicted_category": (
                source["predicted_category"] == packaged["predicted_category"]
            ),
            "class_order": source["class_order"] == packaged["class_order"] == contract["class_order"],
            "accepted": source["accepted"] == packaged["accepted"],
            "uncertain": source["uncertain"] == packaged["uncertain"],
            "threshold": source["threshold"] == packaged["threshold"] == contract["threshold"],
        }
        passed = all(exact_fields.values()) and maximum <= tolerance
        results.append(
            {
                "file": image_record["path"],
                "sha256": actual_image_hash,
                "source": source,
                "packaged": packaged,
                "absolute_probability_differences": differences,
                "maximum_absolute_probability_difference": maximum,
                "exact_match_checks": exact_fields,
                "status": "PASS" if passed else "FAIL",
            }
        )
    return results, global_maximum


def _probability_summary(result: list) -> dict:
    label_output = result[1] if len(result) > 1 else {}
    confidences = label_output.get("confidences", []) if label_output else []
    values = {
        str(item["label"]): float(item["confidence"])
        for item in confidences
    }
    return {
        "labels": sorted(values),
        "count": len(values),
        "sum": sum(values.values()),
    }


def verify_gradio_interface(url: str) -> dict:
    """Exercise the packaged Gradio endpoints like the visible controls do."""
    from gradio_client import Client, handle_file

    expected_labels = sorted(
        ["Display", "Handwriting", "Monospace", "Sans serif", "Serif"]
    )
    with urlopen(f"{url}config", timeout=10) as response:
        config = json.loads(response.read().decode("utf-8"))
    model_components = [
        component
        for component in config.get("components", [])
        if component.get("props", {}).get("label") == "Prediction model"
    ]
    model_props = model_components[0].get("props", {}) if model_components else {}
    raw_choices = model_props.get("choices", [])
    choices = [
        choice[1] if isinstance(choice, list) and len(choice) > 1 else choice
        for choice in raw_choices
    ]
    final_cnn_only = choices == ["Final CNN"] and model_props.get("visible") is False

    client = Client(url, verbose=False)
    source_png = ROOT / "data" / "sample" / "serif__DejaVu_Serif_0000.png"
    with tempfile.TemporaryDirectory(prefix="fontsense-ui-check-") as temporary:
        temporary_root = Path(temporary)
        jpeg = temporary_root / "sample.jpg"
        blank = temporary_root / "blank.png"
        too_small = temporary_root / "too-small.png"
        corrupted = temporary_root / "corrupted.png"
        with Image.open(source_png) as image:
            image.convert("RGB").save(jpeg, format="JPEG", quality=92)
        Image.new("RGB", (224, 96), "white").save(blank)
        Image.new("RGB", (10, 10), "black").save(too_small)
        corrupted.write_bytes(b"this is not a valid image")

        png_result = client.predict(
            handle_file(source_png), "Final CNN", api_name="/predict"
        )
        repeated_result = client.predict(
            handle_file(source_png), "Final CNN", api_name="/predict"
        )
        jpeg_result = client.predict(
            handle_file(jpeg), "Final CNN", api_name="/predict"
        )
        blank_result = client.predict(
            handle_file(blank), "Final CNN", api_name="/predict"
        )
        too_small_result = client.predict(
            handle_file(too_small), "Final CNN", api_name="/predict"
        )
        no_image_result = client.predict(None, "Final CNN", api_name="/predict")
        reset_result = client.predict(api_name="/reset")
        corrupted_rejected = False
        try:
            client.predict(
                handle_file(corrupted), "Final CNN", api_name="/predict"
            )
        except Exception:
            corrupted_rejected = True
        with urlopen(url, timeout=10) as response:
            server_available_after_invalid_input = response.status == 200

    png_summary = _probability_summary(png_result)
    jpeg_summary = _probability_summary(jpeg_result)
    checks = {
        "final_cnn_is_only_visible_model": final_cnn_only,
        "png_prediction_has_category_and_confidence": (
            "category:" in png_result[0]
            and "Confidence:" in png_result[0]
        ),
        "png_has_five_probabilities": (
            png_summary["labels"] == expected_labels
            and abs(png_summary["sum"] - 1.0) <= 1e-4
        ),
        "jpeg_prediction_has_category_and_confidence": (
            "category:" in jpeg_result[0]
            and "Confidence:" in jpeg_result[0]
        ),
        "jpeg_has_five_probabilities": (
            jpeg_summary["labels"] == expected_labels
            and abs(jpeg_summary["sum"] - 1.0) <= 1e-4
        ),
        "accepted_or_uncertain_status_present": (
            any(word in png_result[2] for word in ("Accepted", "Uncertain"))
            and any(word in jpeg_result[2] for word in ("Accepted", "Uncertain"))
        ),
        "repeated_prediction_matches": png_result == repeated_result,
        "blank_image_rejected": (
            "Could not classify image" in blank_result[0]
            and "blank or nearly blank" in blank_result[2]
        ),
        "too_small_image_rejected": (
            "Could not classify image" in too_small_result[0]
            and "too small" in too_small_result[2]
        ),
        "missing_image_rejected": (
            "No image uploaded" in no_image_result[0]
        ),
        "corrupted_image_rejected": corrupted_rejected,
        "server_available_after_invalid_input": (
            server_available_after_invalid_input
        ),
        "reset_returns_initial_state": (
            reset_result[0] is None
            and "No prediction yet" in reset_result[1]
            and "readable text crop" in reset_result[3]
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "png_probability_summary": png_summary,
        "jpeg_probability_summary": jpeg_summary,
    }


def smoke_server(exe: Path, port: int, timeout: int = 120) -> dict:
    url = f"http://127.0.0.1:{port}/"
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    process = subprocess.Popen(
        [str(exe), "--no-browser", "--port", str(port)],
        cwd=exe.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=creationflags,
    )
    started = time.perf_counter()
    status_code = None
    body_has_fontsense = False
    functional_checks = None
    error = None
    try:
        while time.perf_counter() - started < timeout:
            if process.poll() is not None:
                error = f"Server exited early with code {process.returncode}."
                break
            try:
                with urlopen(url, timeout=4) as response:
                    status_code = response.status
                    body = response.read().decode("utf-8", errors="replace")
                    body_has_fontsense = "FontSense" in body or "gradio" in body.lower()
                    if status_code == 200 and body_has_fontsense:
                        functional_checks = verify_gradio_interface(url)
                        break
            except (URLError, TimeoutError, ConnectionError):
                time.sleep(1)
        else:
            error = f"Server did not become reachable within {timeout} seconds."
    finally:
        if process.poll() is None:
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT)
                process.wait(timeout=15)
            except (AttributeError, OSError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        output = process.stdout.read() if process.stdout else ""

    startup_console_checks = {
        "friendly_start_message": "FontSense is starting" in output,
        "model_loading_message": "Loading the final CNN" in output,
        "browser_fallback_message": "Open this address in your browser" in output,
        "local_url_printed": url.rstrip("/") in output,
    }
    passed = (
        status_code == 200
        and body_has_fontsense
        and functional_checks is not None
        and functional_checks["status"] == "PASS"
        and all(startup_console_checks.values())
        and error is None
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "url": url,
        "http_status": status_code,
        "body_identified_as_fontsense": body_has_fontsense,
        "functional_checks": functional_checks,
        "startup_console_checks": startup_console_checks,
        "startup_seconds": time.perf_counter() - started,
        "process_exit_code": process.returncode,
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--port", type=int, default=7867)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    expected_hash = contract["checkpoint_sha256"]
    source_checkpoint = ROOT / "artifacts" / "cnn" / "cnn_model.pt"
    if sha256_file(source_checkpoint) != expected_hash:
        raise RuntimeError("The source checkpoint does not match the frozen hash.")

    source_python = ROOT / ".venv" / "Scripts" / "python.exe"
    audit = audit_distribution(args.exe, expected_hash)
    self_test = run_json([str(args.exe), "--self-test"])
    self_test["checkpoint_path"] = repository_relative(
        self_test["checkpoint_path"]
    )
    comparisons, global_maximum = compare_predictions(
        contract,
        args.exe,
        source_python,
    )
    server = smoke_server(args.exe, args.port)
    tolerance = float(contract["maximum_absolute_probability_difference"])
    overall_pass = (
        audit["status"] == "PASS"
        and self_test.get("status") == "PASS"
        and all(item["status"] == "PASS" for item in comparisons)
        and global_maximum <= tolerance
        and server["status"] == "PASS"
    )
    report = {
        "status": "PASS" if overall_pass else "FAIL",
        "purpose": "Windows packaging verification, not model evaluation.",
        "contract_path": str(args.contract.relative_to(ROOT)),
        "predeclared_tolerance": tolerance,
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "distribution_audit": audit,
        "self_test": self_test,
        "comparisons": comparisons,
        "global_maximum_absolute_probability_difference": global_maximum,
        "server_smoke_test": server,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
