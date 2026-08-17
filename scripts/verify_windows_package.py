from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


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

    passed = status_code == 200 and body_has_fontsense and error is None
    return {
        "status": "PASS" if passed else "FAIL",
        "url": url,
        "http_status": status_code,
        "body_identified_as_fontsense": body_has_fontsense,
        "startup_seconds": time.perf_counter() - started,
        "process_exit_code": process.returncode,
        "error": error,
        "output_tail": output[-4000:],
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
