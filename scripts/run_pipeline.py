from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str]) -> None:
    print("\n$", " ".join(args))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(args, cwd=ROOT, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FontSense project pipeline.")
    parser.add_argument("--source", choices=["system", "google"], default="system")
    parser.add_argument("--images-per-family", type=int, default=20)
    parser.add_argument("--cnn-epochs", type=int, default=8)
    parser.add_argument("--max-fonts-per-category", type=int, default=None)
    parser.add_argument("--proof-work-dir", type=Path, default=None)
    parser.add_argument("--skip-cnn", action="store_true")
    args = parser.parse_args()
    if args.max_fonts_per_category is not None and args.max_fonts_per_category < 3:
        parser.error("--max-fonts-per-category must be at least 3 for train/validation/test family splits")
    if args.proof_work_dir is not None and args.source != "system":
        parser.error("--proof-work-dir is only available for the local system-font proof pipeline")

    python = sys.executable
    if args.source == "system":
        if args.proof_work_dir is not None:
            proof_root = args.proof_work_dir.resolve()
            font_manifest = str(proof_root / "system_fonts_manifest.csv")
            output_dir = str(proof_root / "dataset")
            artifact_dir = str(proof_root / "artifacts")
            report_dir = str(proof_root / "reports")
        else:
            font_manifest = "data/interim/system_fonts_manifest.csv"
            output_dir = "data/processed/fontsense_system"
            artifact_dir = "artifacts/proof"
            report_dir = "reports/proof"
        audit_command = [python, "-m", "fontsense.font_audit"]
        if args.max_fonts_per_category is not None:
            audit_command.extend(["--max-usable-per-category", str(args.max_fonts_per_category)])
        if args.proof_work_dir is not None:
            audit_command.extend(["--output", font_manifest])
        run(audit_command)
    else:
        run([python, "scripts/download_google_fonts.py", "--max-per-category", "35"])
        font_manifest = "data/interim/google_fonts_manifest.csv"
        output_dir = "data/processed/fontsense_google"
        artifact_dir = "artifacts"
        report_dir = "reports"

    run([python, "-m", "fontsense.generate_dataset", "--font-manifest", font_manifest, "--output-dir", output_dir, "--images-per-family", str(args.images_per_family)])
    manifest = f"{output_dir}/manifest.csv"
    run([python, "-m", "fontsense.train_hog", "--manifest", manifest, "--artifact-dir", artifact_dir, "--report-dir", report_dir])
    run([python, "-m", "fontsense.evaluate", "--manifest", manifest, "--model", "hog", "--artifact-dir", artifact_dir, "--report-dir", report_dir])
    if not args.skip_cnn:
        run([python, "-m", "fontsense.train_cnn", "--manifest", manifest, "--epochs", str(args.cnn_epochs), "--artifact-dir", artifact_dir, "--report-dir", report_dir])
        run([python, "-m", "fontsense.evaluate", "--manifest", manifest, "--model", "cnn", "--artifact-dir", artifact_dir, "--report-dir", report_dir])
    print("\nPipeline complete. Review reports/ and open app.py or demo.ipynb.")


if __name__ == "__main__":
    main()
