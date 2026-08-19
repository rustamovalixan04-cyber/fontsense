from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from .font_audit import parse_metadata_pb
from .google_fonts import (
    CATEGORY_FROM_METADATA,
    CATEGORY_ORDER,
    LICENSE_CODE_BY_FOLDER,
    LICENSE_FILE_BY_CODE,
    OFFICIAL_SOURCE,
    choose_font_file,
    request_with_retry,
    validate_font_file,
)
from .split import assert_no_family_leakage
from .utils import project_root, save_json

V2_SEED = 42
V2_TARGET_FAMILIES_PER_CATEGORY = 40
V2_MINIMUM_FALLBACK_FAMILIES = 30
V2_PRIMARY_SPLIT_COUNTS = {"train": 28, "validation": 6, "test": 6}
V2_MINIMUM_IMAGES = 20_000
V2_LICENSE_FILE_BY_CODE = {
    **LICENSE_FILE_BY_CODE,
    # The official google/fonts UFL directories use British spelling.
    "UFL": "LICENCE.txt",
}

DISCOVERY_COLUMNS = [
    "family",
    "category",
    "slug",
    "repository_folder",
    "license",
    "latin_support",
    "subsets",
    "font_filename",
    "source_commit",
    "metadata_sha256",
    "metadata_url",
    "font_url",
    "license_url",
    "source",
    "candidate_rank",
]

AUDIT_COLUMNS = [
    "family",
    "category",
    "source",
    "license",
    "path",
    "latin_support",
    "validation_status",
    "failure_reason",
    "usable",
    "slug",
    "repository_folder",
    "source_commit",
    "metadata_url",
    "font_url",
    "license_url",
    "license_path",
    "subsets",
    "font_sha256",
    "font_size_bytes",
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _repository_commit(metadata_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(metadata_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def discover_official_candidates(
    metadata_root: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    limit_per_category: int = 70,
    seed: int = V2_SEED,
) -> tuple[pd.DataFrame, dict]:
    """Create a deterministic catalog from a pinned google/fonts checkout."""
    metadata_root = Path(metadata_root)
    if not metadata_root.is_dir():
        raise FileNotFoundError(f"Official metadata checkout not found: {metadata_root}")
    if limit_per_category < V2_TARGET_FAMILIES_PER_CATEGORY:
        raise ValueError("Candidate discovery must retain at least 40 families per category")

    source_commit = _repository_commit(metadata_root)
    eligible: dict[str, list[dict]] = {category: [] for category in CATEGORY_ORDER}
    rejected_reasons: dict[str, int] = {}
    scanned = 0

    for metadata_path in sorted(metadata_root.rglob("METADATA.pb")):
        scanned += 1
        base = metadata_path.parent.parent.name
        slug = metadata_path.parent.name
        text = metadata_path.read_text(encoding="utf-8")
        metadata = parse_metadata_pb(text)
        family = str(metadata.get("family", "")).strip()
        metadata_category = str(metadata.get("category", "")).strip()
        category = CATEGORY_FROM_METADATA.get(metadata_category, "")
        subsets = sorted({str(value).strip() for value in metadata.get("subsets", [])})
        license_code = str(metadata.get("license", "")).strip()
        filename = choose_font_file(list(metadata.get("filenames", [])))

        reason = ""
        if not family:
            reason = "missing family name"
        elif not category:
            reason = "unsupported category"
        elif "latin" not in subsets:
            reason = "no Latin subset"
        elif LICENSE_CODE_BY_FOLDER.get(base) != license_code:
            reason = "licence metadata does not match repository folder"
        elif license_code not in V2_LICENSE_FILE_BY_CODE:
            reason = "unsupported licence"
        elif filename is None:
            reason = "no TTF or OTF font file"
        if reason:
            rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
            continue

        raw_root = f"https://raw.githubusercontent.com/google/fonts/{source_commit}"
        encoded_filename = quote(str(filename), safe="")
        license_filename = V2_LICENSE_FILE_BY_CODE[license_code]
        eligible[category].append(
            {
                "family": family,
                "category": category,
                "slug": slug,
                "repository_folder": base,
                "license": license_code,
                "latin_support": True,
                "subsets": ",".join(subsets),
                "font_filename": filename,
                "source_commit": source_commit,
                "metadata_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "metadata_url": f"{raw_root}/{base}/{slug}/METADATA.pb",
                "font_url": f"{raw_root}/{base}/{slug}/{encoded_filename}",
                "license_url": f"{raw_root}/{base}/{slug}/{license_filename}",
                "source": OFFICIAL_SOURCE,
            }
        )

    selected_rows: list[dict] = []
    eligible_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    for category_index, category in enumerate(CATEGORY_ORDER):
        unique: dict[str, dict] = {}
        for row in sorted(eligible[category], key=lambda item: item["family"].casefold()):
            unique.setdefault(row["family"].casefold(), row)
        rows = list(unique.values())
        eligible_counts[category] = len(rows)
        if len(rows) < V2_TARGET_FAMILIES_PER_CATEGORY:
            raise RuntimeError(
                f"Official metadata has only {len(rows)} eligible {category} families; 40 are required"
            )
        rng = random.Random(seed + category_index * 10_007)
        rng.shuffle(rows)
        chosen = rows[:limit_per_category]
        for rank, row in enumerate(chosen, start=1):
            selected_rows.append({**row, "candidate_rank": rank})
        selected_counts[category] = len(chosen)

    frame = pd.DataFrame(selected_rows, columns=DISCOVERY_COLUMNS)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    report = {
        "status": "passed",
        "purpose": "Reproducible V2 candidate discovery from official Google Fonts metadata.",
        "official_repository": "https://github.com/google/fonts.git",
        "source_commit": source_commit,
        "metadata_files_scanned": scanned,
        "eligible_latin_families": eligible_counts,
        "selected_candidates": selected_counts,
        "candidate_limit_per_category": limit_per_category,
        "seed": seed,
        "rejected_metadata_entries": rejected_reasons,
        "catalog_path": _portable_path(output_path, project_root()),
        "catalog_sha256": sha256_file(output_path),
    }
    save_json(report, report_path)
    return frame, report


def _safe_family(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def audit_official_candidates(
    catalog_path: str | Path,
    output_dir: str | Path,
    audit_path: str | Path,
    summary_path: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """Download, licence-check, open, and render every V2 candidate font."""
    root = project_root()
    catalog = pd.read_csv(catalog_path, keep_default_na=False)
    missing = set(DISCOVERY_COLUMNS) - set(catalog.columns)
    if missing:
        raise ValueError(f"V2 candidate catalog is missing columns: {sorted(missing)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for candidate in catalog.itertuples(index=False):
        row = {
            "family": candidate.family,
            "category": candidate.category,
            "source": candidate.source,
            "license": candidate.license,
            "path": "",
            "latin_support": bool(candidate.latin_support),
            "validation_status": "failed",
            "failure_reason": "",
            "usable": False,
            "slug": candidate.slug,
            "repository_folder": candidate.repository_folder,
            "source_commit": candidate.source_commit,
            "metadata_url": candidate.metadata_url,
            "font_url": candidate.font_url,
            "license_url": candidate.license_url,
            "license_path": "",
            "subsets": candidate.subsets,
            "font_sha256": "",
            "font_size_bytes": 0,
        }
        font_path: Path | None = None
        try:
            expected_license = LICENSE_CODE_BY_FOLDER.get(candidate.repository_folder)
            if candidate.license != expected_license:
                raise ValueError("licence metadata does not match official repository folder")
            if str(candidate.latin_support).casefold() not in {"true", "1"}:
                raise ValueError("official metadata does not declare Latin support")
            category_dir = output_dir / candidate.category
            category_dir.mkdir(parents=True, exist_ok=True)
            safe = _safe_family(candidate.family)
            filename = Path(candidate.font_filename).name
            font_path = category_dir / f"{safe}__{filename}"
            license_path = category_dir / f"{safe}__{V2_LICENSE_FILE_BY_CODE[candidate.license]}"

            font_bytes = (
                font_path.read_bytes()
                if font_path.is_file()
                else request_with_retry(candidate.font_url, binary=True)
            )
            license_text = (
                license_path.read_text(encoding="utf-8")
                if license_path.is_file()
                else str(request_with_retry(candidate.license_url))
            )
            if not license_text.strip():
                raise ValueError("official licence file is empty")
            font_path.write_bytes(font_bytes)
            license_path.write_text(license_text, encoding="utf-8")
            valid, reason = validate_font_file(font_path)
            if not valid:
                raise ValueError(f"Pillow render validation failed: {reason}")
            row.update(
                path=_portable_path(font_path, root),
                validation_status="passed",
                failure_reason="",
                usable=True,
                license_path=_portable_path(license_path, root),
                font_sha256=sha256_file(font_path),
                font_size_bytes=font_path.stat().st_size,
            )
        except Exception as exc:
            if font_path is not None:
                font_path.unlink(missing_ok=True)
            row["failure_reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    audit = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    audit_path = Path(audit_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)
    usable = audit.loc[audit["usable"].astype(bool)]
    counts = {
        category: int(usable.loc[usable["category"] == category, "family"].nunique())
        for category in CATEGORY_ORDER
    }
    if min(counts.values(), default=0) < V2_MINIMUM_FALLBACK_FAMILIES:
        raise RuntimeError(f"V2 audit did not find at least 30 usable families per category: {counts}")
    summary = {
        "status": "passed",
        "candidates_audited": len(audit),
        "usable_independent_families": counts,
        "failed_candidates": int((~audit["usable"].astype(bool)).sum()),
        "validation": "Every usable file opened and rendered visible Latin text with Pillow.",
        "audit_path": _portable_path(audit_path, root),
        "audit_sha256": sha256_file(audit_path),
    }
    save_json(summary, summary_path)
    return audit, summary


def _split_counts(families_per_category: int) -> dict[str, int]:
    if families_per_category == V2_TARGET_FAMILIES_PER_CATEGORY:
        return dict(V2_PRIMARY_SPLIT_COUNTS)
    validation = max(5, round(families_per_category * 0.15))
    test = max(5, round(families_per_category * 0.15))
    return {"train": families_per_category - validation - test, "validation": validation, "test": test}


def create_v2_family_split(
    audit_path: str | Path,
    v1_split_path: str | Path,
    output_path: str | Path,
    exclusions_path: str | Path,
    summary_path: str | Path,
    *,
    seed: int = V2_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Freeze V2 families while keeping every historical V1 test family out."""
    root = project_root()
    audit = pd.read_csv(audit_path, keep_default_na=False)
    usable = audit.loc[audit["usable"].astype(str).str.casefold().isin({"true", "1"})].copy()
    v1 = pd.read_csv(v1_split_path, keep_default_na=False)
    assert_no_family_leakage(v1)
    v1_selected = set(v1["family"].astype(str).str.casefold())
    v1_test = set(v1.loc[v1["split"] == "test", "family"].astype(str).str.casefold())

    available_counts = {
        category: int(
            usable.loc[
                (usable["category"] == category)
                & ~usable["family"].astype(str).str.casefold().isin(v1_test),
                "family",
            ].nunique()
        )
        for category in CATEGORY_ORDER
    }
    families_per_category = min(V2_TARGET_FAMILIES_PER_CATEGORY, min(available_counts.values()))
    if families_per_category < V2_MINIMUM_FALLBACK_FAMILIES:
        raise RuntimeError(
            "Fewer than 30 audited families remain after protecting V1 test families: "
            f"{available_counts}"
        )
    counts = _split_counts(families_per_category)
    selected_rows: list[pd.DataFrame] = []
    excluded_rows: list[dict] = []

    for category_index, category in enumerate(CATEGORY_ORDER):
        category_rows = (
            usable.loc[usable["category"] == category]
            .drop_duplicates("family")
            .sort_values("family", key=lambda values: values.str.casefold())
            .reset_index(drop=True)
        )
        category_rows["family_key"] = category_rows["family"].str.casefold()
        protected = category_rows.loc[category_rows["family_key"].isin(v1_test)]
        for row in protected.to_dict("records"):
            excluded_rows.append({**row, "exclusion_reason": "protected historical V1 test family"})
        pool = category_rows.loc[~category_rows["family_key"].isin(v1_test)].copy()

        fresh = pool.loc[~pool["family_key"].isin(v1_selected)].copy()
        fresh_records = fresh.to_dict("records")
        random.Random(seed + category_index * 20_011 + 1).shuffle(fresh_records)
        if len(fresh_records) < counts["test"]:
            raise RuntimeError(
                f"Category {category} has only {len(fresh_records)} fresh test candidates; "
                f"{counts['test']} are required"
            )
        test_keys = {row["family_key"] for row in fresh_records[: counts["test"]]}
        test = pool.loc[pool["family_key"].isin(test_keys)].copy()

        remaining_records = pool.loc[~pool["family_key"].isin(test_keys)].to_dict("records")
        random.Random(seed + category_index * 20_011 + 2).shuffle(remaining_records)
        needed = counts["train"] + counts["validation"]
        chosen_rest = pd.DataFrame(remaining_records[:needed])
        train = chosen_rest.iloc[: counts["train"]].copy()
        validation = chosen_rest.iloc[counts["train"] :].copy()
        train["split"] = "train"
        validation["split"] = "validation"
        test["split"] = "test"
        selected_rows.extend([train, validation, test])

        selected_keys = set(pd.concat([train, validation, test])["family_key"])
        for row in pool.loc[~pool["family_key"].isin(selected_keys)].to_dict("records"):
            excluded_rows.append({**row, "exclusion_reason": "not selected for balanced V2 family sample"})

    selected = pd.concat(selected_rows, ignore_index=True)
    selected["selection_seed"] = seed
    selected["was_in_v1_selected"] = selected["family_key"].isin(v1_selected)
    selected["was_in_v1_test"] = selected["family_key"].isin(v1_test)
    selected["fresh_v2_test_family"] = (selected["split"] == "test") & ~selected["family_key"].isin(v1_selected)
    selected = selected.drop(columns=["family_key"])
    excluded = pd.DataFrame(excluded_rows)
    if not excluded.empty:
        excluded = excluded.drop(columns=["family_key"], errors="ignore")

    assert_no_family_leakage(selected)
    if selected["family"].nunique() != families_per_category * len(CATEGORY_ORDER):
        raise AssertionError("V2 selected family total is incorrect")
    per_category = selected.groupby("category")["family"].nunique().to_dict()
    if per_category != {category: families_per_category for category in CATEGORY_ORDER}:
        raise AssertionError(f"V2 category family counts are not balanced: {per_category}")
    expected_splits = {name: count * len(CATEGORY_ORDER) for name, count in counts.items()}
    if selected.groupby("split")["family"].nunique().to_dict() != expected_splits:
        raise AssertionError("V2 split family counts are incorrect")
    if selected["was_in_v1_test"].any():
        raise AssertionError("A protected V1 test family entered the V2 split")
    if not selected.loc[selected["split"] == "test", "fresh_v2_test_family"].all():
        raise AssertionError("Every V2 test family must be fresh relative to all V1 selected families")

    output_path = Path(output_path)
    exclusions_path = Path(exclusions_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exclusions_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)
    excluded.to_csv(exclusions_path, index=False)
    images_per_family = max(100, math.ceil(V2_MINIMUM_IMAGES / len(selected)))
    summary = {
        "status": "passed",
        "configuration": "primary" if families_per_category == 40 else "fallback",
        "seed": seed,
        "families_total": int(selected["family"].nunique()),
        "families_per_category": families_per_category,
        "families_per_split": expected_splits,
        "families_per_category_and_split": counts,
        "images_per_family_required": images_per_family,
        "planned_images_total": images_per_family * len(selected),
        "family_overlap_count": 0,
        "protected_v1_test_families_selected": 0,
        "fresh_v2_test_families": int(selected.loc[selected["split"] == "test", "family"].nunique()),
        "v2_test_families_seen_in_v1_selected": 0,
        "split_path": _portable_path(output_path, root),
        "split_sha256": sha256_file(output_path),
        "v1_split_path": _portable_path(Path(v1_split_path), root),
        "v1_split_sha256": sha256_file(v1_split_path),
    }
    save_json(summary, summary_path)
    return selected, excluded, summary


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="FontSense V2 official font data preparation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover")
    discover.add_argument("--metadata-root", default=str(root / "data/v2/cache/google-fonts"))
    discover.add_argument("--output", default=str(root / "data/v2/google_fonts_candidates.csv"))
    discover.add_argument("--report", default=str(root / "reports/v2/data/candidate_discovery.json"))
    discover.add_argument("--limit-per-category", type=int, default=70)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--catalog", default=str(root / "data/v2/google_fonts_candidates.csv"))
    audit.add_argument("--output-dir", default=str(root / "data/v2/raw/fonts"))
    audit.add_argument("--output", default=str(root / "data/v2/google_fonts_audit.csv"))
    audit.add_argument("--summary", default=str(root / "reports/v2/data/font_audit_summary.json"))

    split = subparsers.add_parser("split")
    split.add_argument("--audit", default=str(root / "data/v2/google_fonts_audit.csv"))
    split.add_argument("--v1-split", default=str(root / "data/interim/google_fonts_final_family_split.csv"))
    split.add_argument("--output", default=str(root / "data/v2/frozen_family_split.csv"))
    split.add_argument("--exclusions", default=str(root / "data/v2/family_exclusions.csv"))
    split.add_argument("--summary", default=str(root / "reports/v2/data/family_split_summary.json"))

    args = parser.parse_args()
    if args.command == "discover":
        _, result = discover_official_candidates(
            args.metadata_root, args.output, args.report, limit_per_category=args.limit_per_category
        )
    elif args.command == "audit":
        _, result = audit_official_candidates(args.catalog, args.output_dir, args.output, args.summary)
    else:
        _, _, result = create_v2_family_split(
            args.audit, args.v1_split, args.output, args.exclusions, args.summary
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
