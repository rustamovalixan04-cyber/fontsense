"""Auditable commercial-product font catalog and preserved family boundaries."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import string
import time

from fontTools.ttLib import TTFont
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import requests

from .google_fonts import CATEGORY_ORDER
from .utils import load_json, project_root, save_json
from .v2_data import discover_official_candidates, sha256_file, _safe_family


def family_key(value: str) -> str:
    return " ".join(value.casefold().split())


def _download(url: str, limit: int) -> bytes:
    if not url.startswith("https://raw.githubusercontent.com/google/fonts/"):
        raise ValueError("Font downloads must come from the pinned official Google Fonts repository")
    error = None
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(15, 50), headers={"User-Agent": "FontSense/0.3"}) as response:
                response.raise_for_status()
                payload = bytearray()
                for block in response.iter_content(65536):
                    payload.extend(block)
                    if len(payload) > limit:
                        raise ValueError("Download exceeds the configured size limit")
                return bytes(payload)
        except (requests.RequestException, OSError) as exc:
            error = exc
            time.sleep(1 + attempt)
    raise RuntimeError(f"Download failed: {error}")


def validate_product_font(path: Path) -> dict:
    """Require real Latin glyphs, not a visible missing-glyph box."""
    with TTFont(path, lazy=True) as font:
        cmap = font.getBestCmap() or {}
        required = set(string.ascii_letters + string.digits + " ")
        missing = sorted(required - {chr(codepoint) for codepoint in cmap})
        if missing:
            missing_codes = ', '.join(f'U+{ord(character):04X}' for character in missing)
            raise ValueError(f"Missing required Latin characters: {missing_codes}")
        axes = [dict(tag=axis.axisTag, minimum=float(axis.minValue), default=float(axis.defaultValue), maximum=float(axis.maxValue)) for axis in font['fvar'].axes] if 'fvar' in font else []
    face = ImageFont.truetype(str(path), 32)
    probe = Image.new("L", (1800, 120), 0)
    draw = ImageDraw.Draw(probe)
    draw.text((8, 0), string.ascii_uppercase, font=face, fill=255)
    draw.text((8, 56), string.ascii_lowercase + string.digits, font=face, fill=255)
    if probe.getbbox() is None:
        raise ValueError("Pillow rendered no visible Latin glyphs")
    return {"glyph_probe_sha256": hashlib.sha256(probe.tobytes()).hexdigest(), "variable_axes": json.dumps(axes, sort_keys=True), "latin_glyphs_verified": True}


def _audit_one(candidate: dict, config: dict, old: dict) -> dict:
    root = project_root()
    result = {**candidate, "path": "", "license_path": "", "font_sha256": "", "font_size_bytes": 0, "glyph_probe_sha256": "", "variable_axes": "[]", "latin_glyphs_verified": False, "usable": False, "validation_status": "failed", "failure_reason": ""}
    if candidate['license'] not in config['allowed_licenses']:
        result['failure_reason'] = "Licence outside this product track's configured licence set"
        return result
    cache = root / "data/v3/cache/audit" / f"{candidate['slug']}.json"
    if cache.is_file():
        previous = load_json(cache)
        if previous.get('source_commit') == candidate['source_commit'] and previous.get('usable') and (root / previous['path']).is_file() and sha256_file(root / previous['path']) == previous['font_sha256']:
            return previous
    try:
        prior = old.get(family_key(candidate['family']))
        if prior and prior['source_commit'] == candidate['source_commit'] and (root / prior['path']).is_file() and sha256_file(root / prior['path']) == prior['font_sha256']:
            path = root / prior['path']
            license_path = root / prior['license_path']
        else:
            directory = root / "data/v3/raw/fonts" / candidate['slug']
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / Path(candidate['font_filename']).name
            license_path = directory / ("OFL.txt" if candidate['license'] == 'OFL' else "LICENSE.txt")
            if not path.is_file():
                path.write_bytes(_download(candidate['font_url'], config['max_font_download_bytes']))
            if not license_path.is_file():
                license_path.write_bytes(_download(candidate['license_url'], 1024 * 1024))
        license_text = license_path.read_text(encoding='utf-8-sig')
        if candidate['license'] == 'OFL' and 'OPEN FONT LICENSE' not in license_text.upper():
            raise ValueError("OFL licence text could not be verified")
        if candidate['license'] == 'APACHE2' and ('Apache License' not in license_text or 'Version 2.0' not in license_text):
            raise ValueError("Apache 2 licence text could not be verified")
        verified = validate_product_font(path)
        result.update(**verified, path=path.relative_to(root).as_posix(), license_path=license_path.relative_to(root).as_posix(), font_sha256=sha256_file(path), font_size_bytes=path.stat().st_size, usable=True, validation_status='passed')
    except Exception as exc:
        result['failure_reason'] = f"{type(exc).__name__}: {exc}"
    save_json(result, cache)
    return result


def audit_catalog(config: dict, workers: int = 8) -> pd.DataFrame:
    root = project_root()
    source_root = root / config['metadata_root']
    catalog_path = root / "data/v3/font_candidates.csv"
    frame, discovery = discover_official_candidates(source_root, catalog_path, root / "reports/v3/font_discovery.json", limit_per_category=10000)
    if discovery['source_commit'] != config['source_commit']:
        raise ValueError("Official metadata checkout differs from the pinned product source")
    old_frame = pd.read_csv(root / "data/v2/google_fonts_audit.csv", keep_default_na=False)
    old = {family_key(row['family']): row for row in old_frame.to_dict('records') if str(row['usable']).casefold() == 'true'}
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = [executor.submit(_audit_one, candidate, config, old) for candidate in frame.to_dict('records')]
        for job in as_completed(jobs):
            rows.append(job.result())
            if len(rows) % 25 == 0:
                save_json({"stage": "font_audit", "status": "running", "processed": len(rows), "total": len(frame), "usable": sum(item['usable'] for item in rows)}, root / "reports/v3/progress.json")
                print(f"Font audit {len(rows)}/{len(frame)}; usable={sum(item['usable'] for item in rows)}", flush=True)
    audit = pd.DataFrame(rows).sort_values(['category', 'family']).reset_index(drop=True)
    audit.to_csv(root / "data/v3/font_audit.csv", index=False)
    save_json({"status": "completed", "candidates": len(audit), "usable_families_per_category": audit.loc[audit['usable']].groupby('category').size().to_dict(), "failed": int((~audit['usable']).sum()), "source_commit": config['source_commit'], "audit_sha256": sha256_file(root / "data/v3/font_audit.csv")}, root / "reports/v3/font_audit_summary.json")
    return audit


def build_product_split(audit: pd.DataFrame, v2: pd.DataFrame, v1: pd.DataFrame, seed: int = 42, excluded_family_prefixes: tuple[str, ...] = (), exclusion_reason: str = 'Outside readable Latin-text classification scope') -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = audit.loc[audit['usable'].astype(str).str.casefold().isin({'true','1'})].copy()
    policy_excluded = usable['family'].map(lambda family: any(family.casefold().startswith(prefix.casefold()) for prefix in excluded_family_prefixes))
    excluded = [{**row, 'exclusion_reason': exclusion_reason} for row in usable.loc[policy_excluded].to_dict('records')]
    usable = usable.loc[~policy_excluded]
    v2_map = {family_key(row.family): row.split for row in v2.itertuples()}
    v1_test = {family_key(row.family) for row in v1.itertuples() if row.split == 'test'}
    selected = []
    for _, group in usable.groupby('glyph_probe_sha256', sort=True):
        rows = group.to_dict('records')
        keys = {family_key(row['family']) for row in rows}
        folds = {v2_map[key] for key in keys if key in v2_map}
        reason = ''
        if keys & v1_test:
            reason = 'protected V1 test family or exact glyph alias'
        elif len(folds) > 1:
            reason = 'exact glyph alias spans historical V2 folds; quarantined'
        elif len({row['category'] for row in rows}) > 1:
            reason = 'exact glyph alias has conflicting category metadata'
        if reason:
            excluded.extend({**row, 'exclusion_reason': reason} for row in rows)
            continue
        rows.sort(key=lambda row: (family_key(row['family']) not in v2_map, family_key(row['family'])))
        chosen = rows[0]
        chosen['aliases'] = json.dumps([row['family'] for row in rows[1:]])
        chosen['split'] = next(iter(folds)) if folds else ''
        selected.append(chosen)
        excluded.extend({**row, 'exclusion_reason': 'exact rendered glyph alias; canonical family retained'} for row in rows[1:])
    result = pd.DataFrame(selected)
    if result.empty:
        raise ValueError('Zero usable independent product families')
    for index, category in enumerate(CATEGORY_ORDER):
        fresh_indices = result.index[(result['category'] == category) & (result['split'] == '')].tolist()
        random.Random(seed + index * 10007).shuffle(fresh_indices)
        n = len(fresh_indices)
        per_holdout = int(n * 0.10) if n >= 10 else 0
        for position, row_index in enumerate(fresh_indices):
            result.at[row_index, 'split'] = 'validation' if position < per_holdout else 'test' if position < per_holdout * 2 else 'train'
        for split in ('train','validation','test'):
            if not ((result['category'] == category) & (result['split'] == split)).any():
                raise ValueError(f'No usable {category} families in {split}')
    if result.groupby(result['family'].map(family_key))['split'].nunique().max() != 1:
        raise ValueError('Product family leakage detected')
    return result.sort_values(['category','split','family']).reset_index(drop=True), pd.DataFrame(excluded)


def prepare(config_path: Path, workers: int) -> dict:
    root = project_root()
    config = load_json(config_path)
    audit = audit_catalog(config, workers)
    v2_path = root / "data/v2/frozen_family_split.csv"
    v1_path = root / "data/interim/google_fonts_final_family_split.csv"
    before = sha256_file(v2_path)
    split, exclusions = build_product_split(audit, pd.read_csv(v2_path), pd.read_csv(v1_path), config['seed'], tuple(config.get('excluded_family_prefixes', [])), config.get('family_exclusion_reason', 'Outside readable Latin-text classification scope'))
    output = root / config.get('split_path', "data/v3/frozen_family_split.csv")
    content = split.to_csv(index=False, lineterminator='\n')
    if output.is_file() and output.read_text(encoding='utf-8') != content:
        raise ValueError('Existing V3 split would change; use a new dataset version instead')
    output.write_text(content, encoding='utf-8', newline='')
    exclusions.to_csv(root / "data/v3/font_exclusions.csv", index=False)
    if sha256_file(v2_path) != before:
        raise ValueError('Historical V2 split changed')
    summary = {'status':'passed','families_total':len(split),'families_per_category':split.groupby('category').size().to_dict(),'families_per_category_and_split':{category: group.groupby('split').size().to_dict() for category,group in split.groupby('category')},'family_overlap_count':0,'split_sha256':sha256_file(output),'historical_v2_split_sha256':before,'target_images':config['target_images'],'raw_font_redistribution':False,'source_commit':config['source_commit']}
    save_json(summary, root / "reports/v3/family_split_summary.json")
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=project_root()/'config/v3/dataset.json')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(prepare(args.config, args.workers), indent=2))
