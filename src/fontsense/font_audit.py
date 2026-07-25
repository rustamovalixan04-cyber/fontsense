from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

import pandas as pd
from PIL import ImageFont

from .utils import project_root

VALID_EXTENSIONS = {".ttf", ".otf", ".ttc"}
AUDIT_COLUMNS = ["family", "category", "path", "usable", "reason"]
FontResolver = Callable[[str], str | None]


def _normalize_family_name(family: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", family.casefold())


def font_search_directories(system_name: str | None = None) -> tuple[Path, ...]:
    """Return common system and per-user font directories."""
    system_name = system_name or platform.system()
    if system_name == "Windows":
        windows_root = Path(os.environ.get("WINDIR", r"C:\Windows"))
        local_app_data = os.environ.get("LOCALAPPDATA")
        directories = [windows_root / "Fonts"]
        if local_app_data:
            directories.append(Path(local_app_data) / "Microsoft" / "Windows" / "Fonts")
        return tuple(directories)
    if system_name == "Darwin":
        return (
            Path("/System/Library/Fonts"),
            Path("/Library/Fonts"),
            Path.home() / "Library" / "Fonts",
        )
    return (
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
        Path.home() / ".local" / "share" / "fonts",
    )


def _validated_family_name(font_path: str | Path) -> str | None:
    """Open a font with Pillow and return its declared family name."""
    try:
        font = ImageFont.truetype(str(font_path), size=32)
        family_name = str(font.getname()[0]).strip()
        bbox = font.getbbox("Typography 123")
        if not family_name or bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None
        return family_name
    except Exception:
        return None


@lru_cache(maxsize=16)
def _directory_font_index(directory_names: tuple[str, ...]) -> dict[str, str]:
    """Build a validated family-to-file index for the selected directories."""
    index: dict[str, str] = {}
    for directory_name in directory_names:
        directory = Path(directory_name)
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*"), key=lambda item: str(item).casefold()):
            if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            family_name = _validated_family_name(path)
            if family_name:
                index.setdefault(_normalize_family_name(family_name), str(path.resolve()))
    return index


def _resolve_with_fontconfig(family: str) -> str | None:
    """Use fontconfig where available, while rejecting fallback families."""
    executable = shutil.which("fc-match")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "-f", "%{file}\n", family],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    path = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not path or not Path(path).is_file() or Path(path).suffix.lower() not in VALID_EXTENSIONS:
        return None
    matched_family = _validated_family_name(path)
    if matched_family is None or _normalize_family_name(matched_family) != _normalize_family_name(family):
        return None
    return path


def resolve_system_font(
    family: str,
    *,
    system_name: str | None = None,
    search_dirs: Sequence[str | Path] | None = None,
) -> str | None:
    """Resolve an installed family without silently substituting another family."""
    system_name = system_name or platform.system()
    if system_name != "Windows":
        fontconfig_path = _resolve_with_fontconfig(family)
        if fontconfig_path is not None:
            return fontconfig_path

    directories = tuple(Path(path) for path in search_dirs) if search_dirs is not None else font_search_directories(system_name)
    directory_names = tuple(str(path.resolve()) for path in directories)
    return _directory_font_index(directory_names).get(_normalize_family_name(family))


def can_render_latin(font_path: str, text: str = "Typography 123") -> tuple[bool, str]:
    try:
        font = ImageFont.truetype(font_path, size=48)
        bbox = font.getbbox(text)
        if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return False, "empty glyph bounding box"
        return True, "ok"
    except Exception as exc:  # font files can fail for many library-specific reasons
        return False, str(exc)


def audit_system_fonts(
    config_path: str | Path,
    output_path: str | Path,
    resolver: FontResolver | None = None,
    max_usable_per_category: int | None = None,
) -> pd.DataFrame:
    """Audit configured font families and save one consistent CSV schema."""
    if max_usable_per_category is not None and max_usable_per_category < 1:
        raise ValueError("max_usable_per_category must be positive")
    resolver = resolver or resolve_system_font
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("System font config must map categories to font family lists")

    rows: list[dict] = []
    for raw_category, families in config.items():
        category = str(raw_category).strip()
        if not category or not isinstance(families, list):
            raise ValueError("Every category must have a name and a list of font families")
        usable_count = 0
        for raw_family in families:
            if max_usable_per_category is not None and usable_count >= max_usable_per_category:
                break
            family = str(raw_family).strip()
            if not family:
                raise ValueError(f"Category '{category}' contains a blank font family")
            path = resolver(family)
            if path is None:
                rows.append({"family": family, "category": category, "path": "", "usable": False, "reason": "not found"})
                continue
            path = str(Path(path).resolve())
            usable, reason = can_render_latin(path)
            rows.append({"family": family, "category": category, "path": path, "usable": usable, "reason": reason})
            if usable:
                usable_count += 1

    frame = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    if frame.empty or not frame["usable"].astype(bool).any():
        raise RuntimeError(
            "No usable fonts were found. Install configured fonts or update config/system_fonts.json "
            "before generating a dataset."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, AUDIT_COLUMNS].to_csv(output_path, index=False)
    return frame


def parse_metadata_pb(text: str) -> dict:
    def first(pattern: str, default: str = "") -> str:
        match = re.search(pattern, text, flags=re.MULTILINE)
        return match.group(1) if match else default

    filenames = re.findall(r'filename:\s*"([^"]+)"', text)
    subsets = re.findall(r'subsets:\s*"([^"]+)"', text)
    return {
        "family": first(r'^name:\s*"([^"]+)"'),
        "category": first(r'^category:\s*"([^"]+)"'),
        "license": first(r'^license:\s*"([^"]+)"'),
        "filenames": filenames,
        "subsets": subsets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit locally installed fonts for the proof pipeline.")
    parser.add_argument("--config", default=str(project_root() / "config/system_fonts.json"))
    parser.add_argument("--output", default=str(project_root() / "data/interim/system_fonts_manifest.csv"))
    parser.add_argument("--max-usable-per-category", type=int, default=None)
    args = parser.parse_args()
    frame = audit_system_fonts(args.config, args.output, max_usable_per_category=args.max_usable_per_category)
    print(frame.groupby(["category", "usable"]).size().unstack(fill_value=0))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
