"""Audit pinned Google Fonts metadata and build licensed style references.

The checked-in manifest allows ordinary builds to verify/download assets without
rerunning discovery. This script deliberately pins the repository revision.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
REVISION = "6c1dbe27dcf7d817d4b62cec033ba77fa8e2a5ec"
FAMILIES = {
    "serif": "tinos gelasio librebaskerville librecaslontext lora ebgaramond merriweather crimsonpro spectral notoserif ptserif cormorantgaramond",
    "sans_serif": "arimo roboto inter opensans lato montserrat nunito poppins worksans carlito sourcesans3 notosans ptsans",
    "monospace": "cousine robotomono ibmplexmono inconsolata spacemono jetbrainsmono anonymouspro dmmono",
    "display": "anton bebasneue bungee abrilfatface alfaslabone blackopsone monoton righteous",
    "handwriting": "caveat patrickhand dancingscript pacifico shadowsintolight kalam sacramento comicneue handlee allura architectsdaughter comicrelief charm italianno niconne",
}
# Independent family holdout. Never added to the retrieval library or used to
# choose thresholds. It is rendered only after the implementation is selected.
VALIDATION_FAMILIES = {
    "serif": "bitter domine alegreya cardo",
    "sans_serif": "barlow karla quicksand figtree",
    "monospace": "chivomono oxygenmono ptmono cutivemono",
    "display": "bangers creepster fasterone stardosstencil",
    "handwriting": "courgette marckscript mali indieflower",
}
FINAL_HOLDOUT = {
    "serif": "alice arapey breeserif castoro",
    "sans_serif": "alef andika assistant manrope",
    "monospace": "azeretmono overpassmono redditmono fragmentmono",
    "display": "aladin audiowide frederickathegreat gruppo",
    "handwriting": "charmonman delius giveyouglory delicioushandrawn",
}
# Google labels intended display use; FontSense's exclusive five-class scheme
# groups informal comic-style hand lettering with handwriting. Record this
# explicit visual taxonomy audit rather than silently editing upstream labels.
CATEGORY_AUDITS = {
    "comicrelief": "Casual comic-style hand lettering maps to FontSense handwriting; Google Fonts lists intended use as DISPLAY.",
}


def fetch(url):
    cache = ROOT.parent / "font-download-cache" / hashlib.sha256(url.encode()).hexdigest()
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        return cache.read_bytes()
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "FontSense-catalog-audit"}), timeout=45) as response:
                data = response.read()
            cache.write_bytes(data)
            return data
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1)


def build_family(item, output):
    category, slug = item
    folder = "ofl/" + slug
    base = f"https://raw.githubusercontent.com/google/fonts/{REVISION}/{folder}"
    metadata = fetch(base + "/METADATA.pb").decode()
    family = re.search(r'^name: "([^"]+)"', metadata).group(1)
    if not re.search(r'^license: "OFL"', metadata, re.M):
        raise ValueError("Only verified OFL families may be bundled: " + slug)
    if 'subsets: "latin"' not in metadata:
        raise ValueError("No recorded Latin subset: " + slug)
    declared = [value.lower() for value in re.findall(r'^category: "([^"]+)"', metadata, re.M)]
    if category not in declared and category != "monospace" and slug not in CATEGORY_AUDITS:
        raise ValueError(f"Category audit mismatch: {slug}: {declared} vs {category}")
    blocks = re.findall(r'fonts \{(.*?)\n\}', metadata, re.S)
    license_url = base + "/OFL.txt"
    if slug == "tinos":
        license_url = "https://raw.githubusercontent.com/googlefonts/tinos/3b4482a99b80ea5fc75f187b1be3120a3f5905b3/OFL.txt"
    license_text = fetch(license_url)
    folder_out = output / "fonts"
    folder_out.mkdir(parents=True, exist_ok=True)
    records = []
    source_files = {}
    for italic, target_weight, suffix, style in (
        (False, 400, "", "Regular"), (False, 700, "-bold", "Bold"),
        (True, 400, "-italic", "Italic"), (True, 700, "-bolditalic", "Bold Italic"),
    ):
        candidates = [b for b in blocks if f'style: "{"italic" if italic else "normal"}"' in b]
        if not candidates:
            continue
        block = min(candidates, key=lambda b: abs(int(re.search(r'weight: (\d+)', b).group(1)) - target_weight))
        filename = re.search(r'filename: "([^"]+)"', block).group(1)
        actual_weight = int(re.search(r'weight: (\d+)', block).group(1))
        url = base + "/" + quote(filename)
        data = fetch(url)
        font = ImageFont.truetype(BytesIO(data), 64)
        if category == "monospace":
            advances = [font.getlength(char) for char in "iWm012"]
            # Basic FreeType layout rounds hinted advances to whole pixels.
            if max(advances) - min(advances) > 1.1:
                raise ValueError("Monospace advance audit failed: " + slug)
        try:
            axes = font.get_variation_axes()
        except OSError:
            axes = []
        weight_axis = next((a for a in axes if a["name"] == b"Weight"), None)
        if weight_axis:
            if not weight_axis["minimum"] <= target_weight <= weight_axis["maximum"]:
                continue
        elif abs(actual_weight - target_weight) > 150:
            continue
        # Shared variable files are stored once; each style has a separate ID.
        if filename not in source_files:
            source_files[filename] = "fonts/" + slug + suffix + ".ttf"
            (output / source_files[filename]).write_bytes(data)
        font_id = slug + suffix
        (folder_out / (font_id + "-OFL.txt")).write_bytes(license_text)
        records.append({
            "id": font_id, "family_id": slug, "family": family, "category": category,
            "style": style, "weight": target_weight if weight_axis else actual_weight,
            "upstream_categories": declared, "category_audit": CATEGORY_AUDITS.get(slug, "Upstream category; monospace additionally checked by glyph advances."),
            "variable_weight": bool(weight_axis), "italic": italic,
            "file": source_files[filename], "source": f"https://github.com/google/fonts/tree/{REVISION}/{folder}",
            "sha256": hashlib.sha256(data).hexdigest(), "license": "SIL Open Font License 1.1",
            "download_url": url, "license_url": license_url,
            "license_sha256": hashlib.sha256(license_text).hexdigest(),
        })
    print(f"Audited {family}: {len(records)} real styles", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--holdout", action="store_true", help="Earlier external check, now development evidence")
    group.add_argument("--final-holdout", action="store_true", help="Fresh independent families; do not use for tuning")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "artifacts/web"
    families = FINAL_HOLDOUT if args.final_holdout else VALIDATION_FAMILIES if args.holdout else FAMILIES
    jobs = [(category, slug) for category, names in families.items() for slug in names.split()]
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = [record for batch in pool.map(lambda item: build_family(item, output), jobs) for record in batch]
    manifest = {"revision": REVISION, "fonts": records}
    (output / "font_catalog.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Saved", len(jobs), "families /", len(records), "styles")


if __name__ == "__main__":
    main()
