"""Whole-word source rendering shared by name-model development scripts."""
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "name-data"
WORDS = "river forest garden paper design studio modern bright clear vision future north golden marble cloud deep focus fresh craft open quiet summer winter silver rocket planet window valley ocean bridge simple smooth bold little great honest blue green orange purple shape line picture mountain autumn copper willow meadow travel book music stone light night morning evening city water space motion circle square narrow wide velvet canvas orange pencil poster print table chair flower nature wonder memory signal field road beach lane letter detail screen image chapter classic sample subtle strong reason rhythm story pixel vector banner maple bird sun moon type word text form clean style work small large happy gentle brave swift calm vivid sky dream journey camera frame local urban coral olive lemon grape peach apple cherry melon kiwi lemon cobalt amber ivory slate cream wheat walnut cotton linen spring frost glow spark stream lake wild soft sharp warm cool lucid rapid fluent final first novel ready still breeze modernist samplebook northbound skyline coastline hillside woodland workshop playground bookstore sunlight fireworks blueprint notebook postcard fountain lantern passage headline daylight nightfall wonderful readable perfectly proportion structure geometry contrast balance welcome creative lovely careful visible curious hopeful patient natural enduring elegant productive colorful fantastic tomorrow yesterday weekend holiday teamwork freedom history harmony solid precision mission compass season village portrait success welcome flourish travelmaker storybook newsprint wordsmith typeface creativework".split()
REFERENCE_WORDS = ["Bright river", "Future craft", "North field", "Forest lane", "Golden hour", "Cloud design", "Marble garden", "Urban studio", "Clear vision", "Modern paper", "Vector forms", "Deep focus"]
VALIDATION_WORDS = ["Studio lights", "Clean shapes", "Book covers", "Summer garden", "Copper window", "Fresh ideas"]


@lru_cache(maxsize=384)
def face(record_id, size, manifest_path=str(DATA / "catalog.json"), windows_path="C:/Windows/Fonts"):
    import json
    records = json.loads(Path(manifest_path).read_text())["fonts"]
    record = next(record for record in records if record["id"] == record_id)
    local_paths = DATA / "local-paths.json"
    if local_paths.exists():
        windows_path = json.loads(local_paths.read_text())["windows_fonts"]
    base = ROOT / "artifacts/web" if record["asset_root"] == "main" else DATA if record["asset_root"] == "extra" else Path(windows_path)
    font = ImageFont.truetype(str(base / record["file"]), size)
    if record.get("variable_weight"):
        font.set_variation_by_axes([record["weight"] if a["name"] == b"Weight" else a["default"] for a in font.get_variation_axes()])
    return font


def render(record, text, size=64):
    font = face(record["id"], size)
    box = font.getbbox(text)
    image = Image.new("RGB", (box[2] - box[0] + 80, box[3] - box[1] + 60), "white")
    ImageDraw.Draw(image).text((40 - box[0], 30 - box[1]), text, font=font, fill="black")
    return image
