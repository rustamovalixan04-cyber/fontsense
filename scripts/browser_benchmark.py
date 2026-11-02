"""Local-only known-font specimen page for genuine browser screenshot captures.

Run then capture through the browser UI. Never substitutes a PIL reconstruction
for a browser screenshot. No native or proprietary font program is served.
"""
import html
import json
from pathlib import Path
import random
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.extended_match import catalog
from scripts.retrieval_data import DATA, VALIDATION_TEXTS, TEST_TEXTS

app = FastAPI(docs_url=None, redoc_url=None)


def cases(split):
    if split in ("native", "native2"):
        from scripts.build_retrieval_index import native_sources
        from scripts.calibrate_retrieval_confidence import NATIVE_FINAL_FAMILIES
        result = []
        sources = [face for face in native_sources() if face["family"] in NATIVE_FINAL_FAMILIES]
        # Outside-gallery fonts test rejection, not identification coverage.
        for family, filename in (("Candara", "candara.ttf"), ("Corbel", "corbel.ttf"), ("Constantia", "constan.ttf"),
                                 ("Bahnschrift", "bahnschrift.ttf"), ("Ebrima", "ebrima.ttf"), ("Gadugi", "gadugi.ttf")):
            import hashlib
            path = Path("C:/Windows/Fonts") / filename
            if path.exists():
                sources.append({"family": family, "id": "unknown-" + family.lower(), "style": "Regular",
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "unknown": True})
        for face in sources:
            for number, phrase in enumerate(("Cobalt Horizon", "Fuchsia Station") if split == "native2" else TEST_TEXTS):
                result.append({"family": face["family"], "font_id": face["id"], "style": face["style"],
                               "weight": 900 if "Black" in face["style"] else 700 if "Bold" in face["style"] else 400,
                               "italic": "Italic" in face["style"], "phrase": phrase, "split": "native_test2" if split == "native2" else "native_test",
                               "dark": bool(number % 2), "case_id": len(result), "native": True,
                               "known_reference": not face.get("unknown", False),
                               "source_font_sha256": face["sha256"]})
        return result
    manifest = json.loads((DATA / "splits.json").read_text())
    splits = manifest["families"]
    source_split = "test" if split == "test2" else split
    names = sorted(name for name, value in splits.items() if value == source_split)
    random.Random(8300 + len(source_split)).shuffle(names)
    if split == "test2":
        excluded = set(names[:80])
        for group in manifest["duplicate_groups"]:
            if excluded.intersection(group): excluded.update(group)
        names = [name for name in names if name not in excluded]
    names = names[:96 if split == "train" else 60 if split == "validation" else 80]
    phrases = ("Bright river",) if split == "train" else VALIDATION_TEXTS if split == "validation" else ("Cobalt Horizon", "Fuchsia Station") if split == "test2" else TEST_TEXTS
    result = []
    for index, family in enumerate(names):
        faces = [font for font in catalog() if font["family"] == family]
        face = faces[index % len(faces)]
        for number, phrase in enumerate(phrases):
            result.append({"family": family, "font_id": face["id"], "style": face["style"],
                           "weight": face["weight"], "italic": face["italic"],
                           "phrase": phrase, "split": split, "dark": bool(number % 2), "case_id": len(result)})
    return result


@app.get("/asset/{font_id}")
def asset(font_id):
    face = next((font for font in catalog() if font["id"] == font_id), None)
    if face is None: raise HTTPException(404)
    return FileResponse(ROOT / "artifacts/web" / face["file"], media_type="font/ttf")


@app.get("/", response_class=HTMLResponse)
def page(split: str = "train", page: int = 0):
    if split not in ("train", "validation", "test", "test2", "native", "native2") or page < 0: raise HTTPException(400)
    rows = cases(split)[page * 40:(page + 1) * 40]
    css, body = [], []
    for row in rows:
        family = row["family"] if row.get("native") else "FS" + str(row["case_id"])
        if not row.get("native"):
            css.append(f"@font-face{{font-family:{family};src:url('/asset/{row['font_id']}');font-weight:100 900;font-style:normal;}}")
        background, foreground = ("#202026", "#f3f0e9") if row["dark"] else ("#faf9f5", "#202026")
        body.append(f'<div class="specimen" data-case="{html.escape(json.dumps(row), quote=True)}" data-native="{str(row.get("native", False)).lower()}" data-family="{family}" data-weight="{row["weight"]}" style="background:{background};color:{foreground}"><span style="font-family:\'{family}\';font-weight:{row["weight"]};font-style:{"italic" if row.get("native") and row["italic"] else "normal"};letter-spacing:{.3 if row["dark"] else 0}px">{html.escape(row["phrase"])}</span></div>')
    return '<!doctype html><html><head><meta charset="utf-8"><title>FontSense browser benchmark</title><style>body{margin:16px;background:#ccc}.grid{display:grid;grid-template-columns:560px 560px;gap:16px}.specimen{width:560px;height:130px;display:flex;align-items:center;justify-content:center;overflow:hidden}.specimen span{font-size:52px;white-space:pre;line-height:1.2}</style><style>' + ''.join(css) + '</style></head><body><main class="grid">' + ''.join(body) + '''</main><script>
Promise.all([...document.querySelectorAll('.specimen')].map(async node=>{
try{let loaded;if(node.dataset.native==='true'){
const measure=family=>{const probe=document.createElement('span');probe.textContent='mmmmmmmmmmlliWW0123';probe.style.cssText='position:absolute;left:-10000px;white-space:pre;font-size:52px;font-family:'+family;document.body.append(probe);const width=probe.getBoundingClientRect().width;probe.remove();return width;};
loaded=['monospace','sans-serif'].every(fallback=>Math.abs(measure('"'+node.dataset.family+'",'+fallback)-measure(fallback))>.2);
}else{const fonts=await document.fonts.load(node.dataset.weight+' 52px '+node.dataset.family);loaded=fonts.length>0&&fonts.every(font=>font.status==='loaded');}const span=node.querySelector('span');const width=span.getBoundingClientRect().width;if(width>510)span.style.fontSize=(52*510/width)+'px';node.dataset.loaded=String(loaded);}catch(e){node.dataset.loaded='false';}
})).then(()=>document.body.dataset.ready='true');
</script></body></html>'''


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8771)
