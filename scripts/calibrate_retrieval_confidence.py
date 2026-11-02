"""Fit a correctness filter and select its gate on separate validation families.

Unknowns are simulated by removing the query's true family from the enrollment
result; actual outside-gallery fonts are checked separately in the final audit.
"""
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.compact import text_crop
from api.font_match import read_text, straighten_line
from api.retrieval_match import rank, confidence_features
from scripts.retrieval_data import DATA, VALIDATION_TEXTS, augment, SEED
from api.extended_match import primary_styles, render_reference
from api.retrieval_index import candidates as index_candidates, native_candidates
from scripts.build_retrieval_index import native_sources, native_render

NATIVE_FINAL_FAMILIES = {"Arial Black", "Palatino Linotype", "Lucida Console", "Lucida Sans Unicode",
                         "Franklin Gothic Medium", "Gabriola", "Segoe Print", "Segoe Script", "Sylfaen"}
NATIVE_UNKNOWN_EXCLUSIONS = []


def native_unknown_rows():
    # Different sources from the six final outside-gallery fonts. These include
    # close Latin companions, not only easy decorative negative examples.
    files = ("leelawui.ttf", "malgun.ttf", "taile.ttf", "phagspa.ttf", "ntailu.ttf", "monbaiti.ttf", "javatext.ttf")
    from PIL import ImageFont
    rows = []
    for i, filename in enumerate(files):
        path = Path("C:/Windows/Fonts") / filename
        if not path.exists(): continue
        try: family = ImageFont.truetype(str(path), 64).getname()[0]
        except OSError as error:
            NATIVE_UNKNOWN_EXCLUSIONS.append({"source": filename, "reason": str(error)}); continue
        source = {"local_file": filename}
        for text in VALIDATION_TEXTS:
            try:
                image = native_render(source, text, 73); text_crop(image)
            except (OSError, ValueError) as error:
                NATIVE_UNKNOWN_EXCLUSIONS.append({"source": filename, "reason": str(error)}); continue
            for condition in range(2):
                current = image if condition == 0 else augment(image, random.Random(SEED + i * 17))
                ranked = native_candidates(index_candidates(current, text))
                if ranked:
                    rows.append({"partition": "fit" if i % 3 == 0 else "selection", "family": family,
                                 "correct": False, "source": "actual_unknown", "similarity": ranked[0]["embedding_similarity"],
                                 "margin": ranked[0]["prototype_margin"], "source_font_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return rows


def choose_native_gate(rows, families, fit_families):
    selected = [row for row in rows if row["partition"] == "selection"]
    best = None
    for minimum in (.90, .92, .94, .96, .97, .98):
        for margin in (.005, .01, .015, .02, .03, .05, .08):
            accepted = [row for row in selected if row["similarity"] >= minimum and row["margin"] >= margin]
            if len(accepted) < 8 or not all(row["correct"] for row in accepted): continue
            candidate = {"minimum_similarity": minimum, "minimum_margin": margin, "accepted": len(accepted),
                         "correct": len(accepted), "queries": len(selected),
                         "selection_families": sorted(set(families) - fit_families),
                         "actual_unknown_queries": sum(row.get("source") == "actual_unknown" for row in selected),
                         "final_families_excluded": sorted(NATIVE_FINAL_FAMILIES)}
            if best is None or len(accepted) > best["accepted"]: best = candidate
    return best


def candidate_floor(rows):
    scores = [row["similarity"] for row in rows if row["partition"] == "fit" and row["correct"]]
    return {"minimum_similarity": float(np.quantile(scores, .025)), "maximum_family_rank": 10,
            "role": "Uncertain candidate display only; floor is the 2.5th percentile of correct fitting-native scores. Never a likely-name gate or a probability."}


def calibrate_native(negatives):
    sources = [row for row in native_sources() if row["family"] not in NATIVE_FINAL_FAMILIES]
    families = sorted({row["family"] for row in sources}); random.Random(SEED + 99).shuffle(families)
    fit_families = set(families[:len(families) // 2])
    rows = [{**row, "partition": ("fit" if row["family"] in fit_families else "selection") if row["family"] in set(families) else row["partition"]} for row in negatives]
    for i, source in enumerate(sources):
        for text in VALIDATION_TEXTS:
            image = native_render(source, text, 73)
            for condition in range(2):
                current = image if condition == 0 else augment(image, random.Random(SEED + i))
                ranked = native_candidates(index_candidates(current, text))
                if not ranked: continue
                rows.append({"partition": "fit" if source["family"] in fit_families else "selection",
                             "family": source["family"], "correct": ranked[0]["family"] == source["family"],
                             "similarity": ranked[0]["embedding_similarity"], "margin": ranked[0]["prototype_margin"]})
    rows.extend(native_unknown_rows())
    best = choose_native_gate(rows, families, fit_families)
    (DATA / "native-confidence-rows.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return best


def main():
    samples = [row for row in json.loads((DATA / "browser/samples.json").read_text()) if row["split"] == "validation"]
    queries = [(row["family"], Image.open(DATA / "browser" / row["file"]).convert("RGB"), row["phrase"], "browser") for row in samples]
    # Add independent synthesized conditions from other validation families.
    reserved = {row["family"] for row in samples}
    splits = json.loads((DATA / "splits.json").read_text())["families"]
    extra = [font for font in primary_styles() if splits[font["family"]] == "validation" and font["family"] not in reserved]
    random.Random(SEED + 703).shuffle(extra)
    for i, font in enumerate(extra[:60]):
        for text in VALIDATION_TEXTS:
            try:
                image = augment(render_reference(font["id"], text, 73), random.Random(SEED + i * 13))
                text_crop(image)
                queries.append((font["family"], image, text, "synthetic_damage"))
            except (OSError, ValueError): continue
    # Native fonts are outside the open-font rendering bank. Include their
    # distinct-phrase queries as real negative examples for open-font naming.
    for i, source in enumerate(native_sources()):
        if source["family"] in NATIVE_FINAL_FAMILIES: continue
        for text in VALIDATION_TEXTS:
            queries.append((source["family"], native_render(source, text, 73).convert("RGB"), text, "native_graphic"))
    families = sorted({row[0] for row in queries}); random.Random(SEED + 81).shuffle(families)
    fitting = set(families[:round(len(families) * .65)])
    rows, native_negatives = [], []
    for i, (family, image, visible_text, source) in enumerate(queries):
        normalized = straighten_line(text_crop(image))
        text, confidence = read_text(normalized)
        if confidence < .55 or len(text) < 2: continue
        ranked = rank(normalized, text, limit=30)
        if len(ranked) < 2: continue
        native = ranked[0].get("native_candidates", [])
        if native:
            native_negatives.append({"partition": "fit" if family in fitting else "selection", "family": family,
                                     "correct": native[0]["family"] == family, "similarity": native[0]["embedding_similarity"], "margin": native[0]["prototype_margin"]})
        rows.append({"family": family, "partition": "fit" if family in fitting else "selection", "source": source,
                     "unknown": False, "correct": ranked[0]["family"] == family,
                     "features": confidence_features(ranked, text).tolist()})
        unknown = [font for font in ranked if font["family"] != family]
        if len(unknown) >= 2:
            rows.append({"family": family, "partition": "fit" if family in fitting else "selection", "source": source,
                         "unknown": True, "correct": False, "features": confidence_features(unknown, text).tolist()})
        if i % 20 == 0: print("Calibration images", i, "/", len(queries), flush=True)
    (DATA / "confidence-rows.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    fit = [row for row in rows if row["partition"] == "fit"]
    check = [row for row in rows if row["partition"] == "selection"]
    x = np.asarray([row["features"] for row in fit]); y = np.asarray([row["correct"] for row in fit])
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1., max_iter=2000, random_state=SEED).fit(scaler.transform(x), y)
    cx = np.asarray([row["features"] for row in check]); cy = np.asarray([row["correct"] for row in check]); unknown = np.asarray([row["unknown"] for row in check])
    probability = model.predict_proba(scaler.transform(cx))[:, 1]
    best = None
    for floor in (.86, .88, .90, .92, .94, .96):
        for threshold in sorted(set(probability.tolist() + [.9, .95, .98, .99, .995])):
            accepted = (probability >= threshold) & (cx[:, 1] >= floor)
            count = int(accepted.sum())
            if count < 10: continue
            precision = float(cy[accepted].mean())
            unknown_accepted = int((accepted & unknown).sum())
            # Select zero observed naming errors on the selection subset. This
            # is empirical evidence, not a guarantee beyond the checked sample.
            if precision < 1. or unknown_accepted != 0: continue
            candidate = {"accepted": count, "correct": int(cy[accepted].sum()), "precision": precision,
                         "unknown_accepted": unknown_accepted, "unknown_queries": int(unknown.sum()),
                         "queries": len(check), "threshold": float(threshold), "minimum_pixel_score": floor}
            if best is None or count > best["accepted"]: best = candidate
    if best is None:
        raise RuntimeError("No calibrated gate returned at least ten names without observed selection errors")
    contract = {"version": "retrieval-correctness-1", "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                "coefs": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
                "threshold": best["threshold"], "minimum_pixel_score": best["minimum_pixel_score"],
                "selection": best, "fitting_families": sorted(fitting), "selection_families": sorted(set(families) - fitting),
                "limits": "Empirical correctness filter under a constructed validation mixture. Scores are not universal probabilities and are not displayed as confidence percentages.",
                "index_sha256": json.loads((ROOT / "artifacts/web/retrieval_index.json").read_text())["sha256"]}
    contract["native"] = calibrate_native(native_negatives)
    contract["native_unknown_exclusions"] = NATIVE_UNKNOWN_EXCLUSIONS
    contract["native_candidate"] = candidate_floor(json.loads((DATA / "native-confidence-rows.json").read_text()))
    (ROOT / "artifacts/web/retrieval_confidence.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print("CALIBRATED", json.dumps(best), flush=True)


if __name__ == "__main__":
    if "--refresh-native" in sys.argv:
        rows = [row for row in json.loads((DATA / "native-confidence-rows.json").read_text()) if row.get("source") != "actual_unknown"]
        rows.extend(native_unknown_rows())
        families = sorted({row["family"] for row in native_sources() if row["family"] not in NATIVE_FINAL_FAMILIES})
        random.Random(SEED + 99).shuffle(families)
        result = choose_native_gate(rows, families, set(families[:len(families) // 2]))
        path = ROOT / "artifacts/web/retrieval_confidence.json"
        contract = json.loads(path.read_text()); contract["native"] = result; contract["native_unknown_exclusions"] = NATIVE_UNKNOWN_EXCLUSIONS
        contract["native_candidate"] = candidate_floor(rows)
        path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
        (DATA / "native-confidence-rows.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print("REFRESHED NATIVE", json.dumps(result), flush=True)
    else: main()
