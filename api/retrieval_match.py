"""Trained shortlist followed by text/glyph comparisons and selected gates.

The original exhaustive matcher remains available as a fallback.
"""

import numpy as np

from api import extended_match
from api.glyph_match import glyphs, agreement
from api.retrieval_index import candidates as index_candidates
from api.retrieval_index import native_candidates
from api.font_variations import fitting_axes, render_variation


def rank(image, text, limit=8, candidate_limit=400):
    from api.font_match import descriptor, spacing_descriptor
    shortlisted = index_candidates(image, text, limit=candidate_limit)
    native = native_candidates(shortlisted)
    query, aspect = descriptor(image)
    compact_query, compact_aspect = spacing_descriptor(image)
    query_glyphs = glyphs(image)
    found = {}
    def compare(record, rendered, axes=None):
        reference, ref_aspect = descriptor(rendered)
        compact, ref_compact_aspect = spacing_descriptor(rendered)
        full = float(reference @ query) - .08 * min(abs(np.log(ref_aspect / aspect)), 1.5)
        spaced = float(compact @ compact_query) - .08 * min(abs(np.log(ref_compact_aspect / compact_aspect)), 1.5)
        pixel = max(full, .5 * (full + spaced) - .04)
        glyph = agreement(query_glyphs, glyphs(rendered))
        score = .8 * pixel + .2 * glyph if glyph is not None else pixel
        row = {**record, "score": score, "pixel_score": pixel, "glyph_score": glyph,
               "glyph_count": len(query_glyphs[0]) if query_glyphs is not None else 0,
               "method": "trained-shortlist-and-glyphs"}
        if axes: row["fit_axes"] = axes
        previous = found.get(record["family"])
        if previous is None or score > previous["score"]: found[record["family"]] = row
    for record in shortlisted:
        if record.get("native"): continue
        try: compare(record, extended_match.render_reference(record["id"], text, 96))
        except (ValueError, OSError): continue
    for record in sorted(found.values(), key=lambda row: -row["score"])[:5]:
        if record["score"] < .65: continue
        for axes in fitting_axes(record["id"]):
            try: compare(record, render_variation(record["id"], text, **axes), axes)
            except (ValueError, OSError): continue
    result = sorted(found.values(), key=lambda row: -row["score"])
    appearance, seen = [], set()
    for record in shortlisted:
        if record.get('native') or record['family'] in seen: continue
        seen.add(record['family'])
        appearance.append({**record, 'score': record['embedding_similarity'],
                           'method': 'trained-appearance-alternatives', 'native_candidates': native[:5]})
        if len(appearance) == limit: break
    for row in result:
        row["native_candidates"] = native[:5]
        row['appearance_candidates'] = appearance
    return result[:limit]


def likely_native(ranked, text):
    if sum(char.isalpha() for char in text) < 5 or len(set(text.casefold())) < 4: return None
    if not ranked or not ranked[0].get("native_candidates"): return None
    from api.retrieval_confidence import parameters
    gate = parameters().get("native")
    if gate is None: return None
    best = ranked[0]["native_candidates"][0]
    if best["embedding_similarity"] < gate["minimum_similarity"] or best["prototype_margin"] < gate["minimum_margin"]:
        return None
    return {"family": best["family"], "category": best["category"], "url": best["url"],
            "strength": "likely", "method": "trained-native-whole-word-references"}


def confidence_features(ranked, text):
    best = ranked[0]
    other = ranked[1] if len(ranked) > 1 else best
    return np.asarray([best["score"], best["pixel_score"], best["score"] - other["score"],
                       best["pixel_score"] - other["pixel_score"],
                       best.get("glyph_score") or best["pixel_score"],
                       min(best.get("glyph_count", 0), 20) / 20,
                       best["embedding_similarity"], best["embedding_similarity"] - other["embedding_similarity"],
                       min(sum(char.isalpha() for char in text), 30) / 30], dtype=np.float64)


def likely_match(ranked, text):
    if len(ranked) < 2 or sum(char.isalpha() for char in text) < 5 or len(set(text.casefold())) < 4:
        return None
    # Calibration is fitted only on development data by a separate script.
    from api.retrieval_confidence import accept
    if not accept(confidence_features(ranked, text)): return None
    best = ranked[0]
    return {"family": best["family"], "category": best["category"], "url": best["url"],
            "style": best["style"], "strength": "likely", "method": "trained-shortlist-and-glyphs"}
