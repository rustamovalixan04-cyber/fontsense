"""Selective second OCR opinion, accepted only with better rendered evidence."""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / 'artifacts/web'


@lru_cache(maxsize=1)
def reader():
    import onnxruntime as ort
    source = json.loads((ROOT / 'ocr_layout_source.json').read_text(encoding='utf-8'))['secondary_reader']
    path = ROOT / source['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
        raise RuntimeError('Secondary OCR checksum mismatch')
    options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    characters = session.get_modelmeta().custom_metadata_map['character'].rstrip('\n').split('\n')
    count = session.get_outputs()[0].shape[-1]
    if len(characters) == count - 2: characters.append(' ')
    if len(characters) != count - 1: raise RuntimeError('Secondary OCR character dictionary mismatch')
    return session, [''] + characters


def second_read(image):
    session, characters = reader()
    resized_width = min(1536, math.ceil(48 * image.width / image.height))
    width = max(320, resized_width)
    resized = image.resize((resized_width,48),Image.Resampling.BILINEAR).convert('RGB')
    tensor = np.zeros((1,3,48,width),dtype=np.float32)
    tensor[0,:,:,:resized_width] = (np.asarray(resized,dtype=np.float32)/127.5-1).transpose(2,0,1)
    scores = session.run(None,{session.get_inputs()[0].name:tensor})[0][0]
    indices = scores.argmax(1)
    chosen = (indices != 0) & np.r_[True,indices[1:] != indices[:-1]]
    return ''.join(characters[i] for i in indices[chosen]).strip()[:80], float(scores.max(1)[chosen].mean()) if chosen.any() else 0.


def shape_score(image, text, fonts):
    from api.font_match import descriptor, spacing_descriptor
    from api.extended_match import render_reference
    query, aspect = descriptor(image); compact_query, compact_aspect = spacing_descriptor(image)
    best = -1.
    for font in fonts:
        try:
            reference = render_reference(font['id'],text,96)
            full, reference_aspect = descriptor(reference)
            compact, reference_compact_aspect = spacing_descriptor(reference)
        except (ValueError,OSError): continue
        score = float(query@full) - .08*min(abs(np.log(reference_aspect/aspect)),1.5)
        spaced = float(compact_query@compact) - .08*min(abs(np.log(reference_compact_aspect/compact_aspect)),1.5)
        best = max(best,score,.5*(score+spaced)-.04)
    return best


def refine(image, basic):
    text, confidence = basic
    if confidence >= .98: return basic
    alternate, other_confidence = second_read(image)
    if alternate == text or other_confidence < .75 or not alternate or any(not 32 <= ord(c) <= 126 for c in alternate):
        return basic
    from api.retrieval_index import candidates
    fonts, seen = [], set()
    for font in candidates(image,text,80):
        if font.get('native') or font['family'] in seen: continue
        seen.add(font['family']); fonts.append(font)
        if len(fonts) == 10: break
    # Never replace text using only the second model's uncalibrated confidence.
    if shape_score(image,alternate,fonts) >= shape_score(image,text,fonts) + .015:
        return alternate, other_confidence
    return basic
