"""Bounded horizontal text detection for posters, with no upload persistence.

The pinned Apache-2.0 PP-OCRv5 detector supplies a probability map. Run-length
components recover horizontal line boxes without an OpenCV deployment dependency.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1] / 'artifacts/web'


@lru_cache(maxsize=1)
def detector():
    import onnxruntime as ort
    source = json.loads((ROOT / 'ocr_layout_source.json').read_text(encoding='utf-8'))['detector']
    path = ROOT / source['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
        raise RuntimeError('Text detector checksum mismatch')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def components(mask):
    """Eight-connected row runs; work is bounded by a 640-side detection map."""
    parents, bounds = [], []
    def root(label):
        while parents[label] != label:
            parents[label] = parents[parents[label]]
            label = parents[label]
        return label
    previous = []
    for y, row in enumerate(mask):
        padded = np.r_[False, row, False].astype(np.int8)
        changes = np.diff(padded)
        current = []
        for left, right in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            touching = {root(label) for a, b, label in previous if a <= right and b >= left}
            if touching:
                label = min(touching)
                for other in touching:
                    if other == label: continue
                    parents[other] = label
                    old = bounds[other]
                    b = bounds[label]
                    bounds[label] = [min(b[0], old[0]), min(b[1], old[1]), max(b[2], old[2]), max(b[3], old[3])]
                b = bounds[label]
                bounds[label] = [min(b[0], int(left)), b[1], max(b[2], int(right)), y + 1]
            else:
                label = len(parents)
                parents.append(label)
                bounds.append([int(left), y, int(right), y + 1])
            current.append((int(left), int(right), label))
        previous = current
    return [bounds[i] for i in range(len(parents)) if root(i) == i]


def detect(image):
    """Return at most six normalized horizontal text boxes, largest type first."""
    rgb = image.convert('RGB')
    # Bounding both sides limits compute for untrusted large source images.
    ratio = 640 / max(rgb.size)
    size = tuple(max(32, round(side * ratio / 32) * 32) for side in rgb.size)
    resized = np.asarray(rgb.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
    # RapidOCR receives BGR images and uses mean/std=.5 for this model family.
    pixels = resized[:, :, ::-1] / 127.5 - 1
    session = detector()
    probability = session.run(None, {session.get_inputs()[0].name: pixels.transpose(2, 0, 1)[None].copy()})[0][0, 0]
    mask = np.asarray(Image.fromarray(np.uint8(probability > .3) * 255).filter(ImageFilter.MaxFilter(3))) > 0
    height, width = probability.shape
    found = []
    for left, top, right, bottom in components(mask):
        w, h = right - left, bottom - top
        if h < 4 or w < max(10, h * 1.2): continue
        score = float(probability[top:bottom, left:right].mean())
        if score < .5: continue
        found.append({'box':[left,top,right,bottom],'detection_score':score,'weight':w*h})
    # DB can separate words at spaces. Merge compatible baselines before
    # choosing a line, so a long second word cannot hide the first word.
    found = sorted(found,key=lambda r:-(r['box'][3]-r['box'][1])*np.sqrt(r['box'][2]-r['box'][0])*r['detection_score'])[:128]
    changed = True
    while changed:
        changed = False
        for i in range(len(found)):
            a = found[i]['box']; aw, ah = a[2]-a[0], a[3]-a[1]
            for j in range(i+1,len(found)):
                b = found[j]['box']; bw, bh = b[2]-b[0], b[3]-b[1]
                overlap = min(a[3],b[3])-max(a[1],b[1])
                gap = max(a[0],b[0])-min(a[2],b[2])
                if (overlap < .55*min(ah,bh) or max(ah,bh)>1.8*min(ah,bh)
                        or abs((a[1]+a[3])-(b[1]+b[3]))>.7*max(ah,bh)
                        or not -.5*min(aw,bw) <= gap <= 2.5*max(ah,bh)):
                    continue
                first, second = found[i],found[j]
                weight = first['weight']+second['weight']
                first['detection_score'] = (first['detection_score']*first['weight']+second['detection_score']*second['weight'])/weight
                first['weight'] = weight
                first['box'] = [min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])]
                found.pop(j); changed = True
                break
            if changed: break
    for row in found:
        left,top,right,bottom = row['box']; w,h = right-left,bottom-top
        # Axis-aligned DB expansion; small rotations are corrected downstream.
        padding = w*h*1.5/(2*(w+h))
        row['box'] = [max(0.,(left-padding)/width),max(0.,(top-padding)/height),
                      min(1.,(right+padding)/width),min(1.,(bottom+padding)/height)]
        row['priority'] = h*np.sqrt(w)*row['detection_score']
    found.sort(key=lambda r: -r['priority'])
    return [{key: row[key] for key in ('box','detection_score')} for row in found[:6]]


def crop(image, box):
    left, top, right, bottom = box
    return image.crop((max(0, int(left * image.width)), max(0, int(top * image.height)),
                       min(image.width, int(np.ceil(right * image.width))), min(image.height, int(np.ceil(bottom * image.height)))))


def needs_detection(normalized):
    # The established one-line path stays unchanged for already cropped lines.
    if normalized.width / normalized.height < 3.2: return True
    ink = 255 - np.asarray(normalized.convert('L'), dtype=np.uint8)
    occupied = (ink > 40).sum(axis=1) >= max(3, normalized.width * .025)
    changes = np.diff(np.r_[False, occupied, False].astype(np.int8))
    bands = [(a, b) for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)) if b - a >= max(8, normalized.height * .15)]
    return len(bands) > 1
