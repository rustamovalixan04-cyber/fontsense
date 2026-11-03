"""One post-selection audit, with fresh synthetic queries and actual unknown fonts."""
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))
from api.extended_match import primary_styles, render_reference
from api.font_match import identify
from api.retrieval_features import text_patches
from scripts.independent_training_data import DATA, SOURCES, render, SEED, sha
from scripts.retrieval_data import augment
from scripts.train_retrieval_encoder import RetrievalModel, vectors


def main():
    torch.set_num_threads(6)
    contract = json.loads((DATA / 'splits.json').read_text())
    exported = json.loads((ROOT / 'artifacts/web/retrieval_encoder.json').read_text())
    if sha(DATA / 'selected-model.pt') != exported['checkpoint_sha256']: raise ValueError('Model is not locked')
    audited = json.loads((SOURCES / 'audited-catalog.json').read_text())
    indie = {row['family']: row for row in reversed(audited['fonts'])}
    faces = list(primary_styles()); names = [row['family'] for row in faces]
    # Build references independently of query images. They necessarily enroll each searchable family.
    refs = []
    for font in faces:
        for phrase in ('Cloud design', 'Golden hour'):
            try: refs.extend(text_patches(render_reference(font['id'], phrase, 64)))
            except (ValueError, OSError): refs.extend(np.full((3, 64, 160), 255, np.uint8))
    old_test = [row for row in faces if row['family'] in contract['old_family_splits'] and contract['old_family_splits'][row['family']] == 'test']
    random.Random(SEED + 440).shuffle(old_test)
    selected = [('old', row) for row in old_test[:40]] + [('independent', row) for row in faces if contract['new_family_splits'].get(row['family']) == 'test']
    queries, manifest = [], []
    for i, (scope, font) in enumerate(selected):
        for phrase in ('Violet Observatory', 'Mango Picnic'):
            image = render_reference(font['id'], phrase, 83)
            for condition in range(2):
                current = image if not condition else augment(image, random.Random(SEED + 776 + i))
                queries.extend(text_patches(current))
                manifest.append({'family': font['family'], 'scope': scope, 'phrase': phrase, 'condition': 'damaged' if condition else 'clean'})
    queries = np.asarray(queries, np.uint8); refs = np.asarray(refs, np.uint8)
    base = ROOT / 'artifacts/training/retrieval_base_state.npz'
    with np.load(base, allow_pickle=False) as archive: old_state = {key: torch.from_numpy(archive[key].copy()) for key in archive.files}
    trained_state = torch.load(DATA / 'selected-model.pt', map_location='cpu', weights_only=True)
    results = {}
    for title, state in [('baseline_expanded', old_state), ('trained_expanded', trained_state)]:
        model = RetrievalModel(state['head'].shape[0]); model.load_state_dict(state); model.eval()
        reference = vectors(model, refs).reshape(-1, 6, 96).mean(1); reference /= np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-6)
        query = vectors(model, queries).reshape(-1, 3, 96).mean(1); query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-6)
        order = np.argsort(-(query @ reference.T), axis=1)
        rows = [{**row, 'top10': [names[j] for j in ranked[:10]], 'rank': int(np.flatnonzero(ranked == names.index(row['family']))[0]) + 1} for row, ranked in zip(manifest, order)]
        scopes = {}
        for scope in ('old', 'independent'):
            group = [row for row in rows if row['scope'] == scope]
            scopes[scope] = {'queries': len(group), 'families': len({row['family'] for row in group}),
                             **{f'top{k}': sum(row['rank'] <= k for row in group) / len(group) for k in (1, 10, 100)},
                             'macro_f1_query_families': float(f1_score([row['family'] for row in group], [row['top10'][0] for row in group], labels=sorted({row['family'] for row in group}), average='macro', zero_division=0))}
        results[title] = {'summary': scopes, 'rows': rows}
        print(title, json.dumps(scopes), flush=True)
    # Complete OCR-to-match audit: separately report the manual text override.
    pipeline = []
    test_sources = [(family, row, True) for family, row in indie.items() if contract['new_family_splits'][family] == 'test']
    test_sources += [(family, row, False) for family, row in indie.items() if contract['new_family_splits'][family] == 'unknown_test']
    for i, (family, font, known) in enumerate(test_sources):
        for phrase in ('Violet Observatory', 'Mango Picnic'):
            for condition in range(2):
                image = render(font, phrase, 83)
                if condition: image = augment(image, random.Random(SEED + 776 + names.index(family) if known else SEED + 1661 + i))
                for mode in ('automatic_ocr', 'supplied_text'):
                    started = time.perf_counter()
                    result = identify(image, phrase) if mode == 'supplied_text' else identify(image)
                    named = result.get('identified_font'); name = named['family'] if named else None
                    top3 = list(dict.fromkeys(([name] if name else []) + [row['family'] for row in result['matches']]))[:3]
                    pipeline.append({'family': family, 'known': known, 'phrase': phrase, 'condition': 'damaged' if condition else 'clean',
                                     'mode': mode, 'read_text': result['text'], 'named': name, 'top3': top3,
                                     'needs_text': result['needs_text'], 'elapsed_ms': round((time.perf_counter() - started) * 1000, 1)})
        print('Pipeline tested independent family', i + 1, '/', len(test_sources), flush=True)
    summaries = {}
    normalize = lambda text: ''.join(c for c in text.casefold() if c.isalnum())
    for mode in ('automatic_ocr', 'supplied_text'):
        group = [row for row in pipeline if row['mode'] == mode]; known = [row for row in group if row['known']]
        unknown = [row for row in group if not row['known']]; named = [row for row in group if row['named']]
        summaries[mode] = {'known_queries': len(known), 'unknown_queries': len(unknown),
                           'top3_correct': sum(row['family'] in row['top3'] for row in known),
                           'names_returned': len(named), 'correct_names': sum(row['known'] and row['named'] == row['family'] for row in named),
                           'unknown_names_withheld': sum(row['named'] is None for row in unknown),
                           'exact_text': sum(normalize(row['read_text']) == normalize(row['phrase']) for row in group),
                           'median_ms': float(np.median([row['elapsed_ms'] for row in group]))}
    report = {'protocol': 'After model selection: fresh phrases rendered by Pillow, plus simulated damage. 11 new held-out known families and five actual non-enrolled families (four source groups). Baseline and trained encoders have identical expanded enrollment. Old families were held out of gradient training but may have historical diagnostic exposure. Not browser/camera accuracy or global font coverage.',
              'encoder_sha256': exported['sha256'], 'selected_epoch': exported['selected_epoch'], 'retrieval': results,
              'pipeline': {'summary': summaries, 'rows': pipeline}}
    path = ROOT / 'reports/independent_fonts/final_evaluation.json'; path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('FINAL', json.dumps(summaries), flush=True)


if __name__ == '__main__': main()
