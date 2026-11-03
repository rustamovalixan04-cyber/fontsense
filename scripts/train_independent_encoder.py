"""Fine-tune the product encoder; select only on family-held-out development data."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))
from api.extended_match import primary_styles, render_reference
from api.retrieval_features import text_patches
from scripts.independent_training_data import DATA, SOURCES, SEED, render, sha
from scripts.retrieval_data import VALIDATION_TEXTS, augment, SEED as OLD_SEED
from scripts.train_retrieval_encoder import RetrievalModel, vectors


def development(contract, cache_path=None):
    """Old validation queries remain unchanged; new test query images are absent."""
    cache = Path(cache_path) if cache_path is not None else ROOT.parent / 'retrieval-training/epoch-validation.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
    else:
        base_faces = [row for row in primary_styles() if row['family'] in contract['old_family_splits']]
        names = [row['family'] for row in base_faces]
        selected = [row for row in base_faces if contract['old_family_splits'][row['family']] == 'validation']
        random.Random(OLD_SEED + 51).shuffle(selected)
        refs, queries, labels = [], [], []
        for row in base_faces:
            for phrase in ('Cloud design', 'Golden hour'):
                try: refs.extend(text_patches(render_reference(row['id'], phrase, 64)))
                except (ValueError, OSError): refs.extend(np.full((3,64,160),255,np.uint8))
        for i, row in enumerate(selected[:80]):
            for phrase in VALIDATION_TEXTS:
                try:
                    image = render_reference(row['id'], phrase, 73); text_patches(image)
                except (ValueError, OSError): continue
                for condition in range(2):
                    current = image if not condition else augment(image, random.Random(OLD_SEED + i * 9 + condition))
                    queries.extend(text_patches(current)); labels.append(names.index(row['family']))
        data = {'reference': np.asarray(refs, np.uint8), 'queries': np.asarray(queries, np.uint8), 'labels': np.asarray(labels), 'family_names': np.asarray(names)}
    audit = json.loads((SOURCES / 'audited-catalog.json').read_text())
    records = {row['family']: row for row in reversed(audit['fonts'])}
    new_names = sorted(name for name, split in contract['new_family_splits'].items() if split in ('train', 'validation', 'test'))
    names = data['family_names'].tolist() + new_names
    references, queries, labels = [], [], []
    for family in new_names:
        for phrase in ('Cloud design', 'Golden hour'):
            references.extend(text_patches(render(records[family], phrase, 64)))
    for i, family in enumerate(new_names):
        if contract['new_family_splits'][family] != 'validation': continue
        for phrase in VALIDATION_TEXTS:
            image = render(records[family], phrase)
            for condition in range(2):
                current = image if not condition else augment(image, random.Random(SEED + i * 19))
                queries.extend(text_patches(current)); labels.append(names.index(family))
    data['reference'] = np.concatenate((data['reference'], np.asarray(references, dtype=np.uint8)))
    data['old_queries'] = len(data['labels'])
    data['queries'] = np.concatenate((data['queries'], np.asarray(queries, dtype=np.uint8)))
    data['labels'] = np.concatenate((data['labels'], np.asarray(labels)))
    data['family_names'] = np.asarray(names)
    return data


def evaluate(model, data):
    refs = vectors(model, data['reference']).reshape(-1, 6, 96).mean(1)
    refs /= np.maximum(np.linalg.norm(refs, axis=1, keepdims=True), 1e-6)
    query = vectors(model, data['queries']).reshape(-1, 3, 96).mean(1)
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-6)
    order = np.argsort(-(query @ refs.T), axis=1)
    boundary = data['old_queries']
    return {f'{scope}_top{k}': float((order[section, :k] == data['labels'][section, None]).any(axis=1).mean())
            for scope, section in [('old', slice(0, boundary)), ('independent', slice(boundary, None))] for k in (1, 10, 100)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--epochs', type=int, default=6); args = parser.parse_args()
    torch.set_num_threads(6); torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    contract = json.loads((DATA / 'splits.json').read_text())
    if sha(DATA / 'training.npz') != contract['training_sha256']:
        raise ValueError('Training data does not match its recorded contract')
    with np.load(DATA / 'training.npz', allow_pickle=False) as archive:
        pixels, labels = archive['pixels'], archive['labels']
    base = ROOT / 'artifacts/training/retrieval_base_state.npz'
    if sha(base) != json.loads(base.with_suffix('.json').read_text())['sha256']:
        raise ValueError('Warm-start checkpoint checksum mismatch')
    with np.load(base, allow_pickle=False) as archive:
        state = {key: torch.from_numpy(archive[key].copy()) for key in archive.files}
    teacher = RetrievalModel(1325); teacher.load_state_dict(state); teacher.eval()
    model = RetrievalModel(contract['training_families'])
    expanded = model.state_dict(); expanded.update({key: value for key, value in state.items() if key != 'head'})
    expanded['head'][:1325] = state['head']; model.load_state_dict(expanded); model.eval()
    positions = [np.flatnonzero(labels == i) for i in range(contract['training_families'])]
    if not all(len(indices) >= 2 for indices in positions): raise ValueError('Insufficient paired examples')
    old_count = int((labels < 1325).sum())
    teacher_vectors = np.zeros((len(pixels), 96), np.float32)
    old_indices = np.flatnonzero(labels < 1325)
    teacher_vectors[old_indices] = vectors(teacher, pixels[old_indices])
    del teacher
    data = development(contract)
    baseline = evaluate(model, data)
    print('BASELINE_EXPANDED', json.dumps(baseline), flush=True)
    best_state, best_epoch, history = copy.deepcopy(model.state_dict()), 0, []
    def score(metrics):
        return sum(metrics[f'{scope}_top{k}'] for scope in ('old', 'independent') for k in (1, 10)) / 4
    best_score = score(baseline)
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': 0.},
        {'params': model.projection.parameters(), 'lr': 0.},
        {'params': [model.head], 'lr': .001}], weight_decay=.0002)
    import mlflow
    mlflow.set_tracking_uri((DATA / 'mlruns').resolve().as_uri()); mlflow.set_experiment('FontSense independent font adaptation')
    with mlflow.start_run(run_name='Independent fonts with old-family replay and embedding preservation') as run:
        mlflow.log_params({'seed': SEED, 'epochs': args.epochs, 'training_images': len(pixels),
                          'old_replay_images': old_count, 'training_families': len(positions),
                          'warm_start_sha256': sha(base), 'old_batch_fraction': .75,
                          'old_embedding_loss_weight': 4., 'feature_lr': .00004, 'projection_lr': .00008,
                          'head_lr': .001, 'family_splits_sha256': sha(DATA / 'splits.json')})
        mlflow.log_metrics({'baseline_' + key: value for key, value in baseline.items()}, step=0)
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter(); model.train(); model.features.eval(); losses = []
            # First epoch adapts the enlarged head, then cautiously updates the image encoder.
            decay = .3 + .7 * (1 - (epoch - 1) / args.epochs)
            optimizer.param_groups[0]['lr'] = .00004 * decay if epoch > 1 else 0.
            optimizer.param_groups[1]['lr'] = .00008 * decay if epoch > 1 else 0.
            for step in range(int(np.ceil(len(pixels) / 96))):
                selected = np.concatenate((rng.integers(0, 1325, 36), rng.integers(1325, len(positions), 12)))
                indices = np.concatenate([rng.choice(positions[int(label)], 2, replace=False) for label in selected])
                tensor = torch.from_numpy((pixels[indices, None].astype(np.float32) / 255 - .5) / .5)
                targets = torch.from_numpy(np.repeat(selected, 2))
                optimizer.zero_grad(set_to_none=True); z = model.embed(tensor)
                cosine = z @ F.normalize(model.head, dim=1).T
                margin = torch.zeros_like(cosine).scatter_(1, targets[:, None], .1)
                old_mask = targets < 1325
                preservation = (1 - (z[old_mask] * torch.from_numpy(teacher_vectors[indices][old_mask])).sum(1)).mean()
                loss = F.cross_entropy(24 * (cosine - margin), targets) + .15 * (1 - (z[::2] * z[1::2]).sum(1)).mean() + 4 * preservation
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
                losses.append(float(loss.detach()))
            model.eval(); metrics = evaluate(model, data)
            eligible = metrics['old_top10'] >= baseline['old_top10'] - .015 and metrics['old_top100'] >= baseline['old_top100'] - .01
            row = {'epoch': epoch, 'loss': float(np.mean(losses)), 'seconds': round(time.perf_counter() - started, 1), 'eligible': eligible, **metrics}
            history.append(row); mlflow.log_metrics({key: float(value) for key, value in row.items()}, step=epoch)
            print(json.dumps(row), flush=True)
            if eligible and score(metrics) > best_score:
                best_score, best_epoch, best_state = score(metrics), epoch, copy.deepcopy(model.state_dict())
            (DATA / 'training-progress.json').write_text(json.dumps(history, indent=2) + '\n')
        model.load_state_dict(best_state); model.eval(); checkpoint = DATA / 'selected-model.pt'; torch.save(best_state, checkpoint)
        destination = ROOT / 'artifacts/web/retrieval_encoder.onnx'
        torch.onnx.export(model, torch.zeros(3, 1, 64, 160), destination, input_names=['images'], output_names=['vectors'],
                          dynamic_axes={'images': {0: 'batch'}, 'vectors': {0: 'batch'}}, opset_version=17, dynamo=False)
        previous = json.loads((ROOT / 'artifacts/training/retrieval_base_encoder.json').read_text())
        metadata = {'version': 'independent-retrieval-1', 'sha256': sha(destination), 'checkpoint_sha256': sha(checkpoint),
                    'seed': SEED, 'epochs': args.epochs, 'selected_epoch': best_epoch, 'training_images': len(pixels),
                    'training_families': len(positions), 'active_training_families': contract['active_training_families'],
                    'family_splits': contract['family_splits'], 'duplicate_groups': previous['duplicate_groups'] +
                    [[name for name in group if contract['new_family_splits'][name] in ('train', 'validation', 'test')]
                     for group in contract['new_groups'] if any(contract['new_family_splits'][name] in ('train', 'validation', 'test') for name in group)],
                    'dimensions': 96, 'input': [160, 64], 'baseline_expanded_validation': baseline,
                    'validation': evaluate(model, data), 'history': history, 'mlflow_run_id': run.info.run_id,
                    'initialization': 'Previous independent product encoder with replay and embedding preservation; original assessed classifier unchanged.',
                    'warm_start_sha256': sha(base), 'limits': 'Family-held-out synthetic development queries; references include enrolled held-out families. No claim of worldwide coverage or 100% accuracy.'}
        path = ROOT / 'artifacts/web/retrieval_encoder.json'; path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        checks = data['queries'][:12]
        np.savez_compressed(ROOT / 'tests/fixtures/retrieval_encoder_parity.npz', pixels=checks, vectors=vectors(model, checks))
        (ROOT / 'reports/independent_fonts/training.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        mlflow.log_artifact(str(path)); print('SELECTED', best_epoch, json.dumps(metadata['validation']), run.info.run_id, flush=True)


if __name__ == '__main__': main()
