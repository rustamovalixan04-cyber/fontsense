"""Prepare indie-family splits, replay and development queries before learning."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from api.retrieval_features import text_patches
from scripts.retrieval_data import augment,SEED as OLD_SEED
from scripts.name_data import WORDS

DATA=ROOT.parent/'independent-training'
SOURCES=ROOT.parent/'independent-fonts'
SEED=20261103


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def render(record,text,size=73):
    file=SOURCES/record['local_subset']
    font=ImageFont.truetype(str(file),size)
    if record['variable_weight']:
        font.set_variation_by_axes([record['weight'] if a['name']==b'Weight' else a['default'] for a in font.get_variation_axes()])
    box=font.getbbox(text)
    image=Image.new('L',(box[2]-box[0]+12,box[3]-box[1]+12),255)
    ImageDraw.Draw(image).text((6-box[0],6-box[1]),text,font=font,fill=0)
    return image.convert('RGB')


def partition(audit,known_audit):
    families=sorted({r['family'] for r in audit['fonts']});parent={n:n for n in families}
    def find(n):
        while parent[n]!=n: parent[n]=parent[parent[n]];n=parent[n]
        return n
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[a]=b
    shared={}
    for row in audit['fonts']:
        tokens=['shape:'+row['signature']]+['source:'+g for g in audit['family_groups'][row['family']]]
        for token in tokens:
            if token in shared:union(row['family'],shared[token])
            else:shared[token]=row['family']
    groups=defaultdict(list)
    for name in families:groups[find(name)].append(name)
    normalize=lambda name:''.join(c for c in name.casefold() if c not in ' _-')
    known_keys={normalize(r['family']) for r in known_audit['fonts']}
    known={name for name in families if normalize(name) in known_keys}
    known_groups=[];unknown_groups=[];excluded=[]
    for group in groups.values():
        if set(group)&known:
            known_groups.append(sorted(set(group)&known));excluded.extend(sorted(set(group)-known))
        elif any(name in audit['forced_splits'] for name in group):excluded.extend(group)
        else:unknown_groups.append(sorted(group))
    rng=random.Random(SEED);rng.shuffle(known_groups);rng.shuffle(unknown_groups)
    if len(unknown_groups)<4:raise RuntimeError('Need at least four independent unsupported-font groups')
    splits={}
    train_end=round(len(known_groups)*.7);val_end=round(len(known_groups)*.85)
    for i,group in enumerate(known_groups):
        assigned={audit['forced_splits'][name] for name in group if name in audit['forced_splits']}
        if len(assigned)>1:raise RuntimeError('Old and new shape groups cross existing partitions')
        split=next(iter(assigned)) if assigned else ('train' if i<train_end else 'validation' if i<val_end else 'test')
        for name in group:splits[name]=split
    for i,group in enumerate(unknown_groups):
        for name in group:splits[name]='unknown_validation' if i<max(2,round(len(unknown_groups)/3)) else 'unknown_test'
    for name in excluded:splits[name]='excluded_related'
    return splits,[sorted(g) for g in groups.values() if len(g)>1]


def main():
    DATA.mkdir(parents=True,exist_ok=True)
    audit_path=SOURCES/'audited-catalog.json'
    reports=ROOT/'reports/independent_fonts'
    audit=json.loads((audit_path if audit_path.exists() else reports/'source_audit.json').read_text())
    from scripts.build_independent_catalog import apply_category_review
    apply_category_review(audit['fonts'])
    if not (SOURCES/'known-source-audit.json').exists():
        saved=json.loads((reports/'data_contract.json').read_text())
        splits,groups=saved['new_family_splits'],saved['new_groups']
    else:
        known=json.loads((SOURCES/'known-source-audit.json').read_text())
        splits,groups=partition(audit,known)
    # Fresh checkouts can restore every source, including non-enrolled audit fonts.
    from scripts.independent_font_assets import prepare
    for source in audit['sources']:
        local=SOURCES/'subsets'/Path(source['file']).name
        license_local=SOURCES/'licenses'/(source['id']+'.txt')
        if not local.exists() or not license_local.exists():
            restored=SOURCES/'restored-assets'
            prepare(source,restored)
            local.parent.mkdir(parents=True,exist_ok=True);license_local.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(restored/source['file'],local)
            shutil.copyfile(restored/source['license_file'],license_local)
    SOURCES.mkdir(parents=True,exist_ok=True)
    audit_path.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    base=DATA/'baseline-artifacts';base.mkdir(exist_ok=True)
    web=ROOT/'artifacts/web'
    base_contract=ROOT/'artifacts/training/retrieval_base_encoder.json'
    if not base_contract.exists():
        for name in ('retrieval_encoder.onnx','retrieval_encoder.json','retrieval_index.npz','retrieval_index.json','retrieval_confidence.json','retrieval_policy.json'):
            if not (base/name).exists():shutil.copyfile(web/name,base/name)
    old=json.loads((base_contract if base_contract.exists() else base/'retrieval_encoder.json').read_text())
    if old['training_families']!=1325:raise RuntimeError('Expected the documented pre-expansion baseline')
    base_contract.parent.mkdir(parents=True,exist_ok=True)
    if not base_contract.exists():base_contract.write_text(json.dumps(old,indent=2)+'\n',encoding='utf-8')
    contract={'seed':SEED,'new_family_splits':splits,'new_groups':groups,'old_family_splits':old['family_splits'],
              'family_splits':{**old['family_splits'],**splits},'counts':dict(Counter(splits.values())),
              'baseline_encoder_sha256':old['sha256'],'protocol':'Family/source/shared-Latin groups separated before query generation. Unknown pools have no gradient or enrollment references. Enrollment references include known validation/test families.'}
    (DATA/'splits.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf-8')
    active={n for n,s in splits.items() if s in ('train','validation','test')}
    records=[r for r in audit['fonts'] if r['family'] in active]
    used={r['source_id'] for r in records}
    sources=[s for s in audit['sources'] if s['id'] in used]
    target=web/'independent-fonts';target.mkdir(exist_ok=True)
    for source in sources:
        local=SOURCES/'subsets'/Path(source['file']).name
        shutil.copyfile(local,web/source['file'])
        shutil.copyfile(SOURCES/'licenses'/(source['id']+'.txt'),web/source['license_file'])
    catalog={'version':'independent-catalog-1','families':len(active),'styles':len(records),'sources':sources,
             'fonts':[{k:v for k,v in r.items() if k not in ('signature','local_subset','group')} for r in records],
             'scope':'Independent OFL/CC0 references from FontSpace and original designers. Unknown evaluation pools are not enrolled.'}
    (web/'independent_catalog.json').write_text(json.dumps(catalog,indent=2)+'\n',encoding='utf-8')
    # Small, fully documented warm-start checkpoint; never deployed.
    import torch
    training=ROOT/'artifacts/training';training.mkdir(exist_ok=True)
    baseline_path=training/'retrieval_base_state.npz'
    if not baseline_path.exists():
        state=torch.load(ROOT.parent/'retrieval-training/selected-model.pt',map_location='cpu',weights_only=True)
        np.savez_compressed(baseline_path,**{k:v.numpy() for k,v in state.items()})
    (training/'retrieval_base_state.json').write_text(json.dumps({'sha256':sha(baseline_path),'encoder_sha256':old['sha256'],'mlflow_run_id':old['mlflow_run_id'],'selected_epoch':old['selected_epoch'],'training_families':old['training_families'],'active_training_families':old['active_training_families'],'scope':'Exact previous product checkpoint for reproducible warm-start training. No assessed classifier or font program.'},indent=2)+'\n',encoding='utf-8')
    # Replay preserves old training-family order and never samples old holdouts.
    replay=ROOT.parent/'retrieval-training/training.npz'
    if replay.exists():
        with np.load(replay,allow_pickle=False) as archive:
            original_pixels=archive['pixels'];original_labels=archive['labels']
            assert np.array_equal(original_labels,np.repeat(np.arange(1325),48))
            pixels=original_pixels.reshape(1325,48,64,160)[:,:12].reshape(-1,64,160).copy()
    else:
        from api.extended_match import render_reference
        base_fonts=json.loads((web/'extended_catalog.json').read_text())['fonts']
        by_family=defaultdict(list)
        for face in base_fonts:by_family[face['family']].append(face)
        replay_images=[]
        for index,family in enumerate(old['active_training_families']):
            rng=random.Random(OLD_SEED+index*19)
            for number in range(12):
                face=by_family[family][number%len(by_family[family])]
                text=(str.lower,str.title,str.upper)[number%3](' '.join(rng.sample(WORDS,2 if number%3 else 1)))
                patches=text_patches(augment(render_reference(face['id'],text,rng.choice((40,56,73,96))),rng),count=5)
                replay_images.append(patches[rng.randrange(5)])
        pixels=np.asarray(replay_images,dtype=np.uint8)
    labels=np.repeat(np.arange(1325),12)
    old_names=old['active_training_families'];new_names=sorted(n for n,s in splits.items() if s=='train')
    names=old_names+new_names
    all_records=defaultdict(list)
    for record in audit['fonts']:all_records[record['family']].append(record)
    added=[];added_labels=[]
    for i,family in enumerate(new_names):
        rng=random.Random(SEED+i*29)
        for number in range(64):
            face=all_records[family][number%len(all_records[family])]
            text=(str.lower,str.title,str.upper)[number%3](' '.join(rng.sample(WORDS,1 if number%3==0 else 2)))
            patch=text_patches(augment(render(face,text,rng.choice((40,56,73,96))),rng),count=5)[rng.randrange(5)]
            added.append(patch);added_labels.append(1325+i)
    browser=ROOT.parent/'retrieval-training/browser'
    if not (browser/'samples.json').exists():
        raise FileNotFoundError('Restore the documented fontsense-browser-benchmark.zip corpus under ../retrieval-training/browser before reproducing training')
    browser_rows=[r for r in json.loads((browser/'samples.json').read_text()) if r['split']=='train']
    browser_patches=0
    for row in browser_rows:
        if row['family'] not in old_names:raise RuntimeError('Browser replay includes an old held-out family')
        image=Image.open(browser/row['file']).convert('RGB');rng=random.Random(OLD_SEED+browser_patches)
        for condition in range(4):
            patches=text_patches(image if condition==0 else augment(image,rng))
            added.extend(patches);added_labels.extend([old_names.index(row['family'])]*len(patches));browser_patches+=len(patches)
    pixels=np.concatenate((pixels,np.asarray(added,dtype=np.uint8)));labels=np.concatenate((labels,np.asarray(added_labels,dtype=np.int64)))
    np.savez_compressed(DATA/'training.npz',pixels=pixels,labels=labels)
    contract.update({'training_images':len(pixels),'old_synthetic_replay':15900,'old_browser_replay':browser_patches,'new_training_images':len(new_names)*64,'training_families':len(names),'active_training_families':names,'training_sha256':sha(DATA/'training.npz')})
    (DATA/'splits.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf-8')
    sheet=Image.new('RGB',(900,8*100),'#eeeeee');draw=ImageDraw.Draw(sheet)
    for i,family in enumerate(new_names[:16]):
        image=render(all_records[family][0],'River craft',64);image.thumbnail((430,75))
        x,y=(i%2)*450,(i//2)*100;sheet.paste(image,(x,y));draw.text((x+4,y+78),family,fill='black')
    sheet.save(DATA/'training-contact-sheet.jpg')
    reports=ROOT/'reports/independent_fonts';reports.mkdir(parents=True,exist_ok=True)
    (reports/'source_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    (reports/'data_contract.json').write_text(json.dumps(contract,indent=2)+'\n',encoding='utf-8')
    print('PREPARED',contract['counts'],'training images',len(pixels),'new train families',len(new_names),'runtime new families',len(active),flush=True)


if __name__=='__main__': main()
