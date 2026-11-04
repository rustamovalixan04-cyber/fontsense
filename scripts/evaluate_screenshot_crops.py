"""Real browser text with separately labelled simulated crop defects."""
import argparse
from collections import defaultdict
from io import BytesIO
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image, ImageChops

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from api.font_match import identify
from api.extended_match import catalog

DATA=ROOT.parent/'screenshot-crops'
BROWSER=ROOT.parent/'retrieval-training/browser'
REPORT=ROOT/'reports/screenshot_crops'


def prepare(split):
    records=json.loads((BROWSER/'samples.json').read_text())
    categories={r['family']:r['category'] for r in catalog()}
    grouped=defaultdict(dict)
    for row in records:
        if row['split']!=('test2' if split=='test3' else split):continue
        grouped[categories[row['family']]].setdefault(row['family'],row)
    excluded=set()
    if split=='test3':
        earlier=json.loads((DATA/'test2/manifest.json').read_text())
        excluded={row['family'] for row in earlier['rows']}
    selected=[]
    for category,fonts in sorted(grouped.items()):
        names=sorted(set(fonts)-excluded)
        random.Random((20261105 if split=='test3' else 20261104)+len(category)).shuffle(names)
        selected.extend(fonts[name] for name in names[:3])
    result=[]
    folder=DATA/split;folder.mkdir(parents=True,exist_ok=True)
    for index,row in enumerate(selected):
        original=Image.open(BROWSER/row['file']).convert('RGB')
        array=np.asarray(original.convert('L'));background=float(np.median(np.r_[array[0],array[-1],array[:,0],array[:,-1]]))
        mask=np.abs(array.astype(float)-background)>60;y,x=np.where(mask)
        tight=original.crop((x.min(),y.min(),x.max()+1,y.max()+1))
        bg=tuple(int(v) for v in np.median(np.asarray(original).reshape(-1,3),axis=0))
        fragment=tight.crop((0,min(8,tight.height-4),tight.width,min(12,tight.height)))
        fragment=Image.blend(Image.new('RGB',fragment.size,bg),fragment,.32)
        remnant=Image.new('RGB',(tight.width+20,tight.height+35),bg)
        remnant.paste(fragment,(10,1));remnant.paste(tight,(10,22))
        sparse=Image.new('RGB',(2500,1300),bg);sparse.paste(original,((2500-original.width)//2,(1300-original.height)//2))
        # Match the existing public client's 1800-side JPEG preparation.
        sparse.thumbnail((1800,1800),Image.Resampling.LANCZOS)
        variants={'browser':original,'tight_boundary':tight,'clipped_caption':remnant,'large_margins':sparse}
        for condition,image in variants.items():
            stream=BytesIO();image.save(stream,format='JPEG',quality=95)
            name=f'{index:02d}-{condition}.jpg';(folder/name).write_bytes(stream.getvalue())
            result.append({'file':name,'family':row['family'],'category':categories[row['family']],
                           'text':row['phrase'],'condition':condition,'source_file':row['file'],
                           'source_capture_sha256':row['source_sha256']})
    manifest={'split':split,'method':'Actual saved browser crops. Tight boundaries, faint partial-caption strips and blank margins are simulated transformations, then JPEG quality 95 as in the public client. Not new photographs or fresh font families relative to previous browser audits.',
              'selection':'Three families per available category from the pre-existing gradient-held-out browser split. Validation and test2 families are disjoint. Test3 excludes the first test2 selection and uses a new fixed seed after the decorative-grid regression fix. No weight training.', 'rows':result}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('PREPARED',split,len(result),'images',len(selected),'families',flush=True)


def summarize(rows):
    named=[row for row in rows if row.get('named')]
    f1=[]
    for family in sorted({row['family'] for row in rows}):
        predict=lambda row:row.get('top3',[None])[0] if row.get('top3') else None
        tp=sum(row['family']==family and predict(row)==family for row in rows)
        fp=sum(row['family']!=family and predict(row)==family for row in rows)
        fn=sum(row['family']==family and predict(row)!=family for row in rows)
        f1.append(2*tp/max(2*tp+fp+fn,1))
    return {'images':len(rows),'top3_correct':sum(row['family'] in row.get('top3',[]) for row in rows),
            'names_returned':len(named),'correct_names':sum(row['named']==row['family'] for row in named),
            'top1_macro_f1_query_families':float(np.mean(f1)),
            'exact_text':sum(''.join(row.get('read_text','').casefold().split())==''.join(row['text'].casefold().split()) for row in rows),
            'rejected':sum(bool(row.get('error')) for row in rows),
            'median_ms':float(np.median([row['elapsed_ms'] for row in rows]))}


def classification_details(rows):
    families=sorted({row['family'] for row in rows});per_family={};confusion=defaultdict(lambda:defaultdict(int))
    for row in rows:
        predicted=row.get('top3',[None])[0] if row.get('top3') else '<no result>'
        confusion[row['family']][predicted]+=1
    for family in families:
        tp=confusion[family][family]
        fp=sum(counts.get(family,0) for truth,counts in confusion.items() if truth!=family)
        fn=sum(confusion[family].values())-tp
        per_family[family]={'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),
                            'f1':2*tp/max(2*tp+fp+fn,1),'support':tp+fn}
    return {'labels':'Queried families only; unrelated predicted names and rejected cases remain visible in confusion counts.',
            'per_family':per_family,'confusion_counts':{family:dict(counts) for family,counts in confusion.items()}}


def evaluate(split,mode):
    os.environ['FONTSENSE_CROP_MODE']=mode
    folder=DATA/split;manifest=json.loads((folder/'manifest.json').read_text());rows=[]
    for i,row in enumerate(manifest['rows']):
        started=time.perf_counter()
        try:
            result=identify(Image.open(folder/row['file']).convert('RGB'))
            identity=result.get('identified_font');name=identity['family'] if identity else None
            top=list(dict.fromkeys(([name] if name else [])+[r['family'] for r in result['matches']]))[:3]
            current={**row,'named':name,'top3':top,'read_text':result['text'],'needs_text':result['needs_text']}
        except ValueError as error:current={**row,'error':str(error)}
        current['elapsed_ms']=round((time.perf_counter()-started)*1000,1);rows.append(current)
        if i%8==0:print('EVALUATED',split,mode,i,'/',len(manifest['rows']),flush=True)
    report={'mode':mode,'split':split,'method':manifest['method'],'summary':summarize(rows),
            'top1_classification':classification_details(rows),
            'by_condition':{condition:summarize([r for r in rows if r['condition']==condition]) for condition in sorted({r['condition'] for r in rows})},'rows':rows}
    REPORT.mkdir(parents=True,exist_ok=True);(REPORT/f'{split}-{mode}.json').write_text(json.dumps(report,indent=2)+'\n')
    print('RESULT',json.dumps(report['summary']),flush=True)


def audit_references():
    from api.compact import text_crop
    from api.extended_match import render_reference
    from scripts.retrieval_data import REFERENCE_TEXTS
    changed=[];count=0
    for record in catalog():
        for phrase in REFERENCE_TEXTS:
            for case in (str.lower,str.title,str.upper):
                try:image=render_reference(record['id'],case(phrase),64)
                except (ValueError,OSError):continue
                os.environ['FONTSENSE_CROP_MODE']='legacy';old=text_crop(image)
                os.environ['FONTSENSE_CROP_MODE']='improved';new=text_crop(image)
                count+=1
                if old.size!=new.size or ImageChops.difference(old,new).getbbox():
                    changed.append({'id':record['id'],'family':record['family'],'phrase':case(phrase),
                                    'old_size':old.size,'new_size':new.size})
    report={'comparisons':count,'changed':len(changed),'changed_rows':changed,
            'method':'All available open-font styles, four enrollment phrases and lower/title/upper case; compare normalized pixels from legacy and selected crop behavior. Existing font binaries not committed.'}
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/'reference_parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print('REFERENCE RESULT',count,len(changed),flush=True)


def unknown_regression():
    from api.extended_match import primary_styles
    from scripts.independent_training_data import render
    os.environ['FONTSENSE_CROP_MODE']='improved'
    contract=json.loads((ROOT/'reports/independent_fonts/data_contract.json').read_text())
    audit=json.loads((ROOT.parent/'independent-fonts/audited-catalog.json').read_text())
    fonts={row['family']:row for row in reversed(audit['fonts'])}
    enrolled={row['family'] for row in primary_styles()};rows=[]
    for family,font in sorted(fonts.items()):
        if contract['new_family_splits'].get(family)!='unknown_test':continue
        assert family not in enrolled
        image=render(font,'Copper Meadow',73)
        sparse=Image.new('RGB',(2500,1300),'white');sparse.paste(image,(600,600))
        sparse.thumbnail((1800,1800),Image.Resampling.LANCZOS)
        for condition,current in [('tight',image),('large_margins',sparse)]:
            stream=BytesIO();current.save(stream,format='JPEG',quality=95)
            result=identify(Image.open(BytesIO(stream.getvalue())))
            name=result.get('identified_font')
            rows.append({'family':family,'condition':condition,'named':name['family'] if name else None,
                         'text':result['text'],'needs_text':result['needs_text']})
    report={'method':'Diagnostic reuse of the previously evaluated five non-enrolled independent families (four source groups), with a fresh phrase and tight/margin JPEG conditions. Not an untouched final test or global false-positive estimate.',
            'images':len(rows),'names_withheld':sum(row['named'] is None for row in rows),'rows':rows}
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/'unknown_regression.json').write_text(json.dumps(report,indent=2)+'\n')
    print('UNKNOWN RESULT',report['images'],report['names_withheld'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['validation','test2','test3'])
    parser.add_argument('--audit-references',action='store_true');parser.add_argument('--unknown-regression',action='store_true')
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--mode',choices=['legacy','improved'],default='improved');args=parser.parse_args()
    if args.audit_references:audit_references()
    elif args.unknown_regression:unknown_regression()
    elif not args.split:parser.error('--split is required for preparation or evaluation')
    elif args.prepare:prepare(args.split)
    else:evaluate(args.split,args.mode)
