"""Reproducible synthetic mixed-layout checks, separate from capstone evidence."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'ocr-experiments/posters'
MANIFEST = ROOT / 'reports/poster_matching/query_manifest.json'


def prepare():
    from api.extended_match import primary_styles
    from api.font_match import render_reference
    from api.extended_match import render_reference as extended_render
    contract = json.loads((ROOT / 'artifacts/web/retrieval_encoder.json').read_text())
    seen = set()
    browser_path = ROOT.parent / 'retrieval-training/browser/samples.json'
    if browser_path.exists():
        seen = {row['family'] for row in json.loads(browser_path.read_text()) if row['split'] in ('test','test2')}
    for group in contract['duplicate_groups']:
        if seen & set(group): seen.update(group)
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
    else:
        pools = {}
        for font in primary_styles():
            if contract['family_splits'][font['family']] == 'test' and font['family'] not in seen:
                pools.setdefault(font['category'], []).append(font)
        rng = random.Random(92061)
        chosen = []
        for category, count in (('serif',3),('sans_serif',3),('display',3),('handwriting',2),('monospace',1)):
            fonts = sorted(pools[category], key=lambda row:row['family'])
            rng.shuffle(fonts)
            assert len(fonts) >= count
            chosen.extend(fonts[:count])
        rows = []
        for font in chosen:
            for phrase, dark in (('Amber Grove', False), ('SUMMER HORIZON', True)):
                rows.append({'file': f'{len(rows):03d}.png', 'font_id': font['id'], 'family': font['family'],
                             'category': font['category'], 'text': phrase, 'dark':dark})
        manifest = {'kind':'synthetic mixed-layout graphics; not browser or camera photographs', 'seed':92061,
                    'protocol':'12 reserved test families outside the earlier diagnostic/fresh browser query families and detected shared-Latin groups. No new neural gradient training. Queries chosen before this comparison.', 'rows':rows}
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest,indent=2) + '\n',encoding='utf-8')
    DATA.mkdir(parents=True,exist_ok=True)
    for row in manifest['rows']:
        foreground, background = ('#f0dfb6','#303d51') if row['dark'] else ('#302939','#f4ece1')
        canvas = Image.new('RGB',(640,460),background)
        def paste_text(text_image, y, size):
            text_image = text_image.convert('L')
            text_image.thumbnail(size, Image.Resampling.LANCZOS)
            canvas.paste(Image.new('RGB',text_image.size,foreground),((640-text_image.width)//2,y),ImageOps.invert(text_image))
        paste_text(render_reference('arimo','DESIGN COLLECTION'),70,(510,22))
        paste_text(extended_render(row['font_id'],row['text'],96),155,(580,105))
        drawing = ImageDraw.Draw(canvas)
        drawing.line([(320,440),(320,340)],fill=foreground,width=3)
        for box in ((285,280,355,365),(245,325,325,365),(315,325,395,365),(263,293,325,360),(315,293,377,360)):
            drawing.ellipse(box,outline=foreground,width=3)
        canvas.save(DATA / row['file'])
    print('Prepared',len(manifest['rows']),'synthetic posters from',len({r['family'] for r in manifest['rows']}),'families.',flush=True)


def evaluate(args):
    from api.font_match import identify
    manifest = json.loads(MANIFEST.read_text())
    rows = []
    for number, query in enumerate(manifest['rows']):
        started = time.perf_counter()
        result = identify(Image.open(DATA / query['file']).convert('RGB'))
        named = result.get('identified_font')
        names = ([named['family']] if named else []) + [font['family'] for font in result['matches']]
        rows.append({**query, 'read_text':result['text'], 'named':named['family'] if named else None,
                     'top3':list(dict.fromkeys(names))[:3], 'category_result':result['category'],
                     'needs_text':result['needs_text'],'elapsed_ms':round((time.perf_counter()-started)*1000,1),
                     'regions':result.get('text_regions',[])})
        print('Evaluated poster',number+1,'/',len(manifest['rows']),flush=True)
    clean = lambda text: ''.join(text.casefold().split())
    accepted = [r for r in rows if r['named']]
    summary = {'images':len(rows), 'families':len({r['family'] for r in rows}),
               'exact_text':sum(clean(r['text']) == clean(r['read_text']) for r in rows),
               'top3_correct':sum(r['family'] in r['top3'] for r in rows),
               'category_correct':sum(r['category'] == r['category_result'] for r in rows),
               'names_returned':len(accepted),'correct_names':sum(r['named']==r['family'] for r in accepted),
               'needs_text':sum(r['needs_text'] for r in rows)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'engine':args.engine,'kind':manifest['kind'],'summary':summary,'rows':rows},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--fresh',action='store_true')
    parser.add_argument('--final',action='store_true')
    parser.add_argument('--api-root',type=Path,default=ROOT)
    parser.add_argument('--engine',choices=('baseline','candidate'),default='candidate')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.fresh or args.final:
        initial = MANIFEST
        MANIFEST = ROOT / ('reports/poster_matching/final_query_manifest.json' if args.final else 'reports/poster_matching/fresh_query_manifest.json')
        DATA = ROOT.parent / ('ocr-experiments/final-posters' if args.final else 'ocr-experiments/fresh-posters')
        if args.prepare and not MANIFEST.exists():
            source = json.loads(initial.read_text())
            source['protocol'] = 'Fresh words on the same 12 diagnostically inspected reserved families. New query images were prepared after detector, OCR refinement and matching rules were locked. Families are excluded from new gradient training but are necessarily enrolled in the search index. Not an unseen-family claim relative to diagnostic inspection.'
            light,dark = ('Maple Harbor','SPRING MEADOW') if args.final else ('Cedar Valley','WINTER BLOSSOM')
            source['rows'] = [{**row, 'text':dark if row['dark'] else light} for row in source['rows']]
            MANIFEST.write_text(json.dumps(source,indent=2)+'\n',encoding='utf-8')
    sys.path.insert(0,str(args.api_root.resolve()))
    if args.prepare: prepare()
    else:
        if args.output is None: parser.error('--output is required for evaluation')
        evaluate(args)
