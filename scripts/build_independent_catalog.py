"""Audit explicitly selected indie sources and create reproducible font assets.

FontSpace archives are inspected in memory; no member-controlled path is ever
extracted. Only complete OFL/CC0 grants and basic Latin font programs qualify.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
import logging
from pathlib import Path
import random
import re
import sys
import unicodedata
from urllib.request import Request, urlopen
from urllib.error import URLError
import zipfile

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.extended_font_assets import make_subset
from api.extended_match import render_reference

logging.getLogger('fontTools.subset').setLevel(logging.ERROR)
DATA = ROOT.parent / 'independent-fonts'
PROBE = 'Hamburgefonts 012'
# Curated from enrollment renderings, before final query evaluation. Some
# independent files omit Panose/OS2 genre fields; missing metadata is not proof
# of a display design. These labels never enter the learned image encoder.
REVIEWED_CATEGORIES = {
    **dict.fromkeys(('Bluu Next','Coconat','Halibut','Mazius Display','Sinistre',
                     'Daubenton','VTF Victorianna','Ortica Angular','Ortica Linear'),'serif'),
    **dict.fromkeys(('Drybrush','Mister Brush','Crayon Libre','Erratic Cursive','Help Me'),'handwriting'),
}


def apply_category_review(fonts):
    for row in fonts:
        if row['family'] in REVIEWED_CATEGORIES:
            row['category'] = REVIEWED_CATEGORIES[row['family']]
            row['category_basis'] = 'Manual review of Latin enrollment rendering; not an encoder input'


def digest(data): return hashlib.sha256(data).hexdigest()
def key(name): return re.sub(r'[\s_-]+','',unicodedata.normalize('NFKC',name).casefold())


def fetch(url):
    cache = DATA/'downloads'/(digest(url.encode())+'.bin')
    metadata = cache.with_suffix('.json')
    if cache.exists() and metadata.exists(): return cache.read_bytes(),json.loads(metadata.read_text())['resolved_url']
    for attempt in range(3):
        try:
            with urlopen(Request(url,headers={'User-Agent':'FontSense-source-audit'}),timeout=60) as stream:
                data,resolved = stream.read(32*1024*1024+1),stream.geturl()
            break
        except (URLError,TimeoutError):
            if attempt==2: raise
    if len(data)>32*1024*1024: raise ValueError('Source package exceeds 32 MB')
    cache.parent.mkdir(parents=True,exist_ok=True)
    cache.write_bytes(data);metadata.write_text(json.dumps({'resolved_url':resolved}),encoding='utf-8')
    return data,resolved


def license_type(data):
    encoding='utf-16' if data.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig'
    text=' '.join(data.decode(encoding,errors='replace').upper().split())
    if 'SIL OPEN FONT LICENSE' in text and '1.1' in text: return 'OFL-1.1'
    if 'CC0 1.0 UNIVERSAL' in text: return 'CC0-1.0'
    raise ValueError('No complete OFL 1.1 or CC0 1.0 license')


def category(font,tags):
    lowered=' '.join(tags).lower()
    if 'handwriting' in lowered or 'hand drawn' in lowered or 'brush' in lowered: return 'handwriting','source tags'
    if 'sans serif' in lowered or 'sans-serif' in lowered: return 'sans_serif','source tags'
    if 'serif' in lowered: return 'serif','source tags'
    advances=[font.getlength(char) for char in 'iWm012']
    if max(advances)-min(advances)<.1: return 'monospace','equal reference advances'
    return 'display','decorative default; not an independently measured category'


def inspect(raw,license_data,source,member_name):
    license_id=license_type(license_data)
    with TTFont(BytesIO(raw),recalcTimestamp=False) as original:
        name=original['name']; family=name.getDebugName(16) or name.getDebugName(1)
        style=name.getDebugName(17) or name.getDebugName(2) or 'Regular'
        if not family: raise ValueError('Missing font family metadata')
        cmap=original.getBestCmap() or {}
        alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        if not all(ord(c) in cmap for c in alphabet): raise ValueError('Incomplete basic Latin alphabet')
        if len({cmap[ord(c)] for c in alphabet})<20: raise ValueError('Insufficient distinct Latin glyphs')
        if any(word in family.casefold() for word in ('braille','dummy text','notdef','runic','auseklis','naganoshi')):
            raise ValueError('Latin-encoded symbol/placeholder face outside text scope')
        weight=int(original['OS/2'].usWeightClass) if 'OS/2' in original else 400
        italic=bool(original['head'].macStyle&2) or ('OS/2' in original and bool(original['OS/2'].fsSelection&1))
        sfnt=original.sfntVersion
        panose=original['OS/2'].panose if 'OS/2' in original else None
        serif=panose.bSerifStyle if panose and panose.bFamilyType==2 else 0
        family_class=(original['OS/2'].sFamilyClass>>8) if 'OS/2' in original else 0
    source_id='i-'+digest((source['download_url']+'\0'+member_name).encode())[:18]
    converted=make_subset(raw,source_id)
    suffix='.otf' if sfnt=='OTTO' else '.ttf'
    file='independent-fonts/'+source_id+suffix
    local=DATA/'subsets'/(source_id+suffix);local.parent.mkdir(parents=True,exist_ok=True);local.write_bytes(converted)
    license_file='independent-fonts/'+source_id+'-LICENSE.txt'
    license_local=DATA/'licenses'/(source_id+'.txt');license_local.parent.mkdir(parents=True,exist_ok=True);license_local.write_bytes(license_data)
    runtime_source={**{k:v for k,v in source.items() if k not in ('family_hint','tags','license')},'id':source_id,'file':file,
                    'member':member_name,'source_sha256':digest(raw),'sha256':digest(converted),'license':license_id,
                    'license_file':license_file,'license_sha256':digest(license_data)}
    face=ImageFont.truetype(BytesIO(converted),48)
    try: axes=face.get_variation_axes()
    except OSError: axes=[]
    weight_axis=next((a for a in axes if a['name']==b'Weight'),None)
    weights=[w for w in (400,700) if weight_axis['minimum']<=w<=weight_axis['maximum']] if weight_axis else [weight]
    if not weights: weights=[int(weight_axis['default'])]
    rows=[]
    for w in weights:
        if weight_axis: face.set_variation_by_axes([w if a['name']==b'Weight' else a['default'] for a in axes])
        label,method=category(face,source.get('tags',[]))
        if label=='display' and serif in range(2,11): label,method='serif','font Panose metadata'
        elif label=='display' and (serif in range(11,16) or family_class==8): label,method='sans_serif','font Panose/OS2 metadata'
        elif label=='display' and family_class in (1,2,3,4,5,7): label,method='serif','font OS2 metadata'
        box=face.getbbox(PROBE)
        image=Image.new('L',(box[2]-box[0]+12,box[3]-box[1]+12),255)
        ImageDraw.Draw(image).text((6-box[0],6-box[1]),PROBE,font=face,fill=0)
        if image.width<=12 or image.height<=12: raise ValueError('Empty reference rendering')
        signature=digest(str(image.size).encode()+image.tobytes())
        rows.append({'id':source_id+'-'+str(w),'family':family,'family_id':key(family),'category':label,'category_basis':method,
                     'style':(('Bold' if w>=600 else 'Regular')+(' Italic' if italic else '')) if weight_axis else style,
                     'weight':w,'italic':italic,'variable_weight':bool(weight_axis),'file':file,'url':source['page'],
                     'source_id':source_id,'publisher':source['publisher'],'group':source['group'],'signature':signature,
                     'local_subset':str(local.relative_to(DATA))})
    return runtime_source,rows


def audit(source):
    results,errors=[],[]
    try:
        data,resolved=fetch(source['download_url'])
        if source['kind']=='archive':
            with zipfile.ZipFile(BytesIO(data)) as archive:
                infos=archive.infolist()
                if len(infos)>1000 or sum(i.file_size for i in infos)>64*1024*1024: raise ValueError('Oversized source archive')
                texts=[i.filename for i in infos if not i.is_dir() and any(t in i.filename.lower() for t in ('license','licence','ofl','copying')) and i.file_size<100000]
                licenses=[]
                for name in texts:
                    try: license_type(archive.read(name));licenses.append(name)
                    except ValueError: pass
                if not licenses: raise ValueError('Archive has no complete supported license')
                license_member=min(licenses,key=lambda p:(p.count('/'),len(p)))
                license_data=archive.read(license_member)
                for name in [i.filename for i in infos if i.filename.lower().endswith(('.ttf','.otf')) and i.file_size<16*1024*1024]:
                    try: results.append(inspect(archive.read(name),license_data,{**source,'download_url':resolved,'archive_sha256':digest(data),'license_member':license_member},name))
                    except Exception as error: errors.append({'page':source['page'],'member':name,'reason':str(error)})
        else:
            license_data,_=fetch(source['license_url'])
            if license_type(license_data)!=source['license']: raise ValueError('Declared and actual license disagree')
            results.append(inspect(data,license_data,source,source['member']))
    except Exception as error: errors.append({'page':source['page'],'member':source.get('member'),'reason':str(error)})
    return results,errors


def main(registry):
    DATA.mkdir(parents=True,exist_ok=True)
    raw=json.loads(registry.read_text())['sources']
    sources,fonts,excluded=[],[],[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for number,(found,errors) in enumerate(pool.map(audit,raw)):
            for source,records in found: sources.append(source);fonts.extend(records)
            excluded.extend(errors)
            if number%20==0: print('Audited sources',number,'/',len(raw),'valid faces',len(fonts),flush=True)
    # Source preference does not create separate classes for mirrored releases.
    fonts.sort(key=lambda r:(not r['publisher'].startswith('FontSpace'),r['family'],r['italic'],abs(r['weight']-400)))
    original_fonts=json.loads((ROOT/'artifacts/web/extended_catalog.json').read_text())['fonts']
    old_names={key(row['family']):row['family'] for row in original_fonts}
    old_signatures=defaultdict(set)
    old_splits=json.loads((ROOT/'artifacts/web/retrieval_encoder.json').read_text())['family_splits']
    for row in original_fonts:
        image=render_reference(row['id'],PROBE,48)
        old_signatures[digest(str(image.size).encode()+image.tobytes())].add(row['family'])
    accepted,seen,forced,groups= [],set(),{},defaultdict(set)
    for row in fonts:
        identity=key(row['family']);groups[identity].add(row['group'])
        if identity in old_names:
            excluded.append({'family':row['family'],'reason':'Family already in existing catalog','alias_of':old_names[identity]});continue
        duplicate=old_signatures.get(row['signature'])
        if duplicate:
            partitions={old_splits[name] for name in duplicate}
            if len(partitions)!=1: raise RuntimeError('Existing Latin groups cross partitions')
            forced[identity]=partitions.pop()
        if (identity,row['signature']) in seen: continue
        seen.add((identity,row['signature']));accepted.append(row)
    # Canonicalize spelling of mirrored family names, retaining source groups.
    names={}
    for row in accepted: names.setdefault(key(row['family']),row['family'])
    for row in accepted: row['family']=names[key(row['family'])]
    apply_category_review(accepted)
    used={row['source_id'] for row in accepted}
    sources=list({row['id']:row for row in sources if row['id'] in used}.values())
    result={'version':'independent-source-audit-1','sources':sources,'fonts':accepted,'excluded':excluded,
            'family_groups':{names[k]:sorted(v) for k,v in groups.items() if k in names},
            'forced_splits':{names[k]:v for k,v in forced.items() if k in names},
            'families':len(names),'styles':len(accepted),'scope':'Verified OFL/CC0 source audit. Data partitions are assigned before query generation. Native/proprietary font programs are not imported.'}
    (DATA/'audited-catalog.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('AUDIT COMPLETE',result['families'],'families',result['styles'],'styles',len(excluded),'exclusions',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--registry',type=Path,required=True)
    main(parser.parse_args().registry)
