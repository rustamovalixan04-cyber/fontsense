"""Deterministic, resumable image shards for the 500k product dataset."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from functools import lru_cache
import gzip
import hashlib
from io import BytesIO, TextIOWrapper
import json
import math
from pathlib import Path
import random
import shutil
import sqlite3
import tarfile
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageStat

from .utils import project_root, load_json, save_json
from .v2_data import sha256_file

CLASSES = ('display', 'handwriting', 'monospace', 'sans_serif', 'serif')
WORDS = (
    'Agile Alpine Amber Autumn Balance Basic Bloom Bold Book Brave Bright Calm '
    'Canvas Cedar Circle City Clear Cloud Coastal Color Copper Craft Creative '
    'Crisp Culture Daily Dawn Deep Design Detail Digital Dream Earth Easy Echo '
    'Elegant Evening Explore Fine Flow Focus Forest Form Frame Free Fresh Future '
    'Garden Gentle Golden Good Grace Grand Graphic Green Grid Happy Harbor Hello '
    'Home Honest Horizon Idea Indigo Island Ivory Kind Lake Layout Leaf Letter '
    'Light Line Local Logo Lunar Maple Meadow Message Modern Moon Morning Motion '
    'Natural Neat New Night Note Ocean Open Orange Origin Paper Path Pattern Peace '
    'Pearl People Pixel Plain Planet Point Print Quiet Quick Rain Read Reading '
    'River Road Rock Rose Round Sand Scene Sea Shadow Shape Sharp Shine Simple '
    'Small Smart Smooth Snow Soft Solar Solid Sound Source Space Spring Stone '
    'Story Studio Style Summer Sun Text Timber Tiny Trail Travel Tree True Type '
    'Unity Urban Valley View Violet Vision Visual Voice Warm Water Wave White '
    'Wild Wind Winter Wonder Wood Word Work World Young'
).split()
PHRASES = tuple([word for word in WORDS] + [f'{left} {right}' for left in WORDS for right in WORDS if left != right and len(left)+len(right) <= 17])
FIELDS = ('sample_id','family','category','split','shard','member','seed','font_path','rendered_text','applied_effects','pixel_sha256','image_sha256','dhash','brightness','contrast')
_STATE = {}


def validate_config(config: dict) -> None:
    if sum(config['split_image_counts'].values()) != config['target_images']:
        raise ValueError('Split counts must add up to the target image count')
    size = int(config['images_per_shard'])
    if size <= 0 or size % len(CLASSES):
        raise ValueError('Each shard must contain an equal number from all five categories')
    if set(config['split_image_counts']) != {'train','validation','test'}:
        raise ValueError('Exactly train, validation, and test splits are required')
    if any(count <= 0 or count % size for count in config['split_image_counts'].values()):
        raise ValueError('Each split must contain complete shards')
    if config['storage_format'] != 'JPEG' or not 80 <= config['jpeg_quality'] <= 98:
        raise ValueError('Use readable high-quality JPEG storage')
    if config['rotation_degrees'] > 4 or config['max_shear'] > .025:
        raise ValueError('Text distortions must remain mild')


@lru_cache(maxsize=48)
def _font(path: str, size: int, weight_step: int):
    font = ImageFont.truetype(path, size)
    try:
        axes = font.get_variation_axes()
        values = []
        for axis in axes:
            name = axis['name'].decode('utf-8', errors='ignore').lower()
            value = axis['default']
            if 'weight' in name:
                lo, hi = max(300, axis['minimum']), min(700, axis['maximum'])
                if lo <= hi:
                    value = lo + weight_step / 4 * (hi-lo)
            values.append(value)
        font.set_variation_by_axes(values)
    except (OSError, AttributeError):
        pass
    return font


def render_product_image(font_path: str, seed: int, config: dict) -> tuple[bytes, dict]:
    """The renderer receives no category or split label."""
    rng = random.Random(seed)
    text = rng.choice(PHRASES)
    case = rng.choice(('original','original','upper','lower'))
    if case == 'upper':
        text = text.upper()
    elif case == 'lower':
        text = text.lower()
    if rng.random() < .18:
        text = f'{rng.randrange(10,100)} {text}'
    width, height = config['image_size']
    dark = rng.random() < config['dark_background_probability']
    background = tuple(rng.randint(6,42) if dark else rng.randint(228,255) for _ in range(3))
    foreground = tuple(rng.randint(190,255) if dark else rng.randint(0,90) for _ in range(3))
    requested_size = rng.randint(*config['font_size_range'])
    spacing = rng.choice((0,0,0,1,2))
    stroke = int(rng.random() < .12)
    weight_step = rng.randrange(5)
    size = requested_size
    for _ in range(6):
        font = _font(font_path, size, weight_step)
        bbox = font.getbbox(text, stroke_width=stroke)
        advance = sum(font.getlength(c) for c in text) + spacing*(len(text)-1) if spacing else font.getlength(text)
        text_width = max(advance,bbox[2]+spacing*(len(text)-1))-min(0,bbox[0])
        text_height = bbox[3]-bbox[1]
        factor = min(width*config['max_text_width_fraction']/max(text_width,1), height*config['max_text_height_fraction']/max(text_height,1))
        if factor >= 1:
            break
        size = max(10, math.floor(size*factor*.98))
    else:
        raise ValueError(f'Could not fit readable text with {font_path}')
    dx, dy = rng.randint(-10,10), rng.randint(-4,4)
    x, y = (width-text_width)/2+dx-min(0,bbox[0]), (height-text_height)/2-bbox[1]+dy
    layer = Image.new('RGBA',(width,height))
    draw = ImageDraw.Draw(layer)
    if spacing:
        for character in text:
            if not character.isspace():
                draw.text((x,y), character, font=font, fill=(*foreground,255))
            x += font.getlength(character)+spacing
    else:
        draw.text((x,y), text, font=font, fill=(*foreground,255))
    if stroke:
        # Bounded raster dilation avoids pathological FreeType outline-stroke
        # allocations in decorative fonts such as Rubik Pixels.
        alpha = layer.getchannel('A').filter(ImageFilter.MaxFilter(2*stroke+1))
        layer = Image.new('RGBA',layer.size,(*foreground,0))
        layer.putalpha(alpha)
    angle = round(rng.uniform(-config['rotation_degrees'],config['rotation_degrees']),4)
    layer = layer.rotate(angle,resample=Image.Resampling.BICUBIC)
    ink_bbox = layer.getbbox()
    if ink_bbox is None or ink_bbox[0] <= 1 or ink_bbox[1] <= 1 or ink_bbox[2] >= width-1 or ink_bbox[3] >= height-1:
        raise ValueError('Empty or clipped rendered text')
    gradient = rng.random() < config['gradient_probability']
    canvas = Image.new('RGB',(width,height),background)
    if gradient:
        values = np.tile(np.asarray(background,dtype=np.float32),(height,width,1))
        values += np.linspace(-7,7,width,dtype=np.float32)[None,:,None]
        canvas = Image.fromarray(np.clip(values,0,255).astype('uint8'))
    canvas = Image.alpha_composite(canvas.convert('RGBA'),layer).convert('RGB')
    shear = round(rng.uniform(-config['max_shear'],config['max_shear']),5) if rng.random() < config['perspective_probability'] else None
    if shear is not None:
        canvas = canvas.transform(canvas.size,Image.Transform.AFFINE,(1,shear,-shear*height/2,0,1,0),resample=Image.Resampling.BICUBIC,fillcolor=background)
    blur = round(rng.uniform(*config['blur_radius_range']),3) if rng.random() < config['blur_probability'] else None
    if blur:
        canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
    noise = round(rng.uniform(*config['noise_std_range']),3) if rng.random() < config['noise_probability'] else None
    if noise:
        values = np.asarray(canvas,dtype=np.float32)+np.random.default_rng(seed+31_337).normal(0,noise,(height,width,3))
        canvas = Image.fromarray(np.clip(values,0,255).astype('uint8'))
    quality = rng.randint(80,config['jpeg_quality'])
    encoded = BytesIO()
    canvas.save(encoded,format='JPEG',quality=quality,subsampling=0)
    payload = encoded.getvalue()
    with Image.open(BytesIO(payload)) as decoded:
        decoded.load()
        gray = decoded.convert('L')
        stats = ImageStat.Stat(gray)
        if decoded.size != (width,height) or stats.stddev[0] < 2:
            raise ValueError('Encoded sample is blank or unreadable')
        pixels = decoded.convert('RGB').tobytes()
        small = np.asarray(gray.resize((9,8)),dtype=np.uint8)
        bits = np.packbits((small[:,1:] > small[:,:-1]).reshape(-1)).tobytes().hex()
    effects = dict(background='dark' if dark else 'light',background_rgb=background,foreground_rgb=foreground,requested_font_size=requested_size,actual_font_size=size,letter_spacing=spacing,stroke_width=stroke,weight_step=weight_step,rotation_degrees=angle,translation=[dx,dy],gradient=gradient,shear=shear,blur_radius=blur,noise_std=noise,jpeg_quality=quality,case_style=case,image_size=[width,height],ink_bbox=ink_bbox)
    return payload, dict(rendered_text=text,applied_effects=json.dumps(effects,sort_keys=True),pixel_sha256=hashlib.sha256(pixels).hexdigest(),image_sha256=hashlib.sha256(payload).hexdigest(),dhash=bits,brightness=stats.mean[0],contrast=stats.stddev[0])


def _initialize_worker(root: str, config_path: str, split_path: str, output_dir: str):
    global _STATE
    config = load_json(config_path)
    frame = pd.read_csv(split_path,keep_default_na=False)
    if frame.groupby('family')['split'].nunique().max() != 1:
        raise ValueError('Family overlap in the source split')
    pools = {}
    for split in config['split_image_counts']:
        for category in CLASSES:
            pool = frame.loc[(frame['split']==split)&(frame['category']==category)].sort_values('family').to_dict('records')
            if not pool:
                raise ValueError(f'No families for {split}/{category}')
            random.Random(config['seed']+CLASSES.index(category)*10007).shuffle(pool)
            pools[(split,category)] = pool
    output = Path(output_dir)
    repairs = load_json(output/'repairs.json') if (output/'repairs.json').is_file() else {}
    _STATE = dict(root=Path(root),config=config,pools=pools,output=output,repairs=repairs,plan_sha256=hashlib.sha256((sha256_file(config_path)+sha256_file(split_path)+sha256_file(__file__)).encode()).hexdigest())


def _shard_paths(split: str, index: int) -> tuple[Path,Path,Path]:
    output = _STATE['output']
    key = f'{split}-{index:05d}'
    return output/'shards'/split/f'{key}.tar', output/'manifests'/f'{key}.csv.gz', output/'manifests/receipts'/f'{key}.json'


def _render_shard(task: tuple[str,int]) -> dict:
    split, index = task
    root, config = _STATE['root'], _STATE['config']
    if (_STATE['output']/'STOP').exists():
        return dict(status='intentionally_stopped',images=0,bytes=0)
    archive, metadata, receipt_path = _shard_paths(split,index)
    repairs = {key:value for key,value in _STATE['repairs'].items() if key.startswith(f'{split}-{index:05d}-')}
    signature = hashlib.sha256((_STATE['plan_sha256']+json.dumps(repairs,sort_keys=True)).encode()).hexdigest()
    if receipt_path.is_file() and archive.is_file() and metadata.is_file():
        receipt = load_json(receipt_path)
        if receipt['signature'] == signature and sha256_file(archive)==receipt['archive_sha256'] and sha256_file(metadata)==receipt['metadata_sha256']:
            return receipt
    for path in (archive,metadata,receipt_path):
        path.parent.mkdir(parents=True,exist_ok=True)
    count = config['images_per_shard']
    archive_tmp, metadata_tmp = archive.with_suffix('.tar.tmp'), metadata.with_suffix('.gz.tmp')
    with tarfile.open(archive_tmp,'w',format=tarfile.USTAR_FORMAT) as tar, metadata_tmp.open('wb') as binary:
        with gzip.GzipFile(filename='',fileobj=binary,mode='wb',mtime=0) as gz, TextIOWrapper(gz,encoding='utf-8',newline='') as text_handle:
            writer = csv.DictWriter(text_handle,fieldnames=FIELDS,lineterminator='\n')
            writer.writeheader()
            for offset in range(count):
                category = CLASSES[offset % len(CLASSES)]
                rank = index*(count//len(CLASSES))+offset//len(CLASSES)
                pool = _STATE['pools'][(split,category)]
                family = pool[rank % len(pool)]
                sample_id = f'{split}-{index:05d}-{offset:04d}'
                nonce = int(repairs.get(sample_id,0))
                seed = config['seed']+('train','validation','test').index(split)*1_000_000_007+rank*1009+nonce*10_000_000_019
                font_path = Path(family['path'])
                font_path = font_path if font_path.is_absolute() else root/font_path
                try:
                    payload, metrics = render_product_image(str(font_path),seed,config)
                except Exception as exc:
                    raise RuntimeError(f'Render failed for {sample_id}, family={family["family"]}, seed={seed}: {type(exc).__name__}: {exc}') from exc
                member = f'{sample_id}.jpg'
                info = tarfile.TarInfo(member)
                info.size,info.mtime,info.uid,info.gid,info.mode = len(payload),0,0,0,0o644
                tar.addfile(info,BytesIO(payload))
                writer.writerow(dict(sample_id=sample_id,family=family['family'],category=category,split=split,shard=str(archive.relative_to(_STATE['output']).as_posix()),member=member,seed=seed,font_path=family['path'],**metrics))
    archive_tmp.replace(archive)
    metadata_tmp.replace(metadata)
    receipt = dict(status='completed',split=split,shard_index=index,images=count,signature=signature,archive=archive.relative_to(_STATE['output']).as_posix(),metadata=metadata.relative_to(_STATE['output']).as_posix(),archive_sha256=sha256_file(archive),metadata_sha256=sha256_file(metadata),bytes=archive.stat().st_size+metadata.stat().st_size)
    save_json(receipt,receipt_path)
    return receipt


def validate_manifest(output: Path, tasks: list[tuple[str,int]], config: dict, split_path: Path) -> tuple[dict,dict]:
    database = output/'manifests/manifest.sqlite'
    connection = sqlite3.connect(database)
    connection.execute('DROP TABLE IF EXISTS samples')
    connection.execute('CREATE TABLE samples (sample_id TEXT PRIMARY KEY,family TEXT,category TEXT,split TEXT,shard TEXT,member TEXT,seed INTEGER,font_path TEXT,rendered_text TEXT,applied_effects TEXT,pixel_sha256 TEXT,image_sha256 TEXT,dhash TEXT,brightness REAL,contrast REAL)')
    for split,index in tasks:
        _, metadata, receipt_path = _shard_paths(split,index)
        receipt = load_json(receipt_path)
        if sha256_file(output/receipt['archive']) != receipt['archive_sha256'] or sha256_file(metadata) != receipt['metadata_sha256']:
            raise ValueError(f'Shard integrity failed: {split}/{index}')
        with gzip.open(metadata,'rt',encoding='utf-8',newline='') as handle:
            rows=list(csv.DictReader(handle))
        if len(rows) != config['images_per_shard']:
            raise ValueError('Shard manifest has the wrong row count')
        connection.executemany('INSERT INTO samples VALUES ('+','.join('?' for _ in FIELDS)+')',[[row[key] for key in FIELDS] for row in rows])
    connection.execute('CREATE INDEX pixel_hash_index ON samples(pixel_sha256)')
    connection.commit()
    counts = dict(connection.execute('SELECT split,COUNT(*) FROM samples GROUP BY split'))
    category_counts = dict(connection.execute('SELECT category,COUNT(*) FROM samples GROUP BY category'))
    expected = {category:config['target_images']//len(CLASSES) for category in CLASSES}
    if counts != config['split_image_counts'] or category_counts != expected:
        raise ValueError(f'Dataset counts differ from the plan: {counts}, {category_counts}')
    overlap = connection.execute('SELECT family FROM samples GROUP BY family HAVING COUNT(DISTINCT split)>1').fetchall()
    if overlap:
        raise ValueError('Family overlap detected in rendered samples')
    balance = pd.read_sql_query("SELECT category, AVG(json_extract(applied_effects,'$.background')='dark') AS dark_background, AVG(json_extract(applied_effects,'$.blur_radius') IS NOT NULL) AS blur, AVG(json_extract(applied_effects,'$.gradient')) AS gradient, AVG(json_extract(applied_effects,'$.noise_std') IS NOT NULL) AS noise FROM samples GROUP BY category",connection)
    if any(float(balance[key].max()-balance[key].min())>.02 for key in ('dark_background','blur','gradient','noise')) and config['target_images']>=10000:
        raise ValueError('Serious category-dependent effects imbalance')
    frozen = pd.read_csv(split_path)
    actual = pd.read_sql_query('SELECT DISTINCT family,category,split FROM samples',connection).sort_values('family').reset_index(drop=True)
    if not actual.equals(frozen[['family','category','split']].sort_values('family').reset_index(drop=True)):
        raise ValueError('Rendered family assignments do not match the frozen V3 split')
    duplicates = connection.execute('SELECT pixel_sha256 FROM samples GROUP BY pixel_sha256 HAVING COUNT(*)>1').fetchall()
    replacements = {}
    for (digest,) in duplicates:
        group = connection.execute("SELECT sample_id,split FROM samples WHERE pixel_sha256=? ORDER BY CASE split WHEN 'test' THEN 0 WHEN 'validation' THEN 1 ELSE 2 END,sample_id",(digest,)).fetchall()
        for sample_id,_ in group[1:]:
            replacements[sample_id] = 1
    brightness,contrast = connection.execute('SELECT AVG(brightness),AVG(contrast) FROM samples').fetchone()
    connection.close()
    return dict(images_total=sum(counts.values()),images_per_split=counts,images_per_category=category_counts,families_total=len(frozen),family_overlap_count=0,exact_duplicate_groups=len(duplicates),mean_brightness=brightness,mean_contrast=contrast,effect_rates_per_category=balance.to_dict('records'),manifest_sha256=sha256_file(database)), replacements


def generate(config_path: Path, split_path: Path, output: Path, report_dir: Path, workers: int=4, limit_shards: int|None=None) -> dict:
    config_path,split_path,output,report_dir = (Path(path).resolve() for path in (config_path,split_path,output,report_dir))
    config=load_json(config_path)
    validate_config(config)
    root=project_root()
    output.mkdir(parents=True,exist_ok=True)
    report_dir.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output).free < config['min_free_disk_gib']*1024**3:
        raise RuntimeError('Insufficient free disk space for safe dataset generation')
    split_hash=sha256_file(split_path)
    frozen=pd.read_csv(split_path,keep_default_na=False)
    if 'font_sha256' in frozen:
        for row in frozen[['path','font_sha256']].drop_duplicates().itertuples(index=False):
            font_path=Path(row.path)
            font_path=font_path if font_path.is_absolute() else root/font_path
            if sha256_file(font_path)!=row.font_sha256:
                raise ValueError(f'Font changed after the audit: {row.path}')
    _initialize_worker(str(root),str(config_path),str(split_path),str(output))
    tasks=[(split,index) for split,count in config['split_image_counts'].items() for index in range(count//config['images_per_shard'])]
    selected_tasks=tasks[:limit_shards] if limit_shards else tasks
    started=time.perf_counter()
    receipts=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=_initialize_worker,initargs=(str(root),str(config_path),str(split_path),str(output))) as executor:
        jobs={executor.submit(_render_shard,task):task for task in selected_tasks}
        for job in as_completed(jobs):
            try:
                receipts.append(job.result())
            except Exception as exc:
                (output/'STOP').write_text('Automatic safety stop after a shard error; inspect progress.json before resuming.',encoding='utf-8')
                for pending in jobs:
                    pending.cancel()
                save_json(dict(status='failed',failed_shard=list(jobs[job]),error=str(exc),images_completed=sum(row['images'] for row in receipts),target_images=config['target_images']),report_dir/'progress.json')
                raise
            completed=sum(row['images'] for row in receipts)
            status=dict(status='generating',images_completed=completed,target_images=config['target_images'],shards_completed=len(receipts),elapsed_seconds=time.perf_counter()-started,stored_bytes=sum(row['bytes'] for row in receipts))
            save_json(status,report_dir/'progress.json')
            print(f"Rendered/verified {completed:,}/{config['target_images']:,} images; {len(receipts)} shards",flush=True)
            if shutil.disk_usage(output).free < config['min_free_disk_gib']*1024**3:
                raise RuntimeError('Generation paused at the free-space reserve')
    if any(row['status']=='intentionally_stopped' for row in receipts) or (output/'STOP').exists():
        result=dict(status='intentionally_stopped',target_images=config['target_images'],images_completed=sum(row['images'] for row in receipts))
        save_json(result,report_dir/'progress.json')
        return result
    if limit_shards and len(selected_tasks)<len(tasks):
        result=dict(status='pilot_completed',images_completed=sum(row['images'] for row in receipts),target_images=config['target_images'],seconds=time.perf_counter()-started,stored_bytes=sum(row['bytes'] for row in receipts))
        save_json(result,report_dir/'pilot_summary.json')
        return result
    repairs_path=output/'repairs.json'
    repairs=load_json(repairs_path) if repairs_path.is_file() else {}
    replaced=0
    for _ in range(5):
        _initialize_worker(str(root),str(config_path),str(split_path),str(output))
        result,replacements=validate_manifest(output,tasks,config,split_path)
        if not replacements:
            break
        for key in replacements:
            repairs[key]=int(repairs.get(key,0))+1
        replaced+=len(replacements)
        save_json(repairs,repairs_path)
        _initialize_worker(str(root),str(config_path),str(split_path),str(output))
        affected={('-'.join(key.split('-')[:-2]),int(key.split('-')[-2])) for key in replacements}
        for task in sorted(affected):
            _render_shard(task)
    else:
        raise ValueError('Exact duplicate repair did not converge; dataset is not approved')
    if sha256_file(split_path)!=split_hash:
        raise ValueError('Frozen product split changed during generation')
    inventory=[load_json(_shard_paths(*task)[2]) for task in tasks]
    pd.DataFrame(inventory).to_csv(report_dir/'shard_inventory.csv',index=False)
    save_json(repairs,report_dir/'duplicate_repairs.json')
    import platform
    from PIL import __version__ as pillow_version
    result.update(status='passed',target_images=config['target_images'],all_encoded_images_opened=True,missing_images=0,blank_images=0,corrupted_images=0,duplicate_candidates_replaced=replaced,split_sha256=split_hash,config_sha256=sha256_file(config_path),renderer_sha256=sha256_file(__file__),stored_bytes=sum(row['bytes'] for row in inventory),elapsed_seconds=time.perf_counter()-started,test_model_evaluations=0,software=dict(python=platform.python_version(),numpy=np.__version__,pillow=pillow_version))
    save_json(result,report_dir/'dataset_validation_summary.json')
    save_json(dict(status='completed',images_completed=result['images_total'],target_images=config['target_images']),report_dir/'progress.json')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=project_root()/'config/v3/dataset.json')
    parser.add_argument('--split',type=Path,default=None)
    parser.add_argument('--output',type=Path,default=project_root()/'data/v3')
    parser.add_argument('--reports',type=Path,default=project_root()/'reports/v3')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--limit-shards',type=int,default=None)
    args=parser.parse_args()
    args.split = args.split or project_root()/load_json(args.config).get('split_path','data/v3/frozen_family_split.csv')
    print(json.dumps(generate(args.config,args.split,args.output,args.reports,args.workers,args.limit_shards),indent=2))
