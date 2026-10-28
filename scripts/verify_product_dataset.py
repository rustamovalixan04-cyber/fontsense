"""Independently reopen product archives and reproduce a training-only sample."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import sqlite3
import tarfile
import time

import pandas as pd
from PIL import Image, ImageStat

from fontsense.utils import load_json, project_root, save_json
from fontsense.v2_data import sha256_file
from fontsense.v3_dataset import render_product_image


def verify_archive(data: str, receipt: dict) -> dict:
    data=Path(data)
    archive_path=data/receipt['archive']
    metadata_path=data/receipt['metadata']
    if sha256_file(archive_path)!=receipt['archive_sha256'] or sha256_file(metadata_path)!=receipt['metadata_sha256']:
        raise ValueError('Archive or metadata hash mismatch')
    with gzip.open(metadata_path,'rt',encoding='utf-8',newline='') as handle:
        rows={row['member']:row for row in csv.DictReader(handle)}
    count=0
    names=set()
    with tarfile.open(archive_path,'r|') as archive:
        for member in archive:
            if not member.isfile() or member.name not in rows or member.name in names:
                raise ValueError(f'Unexpected or duplicate TAR member: {member.name}')
            names.add(member.name)
            row=rows[member.name]
            payload=archive.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest()!=row['image_sha256']:
                raise ValueError(f'Encoded hash mismatch: {member.name}')
            with Image.open(BytesIO(payload)) as image:
                image.load()
                if image.size!=(320,128) or image.format!='JPEG' or ImageStat.Stat(image.convert('L')).stddev[0]<2:
                    raise ValueError(f'Wrong dimensions, format, or blank image: {member.name}')
                if hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()!=row['pixel_sha256']:
                    raise ValueError(f'Decoded hash mismatch: {member.name}')
            count+=1
    if count!=receipt['images'] or names!=set(rows):
        raise ValueError('TAR/metadata image count mismatch')
    return dict(split=receipt['split'],images=count)


def main():
    root=project_root()
    data=root/'data/v3'
    summary=load_json(root/'reports/v3/dataset_validation_summary.json')
    config_path=root/'config/v3/dataset.json'
    config=load_json(config_path)
    if summary.get('status')!='passed' or summary.get('images_total')!=500000:
        raise ValueError('Full generation must finish successfully first')
    for path,key in [(config_path,'config_sha256'),(root/config['split_path'],'split_sha256'),(data/'manifests/manifest.sqlite','manifest_sha256')]:
        if sha256_file(path)!=summary[key]:
            raise ValueError(f'Frozen input hash mismatch: {path}')
    inventory=pd.read_csv(root/'reports/v3/shard_inventory.csv').to_dict('records')
    started=time.perf_counter()
    checked=0
    with ProcessPoolExecutor(max_workers=4) as executor:
        jobs=[executor.submit(verify_archive,str(data),receipt) for receipt in inventory]
        for index,job in enumerate(as_completed(jobs),1):
            checked+=job.result()['images']
            if index%50==0:
                print(f'Independently reopened {checked:,}/500,000 JPEGs',flush=True)
    if checked!=500000:
        raise ValueError('Independent image count differs from target')
    with sqlite3.connect(data/'manifests/manifest.sqlite') as connection:
        connection.row_factory=sqlite3.Row
        pool=connection.execute("SELECT * FROM samples WHERE split='train' ORDER BY sample_id").fetchall()
        sample=random.Random(42).sample(pool,100)
        duplicate_groups=connection.execute('SELECT COUNT(*) FROM (SELECT pixel_sha256 FROM samples GROUP BY pixel_sha256 HAVING COUNT(*)>1)').fetchone()[0]
    if duplicate_groups:
        raise ValueError('Exact decoded-pixel duplicates remain')
    reproduced=[]
    for row in sample:
        payload,metadata=render_product_image(str(root/row['font_path']),int(row['seed']),config)
        if hashlib.sha256(payload).hexdigest()!=row['image_sha256'] or metadata['pixel_sha256']!=row['pixel_sha256']:
            raise ValueError(f'Reproducibility failed: {row["sample_id"]}')
        reproduced.append(dict(sample_id=row['sample_id'],image_sha256=row['image_sha256']))
    result=dict(status='passed',images_reopened=checked,missing_images=0,corrupted_images=0,blank_images=0,wrong_dimensions=0,exact_pixel_duplicate_groups=duplicate_groups,reproduced_training_images=len(reproduced),reproduced_samples=reproduced,manifest_sha256=summary['manifest_sha256'],split_sha256=summary['split_sha256'],elapsed_seconds=time.perf_counter()-started,test_images_model_evaluated=0,scope='Image-integrity and reproduction checks only; no model inference')
    save_json(result,root/'reports/v3/independent_image_verification.json')
    print(json.dumps({key:value for key,value in result.items() if key!='reproduced_samples'},indent=2))


if __name__=='__main__':
    main()
