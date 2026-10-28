"""Create inspectable samples from completed product shards; never evaluate a model."""
from pathlib import Path
import csv
import gzip
from io import BytesIO
import json
import tarfile

from PIL import Image, ImageDraw, ImageFont, ImageOps

from fontsense.utils import project_root
from fontsense.v3_dataset import CLASSES


def main():
    root=project_root()
    data=root/'data/v3'
    selected={category:[] for category in CLASSES}
    families={category:set() for category in CLASSES}
    for receipt_path in sorted((data/'manifests/receipts').glob('train-*.json')):
        receipt=json.loads(receipt_path.read_text())
        with gzip.open(data/receipt['metadata'],'rt',encoding='utf-8',newline='') as handle:
            for row in csv.DictReader(handle):
                category=row['category']
                if len(selected[category])<12 and row['family'] not in families[category]:
                    selected[category].append(row);families[category].add(row['family'])
        if all(len(rows)==12 for rows in selected.values()):
            break
    destination=root/'reports/v3/contact_sheets'
    destination.mkdir(parents=True,exist_ok=True)
    font=ImageFont.load_default(size=14)
    for category,rows in selected.items():
        if len(rows)<12:
            raise ValueError(f'Not enough complete {category} samples')
        sheet=Image.new('RGB',(1280,540),'#f0f1f4')
        draw=ImageDraw.Draw(sheet)
        draw.text((12,8),f'FontSense product V3 - {category} - actual rendered training samples',font=font,fill='black')
        for index,row in enumerate(rows):
            with tarfile.open(data/row['shard'],'r') as archive:
                stream=archive.extractfile(row['member'])
                image=Image.open(BytesIO(stream.read())).convert('RGB')
            x=(index%4)*320;y=36+(index//4)*164
            sheet.paste(ImageOps.contain(image,(312,126)),(x+4,y))
            draw.text((x+6,y+128),row['family'][:38],font=font,fill='black')
            draw.text((x+6,y+145),row['rendered_text'][:38],font=font,fill='#394150')
        sheet.save(destination/f'{category}.png',format='PNG')
        print(destination/f'{category}.png')


if __name__=='__main__':
    main()
