import csv
import gzip
import hashlib
from io import BytesIO
from pathlib import Path
import tarfile

from PIL import Image, ImageDraw
import pytest

from fontsense.v2_data import sha256_file
from scripts.verify_product_dataset import verify_archive


def _shard(tmp_path: Path):
    image=Image.new('RGB',(320,128),'white')
    ImageDraw.Draw(image).text((50,50),'FontSense image integrity',fill='black')
    encoded=BytesIO();image.save(encoded,format='JPEG',quality=95)
    payload=encoded.getvalue()
    with Image.open(BytesIO(payload)) as decoded:
        pixel_hash=hashlib.sha256(decoded.convert('RGB').tobytes()).hexdigest()
    with tarfile.open(tmp_path/'one.tar','w') as archive:
        info=tarfile.TarInfo('one.jpg');info.size=len(payload)
        archive.addfile(info,BytesIO(payload))
    with gzip.open(tmp_path/'one.csv.gz','wt',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['member','image_sha256','pixel_sha256'])
        writer.writeheader()
        writer.writerow(dict(member='one.jpg',image_sha256=hashlib.sha256(payload).hexdigest(),pixel_sha256=pixel_hash))
    return dict(archive='one.tar',metadata='one.csv.gz',archive_sha256=sha256_file(tmp_path/'one.tar'),metadata_sha256=sha256_file(tmp_path/'one.csv.gz'),images=1,split='train')


def test_independent_check_reopens_and_hashes_actual_archive_image(tmp_path: Path):
    receipt=_shard(tmp_path)
    assert verify_archive(str(tmp_path),receipt)==dict(split='train',images=1)


def test_independent_check_rejects_damaged_archive(tmp_path: Path):
    receipt=_shard(tmp_path)
    (tmp_path/'one.tar').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='hash mismatch'):
        verify_archive(str(tmp_path),receipt)
