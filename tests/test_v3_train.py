from pathlib import Path
import csv
import gzip
from io import BytesIO
import tarfile

import pandas as pd
from PIL import Image, ImageDraw
import pytest
import torch

from fontsense.v3_train import ProductShardDataset, run_epoch, _update_patience, dataset_content_identity
from fontsense.cnn_model import FontSenseCNN


def test_product_training_loader_refuses_test_shards(tmp_path: Path):
    with pytest.raises(ValueError,match='forbids test-shard access'):
        ProductShardDataset(tmp_path,pd.DataFrame(), 'test',(192,64))


def test_shard_loader_returns_only_pixels_as_model_features(tmp_path: Path):
    image=Image.new('RGB',(320,128),'white')
    ImageDraw.Draw(image).text((30,40),'FontSense',fill='black')
    encoded=BytesIO();image.save(encoded,format='JPEG')
    with tarfile.open(tmp_path/'sample.tar','w') as archive:
        info=tarfile.TarInfo('sample.jpg');info.size=len(encoded.getvalue())
        archive.addfile(info,BytesIO(encoded.getvalue()))
    with gzip.open(tmp_path/'sample.csv.gz','wt',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['member','sample_id','family','category','split'])
        writer.writeheader();writer.writerow(dict(member='sample.jpg',sample_id='id',family='Private family metadata',category='serif',split='validation'))
    inventory=pd.DataFrame([dict(split='validation',shard_index=0,images=1,archive='sample.tar',metadata='sample.csv.gz')])
    dataset=ProductShardDataset(tmp_path,inventory,'validation',(192,64))
    first=next(iter(dataset));second=next(iter(dataset))
    assert tuple(first[0].shape)==(1,64,192)
    assert torch.equal(first[0],second[0])
    assert first[1]==4 and first[2]=='id'
    assert first[3]=='Private family metadata'


def test_product_epoch_updates_weights_and_produces_finite_metrics():
    model=FontSenseCNN(5,width=2,dropout=.1)
    images=torch.zeros(2,1,32,64)
    batch=(images,torch.tensor([0,1]),['a','b'],['f1','f2'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    before=model.classifier[-1].weight.detach().clone()
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    result,frame=run_epoch(model,[batch],optimizer,scaler,torch.device('cpu'),True)
    assert result['images']==2 and 0<=result['macro_f1']<=1
    assert not torch.equal(before,model.classifier[-1].weight)
    assert frame.empty


def test_early_stopping_delta_accumulates_without_discarding_small_checkpoint_gains():
    anchor,stale=_update_patience(.8003,.8000,1,.0005)
    assert anchor==.8000 and stale==2
    anchor,stale=_update_patience(.8006,anchor,stale,.0005)
    assert anchor==.8006 and stale==0


def test_product_loader_rejects_modified_archives_before_decoding(tmp_path: Path):
    (tmp_path/'changed.tar').write_bytes(b'changed after validation')
    inventory=pd.DataFrame([dict(split='train',shard_index=0,images=1,archive='changed.tar',metadata='missing.csv.gz',archive_sha256='0'*64,metadata_sha256='1'*64)])
    dataset=ProductShardDataset(tmp_path,inventory,'train',(192,64))
    with pytest.raises(ValueError,match='changed after validation'):
        next(iter(dataset))


def test_dataset_identity_survives_sqlite_repacking_but_not_sample_changes():
    rows=[dict(split='train',shard_index=index,images=1000,archive=f'{index}.tar',archive_sha256=f'archive{index}',metadata=f'{index}.csv.gz',metadata_sha256=f'metadata{index}') for index in range(2)]
    inventory=pd.DataFrame(rows)
    quality=dict(config_sha256='config',split_sha256='split',manifest_sha256='first physical SQLite layout')
    expected=dataset_content_identity(quality,inventory)
    quality['manifest_sha256']='second equivalent physical SQLite layout'
    assert dataset_content_identity(quality,inventory.iloc[::-1])==expected
    inventory.loc[0,'archive_sha256']='different sample bytes'
    assert dataset_content_identity(quality,inventory)!=expected
