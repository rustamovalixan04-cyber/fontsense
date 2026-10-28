from pathlib import Path
import json

from matplotlib import get_data_path
import pandas as pd
import pytest
from PIL import ImageDraw

from fontsense.utils import project_root
from fontsense.v3_dataset import CLASSES, generate, render_product_image, validate_config


def _config():
    return json.loads((project_root()/'config/v3/dataset.json').read_text())


def test_500k_product_config_has_balanced_explicit_counts():
    config=_config()
    validate_config(config)
    assert config['target_images']==500000
    assert config['split_image_counts']==dict(train=350000,validation=75000,test=75000)


def test_renderer_is_deterministic_and_needs_no_label():
    font=str(Path(get_data_path())/'fonts/ttf/DejaVuSans.ttf')
    first,metadata=render_product_image(font,1234,_config())
    second,again=render_product_image(font,1234,_config())
    assert first==second and metadata==again
    assert metadata['contrast']>=2
    assert 'category' not in metadata['applied_effects']


def test_tiny_shards_are_complete_unique_and_resume_without_changes(tmp_path: Path):
    config=_config()
    config.update(target_images=30,images_per_shard=10,split_image_counts=dict(train=10,validation=10,test=10),min_free_disk_gib=0)
    config_path=tmp_path/'config.json'
    config_path.write_text(json.dumps(config))
    fonts=Path(get_data_path())/'fonts/ttf'
    rows=[dict(family=f'{category}-{split}',category=category,split=split,path=str(fonts/'DejaVuSans.ttf')) for category in CLASSES for split in ('train','validation','test')]
    split_path=tmp_path/'split.csv'
    pd.DataFrame(rows).to_csv(split_path,index=False)
    output=tmp_path/'data'
    result=generate(config_path,split_path,output,tmp_path/'reports',workers=1)
    assert result['images_total']==30 and result['exact_duplicate_groups']==0
    assert result['duplicate_candidates_replaced']>0
    hashes={str(path):path.read_bytes() for path in output.rglob('*.tar')}
    resumed=generate(config_path,split_path,output,tmp_path/'reports',workers=1)
    assert resumed['images_total']==30
    assert all(path.read_bytes()==hashes[str(path)] for path in output.rglob('*.tar'))


def test_generation_rejects_leaking_family_assignments(tmp_path: Path):
    config=_config()
    config.update(target_images=30,images_per_shard=10,split_image_counts=dict(train=10,validation=10,test=10),min_free_disk_gib=0)
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(config))
    split_path=tmp_path/'split.csv'
    pd.DataFrame([dict(family='same',category='serif',split=split,path='font.ttf') for split in ('train','test')]).to_csv(split_path,index=False)
    with pytest.raises(ValueError,match='Family overlap'):
        generate(config_path,split_path,tmp_path/'data',tmp_path/'reports',workers=1)


def test_bold_effect_does_not_use_unbounded_freetype_outline_stroking(monkeypatch):
    original=ImageDraw.ImageDraw.text
    def checked_draw(self,xy,text,*args,**kwargs):
        assert kwargs.get('stroke_width',0)==0
        return original(self,xy,text,*args,**kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw,'text',checked_draw)
    font=str(Path(get_data_path())/'fonts/ttf/DejaVuSans.ttf')
    payload,metadata=render_product_image(font,14629533,_config())
    assert payload
    assert json.loads(metadata['applied_effects'])['stroke_width']==1


def test_render_error_pauses_generation_and_records_actual_context(tmp_path: Path):
    config=_config()
    config.update(target_images=30,images_per_shard=10,split_image_counts=dict(train=10,validation=10,test=10),min_free_disk_gib=0)
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(config))
    split_path=tmp_path/'split.csv'
    pd.DataFrame([dict(family=f'{category}-{split}',category=category,split=split,path=str(tmp_path/'missing-font.ttf')) for category in CLASSES for split in ('train','validation','test')]).to_csv(split_path,index=False)
    output=tmp_path/'data';reports=tmp_path/'reports'
    with pytest.raises(RuntimeError,match='Render failed.*family=.*seed='):
        generate(config_path,split_path,output,reports,workers=1)
    assert (output/'STOP').is_file()
    progress=json.loads((reports/'progress.json').read_text())
    assert progress['status']=='failed'
    assert 'cannot open resource' in progress['error']
    assert not (reports/'dataset_validation_summary.json').exists()
