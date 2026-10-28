from pathlib import Path
import pandas as pd
import pytest
from matplotlib import get_data_path

from fontsense.v3_data import build_product_split, validate_product_font
from fontsense.v3_dataset import CLASSES
from fontsense.utils import load_json, project_root
from fontsense.v2_data import sha256_file


def _families():
    rows=[]
    v2=[]
    for category in CLASSES:
        for split in ('train','validation','test'):
            for index in range(2):
                family=f'{category}-{split}-{index}'
                rows.append(dict(family=family,category=category,usable=True,glyph_probe_sha256=family))
                v2.append(dict(family=family,category=category,split=split))
        for index in range(15):
            family=f'{category}-fresh-{index}'
            rows.append(dict(family=family,category=category,usable=True,glyph_probe_sha256=family))
    return pd.DataFrame(rows),pd.DataFrame(v2)


def test_product_split_preserves_existing_folds_and_excludes_protected_families():
    audit,v2=_families()
    protected=audit.iloc[-1]['family']
    v1=pd.DataFrame([dict(family=protected,split='test')])
    first,excluded=build_product_split(audit,v2,v1)
    second,_=build_product_split(audit,v2,v1)
    assert first.equals(second)
    assert protected not in first['family'].tolist()
    merged=first.merge(v2,on='family',suffixes=('_new','_old'))
    assert (merged['split_new']==merged['split_old']).all()
    assert first.groupby('family')['split'].nunique().eq(1).all()
    assert not excluded.empty


def test_cross_fold_identical_glyph_aliases_are_quarantined():
    audit,v2=_families()
    names=['serif-train-0','serif-validation-0']
    audit.loc[audit['family'].isin(names),'glyph_probe_sha256']='identical'
    selected,excluded=build_product_split(audit,v2,pd.DataFrame(columns=['family','split']))
    assert not set(names)&set(selected['family'])
    assert excluded['exclusion_reason'].str.contains('spans historical').any()


def test_product_font_opens_and_contains_actual_latin_glyphs():
    font=Path(get_data_path())/'fonts/ttf/DejaVuSans.ttf'
    result=validate_product_font(font)
    assert result['latin_glyphs_verified']
    assert len(result['glyph_probe_sha256'])==64


def test_unreadable_utility_fonts_are_excluded_without_inventing_metadata():
    audit,v2=_families()
    utility=pd.DataFrame([dict(family='Libre Barcode 39',category='display',usable=True,glyph_probe_sha256='barcode')])
    audit=pd.concat([audit,utility],ignore_index=True)
    selected,excluded=build_product_split(audit,v2,pd.DataFrame(columns=['family','split']),excluded_family_prefixes=('Libre Barcode',),exclusion_reason='Barcode glyphs, not readable Latin text')
    assert 'Libre Barcode 39' not in selected['family'].tolist()
    row=excluded.loc[excluded['family']=='Libre Barcode 39'].iloc[0]
    assert row['category']=='display'
    assert row['usable']
    assert row['exclusion_reason']=='Barcode glyphs, not readable Latin text'


def test_frozen_reviewed_split_keeps_family_alias_and_historical_boundaries():
    root=project_root()
    config=load_json(root/'config/v3/dataset.json')
    path=root/config['split_path']
    split=pd.read_csv(path)
    summary=load_json(root/'reports/v3/family_split_summary.json')
    assert len(split)==summary['families_total']==1641
    assert sha256_file(path)==summary['split_sha256']
    assert split['family'].str.casefold().is_unique
    assert split['glyph_probe_sha256'].is_unique
    assert set(split['category'])==set(CLASSES)
    assert set(split['split'])=={'train','validation','test'}
    assert not split['family'].map(lambda family: any(family.casefold().startswith(prefix.casefold()) for prefix in config['excluded_family_prefixes'])).any()
    historical=pd.read_csv(root/'data/v2/frozen_family_split.csv')
    matched=split.merge(historical[['family','split']],on='family',suffixes=('_new','_old'))
    assert matched['split_new'].eq(matched['split_old']).all()
    original=pd.read_csv(root/'data/interim/google_fonts_final_family_split.csv')
    assert not set(original.loc[original['split']=='test','family'])&set(split['family'])


def test_missing_space_glyph_is_reported_as_visible_unicode_code(monkeypatch):
    import string
    import fontsense.v3_data as module
    class FontWithoutSpace:
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def getBestCmap(self): return {ord(character):character for character in string.ascii_letters+string.digits}
    monkeypatch.setattr(module,'TTFont',lambda *args,**kwargs:FontWithoutSpace())
    with pytest.raises(ValueError,match='Missing required Latin characters: U\\+0020'):
        validate_product_font(Path('unused.ttf'))
