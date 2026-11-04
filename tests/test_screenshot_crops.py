"""Crop robustness and honest handling of a font absent from the library."""
from pathlib import Path

import numpy as np
from PIL import Image,ImageChops,ImageOps
import pytest

from api.compact import text_crop
from api.extended_match import catalog
from api.font_match import identify,render_reference


@pytest.mark.parametrize('dark',[False,True])
def test_small_text_inside_large_margins_is_not_declared_blank(dark):
    word=render_reference('arimo','Visible letters')
    image=Image.new('RGB',(2500,1300),'white');image.paste(word,(700,600))
    if dark:image=ImageOps.invert(image)
    normalized=text_crop(image)
    assert normalized.width<1000 and normalized.height<150
    assert np.asarray(normalized).min()<30


def test_faint_partial_caption_does_not_change_the_main_letter_pixels():
    main=render_reference('arimo','Minimum glyphs')
    fragment=main.crop((0,20,main.width,24))
    fragment=Image.blend(Image.new('RGB',fragment.size,'white'),fragment,.32)
    screenshot=Image.new('RGB',(main.width+20,main.height+40),'white')
    screenshot.paste(fragment,(10,0));screenshot.paste(main,(10,25))
    assert ImageChops.difference(text_crop(screenshot),text_crop(main)).getbbox() is None


def test_dark_dots_and_accents_are_kept(monkeypatch):
    image=render_reference('tinos','mini naïve ï jjj')
    monkeypatch.setenv('FONTSENSE_CROP_MODE','legacy');previous=text_crop(image)
    monkeypatch.setenv('FONTSENSE_CROP_MODE','improved');current=text_crop(image)
    assert ImageChops.difference(previous,current).getbbox() is None


def test_complete_second_line_is_not_removed_as_a_fragment():
    first=render_reference('arimo','First text line');second=render_reference('arimo','Second text line')
    image=Image.new('RGB',(max(first.width,second.width),first.height+second.height+30),'white')
    image.paste(first,(0,0));image.paste(second,(0,first.height+30))
    assert text_crop(image).height>first.height+second.height


@pytest.mark.parametrize('font_id',['x-jersey20charted','x-jersey25charted','x-jacquard24charted'])
def test_faint_decorative_grid_rows_remain_part_of_the_font(monkeypatch,font_id):
    from api.extended_match import render_reference as reference
    image=reference(font_id,'CLOUD DESIGN',64)
    monkeypatch.setenv('FONTSENSE_CROP_MODE','legacy');previous=text_crop(image)
    monkeypatch.setenv('FONTSENSE_CROP_MODE','improved');current=text_crop(image)
    assert current.size==previous.size
    assert ImageChops.difference(previous,current).getbbox() is None


def test_user_moga_crop_reads_correctly_and_is_not_named_from_its_written_label():
    image=Image.open(Path(__file__).parent/'fixtures/moga_user_screenshot_crop.png')
    result=identify(image)
    assert result['text']=='Moga Regular'
    assert result['category']=='sans_serif'
    assert result['identified_font'] is None
    assert 'Moga' not in {font['family'] for font in catalog()}
    assert len(result['matches'])>=3


def test_thin_letters_remain_matchable_when_the_whole_image_category_check_fails():
    from api.compact import get_model
    from api.extended_match import render_reference as reference
    face=next(row for row in catalog() if row['family']=='Euphoria Script')
    word=reference(face['id'],'Zebra Festival',50).convert('RGB')
    image=Image.new('RGB',(2500,1300),'white');image.paste(word,(800,600))
    with pytest.raises(ValueError):get_model().predict(image)
    result=identify(image,'Zebra Festival')
    names=([result['identified_font']['family']] if result['identified_font'] else [])+[row['family'] for row in result['matches']]
    assert 'Euphoria Script' in names[:3]
