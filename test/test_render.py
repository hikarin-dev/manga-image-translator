import os
import sys
import cv2
import pytest
import numpy as np

from manga_translator.rendering import dispatch as dispatch_rendering, dispatch_eng_render
from manga_translator.utils import (
    TextBlock,
    visualize_textblocks,
)


RENDER_IMAGE_FOLDER = 'test/testdata/render'
os.makedirs(RENDER_IMAGE_FOLDER, exist_ok=True)

def save_result(path, img, regions):
    path = os.path.join(RENDER_IMAGE_FOLDER, path)
    cv2.imwrite(path, visualize_textblocks(img, regions))


@pytest.mark.asyncio
async def test_default_renderer():
    width, height = 1000, 1000
    img = np.zeros((height, width, 3))
    regions = [
        TextBlock(
            [[[10, 10], [200, 10], [10, 400], [200, 400]]],
            texts=['a', 'b','c', 'd', 'e', 'f'],
            translation='aaaaaa bbbbbbbbbbbb cccc ddddddddddd eeeeeeeeeeeeee fff'
        ),
        TextBlock(
            [[[410, 10], [900, 10], [410, 800], [900, 800]]],
            texts=['eng', 'pne'],
            translation=#'aaaaaa bbbbbbbbbbbb cccc' \
                # 'dddddddddddddddddddddddddddddddddddddddddddddddddddd eeeeeeeeeeeeee fff' \
                # 'dddddddddddddddddddddddddddddddddddddddddddddddddddd fff' \
                # 'dddddddddddddddddddddddddddddddddddddddddddddddddddd ' \
                'normal english sentences can be hyphenated! ' \
                'Pneumonoultramicroscopicsilicovolcanoconiosis'
        ),
    ]
    for region in regions:
        region.target_lang = 'ENG'
        region.set_font_colors([255, 255, 255], [200, 200, 200])
        region.font_size = 100

    img_rendered = await dispatch_rendering(img, regions, hyphenate=False)
    save_result('default1.png', img_rendered, regions)


def test_default_and_pillow_renderers_record_the_area_they_lay_text_into():
    import asyncio
    from pathlib import Path
    from manga_translator.rendering import dispatch_eng_render_pillow
    font = str(Path(__file__).resolve().parents[1] / 'fonts' / 'ccvictoryspeech.ttf')
    page = np.full((240, 320, 3), 255, np.uint8)
    cv2.ellipse(page, (160, 120), (130, 90), 0, 0, 360, (0, 0, 0), 3)

    def regions():
        return [TextBlock([[[110, 90], [210, 90], [210, 150], [110, 150]]], texts=['source'],
                          translation='Hello there', font_size=20, target_lang='ENG',
                          fg_color=(0, 0, 0), bg_color=(255, 255, 255))]

    default = regions()
    asyncio.run(dispatch_rendering(page.copy(), default, font, hyphenate=False))
    # The quadrilateral the text is warped into — here widened past the source box to fit it.
    assert default[0]._drawn_shape == [(110, 90), (310, 90), (310, 150), (110, 150)]
    pillow = regions()
    asyncio.run(dispatch_eng_render_pillow(page.copy(), page, pillow, font))
    xs, ys = zip(*pillow[0]._drawn_shape)
    # The balloon around the text, not the text's own box.
    assert min(xs) < 110 and max(xs) > 210 and min(ys) < 90 and max(ys) > 150
