__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os

import cv2
import numpy as np
import pytest

from jina.executors import BaseExecutor
from jina.executors.metas import get_default_metas

from .. import AlbumentationsCrafter as AC

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_img():
    return cv2.imread(os.path.join(cur_dir, '../tests/rubi.png'))[:, :, ::-1]


def flip_img():
    return cv2.imread(os.path.join(cur_dir, '../tests/rubi_flip.png'))[:, :, ::-1]


def crop_img():
    return cv2.imread(os.path.join(cur_dir, '../tests/rubi_crop.png'))[:, :, ::-1]


def center_crop_img():
    return cv2.imread(os.path.join(cur_dir, '../tests/rubi_center_crop.png'))[:, :, ::-1]


def resize_img():
    return cv2.imread(os.path.join(cur_dir, '../tests/rubi_resize.png'))[:, :, ::-1]


def normalize_transform():
    transform = {
        'Normalize': dict(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255)
    }
    return transform


def flip_transform():
    return 'VerticalFlip'


def crop_transform():
    transform = {'Crop': dict(x_min=0, y_min=0, x_max=106, y_max=172)}
    return transform


def center_crop_transform():
    transform = {'CenterCrop': dict(height=100, width=100)}
    return transform


def resize_transform():
    transform = {'Resize': dict(height=100, width=200)}
    return transform


@pytest.mark.parametrize('stack', [False, True])
@pytest.mark.parametrize('transform, inputs, expected',
                         [
                             (normalize_transform(), [test_img(), test_img()], test_img() / 255),
                             (flip_transform(), [test_img(), test_img()], flip_img()),
                             (crop_transform(), [test_img(), test_img()], crop_img()),
                             (center_crop_transform(), [test_img(), test_img()], center_crop_img()),
                             (resize_transform(), [test_img(), test_img()], resize_img()),
                         ])
def test_transform_batch(stack, transform, inputs, expected):
    crafter = AC([transform])
    crafted_imgs = crafter.craft(np.stack(inputs) if stack else inputs)

    assert len(crafted_imgs) == 2
    for crafted_img in crafted_imgs:
        np.testing.assert_almost_equal(expected, crafted_img['blob'])


@pytest.mark.parametrize('transform, inputs, expected',
                         [
                             (normalize_transform(), test_img(), test_img() / 255),
                             (flip_transform(), test_img(), flip_img()),
                             (crop_transform(), test_img(), crop_img()),
                             (center_crop_transform(), test_img(), center_crop_img()),
                             (resize_transform(), test_img(), resize_img()),
                         ])
def test_transform_single_kwargs(transform, inputs, expected):
    crafter = AC([transform])
    crafted_img = crafter.craft(blob=inputs)
    np.testing.assert_almost_equal(expected, crafted_img['blob'])


def test_wrong_transforms():
    # Transforms not a list
    with pytest.raises(ValueError):
        AC('VerticalFlip')

    # Item in transforms not a dict/str
    with pytest.raises(ValueError):
        AC([['VerticalFlip']])

    # Transform not existing
    with pytest.raises(ValueError):
        AC(['FakeTransform'])

    # Wrong args for transform
    with pytest.raises(ValueError):
        AC([{'VerticalFlip': {'width': 100}}])


def test_save_load_config(tmp_path):
    transforms = ['VerticalFlip', {'Resize': {'width': 200, 'height': 300}}]

    metas = get_default_metas()
    metas['workspace'] = str(tmp_path)

    orig_crafter = AC(transforms, metas=metas)
    orig_crafter.save_config()
    orig_trs = orig_crafter.transforms._to_dict()

    load_crafter1 = BaseExecutor.load_config(os.path.join(cur_dir, '../tests/config.yaml'))
    load_crafter2 = BaseExecutor.load_config(orig_crafter.config_abspath)

    assert orig_trs == load_crafter1.transforms._to_dict()
    assert orig_trs == load_crafter2.transforms._to_dict()
