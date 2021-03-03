__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import cv2
import numpy as np
import pytest
from jina.executors import BaseExecutor
from jina.executors.metas import get_default_metas

from .. import AlbumentationsCrafter as AC


@pytest.fixture(scope='module')
def test_img():
    return cv2.imread('tests/rubi.png')[:, :, ::-1]


@pytest.fixture
def flip_img():
    return cv2.imread('tests/rubi_flip.png')[:, :, ::-1]


@pytest.fixture
def crop_img():
    return cv2.imread('tests/rubi_crop.png')[:, :, ::-1]


@pytest.fixture
def center_crop_img():
    return cv2.imread('tests/rubi_center_crop.png')[:, :, ::-1]


@pytest.fixture
def resize_img():
    return cv2.imread('tests/rubi_resize.png')[:, :, ::-1]


def test_normalize_transform(test_img):
    transform = {
        'Normalize': dict(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255)
    }
    crafter = AC([transform])
    crafted_img = crafter.craft(test_img)

    np.testing.assert_almost_equal(test_img / 255, crafted_img)


def test_flip_transform(test_img, flip_img):
    crafter = AC(['VerticalFlip'])
    crafted_img = crafter.craft(test_img)

    np.testing.assert_almost_equal(flip_img, crafted_img)


def test_crop_transform(test_img, crop_img):
    transform = {'Crop': dict(x_min=0, y_min=0, x_max=106, y_max=172)}
    crafter = AC([transform])
    crafted_img = crafter.craft(test_img)

    np.testing.assert_almost_equal(crop_img, crafted_img)


def test_center_crop_transform(test_img, center_crop_img):
    transform = {'CenterCrop': dict(height=100, width=100)}
    crafter = AC([transform])
    crafted_img = crafter.craft(test_img)

    np.testing.assert_almost_equal(center_crop_img, crafted_img)


def test_resize_transform(test_img, resize_img):
    transform = {'Resize': dict(height=100, width=200)}
    crafter = AC([transform])
    crafted_img = crafter.craft(test_img)

    np.testing.assert_almost_equal(resize_img, crafted_img)


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

    load_crafter1 = BaseExecutor.load_config('tests/config.yaml')
    load_crafter2 = BaseExecutor.load_config(orig_crafter.config_abspath)

    assert orig_trs == load_crafter1.transforms._to_dict()
    assert orig_trs == load_crafter2.transforms._to_dict()
