import io
import os

import numpy as np
from PIL import Image
import pytest

from .. import ImageReader


def create_pil_image(size_width, size_height):
    from PIL import Image
    image = Image.new('RGB', size=(size_width, size_height), color=(50, 100, 150))
    return image


def create_test_image(filename, size_width, size_height):
    image = create_pil_image(size_width, size_height)
    with open(filename, "wb") as f:
        image.save(f, 'jpeg')


def test_io_uri():
    crafter = ImageReader()
    filename = os.path.join(crafter.current_workspace, 'test.jpeg')
    img_h = 20
    img_w = 30
    create_test_image(filename, size_width=img_w, size_height=img_h)
    test_doc = crafter.craft(buffer=None, uri=filename, blob=None)
    assert test_doc['blob'].shape == (img_h, img_w, 3)


def test_io_buffer():
    crafter = ImageReader()
    filename = os.path.join(crafter.current_workspace, 'test.jpeg')
    img_h = 20
    img_w = 30
    create_test_image(filename, size_width=img_w, size_height=img_h)
    image_buffer = io.BytesIO()
    img = Image.open(filename)
    img.save(image_buffer, format='PNG')
    image_buffer.seek(0)
    test_doc = crafter.craft(buffer=image_buffer.getvalue(), uri=None, blob=None)
    assert test_doc['blob'].shape == (img_h, img_w, 3)
    np.testing.assert_almost_equal(test_doc['blob'], np.array(img).astype('float32'))


def test_io_blob():
    crafter = ImageReader()
    img_h = 20
    img_w = 30
    numpy_image = np.random.randint(0, 255, size=(img_h, img_w, 3), dtype=np.uint8)
    test_doc = crafter.craft(buffer=None, uri=None, blob=numpy_image)
    assert test_doc['blob'].shape == (img_h, img_w, 3)


def test_blob_no_color_channel():
    crafter = ImageReader()
    img_h = 20
    img_w = 30
    numpy_image = np.random.randint(0, 255, size=(img_h, img_w), dtype=np.uint8)
    test_doc = crafter.craft(buffer=None, uri=None, blob=numpy_image)
    assert test_doc['blob'].shape == (img_h, img_w, 3)


def test_blob_alpha_channel():
    crafter = ImageReader()
    img_h = 20
    img_w = 30
    numpy_image = np.random.randint(0, 255, size=(img_h, img_w, 4), dtype=np.uint8)
    test_doc = crafter.craft(buffer=None, uri=None, blob=numpy_image)
    assert test_doc['blob'].shape == (img_h, img_w, 3)


def test_blob_incorrect_dtype():
    crafter = ImageReader()
    numpy_image = np.random.rand(10, 20, 3)
    with pytest.raises(TypeError):
        crafter.craft(buffer=None, uri=None, blob=numpy_image)


def test_blob_color_dim_first():
    crafter = ImageReader(color_channel_axis=0)
    img_h = 20
    img_w = 30
    numpy_image = np.random.randint(0, 255, size=(img_h, img_w, 3), dtype=np.uint8)
    test_doc = crafter.craft(buffer=None, uri=None, blob=numpy_image)
    assert test_doc['blob'].shape == (3, img_h, img_w)


def test_blob_preserve_color_channels():
    crafter = ImageReader()
    img_h = 20
    img_w = 30
    numpy_image = np.repeat([np.repeat(np.array([[50, 100, 150]], dtype=np.uint8), img_w, axis=0)], img_h, axis=0)
    test_doc = crafter.craft(buffer=None, uri=None, blob=numpy_image)
    assert np.array_equal(test_doc['blob'][0, 0], [50, 100, 150])
