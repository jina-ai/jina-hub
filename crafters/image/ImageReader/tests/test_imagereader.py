import io

import numpy as np
from PIL import Image

from .. import ImageReader


def create_test_image(output_fn, size_width=50, size_height=50):
    from PIL import Image
    image = Image.new('RGB', size=(size_width, size_height), color=(155, 0, 0))
    with open(output_fn, "wb") as f:
        image.save(f, 'jpeg')


def test_io_uri():
    crafter = ImageReader()
    tmp_fn = crafter.get_file_from_workspace('test.jpeg')
    img_size = 50
    create_test_image(tmp_fn, size_width=img_size, size_height=img_size)
    test_docs = crafter.craft([None, None], np.stack([tmp_fn, tmp_fn]))
    assert len(test_docs) == 2
    for test_doc in test_docs:
        assert test_doc['blob'].shape == (img_size, img_size, 3)


def test_io_buffer():
    crafter = ImageReader()
    tmp_fn = crafter.get_file_from_workspace('test.jpeg')
    img_size = 50
    create_test_image(tmp_fn, size_width=img_size, size_height=img_size)
    image_buffer = io.BytesIO()
    img = Image.open(tmp_fn)
    img.save(image_buffer, format='PNG')
    image_buffer.seek(0)
    test_docs = crafter.craft(np.stack([image_buffer.getvalue(), image_buffer.getvalue()]), [None, None])
    assert len(test_docs) == 2
    for test_doc in test_docs:
        assert test_doc['blob'].shape == (img_size, img_size, 3)
        np.testing.assert_almost_equal(test_doc['blob'], np.array(img).astype('float32'))
