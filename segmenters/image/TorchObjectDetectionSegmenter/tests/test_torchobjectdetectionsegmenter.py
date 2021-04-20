import os
from unittest.mock import patch

import numpy as np
from PIL import Image

from .. import TorchObjectDetectionSegmenter

cur_dir = os.path.dirname(os.path.abspath(__file__))


def create_test_image(output_fn, size_width=50, size_height=50):
    image = Image.new('RGB', size=(size_width, size_height), color=(155, 0, 0))
    with open(output_fn, "wb") as f:
        image.save(f, 'jpeg')


def create_random_img_array(img_height, img_width):
    import numpy as np
    return np.random.randint(0, 256, (img_height, img_width, 3))


class MockModel:
    def __init__(self):
        pass

    def __call__(self, input_ids, *args, **kwargs):
        import torch
        bbox_1 = [10, 15, 30, 40]
        bbox_2 = [-1, -1, -1, -1]
        bbox_3 = [20, 10, 30, 40]
        score_1 = 0.91
        score_2 = 0.87
        score_3 = 0.909
        label_1 = 1
        label_2 = 2
        label_3 = 3
        predictions = [{
            'boxes': torch.Tensor([bbox_1, bbox_2, bbox_3]),
            'scores': torch.Tensor([score_1, score_2, score_3]),
            'labels': torch.Tensor([label_1, label_2, label_3])
        } for _ in input_ids]
        return predictions

    def eval(self):
        return self

    def to(self, device):
        return self


def test_encoding_mock_model_results():
    import torchvision.models.detection as detection_models
    img_array = create_random_img_array(128, 64)
    img_array = img_array / 255
    with patch.object(detection_models, 'fasterrcnn_resnet50_fpn', return_value=MockModel()):
        segmenter = TorchObjectDetectionSegmenter(channel_axis=-1, confidence_threshold=0.9,
                                                  label_name_map={0: 'zero',
                                                                  1: 'one',
                                                                  2: 'two',
                                                                  3: 'three'})
        docs_chunks = segmenter.segment(np.stack([img_array, img_array]))
        assert len(docs_chunks) == 2
        for chunks in docs_chunks:
            assert len(chunks) == 2
            assert chunks[0]['blob'].shape == (25, 20, 3)
            assert chunks[0]['location'] == (15, 10)
            assert chunks[0]['tags']['label'] == 'one'

            assert chunks[1]['blob'].shape == (30, 10, 3)
            assert chunks[1]['location'] == (10, 20)
            assert chunks[1]['tags']['label'] == 'three'


def test_encoding_fasterrcnn_results():
    img_array = create_random_img_array(128, 64)
    img_array = img_array / 255
    segmenter = TorchObjectDetectionSegmenter(channel_axis=-1, confidence_threshold=0.98)
    docs_chunks = segmenter.segment(np.stack([img_array, img_array]))
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:
        assert len(chunks) == 0


def test_encoding_fasterrcnn_results_real_image():
    """
    Credit for the image used in this test: 
    Photo by <a href="/photographer/createsima-47728">createsima</a> from <a href="https://freeimages.com/">FreeImages</a>
    https://www.freeimages.com/photo/cars-1407390
    TorchObjectDete@29513[I]:detected person with confidence 0.9911105632781982 at position (541, 992) and size (67, 24)
    TorchObjectDete@29513[I]:detected car with confidence 0.9843265414237976 at position (496, 201) and size (104, 161)
    TorchObjectDete@29513[I]:detected car with confidence 0.9835659861564636 at position (524, 574) and size (77, 131)
    TorchObjectDete@29513[I]:detected person with confidence 0.9795390367507935 at position (539, 969) and size (66, 27)
    TorchObjectDete@29513[I]:detected person with confidence 0.9787288904190063 at position (530, 934) and size (74, 18)
    TorchObjectDete@29513[I]:detected car with confidence 0.9717466831207275 at position (517, 377) and size (82, 154)
    TorchObjectDete@29513[I]:detected person with confidence 0.9682216048240662 at position (532, 919) and size (70, 19)
    TorchObjectDete@29513[I]:detected truck with confidence 0.964297354221344 at position (498, 702) and size (106, 169)
    TorchObjectDete@29513[I]:detected car with confidence 0.9574888944625854 at position (522, 1046) and size (88, 164)
    TorchObjectDete@29513[I]:detected person with confidence 0.9304793477058411 at position (536, 962) and size (70, 17)
    """
    path = os.path.join(cur_dir, 'imgs/cars.jpg')
    img = Image.open(path)
    img = img.convert('RGB')
    img_array = np.array(img).astype('float32') / 255
    segmenter = TorchObjectDetectionSegmenter(channel_axis=-1, confidence_threshold=0.9)
    docs_chunks = segmenter.segment(np.stack([img_array, img_array]))
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:
        assert len(chunks) == 10
        assert chunks[0]['tags']['label'] == 'person'
        img = Image.open(os.path.join(cur_dir, 'imgs/faster_rcnn/person-0.png'))
        assert chunks[0]['location'] == (541, 992)
        # check that the shape of retrieved is the same as the expected image (was computed and stored once)
        blob = chunks[0]['blob']
        assert (blob.shape[1], blob.shape[0]) == img.size
        assert blob.shape == (67, 24, 3)
        array = np.array(img)
        np.testing.assert_array_almost_equal(blob, array)

        assert chunks[1]['tags']['label'] == 'car'
        img = Image.open(os.path.join(cur_dir, 'imgs/faster_rcnn/car-1.png'))
        assert chunks[1]['location'] == (496, 201)
        # check that the shape of retrieved is the same as the expected image (was computed and stored once)
        blob = chunks[1]['blob']
        assert (blob.shape[1], blob.shape[0]) == img.size
        assert blob.shape == (104, 161, 3)
        array = np.array(img)
        np.testing.assert_array_almost_equal(blob, array)

        assert chunks[2]['tags']['label'] == 'car'
        img = Image.open(os.path.join(cur_dir, 'imgs/faster_rcnn/car-2.png'))
        assert chunks[2]['location'] == (524, 574)
        # check that the shape of retrieved is the same as the expected image (was computed and stored once)
        blob = chunks[2]['blob']
        assert (blob.shape[1], blob.shape[0]) == img.size
        assert blob.shape == (77, 131, 3)
        array = np.array(img)
        np.testing.assert_array_almost_equal(blob, array)

        assert chunks[3]['tags']['label'] == 'person'
        img = Image.open(os.path.join(cur_dir, 'imgs/faster_rcnn/person-3.png'))
        assert chunks[3]['location'] == (539, 969)
        # check that the shape of retrieved is the same as the expected image (was computed and stored once)
        blob = chunks[3]['blob']
        assert (blob.shape[1], blob.shape[0]) == img.size
        assert blob.shape == (66, 27, 3)
        array = np.array(img)
        np.testing.assert_array_almost_equal(blob, array)

        assert chunks[4]['tags']['label'] == 'person'
        img = Image.open(os.path.join(cur_dir, 'imgs/faster_rcnn/person-4.png'))
        assert chunks[4]['location'] == (530, 934)
        # check that the shape of retrieved is the same as the expected image (was computed and stored once)
        blob = chunks[4]['blob']
        assert (blob.shape[1], blob.shape[0]) == img.size
        assert blob.shape == (74, 18, 3)
        array = np.array(img)
        np.testing.assert_array_almost_equal(blob, array)

        assert chunks[5]['tags']['label'] == 'car'
        img = Image.open(os.path.join(cur_dir, 'imgs/faster_rcnn/car-5.png'))
        assert chunks[5]['location'] == (517, 377)
        # check that the shape of retrieved is the same as the expected image (was computed and stored once)
        blob = chunks[5]['blob']
        assert (blob.shape[1], blob.shape[0]) == img.size
        assert blob.shape == (82, 154, 3)
        array = np.array(img)
        np.testing.assert_array_almost_equal(blob, array)

        assert chunks[6]['tags']['label'] == 'person'
        img = Image.open(os.path.join(cur_dir, 'imgs/faster_rcnn/person-6.png'))
        assert chunks[6]['location'] == (532, 919)
        # check that the shape of retrieved is the same as the expected image (was computed and stored once)
        blob = chunks[6]['blob']
        assert (blob.shape[1], blob.shape[0]) == img.size
        assert blob.shape == (70, 19, 3)
        array = np.array(img)
        np.testing.assert_array_almost_equal(blob, array)

        # it missclassifies as truck (but is a fairly big car)
        assert chunks[7]['tags']['label'] == 'truck'
        img = Image.open(os.path.join(cur_dir, 'imgs/faster_rcnn/car-7.png'))
        assert chunks[7]['location'] == (498, 702)
        # check that the shape of retrieved is the same as the expected image (was computed and stored once)
        blob = chunks[7]['blob']
        assert (blob.shape[1], blob.shape[0]) == img.size
        assert blob.shape == (106, 169, 3)
        array = np.array(img)
        np.testing.assert_array_almost_equal(blob, array)

        assert chunks[8]['tags']['label'] == 'car'
        img = Image.open(os.path.join(cur_dir, 'imgs/faster_rcnn/car-8.png'))
        assert chunks[8]['location'] == (522, 1046)
        # check that the shape of retrieved is the same as the expected image (was computed and stored once)
        blob = chunks[8]['blob']
        assert (blob.shape[1], blob.shape[0]) == img.size
        assert blob.shape == (88, 164, 3)
        array = np.array(img)
        np.testing.assert_array_almost_equal(blob, array)

        assert chunks[9]['tags']['label'] == 'person'
        img = Image.open(os.path.join(cur_dir, 'imgs/faster_rcnn/person-9.png'))
        assert chunks[9]['location'] == (536, 962)
        # check that the shape of retrieved is the same as the expected image (was computed and stored once)
        blob = chunks[9]['blob']
        assert (blob.shape[1], blob.shape[0]) == img.size
        assert blob.shape == (70, 17, 3)
        array = np.array(img)
        np.testing.assert_array_almost_equal(blob, array)


def test_encoding_maskrcnn_results():
    img_array = create_random_img_array(128, 64)
    img_array = img_array / 255
    segmenter = TorchObjectDetectionSegmenter(model_name='maskrcnn_resnet50_fpn',
                                              channel_axis=-1, confidence_threshold=0.98)
    docs_chunks = segmenter.segment(np.stack([img_array, img_array]))
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:
        assert len(chunks) == 0


def test_encoding_maskrcnn_results_real_image():
    """
    Credit for the image used in this test:
    Photo by <a href="/photographer/createsima-47728">createsima</a> from <a href="https://freeimages.com/">FreeImages</a>
    https://www.freeimages.com/photo/cars-1407390
    TorchObjectDete@31595[I]:detected car with confidence 0.996136486530304 at position (518, 379) and size (85, 152)
    TorchObjectDete@31595[I]:detected car with confidence 0.9923352599143982 at position (527, 572) and size (74, 134)
    TorchObjectDete@31595[I]:detected person with confidence 0.9859431982040405 at position (541, 993) and size (66, 23)
    TorchObjectDete@31595[I]:detected person with confidence 0.9840929508209229 at position (531, 934) and size (74, 19)
    TorchObjectDete@31595[I]:detected car with confidence 0.9836736917495728 at position (499, 196) and size (103, 165)
    TorchObjectDete@31595[I]:detected person with confidence 0.9716113805770874 at position (532, 917) and size (71, 20)
    TorchObjectDete@31595[I]:detected person with confidence 0.9690250754356384 at position (539, 968) and size (68, 27)
    TorchObjectDete@31595[I]:detected truck with confidence 0.9677107334136963 at position (499, 700) and size (107, 167)
    TorchObjectDete@31595[I]:detected car with confidence 0.9577637314796448 at position (534, 142) and size (64, 65)
    TorchObjectDete@31595[I]:detected car with confidence 0.9379956126213074 at position (521, 1037) and size (90, 175)

    """
    path = os.path.join(cur_dir, 'imgs/cars.jpg')
    img = Image.open(path)
    img = img.convert('RGB')
    img_array = np.array(img).astype('float32') / 255
    segmenter = TorchObjectDetectionSegmenter(model_name='maskrcnn_resnet50_fpn',
                                              channel_axis=-1, confidence_threshold=0.9)

    docs_chunks = segmenter.segment(np.stack([img_array, img_array]))
    assert len(docs_chunks) == 2
    for chunks in docs_chunks:
        assert len(chunks) == 10
        assert chunks[0]['tags']['label'] == 'car'
        img = Image.open(os.path.join(cur_dir, 'imgs/mask_rcnn/car-0.png'))
        assert chunks[0]['location'] == (518, 379)
        # check that the shape of retrieved is the same as the expected image (was computed and stored once)
        blob = chunks[0]['blob']
        assert (blob.shape[1], blob.shape[0]) == img.size
        assert blob.shape == (85, 152, 3)
        array = np.array(img)
        np.testing.assert_array_almost_equal(blob, array)

        assert chunks[1]['tags']['label'] == 'car'
        img = Image.open(os.path.join(cur_dir, 'imgs/mask_rcnn/car-1.png'))
        assert chunks[1]['location'] == (527, 572)
        # check that the shape of retrieved is the same as the expected image (was computed and stored once)
        blob = chunks[1]['blob']
        assert (blob.shape[1], blob.shape[0]) == img.size
        assert blob.shape == (74, 134, 3)
        array = np.array(img)
        np.testing.assert_array_almost_equal(blob, array)

        assert chunks[2]['tags']['label'] == 'person'
        img = Image.open(os.path.join(cur_dir, 'imgs/mask_rcnn/person-2.png'))
        assert chunks[2]['location'] == (541, 993)
        # check that the shape of retrieved is the same as the expected image (was computed and stored once)
        blob = chunks[2]['blob']
        assert (blob.shape[1], blob.shape[0]) == img.size
        assert blob.shape == (66, 23, 3)
        array = np.array(img)
        np.testing.assert_array_almost_equal(blob, array)

        assert chunks[3]['tags']['label'] == 'person'
        img = Image.open(os.path.join(cur_dir, 'imgs/mask_rcnn/person-3.png'))
        assert chunks[3]['location'] == (531, 934)
        # check that the shape of retrieved is the same as the expected image (was computed and stored once)
        blob = chunks[3]['blob']
        assert (blob.shape[1], blob.shape[0]) == img.size
        assert blob.shape == (74, 19, 3)
        array = np.array(img)
        np.testing.assert_array_almost_equal(blob, array)

        assert chunks[4]['tags']['label'] == 'car'
        img = Image.open(os.path.join(cur_dir, 'imgs/mask_rcnn/car-4.png'))
        assert chunks[4]['location'] == (499, 196)
        # check that the shape of retrieved is the same as the expected image (was computed and stored once)
        blob = chunks[4]['blob']
        assert (blob.shape[1], blob.shape[0]) == img.size
        assert blob.shape == (103, 165, 3)
        array = np.array(img)
        np.testing.assert_array_almost_equal(blob, array)

        assert chunks[5]['tags']['label'] == 'person'
        img = Image.open(os.path.join(cur_dir, 'imgs/mask_rcnn/person-5.png'))
        assert chunks[5]['location'] == (532, 917)
        # check that the shape of retrieved is the same as the expected image (was computed and stored once)
        blob = chunks[5]['blob']
        assert (blob.shape[1], blob.shape[0]) == img.size
        assert blob.shape == (71, 20, 3)
        array = np.array(img)
        np.testing.assert_array_almost_equal(blob, array)

        assert chunks[6]['tags']['label'] == 'person'
        img = Image.open(os.path.join(cur_dir, 'imgs/mask_rcnn/person-6.png'))
        assert chunks[6]['location'] == (539, 968)
        # check that the shape of retrieved is the same as the expected image (was computed and stored once)
        blob = chunks[6]['blob']
        assert (blob.shape[1], blob.shape[0]) == img.size
        assert blob.shape == (68, 27, 3)
        array = np.array(img)
        np.testing.assert_array_almost_equal(blob, array)

        # it missclassifies as truck (but is a fairly big car)
        assert chunks[7]['tags']['label'] == 'truck'
        img = Image.open(os.path.join(cur_dir, 'imgs/mask_rcnn/car-7.png'))
        assert chunks[7]['location'] == (499, 700)
        # check that the shape of retrieved is the same as the expected image (was computed and stored once)
        blob = chunks[7]['blob']
        assert (blob.shape[1], blob.shape[0]) == img.size
        assert blob.shape == (107, 167, 3)
        array = np.array(img)
        np.testing.assert_array_almost_equal(blob, array)

        assert chunks[8]['tags']['label'] == 'car'
        img = Image.open(os.path.join(cur_dir, 'imgs/mask_rcnn/car-8.png'))
        assert chunks[8]['location'] == (534, 142)
        # check that the shape of retrieved is the same as the expected image (was computed and stored once)
        blob = chunks[8]['blob']
        assert (blob.shape[1], blob.shape[0]) == img.size
        assert blob.shape == (64, 65, 3)
        array = np.array(img)
        np.testing.assert_array_almost_equal(blob, array)

        assert chunks[9]['tags']['label'] == 'car'
        img = Image.open(os.path.join(cur_dir, 'imgs/mask_rcnn/car-9.png'))
        assert chunks[9]['location'] == (521, 1037)
        # check that the shape of retrieved is the same as the expected image (was computed and stored once)
        blob = chunks[9]['blob']
        assert (blob.shape[1], blob.shape[0]) == img.size
        assert blob.shape == (90, 175, 3)
        array = np.array(img)
        np.testing.assert_array_almost_equal(blob, array)
