import os

import cv2
import numpy as np
import pytest

from .. import ImageTorchTransformation

cur_dir = os.path.dirname(os.path.abspath(__file__))


def org_img():
    # using the same image from `AlbumentationsCrafter`
    img = cv2.imread(os.path.join(cur_dir, '../tests/original.png'))[:, :, ::-1]
    return img


def get_mean_std():
    return dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def normalize_image(img):
    # Normalie images of range [0,255]
    params = get_mean_std()
    return ((img / 255.0) - params['mean']) / params['std']


def normalize_img():
    img = cv2.imread(os.path.join(cur_dir, '../tests/original.png'))[:, :, ::-1]
    return normalize_image(img)


def resize_img():
    # Return img of range [0,1] to match range of crafter
    img = cv2.imread(os.path.join(cur_dir, '../tests/resize_exp.png'))[:, :, ::-1]
    return img / 255.0


def center_crop_img():
    img = cv2.imread(os.path.join(cur_dir, '../tests/center_crop_exp.png'))[:, :, ::-1]
    return img / 255.0


def vflip_img():
    img = cv2.imread(os.path.join(cur_dir, '../tests/vflip_exp.png'))[:, :, ::-1]
    return img / 255.0


def normalize_transform():
    return {'Normalize': get_mean_std()}


def resize_transform():
    return {'Resize': dict(size=(100, 200))}


def center_crop_transform():
    return {'CenterCrop': dict(size=(100, 100))}


def flip_transform():
    return 'RandomVerticalFlip'


@pytest.mark.parametrize(
    'transforms, inputs',
    [
        (['ColorJitter'], org_img()),
        (['ColorJitter'], np.stack([org_img()] * 3)),
        ([{'GaussianBlur': dict(kernel_size=(3, 3))}], [org_img(), org_img()]),
    ],
)
def test_allowed_argument_types(transforms, inputs):
    # All valid formats of transforms and inputs
    crafter = ImageTorchTransformation(transforms)
    crafted_imgs = crafter.craft(inputs)
    for crafted_img in crafted_imgs:
        assert crafted_img['blob'].ndim == 3


def test_default_transform():
    crafter = ImageTorchTransformation()
    crafted_imgs = crafter.craft(org_img())
    assert crafted_imgs[0]['blob'].shape == (224, 224, 3)


@pytest.mark.parametrize(
    'transforms, input, expected',
    [
        ([flip_transform()], org_img(), vflip_img()),
        ([center_crop_transform()], org_img(), center_crop_img()),
        ([normalize_transform()], org_img(), normalize_img()),
    ],
)
def test_common_transforms(transforms, input, expected):
    crafter = ImageTorchTransformation(transforms)
    crafted_imgs = crafter.craft(input)
    np.testing.assert_almost_equal(expected, crafted_imgs[0]['blob'], decimal=6)


@pytest.mark.parametrize(
    'transforms, input, expected',
    [
        ([resize_transform()], org_img(), resize_img()),
    ],
)
def test_transform_visually(transforms, input, expected):
    # visual inspection only - WIP https://github.com/pytorch/vision/issues/2950
    crafter = ImageTorchTransformation(transforms)
    crafted_imgs = crafter.craft(input)
    cv2.imwrite(
        f'{cur_dir}/{list(transforms[0].keys())[0].lower()}_out.png',
        cv2.cvtColor(crafted_imgs[0]['blob'] * 255, cv2.COLOR_RGB2BGR),
    )


@pytest.mark.parametrize(
    'transforms, inputs, expected',
    [
        (
            [center_crop_transform(), normalize_transform()],
            [org_img(), org_img()],
            normalize_image(center_crop_img() * 255),
        ),
        (
            [flip_transform(), normalize_transform()],
            [org_img(), org_img()],
            normalize_image(vflip_img() * 255),
        ),
    ],
)
def test_sequential_transforms(transforms, inputs, expected):
    crafter = ImageTorchTransformation(transforms)
    crafted_imgs = crafter.craft(inputs)

    assert len(crafted_imgs) == 2
    for crafted_img in crafted_imgs:
        np.testing.assert_almost_equal(expected, crafted_img['blob'], decimal=6)


@pytest.mark.parametrize(
    'transforms, input, expected',
    [
        ([{'RandomVerticalFlip': dict(p=0.0)}], org_img(), vflip_img()),
        ([{'RandomVerticalFlip': dict(p=1.0)}], org_img(), vflip_img()),
    ],
)
def test_random_transforms(transforms, input, expected):
    # Here randomness is in if or not to do transform. It is overrided to always do.
    crafter = ImageTorchTransformation(transforms)
    crafted_imgs = crafter.craft(input)
    np.testing.assert_almost_equal(expected, crafted_imgs[0]['blob'], decimal=6)


def test_incorrect_transforms():
    # transform should be list
    with pytest.raises(ValueError):
        ImageTorchTransformation('RandomVerticalFlip')
    # incorrect transform name
    with pytest.raises(ValueError):
        ImageTorchTransformation(['Fake'])
    # incorrect arg
    with pytest.raises(ValueError):
        ImageTorchTransformation([{'RandomVerticalFlip': dict(Fake=1.0)}])
    # allowed single image array (H, B, C), list of image arrays, image batch array (B, H, W, C)
    with pytest.raises(AssertionError):
        ImageTorchTransformation(['RandomVerticalFlip']).craft(np.zeros((100, 100)))
    # transform not supported for tensor
    with pytest.raises(ValueError):
        ImageTorchTransformation(['RandomChoice'])


def test_example():
    image_batch = np.stack([org_img(), org_img()])

    transforms = [
        {'CenterCrop': dict(size=(300))},
        {'Resize': dict(size=(244, 244))},
        'RandomVerticalFlip',
        {'Normalize': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))},
    ]

    crafter = ImageTorchTransformation(transforms)
    crafted_imgs = crafter.craft(image_batch)

    import torchvision.transforms as T

    transforms = T.Compose(
        [
            T.ToTensor(),
            T.CenterCrop(300),
            T.Resize((244, 244)),
            T.RandomVerticalFlip(p=1.0),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    transformed_img = transforms(image_batch[0]).cpu().numpy()
    transformed_img = np.transpose(transformed_img, (1, 2, 0))

    np.testing.assert_almost_equal(transformed_img, crafted_imgs[0]['blob'], decimal=6)


def test_save_load_config(tmp_path):
    from jina.executors import BaseExecutor
    from jina.executors.metas import get_default_metas

    transforms = [{'RandomVerticalFlip': dict(p=1.0)}]

    metas = get_default_metas()
    metas['workspace'] = str(tmp_path)

    orig_crafter = ImageTorchTransformation(transforms, metas=metas)
    orig_crafter.save_config()
    orig_trs = orig_crafter.transforms_specification

    load_crafter1 = BaseExecutor.load_config(
        os.path.join(cur_dir, '../tests/config.yaml')
    )
    load_crafter2 = BaseExecutor.load_config(orig_crafter.config_abspath)

    assert orig_trs == load_crafter1.transforms_specification
    assert orig_trs == load_crafter2.transforms_specification
