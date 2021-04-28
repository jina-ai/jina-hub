from typing import Dict, List, Union

import numpy as np
from jina.executors.crafters import BaseCrafter

if False:
    import torch


class ImageTorchTransformation(BaseCrafter):
    """Apply torchvision transforms on image batches.

    This crafter creates a `Compose` transform using the list of transforms provided.
    The numpy input is converted to tensor before applying the transform. Hence `ToTensor()` is not required.

    Note: Not recommended to use inherently random transforms and
    make sure not to use transforms that do not support tensor inputs.

    Example:
        .. code-block:: python
            from jina.hub.crafters.image.ImageTorchTransformation import ImageTorchTransformation
            import numpy

            transforms = [
                {'CenterCrop': dict(size=(300))},
                {'Resize': dict(size=(244, 244))},
                'RandomVerticalFlip',
                {'Normalize': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))},
            ]

            crafter = ImageTorchTransformation(transforms)
            crafted_imgs = crafter.craft(image_batch)

        The torch equivalent for this would be recursively apply `Compose` on each image.

        .. code-block:: python
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

            crafted_img = transforms(image_batch[0]).cpu().numpy()
            crafted_img = np.transpose(crafted_img, (1, 2, 0))


    :param transforms: List of transforms that are applied sequentially
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        transforms: List[Union[str, Dict[str, Dict]]] = [
            {'Resize': dict(size=(224, 224))},
            {'Normalize': dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))},
        ],
        *args,
        **kwargs,
    ):
        """Set constructor."""
        super().__init__(*args, **kwargs)
        if not isinstance(transforms, list):
            self.logger.error('The `transforms` argument has to be a list.')
            raise ValueError

        self.transforms_specification = transforms

    def post_init(self):
        from torchvision import transforms as T

        transforms_list = []
        for transform in self.transforms_specification:
            if isinstance(transform, str):
                tr_name, tr_kwargs = transform, {}
            elif isinstance(transform, dict):
                tr_name, tr_kwargs = next(iter(transform.items()))
            else:
                self.logger.error(
                    'Each element of the `transform` has to be either a dict or  a string'
                )
                raise ValueError

            try:
                tr_class = getattr(T, tr_name)
            except AttributeError:
                self.logger.error(
                    f'The transform class `{tr_name}` does not seem to exist,'
                    'check that you have not made a typo.'
                )
                raise ValueError

            try:
                transform_instance = tr_class(**tr_kwargs)
            except:
                self.logger.error(
                    f'Error instantiating torchvision transforms class `{tr_name}`,'
                    ' check if all kwargs are valid and if transform is valid for tensors.'
                )
                raise ValueError

            # Removes randomness in transforms
            # does not handle inherently random transform like `RandomErasing`
            # where p is probabilty of being random
            transform_instance.__dict__.update({'p': 1.0})
            transforms_list.append(transform_instance)
        self.transforms = T.Compose(transforms_list)

    def craft(
        self, blob: Union[List['np.ndarray'], 'np.ndarray'], *args, **kwargs
    ) -> List[Dict]:
        """Apply the transforms to a batch of numpy images
        Note: The numpy arrays are standardize to range [0, 1] before applying transforms

        :param blob: numpy images in formats `(..., H, W, C)`, `(H, W, C)`, `list[(H, W, C)]`
            with value [0, 255] range
        :return: list of dictionaries, each containing a transformed image
        """
        if isinstance(blob, list):
            blob = np.stack(blob)
        elif isinstance(blob, np.ndarray):
            if blob.ndim == 3:
                blob = np.stack([blob])
        else:
            self.logger.error(f'Expected numpy array or list of numpy arrays')
            raise ValueError
        assert (
            blob.ndim == 4
        ), 'numpy should be of shape (..., H, W, C) or list[(H, W, C)]'

        blob_ = self._to_tensor(blob)
        blob_ = self.transforms(blob_)
        blob_ = self._to_numpy(blob_)

        dict_list = []
        for sample in blob_:
            dict_list.append({'blob': sample})
        return dict_list

    @staticmethod
    def _to_tensor(array: 'np.ndarray') -> 'torch.Tensor':
        """Convert numpy to tensor
        Replaces `transforms.ToTensor` to allow batching

        :param array: numpy array of format `[B, H, W, C]`
        :return: standardized tensor in the format `[B, C, H, W]` as required by torchvision transforms
        """
        import torch

        array = np.transpose(array, (0, 3, 1, 2))
        if not array.data.contiguous:
            array = np.ascontiguousarray(array)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.from_numpy(array).float().to(device).div(255)

    @staticmethod
    def _to_numpy(tensor: 'torch.Tensor') -> 'np.ndarray':
        """Convert tensor to numpy

        :param tensor: tensor in the format `[B, C, H, W]`
        :return: numpy array of format `[B, H, W, C]`
        """

        array = tensor.detach().cpu().numpy()
        array = np.transpose(array, (0, 2, 3, 1))
        return array
