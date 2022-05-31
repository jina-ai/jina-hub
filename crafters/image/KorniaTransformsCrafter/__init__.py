__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Dict, List, Union

import numpy as np

from jina.executors.decorators import batching
from jina.executors.crafters import BaseCrafter


class KorniaTransformsCrafter(BaseCrafter):

    def __init__(
            self,
            transforms: List[Union[str, Dict[str, Dict]]] = ['HorizontalFlip'],
            *args,
            **kwargs
    ):
        """Set constructor."""
        super().__init__(*args, **kwargs)

        if not isinstance(transforms, list):
            self.logger.error('The `transform` argument has to be a list.')
            raise ValueError

        self.transforms_specification = transforms

    def post_init(self):
        import kornia
        import torch.nn as nn
        transforms_list = []
        for transform in self.transforms_specification:

            if isinstance(transform, str):
                tr_name, tr_kwargs = transform, {}
            elif isinstance(transform, dict):
                tr_name, tr_kwargs = next(iter(transform.items()))
            else:
                self.logger.error(
                    'Each element of the `transform` has to be either a dict'
                    ' or a string.'
                )
                raise ValueError

            try:
                tr_class = getattr(kornia, tr_name)
            except:
                self.logger.error(
                    f'The albumentations class `{tr_name}` does not seem to exist,'
                    ' check that you have not made a typo.'
                )
                raise ValueError

            try:
                kornia_transformation = tr_class(**tr_kwargs)
            except:
                self.logger.error(
                    f'Error instantiating kornia class `{tr_name}`,'
                    ' check that all kwargs are valid.'
                )
                raise ValueError

            transforms_list.append(kornia_transformation)

        self.transform = nn.Sequential(
            kornia.geometry.Resize((224, 224)),
            kornia.augmentation.Normalize(0., 255.))

    @batching
    def craft(self, blob: 'np.ndarray', *args, **kwargs) -> List[Dict]:
        """Apply transformations to the image.

        :param blob: The image to transform, should be in ``[B, H, W, C]`` format,
            where ``C`` is the color channel, which is either RGB (siz 3), or BW
            (size 1).
        :return: The transformed image
        """
        images = self.transform(blob)
        result = []
        for image in images:
            result.apppend({'blob': image})
