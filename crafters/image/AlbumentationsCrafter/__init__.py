from typing import Dict, List, Union

import numpy as np

from jina.executors.crafters import BaseCrafter


class AlbumentationsCrafter(BaseCrafter):
    """Applies image transforms from the Albumentations package to the image.

    This crafter allows you to apply any of the Albumenation's transforms to
    an image. You can also compose as many transforms as you would like inside
    a single object. For a full list of available transforms, visit
    `Albumentations's GitHub page <https://github.com/albumentations-team/albumentations/#list-of-augmentations>`__
    or see `the documentation <https://albumentations.ai/docs/api_reference/augmentations/transforms/>`__.

    .. attention::
        Albumentations' transforms were created for the purpose of image augmentation,
        where the transforms are applied randomly. However in indexing/search
        context you will want the transforms to be applied deterministically.

        For that reason pay attention to the parameters and set them in a way that
        any randomness is removed. For example, if you want to rotate the image 45
        degrees, set the ``limit`` parameter in the ``Rotate`` transformation to
        ``[45,45]``. And you don't want to use some inherently random transforms,
        such as ``RandomRotate90``.

        We take care of some of this automatically: we set ``always_apply=True`` for
        all transforms, so that they are always applied.

    Example:
        The transforms that you provide will be applied in the order provided using
        ``A.Compose``. For example, the ``crafter`` in the example below

        .. code-block:: python

            from jina.hub.crafters.image.AlbumentationsCrafter import AlbumentationsCrafter

            transforms = [
                'HorizontalFlip',
                {'Rotate': {'limit': [45, 45]}}
            ]
            crafter = AlbumentationsCrafter(transforms)

        will have the equivalent effect as the ``transform`` below

        .. code-block:: python

            import albumentations as A

            transform = A.Compose([
                A.HorizontalFlip(always_apply=True),
                A.Rotate(limit=[45,45], always_apply=True)
            ])

    Args:
        transforms: A list of transformations that should be applied. Each item in
            the list should be of the form ``{'TransformClass': kwargs}`` or
            ``'TransformClass'``, where ``kwargs`` is a dictionary with keyword
            arguments that will be passed to ``TransformClass`` at initialization
            (or an empty dict if there are no such arguments).

            The argument ``always_apply=True`` will be added to all kwargs
            automatically.
    """

    def __init__(
        self,
        transforms: List[Union[str, Dict[str, Dict]]] = ['HorizontalFlip'],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if not isinstance(transforms, list):
            self.logger.error('The `transform` argument has to be a list.')
            raise ValueError

        self.transforms_specification = transforms

    def post_init(self):
        import albumentations as A

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

            tr_kwargs.update({'always_apply': True})

            try:
                tr_class = getattr(A, tr_name)
            except:
                self.logger.error(
                    f'The albumentations class `{tr_name}` does not seem to exist,'
                    ' check that you have not made a typo.'
                )
                raise ValueError

            try:
                alb_transform = tr_class(**tr_kwargs)
            except:
                self.logger.error(
                    f'Error instantiating albumentations class `{tr_name}`,'
                    ' check that all kwargs are valid.'
                )
                raise ValueError

            transforms_list.append(alb_transform)

        self.transforms = A.Compose(transforms_list)

    def craft(self, blob: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """Applies the transformations to the image.

        Args:
            blob: The image to transform, should be in ``[H, W, C]`` format, where
                ``C`` is the color channel, which is either RGB (siz 3), or BW
                (size 1).
        """

        return self.transforms(image=blob)['image']
