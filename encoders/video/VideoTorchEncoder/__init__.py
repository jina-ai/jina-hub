__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np

from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders import BaseVideoEncoder
from jina.executors.encoders.frameworks import BaseTorchEncoder


class VideoTorchEncoder(BaseTorchEncoder, BaseVideoEncoder):
    """
    Encode data from a ndarray, using the models from `torchvision.models`.

    :class:`VideoTorchEncoder` encodes data from a ndarray, potentially
    B x T x (Channel x Height x Width) into an ndarray of `B x D`.
    Internally, :class:`VideoTorchEncoder` wraps the models from
    `torchvision.models`: https://pytorch.org/docs/stable/torchvision/models.html

    :param model_name: the name of the model.
        Supported models include ``r3d_18``, ``mc3_18``, ``r2plus1d_18``
        Default is ``r3d_18``.
    :param pool_strategy: the pooling strategy
        - `None` means that the output of the model will be the 4D tensor
            output of the last convolutional block.
        - `mean` means that global average pooling will be applied to the
            output of the last convolutional block, and thus the output
            of the model will be a 2D tensor.
        - `max` means that global max pooling will be applied.
    """

    def __init__(self,
                 model_name: str = 'r3d_18',
                 channel_axis: int = 1,
                 pool_strategy: str = 'mean',
                 *args, **kwargs):
        """Set Constructor."""
        super().__init__(*args, **kwargs)
        self.channel_axis = channel_axis
        self.model_name = model_name
        self._default_channel_axis = 2
        if pool_strategy not in ('mean', 'max', None):
            raise NotImplementedError(f'unknown pool_strategy: {self.pool_strategy}')
        self.pool_strategy = pool_strategy

    def post_init(self):
        super().post_init()
        import torchvision.models.video as models
        if self.pool_strategy is not None:
            self.pool_fn = getattr(np, self.pool_strategy)
        self.model = getattr(models, self.model_name)(pretrained=True).eval()
        self.to_device(self.model)

    def _get_features(self, x):
        x = self.model.stem(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.flatten(1)
        return x

    def _get_pooling(self, feature_map: 'np.ndarray') -> 'np.ndarray':
        if feature_map.ndim == 2 or self.pool_strategy is None:
            return feature_map
        return self.pool_fn(feature_map, axis=(2, 3))

    @batching
    @as_ndarray
    def encode(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
         Encodes data from a ndarray.

         :param data: a `B x T x (Channel x Height x Width)` numpy ``ndarray``,
            `B` is the size of the batch, `T` is the number of frames
         :return: a `B x D` numpy ``ndarray``, `D` is the output dimension
        """
        if self.channel_axis != self._default_channel_axis:
            data = np.moveaxis(data, self.channel_axis, self._default_channel_axis)
        import torch
        _input = torch.from_numpy(data.astype('float32'))
        if self.on_gpu:
            _input = _input.cuda()
        _feature = self._get_features(_input).detach()
        if not self.on_gpu:
            _feature = _feature.cpu()
        _feature = _feature.numpy()
        return self._get_pooling(_feature)