__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from typing import Optional

import numpy as np

from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders.frameworks import BaseTorchEncoder
from jina.excepts import PretrainedModelFileDoesNotExist


class CustomImageTorchEncoder(BaseTorchEncoder):
    """
    :class:`CustomImageTorchEncoder` encodes data from a ndarray,
    potentially B x (Channel x Height x Width) into a ndarray of `B x D`.

    Internally, :class:`CustomImageTorchEncoder` wraps any custom torch
    model not part of models from `torchvision.models`.
    https://pytorch.org/docs/stable/torchvision/models.html

    :param model_path: The path where the model is stored.
    :param layer_name: Name of the layer from where to extract the feature map.
    :param pool_strategy: Pooling strategy for the encoder. Default is `mean`.
    :param channel_axis: The axis of the channel, default is 1.
    :param args:  Additional positional arguments.
    :param kwargs: Additional keyword arguments.
    """

    def __init__(self, model_path: Optional[str] = 'models/mobilenet_v2.pth',
                 layer_name: Optional[str] = 'features',
                 pool_strategy: str = 'mean',
                 channel_axis: int = 1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.layer_name = layer_name
        self.channel_axis = channel_axis
        # axis 0 is the batch
        self._default_channel_axis = 1
        self.pool_strategy = pool_strategy

    def post_init(self):
        """Load model. Will raise an exception if the model doesn't exist."""
        super().post_init()
        if self.model_path and os.path.exists(self.model_path):
            import torch
            if self.pool_strategy is not None:
                self.pool_fn = getattr(np, self.pool_strategy)
            self.model = torch.load(self.model_path)
            self.model.eval()
            self.to_device(self.model)
            self.layer = getattr(self.model, self.layer_name)
        else:
            raise PretrainedModelFileDoesNotExist(f'model {self.model_path} does not exist')

    def _get_features(self, data):
        feature_map = None

        def get_activation(model, model_input, output):
            nonlocal feature_map
            feature_map = output.detach()

        handle = self.layer.register_forward_hook(get_activation)
        self.model(data)
        handle.remove()
        return feature_map

    def _get_pooling(self, feature_map: 'np.ndarray') -> 'np.ndarray':
        if feature_map.ndim == 2 or self.pool_strategy is None:
            return feature_map
        return self.pool_fn(feature_map, axis=(2, 3))

    @batching
    @as_ndarray
    def encode(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        Encode input data into `np.ndarray` of size `B x D`.
        Where `B` is the batch size and `D` is the Dimension.

        :param data: An array in size `B`.
        :param args:  Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: An ndarray in size `B x D`.
        """
        if self.channel_axis != self._default_channel_axis:
            data = np.moveaxis(data, self.channel_axis, self._default_channel_axis)
        import torch
        _input = torch.from_numpy(data.astype('float32'))
        if self.on_gpu:
            _input = _input.cuda()
        _feature = self._get_features(_input).detach()
        if self.on_gpu:
            _feature = _feature.cpu()
        _feature = _feature.numpy()
        return self._get_pooling(_feature)
