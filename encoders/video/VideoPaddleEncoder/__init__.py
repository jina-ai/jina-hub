__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np

from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders.frameworks import BasePaddleEncoder


class VideoPaddleEncoder(BasePaddleEncoder):
    """
    Encode ``Document`` content from a ndarray, using the models from `paddlehub`.

    Encodes content from a ndarray, potentially B x T x
    (Channel x Height x Width) into andarray of `B x D`.
    Internally, :class:`VideoPaddleEncoder` wraps the models from `paddlehub`.
    https://github.com/PaddlePaddle/PaddleHub

    param model_name: the name of the model. Supported models include
        ``tsn_kinetics400``, ``stnet_kinetics400``, ``tsm_kinetics400``
    :param output_feature: the name of the layer for feature extraction.
        Please use the following values for the supported models:
            ``tsn_kinetics400``: `@HUB_tsn_kinetics400@reduce_mean_0.tmp_0`
            ``stnet_kinetics400``: ``@HUB_stnet_kinetics400@reshape2_6.tmp_0``
            ``tsm_kinetics400``: ``@HUB_tsm_kinetics400@reduce_mean_0.tmp_0``

    :param pool_strategy: the pooling strategy
        - `None` means that the output of the model will be the output feature.
        - `mean` means that global average pooling will be applied to the output
            feature, and thus the output of the model will be a 2D tensor.
        - `max` means that global max pooling will be applied.
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(self,
                 model_name: str = 'tsn_kinetics400',
                 output_feature: str = '@HUB_tsn_kinetics400@reduce_mean_0.tmp_0',
                 pool_strategy: str = None,
                 channel_axis: int = 2,
                 *args, **kwargs):
        """Set Constructor."""
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.outputs_name = output_feature
        self.pool_strategy = pool_strategy
        self.channel_axis = channel_axis
        self._default_channel_axis = 2
        if pool_strategy not in ('mean', 'max', None):
            raise NotImplementedError(f'unknown pool_strategy: {pool_strategy}')

    def post_init(self):
        """Load VideoPaddleEncoder model"""
        import paddlehub as hub
        module = hub.Module(name=self.model_name)
        inputs, outputs, self.model = module.context(trainable=False)
        self.inputs_name = inputs[0].name
        self.exe = self.to_device()

    def close(self):
        self.exe.close()
        super().close()

    @batching
    @as_ndarray
    def encode(self, content: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        Encode ``Document`` content from a ndarray.

        Potentially B x T x (Channel x Height x Width) into an ndarray of `B x D`.

        :param content: a `B x T x (Channel x Height x Width)` numpy ``ndarray``,
            `B` is the size of the batch, `T` is the number of frames
        :return: a `B x D` numpy ``ndarray``, `D` is the output dimension
        :param args:  Additional positional arguments
        :param kwargs: Additional keyword arguments
        """
        if self.channel_axis != self._default_channel_axis:
            content = np.moveaxis(content, self.channel_axis, self._default_channel_axis)
        feature_map, *_ = self.exe.run(
            program=self.model,
            fetch_list=[self.outputs_name],
            feed={self.inputs_name: content.astype('float32')},
            return_numpy=True
        )
        if feature_map.ndim == 2 or self.pool_strategy is None:
            return feature_map
        return self.get_pooling(feature_map)

    def get_pooling(self, content: 'np.ndarray') -> 'np.ndarray':
        """
        Get np.ndarray with pooling strategy.

        :param content: An np.ndarray of the ``feature_map``
        :return: A `B x D` numpy ``ndarray``, `D` is the output dimension
        """
        _reduce_axis = tuple((i for i in range(len(content.shape)) if i > 1))
        return getattr(np, self.pool_strategy)(content, axis=_reduce_axis)
