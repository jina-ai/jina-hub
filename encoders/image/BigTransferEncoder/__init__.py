__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from typing import Optional

import numpy as np

from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders.frameworks import BaseTFEncoder
from jina.excepts import PretrainedModelFileDoesNotExist


class BigTransferEncoder(BaseTFEncoder):
    """
    :class:`BigTransferEncoder` is Big Transfer (BiT) presented by
    Google (https://github.com/google-research/big_transfer).
    Uses pretrained BiT to encode document content from a ndarray, potentially
    B x (Channel x Height x Width) into a ndarray of `B x D`.
    Internally, :class:`BigTransferEncoder` wraps the models from
    https://storage.googleapis.com/bit_models/.

    .. warning::

        Known issue: this does not work on tensorflow==2.2.0,
        https://github.com/tensorflow/tensorflow/issues/38571

    :param model_path: the path of the model in the `SavedModel` format.
        The pretrained model can be downloaded at
        wget https://storage.googleapis.com/bit_models/Imagenet21k/[model_name]/feature_vectors/saved_model.pb
        wget https://storage.googleapis.com/bit_models/Imagenet21k/[model_name]/feature_vectors/variables/variables.data-00000-of-00001
        wget https://storage.googleapis.com/bit_models/Imagenet21k/[model_name]/feature_vectors/variables/variables.index

        ``[model_name]`` includes `R50x1`, `R101x1`, `R50x3`, `R101x3`, `R152x4`

        The `model_path` should be a directory path, which has the following structure.

        .. highlight:: bash
         .. code-block:: bash

            .
            ├── saved_model.pb
            └── variables
                ├── variables.data-00000-of-00001
                └── variables.index

        :param channel_axis: the axis id of the channel, -1 indicate the color
            channel info at the last axis. If given other, then `
            `np.moveaxis(content, channel_axis, -1)`` is performed before :meth:`encode`.
    """

    def __init__(self,
                 model_path: Optional[str] = '/workspace/pretrained',
                 channel_axis: int = 1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel_axis = channel_axis
        self.model_path = model_path

    def post_init(self):
        """Load model. Raise exception if model doesn't exist"""
        super().post_init()
        if self.model_path and os.path.exists(self.model_path):
            self.to_device()
            import tensorflow as tf
            self.logger.info(f'model_path: {self.model_path}')
            _model = tf.saved_model.load(self.model_path)
            self.model = _model.signatures['serving_default']
            self._get_input = tf.convert_to_tensor
        else:
            raise PretrainedModelFileDoesNotExist(f'model at {self.model_path} does not exist')

    @batching
    @as_ndarray
    def encode(self, content: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        Encode content into a ndarray of `B x D`.
        Where `B` is the batch size and `D` is the Dimension.

        :param content: an array in size `B`
        :return: an ndarray in size `B x D`.
        """
        if self.channel_axis != -1:
            content = np.moveaxis(content, self.channel_axis, -1)
        _output = self.model(self._get_input(content.astype(np.float32)))
        return _output['output_1'].numpy()
