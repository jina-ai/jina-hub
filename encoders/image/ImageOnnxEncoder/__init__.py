__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional
import numpy as np

from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders.frameworks import BaseOnnxEncoder


class ImageOnnxEncoder(BaseOnnxEncoder):
    """
    :class:`ImageOnnxEncoder` encodes data from a ndarray,
    potentially B x (Channel x Height x Width) into a ndarray of `B x D`.

    Internally, :class:`OnnxImageEncoder` wraps the models from `onnxruntime`.

    :param model_path: Path of the model in the format of `.onnx`.
        Check a  list of available pretrained models at
        https://github.com/onnx/models#image_classification and download the
        git LFS to your local path. The ``model_path`` is the local path of the
        ``.onnx`` file, e.g. ``/tmp/onnx/mobilenetv2-1.0.onnx``.
    :param output_feature: Name of the layer for feature extraction.
    :param pool_strategy: the pooling strategy. Options are:
        - `None`: Means that the output of the model will be the 4D tensor
            output of the last convolutional block.
        - `mean`: Means that global average pooling will be applied to the
            output of the last convolutional block and thus the output of
            the model will be a 2D tensor.
        - `max`: Means that global max pooling will be applied.
    """

    def __init__(self,
                 model_path: Optional[str] = 'models/vision/classification/mobilenet/model/mobilenetv2-7.onnx',
                 output_feature: Optional[str] = 'mobilenetv20_features_relu1_fwd',
                 pool_strategy: str = 'mean', *args, **kwargs):
        super().__init__(model_path=model_path, output_feature=output_feature, *args, **kwargs)
        self.pool_strategy = pool_strategy
        if pool_strategy not in ('mean', 'max', None):
            raise NotImplementedError(f'unknown pool_strategy: {self.pool_strategy}')

    @batching
    @as_ndarray
    def encode(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        Encode data into a ndarray of `B x D`. `
        B` is the batch size and `D` is the Dimension.

        :param data: A `B x (Channel x Height x Width)` numpy ``ndarray``,
            `B` is the size of the batch
        :return: a `B x D` numpy ``ndarray``, `D` is the output dimension
        """
        results = []
        for idx in range(data.shape[0]):
            img = np.expand_dims(data[idx, :, :, :], axis=0).astype('float32')
            data_encoded, *_ = self.model.run([self.outputs_name, ], {self.inputs_name: img})
            results.append(data_encoded)
        feature_map = np.concatenate(results, axis=0)
        if feature_map.ndim == 2 or self.pool_strategy is None:
            return feature_map
        return getattr(np, self.pool_strategy)(feature_map, axis=(2, 3))