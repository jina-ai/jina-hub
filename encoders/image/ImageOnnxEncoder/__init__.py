__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional
import numpy as np

from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders.frameworks import BaseOnnxEncoder


class ImageOnnxEncoder(BaseOnnxEncoder):
    """
    :class:`ImageOnnxEncoder` encodes data from a ndarray, potentially B x (Channel x Height x Width) into a
        ndarray of `B x D`.
    Internally, :class:`OnnxImageEncoder` wraps the models from `onnxruntime`.
    """

    def __init__(self,
                 model_path: Optional[str] = 'models/vision/classification/mobilenet/model/mobilenetv2-7.onnx',
                 output_feature: Optional[str] = 'mobilenetv20_features_relu1_fwd',
                 pool_strategy: str = 'mean', *args, **kwargs):
        """
        :param pool_strategy: the pooling strategy
            - `None` means that the output of the model will be the 4D tensor output of the last convolutional block.
            - `mean` means that global average pooling will be applied to the output of the last convolutional block,
            and thus the output of the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        """
        super().__init__(model_path=model_path, output_feature=output_feature, *args, **kwargs)
        self.pool_strategy = pool_strategy
        if pool_strategy not in ('mean', 'max', None):
            raise NotImplementedError(f'unknown pool_strategy: {self.pool_strategy}')

    @batching
    @as_ndarray
    def encode(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        :param data: a `B x (Channel x Height x Width)` numpy ``ndarray``, `B` is the size of the batch
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