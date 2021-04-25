__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import sys
import pickle

from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders.frameworks import BaseTorchEncoder
from jina.excepts import PretrainedModelFileDoesNotExist


class TirgImageEncoder(BaseTorchEncoder):
    """
    Encode BatchSize x (Channel x Height x Width) ndarray into BatchSize * d ndarray.

    :class:`TirgImageEncoder` is originally proposed in the paper *Composing Text and Image for Image Retrieval - An Empirical Odyssey*.
    It can been used for multimodal image retrieval purpose.

    :param model_path: the directory of the TIRG model.
    :param texts_path: the pickled training text of the TIRG model.
    :param channel_axis: The axis of the channel, default -1, will move the axis of input document content from -1 to 1.
    :param args: additional positional arguments.
    :param kwargs: additional positional arguments.
    """

    def __init__(self, model_path: str = 'checkpoint.pth',
                 texts_path: str = 'texts.pkl',
                 channel_axis: int = -1, 
                 *args,
                 **kwargs):
        """
        Class constructor.
        """
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.texts_path = texts_path
        self.channel_axis = channel_axis
        # axis 0 is the batch
        self._default_channel_axis = 1

    def post_init(self):
        """Load `TIRG` model."""
        super().post_init()
        from .img_text_composition_models import TIRG
        import torch
        if self.model_path and os.path.exists(self.model_path):
            with open (self.texts_path, 'rb') as fp:
                texts = pickle.load(fp)
            self.model = TIRG(texts, 512)
            model_sd = torch.load(self.model_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(model_sd['model_state_dict'])
            self.model.eval()
            self.to_device(self.model)
        else:
            raise PretrainedModelFileDoesNotExist(f'model {self.model_path} does not exist')

    def _get_features(self, content):
        return self.model.extract_img_feature(content)

    @batching
    @as_ndarray
    def encode(self, content: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        Encode input document content into `np.ndarray`.

        :param content: Image to be encoded, expected a `np.ndarray` of BatchSize x (Channel x Height x Width).
        :param args: additional positional arguments.
        :param kwargs: additional positional arguments.
        :return feature: Encoded result as `np.ndarray`.
        """
        import numpy as np
        if self.channel_axis != self._default_channel_axis:
            content = np.moveaxis(content, self.channel_axis, self._default_channel_axis)
        import torch
        _input = torch.from_numpy(content.astype('float32'))
        if self.on_gpu:
            _input = _input.cuda()
        feature = self._get_features(_input).detach()
        if self.on_gpu:
            feature = feature.cpu()
        feature = feature.numpy()
        return feature