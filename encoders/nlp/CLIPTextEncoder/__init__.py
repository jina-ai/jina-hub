__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np
import torch
import clip

from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders.frameworks import BaseTorchEncoder
from jina.executors.devices import TorchDevice

class CLIPTextEncoder(BaseTorchEncoder):
    """
    :class:`CLIPImageEncoder` encodes data from a `np.ndarray` (of strings) of shape `BatchSize` into
    a `np.ndarray` of shape `Batchsize x EmbeddingDim`. 

    Internally, :class:`CLIPImageEncoder` wraps the `CLIP` model
    https://github.com/openai/CLIP
    """
    def __init__(self, model_name: str ='ViT-B/32',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name

    def post_init(self):
        """
        :param model_name: the name of the model. Supported models include ``ViT-B/32`` and ``RN50``.
        """
        import clip
        model, _ = clip.load(self.model_name, self.device)
        self.model = model
        #self.preprocess = preprocess

    @batching
    @as_ndarray
    def encode(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':

        input_torchtensor = clip.tokenize(data)
        if self.on_gpu:
            input_torchtensor = input_torchtensor.cuda()

        with torch.no_grad():
            #self.logger.warning(f'text data shape {data.shape}')
            #self.logger.warning(f'text data encoded shape {self.model.encode_text(clip.tokenize(data))}')
            embedded_data = self.model.encode_text(input_torchtensor)

        embedded_data = embedded_data.numpy()
        return embedded_data
    



