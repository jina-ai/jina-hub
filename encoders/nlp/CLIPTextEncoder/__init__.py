__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np
import torch

from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders.frameworks import BaseTorchEncoder
from jina.executors.devices import TorchDevice


class CLIPTextEncoder(BaseTorchEncoder):
    """Encode ``Document`` content from a `np.ndarray` (of strings) of length `BatchSize` into
    a `np.ndarray` of shape `Batchsize x EmbeddingDimension`. 

    Internally, :class:`CLIPImageEncoder` wraps the `CLIP` model from `https://github.com/openai/CLIP`.

    :param model_name: The name of the model. Supported models include ``ViT-B/32`` and ``RN50``.
    :param args: Additional positional arguments.
    :param kwargs: Additional positional arguments.
    """

    def __init__(self, model_name: str = 'ViT-B/32',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name

    def post_init(self):
        """Load a model from clip specified in `model_name`. """
        import clip
        model, _ = clip.load(self.model_name, self.device)
        self.model = model

    @batching
    @as_ndarray
    def encode(self, content: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """Transform a `np.ndarray` of strings of length `BatchSize` into
        a `np.ndarray` of shape `Batchsize x EmbeddingDimension`.

        :param content: A `np.ndarray` of strings.
        :param args: Additional positional arguments.
        :param kwargs: Additional positional arguments.
        :return: A `BachSize x EmbeddingSize` numpy `ndarray`.
        """
        import clip
        input_torchtensor = clip.tokenize(content)
        if self.on_gpu:
            input_torchtensor = input_torchtensor.cuda()

        with torch.no_grad():
            embedded_data = self.model.encode_text(input_torchtensor)

        embedded_data = embedded_data.cpu().numpy()
        return embedded_data
