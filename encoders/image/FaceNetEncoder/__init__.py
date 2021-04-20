__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np
import torch

from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders.frameworks import BaseTorchEncoder
from jina.executors.devices import TorchDevice


class FaceNetEncoder(BaseTorchEncoder, TorchDevice):
    """FaceNetEncoder encodes face images into an embedding with 512 dimensions.
    Works best with outputs from FaceNetSegmenter (e.g. images with faces, 160x160, normalized).

    - Input shape: `BatchSize x (Channels x Height x Width)`
    - Output shape: `BatchSize x 512`

    `Channels` dimension can be changed (e.g. set `channel_axis` to 1 for channels-first mode instead of channels-last).

    :param pretrained_weights: Weights to use for face embedder. Options: "vggface2", "casia-webface"
    :param channel_axis: Axis of channels in the image. Default is 3 (channels-last), use 1 for channels-first.
    """

    def __init__(self,
                 pretrained_weights: str = 'vggface2',
                 channel_axis: int = 1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_weights = pretrained_weights
        self.channel_axis = channel_axis

        self._default_channel_axis = 1

    def post_init(self):
        from facenet_pytorch import InceptionResnetV1

        self.model = InceptionResnetV1(pretrained=self.pretrained_weights,
                                       classify=False,
                                       device=self.device).eval()
        self.model.to(self.device)

    @batching
    @as_ndarray
    def encode(self, content: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """Transform a numpy `ndarray` of shape `BatchSize x (Height x Width x Channel)`
        into a numpy `ndarray` of shape `Batchsize x EmbeddingDimension`.

        :param content: A numpy `ndarray` of strings.
        :param args: Additional positional arguments.
        :param kwargs: Additional positional arguments.
        :return: A `BatchSize x EmbeddingDimension` numpy array.
        """
        if self.channel_axis != self._default_channel_axis:
            content = np.moveaxis(content, self.channel_axis, self._default_channel_axis)

        with torch.no_grad():
            images = torch.from_numpy(content.astype('float32')).to(self.device)
            embedded_faces = self.model(images)
        return embedded_faces.detach().cpu()
