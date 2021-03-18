__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np
import torch

from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders.frameworks import BaseTorchEncoder
from jina.executors.devices import TorchDevice


class FaceNetEncoder(BaseTorchEncoder, TorchDevice):
    """FaceNetEncoder encodes images using the following algorithm:
    * Face detector detects faces on the image
        * If multiple faces are detected the largest is selected (default heuristic)
        * If no faces are detected a dummy face is used (an array with zeroes)
    * The face is encoded to an embedding

    - Input shape: `BatchSize x (Height x Width x Channels)`
    - Output shape: `BatchSize x EmbeddingDimension`

    EmbeddingDimension is equal to 512.

    `Channels` dimension can be changed (e.g. set `channel_axis` to 1 for channels first mode instead of channels last).

    :param selection_method: Heuristic to use to select a single face from the image. Options:
        "probability": highest probability selected
        "largest": largest box selected
        "largest_over_threshold": largest box over a certain probability selected
        "center_weighted_size": box size minus weighted squared offset from image center
    :param pretrained_weights: Weights to use for face embedder. Options: "vggface2", "casia-webface"
    :param channel_axis: Axis of channels in the image. Default is 3 (channels-last), use 1 for channels-first.
    """

    def __init__(self,
                 selection_method: str = 'largest',
                 pretrained_weights: str = 'vggface2',
                 channel_axis: int = 3,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection_method = selection_method
        self.pretrained_weights = pretrained_weights
        self.channel_axis = channel_axis

        self.face_detector = None
        self.face_embedder = None

        self._default_channel_axis = 3

    def post_init(self):
        from facenet_pytorch import MTCNN, InceptionResnetV1

        self.face_detector = MTCNN(selection_method=self.selection_method,
                                   device=self.device).eval()
        self.face_embedder = InceptionResnetV1(pretrained=self.pretrained_weights,
                                               classify=False,
                                               device=self.device).eval()

    @batching
    @as_ndarray
    def encode(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """Transform a numpy `ndarray` of shape `BatchSize x (Height x Width x Channel)`
        into a numpy `ndarray` of shape `Batchsize x EmbeddingDimension`.

        :param data: A numpy `ndarray` of strings.
        :param args: Additional positional arguments.
        :param kwargs: Additional positional arguments.
        :return: A `BatchSize x EmbeddingDimension` numpy array.
        """
        if self.channel_axis != self._default_channel_axis:
            data = np.moveaxis(data, self.channel_axis, self._default_channel_axis)

        with torch.no_grad():
            images = torch.from_numpy(data.astype('float32')).to(self.device)

            faces = self.face_detector(images)
            faces = [face if face is not None else self.face_placeholder()
                     for face in faces]
            faces = torch.stack(faces, dim=0)

            embedded_faces = self.face_embedder(faces)
        return embedded_faces.cpu()

    def face_placeholder(self):
        return torch.zeros(3, self.face_detector.image_size, self.face_detector.image_size,
                           device=self.device)
