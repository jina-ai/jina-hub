__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import List, Dict

import numpy as np
import torch
from jina.executors.decorators import single

from jina.executors.devices import TorchDevice
from jina.executors.segmenters import BaseSegmenter


class FaceNetSegmenter(TorchDevice, BaseSegmenter):
    """FaceNetSegmenter segments faces from an image.

    - Input shape: `(Height x Width x Channels)`
    - Output shape: `NumFaces x (Channels x ImageSize x ImageSize)`

    `Channels` dimension can be changed (e.g. set `channel_axis` to 0 for channels first mode instead of channels last).

    :param image_size: Height and width of a detected face. Smaller faces are upscaled.
    :param margin: Margin to add to bounding box, in terms of pixels in the final image.
    :param selection_method: Heuristic to use to select a single face from the image. Options:
        "probability": highest probability selected
        "largest": largest box selected
        "largest_over_threshold": largest box over a certain probability selected
        "center_weighted_size": box size minus weighted squared offset from image center
    :param post_process: Flag for normalizing the output image. Required if you want to pass
        these face to the FaceNetEmbedder.
    :param keep_all: Flag for keeping all detected faces rather than a single face.
    :param min_face_size: Minimum face size to search for.
    :param channel_axis: Axis of channels in the image. Default is 2 (channels-last), use 0 for channels-first.
    """

    def __init__(self,
                 image_size: int = 160,
                 margin: int = 0,
                 selection_method: str = 'largest',
                 post_process: bool = True,
                 keep_all: bool = False,
                 min_face_size: int = 20,
                 channel_axis: int = 2,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_size = image_size
        self.margin = margin
        self.selection_method = selection_method
        self.post_process = post_process
        self.keep_all = keep_all
        self.min_face_size = min_face_size
        self.channel_axis = channel_axis

        self._default_channel_axis = 2

    def post_init(self):
        from facenet_pytorch import MTCNN

        self.face_detector = MTCNN(selection_method=self.selection_method,
                                   image_size=self.image_size,
                                   margin=self.margin,
                                   device=self.device,
                                   post_process=self.post_process,
                                   min_face_size=self.min_face_size,
                                   keep_all=self.keep_all).eval()

    @single
    def segment(self, data: 'np.ndarray', *args, **kwargs) -> List[Dict]:
        """Transform a numpy `ndarray` of shape `(Height x Width x Channel)`
        into a list with dicts that contain cropped images.

        :param data: A numpy `ndarray` that represents a single image.
        :param args: Additional positional arguments.
        :param kwargs: Additional positional arguments.
        :return: A list with dicts that contain cropped images.
        """
        if self.channel_axis != self._default_channel_axis:
            data = np.moveaxis(data, self.channel_axis, self._default_channel_axis)

        with torch.no_grad():
            image = torch.from_numpy(data.astype('float32')).to(self.device)
            # Create a batch of size 1
            image = image.unsqueeze(0)

            faces, probabilities = self.face_detector(image, return_prob=True)

            if self.keep_all:
                # All faces and probabilities are grouped in the first dimension of the first batch element
                faces = faces[0]
                probabilities = probabilities[0]

            results = [
                dict(
                    offset=0,
                    weight=probability,
                    blob=face.detach().numpy(),
                )
                for face, probability in zip(faces, probabilities)
                if face is not None
            ]
            return results
