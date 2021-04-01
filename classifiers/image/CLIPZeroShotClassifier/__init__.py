from typing import List

from jina.executors.classifiers import BaseClassifier
from jina.executors.decorators import as_ndarray, batching
from jina.executors.devices import TorchDevice
from PIL import Image


class CLIPZeroShotClassifier(TorchDevice, BaseClassifier):
    """
    :class:`ClipZeroShotClassifier` Zero Shot classification for images using OpenAI Clip.

    Internally, :class:`ClipZeroShotClassifier` wraps the `CLIP` modeL from https://github.com/openai/CLIP
    :param labels: labels for the classification task. 
    :param model_name: The name of the model. Supported models include ``ViT-B/32`` and ``RN50``.
    :param args: Additional positional arguments.
    :param kwargs: Additional positional arguments.
    """

    def __init__(self, labels: List[str], model_name: str = 'ViT-B/32',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = labels
        self.model_name = model_name

    def post_init(self):
        """Load a model from clip specified in `model_name`."""
        import clip

        assert self.model_name in clip.available_models(), \
            f'model_name={self.model_name} not in clip.available_models'

        model, preprocess = clip.load(self.model_name, self.device)
        self.model = model
        self.preprocess = preprocess
        self.encode_labels()

    def encode_labels(self):
        import clip
        tokenized_labels = clip.tokenize(self.labels).to(self.device)
        label_features = self.model.encode_text(tokenized_labels)
        label_features /= label_features.norm(dim=-1, keepdim=True)
        self._label_features = label_features

    @batching
    @as_ndarray
    def predict(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        import torch

        input_torchtensor = torch.from_numpy(data.astype('float32'))
        with torch.no_grad():
            image_features = self.model.encode_image(input_torchtensor)
        output = (100.0 * image_features @ self._label_features.T).softmax(dim=-1).round().detach().numpy()
        return output
