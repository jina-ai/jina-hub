from typing import List
import torch
from jina.executors.classifiers import BaseClassifier
from jina.executors.devices import TorchDevice
from jina.executors.decorators import batching, as_ndarray
from PIL import Image

class CLIPZeroShotClassifier(TorchDevice, BaseClassifier):
    """
    :class:`ClipZeroShotClassifier` Zero Shot classification for images using OpenAI Clip.
    """

    def __init__(self, labels: List[str], model_name: str ='ViT-B/32', hypothesis_template: str = "a photo of {}",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = labels
        self.model_name = model_name
        self.hypothesis_template = hypothesis_template

    def post_init(self):
        """Load a model from clip specified in `model_name`."""
        import clip

        assert self.model_name in clip.available_models(),\
            f'model_name={self.model_name} not in clip.available_models'

        model, preprocess = clip.load(self.model_name, self.device)
        self.model = model
        self.preprocess = preprocess
        self.encode_labels()
        
    
    def encode_labels(self):
        import clip
        labels = [self.hypothesis_template.format(label) for label in self.labels]
        tokenized_labels = clip.tokenize(labels).to(self.device)
        label_features = self.model.encode_text(tokenized_labels)
        label_features /= label_features.norm(dim=-1, keepdim=True)
        self._label_features = label_features
        
    @batching
    @as_ndarray
    def predict(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        input_torchtensor = torch.from_numpy(data.astype('float32'))
        with torch.no_grad():
            image_features = self.model.encode_image(input_torchtensor)  
        output = (100.0 * image_features @ self._label_features.T).softmax(dim=-1).round().detach().numpy()
        return output
        

    
