import os

import clip
import numpy as np
from PIL import Image

from .. import CLIPZeroShotClassifier

cur_dir = os.path.dirname(os.path.abspath(__file__))

def test_clipzeroshotclassifier():
    dog_image = os.path.join(cur_dir, 'imgs/dog.jpg')
    labels = ['cat','dog','human']
    executor = CLIPZeroShotClassifier(labels)
    _, preprocess = clip.load('ViT-B/32')
    im = Image.open(dog_image)
    im_tensor_clip_input = preprocess(im).unsqueeze(0)
    im_tensor_clip_np = im_tensor_clip_input.detach().numpy()
    batch = np.vstack([im_tensor_clip_input, im_tensor_clip_input])
    output = executor.predict(batch)
    assert output.shape == (len(batch), len(labels))
    for res in output:
        assert len(res) == len(labels)
        assert res[labels.index('dog')] == 1
    