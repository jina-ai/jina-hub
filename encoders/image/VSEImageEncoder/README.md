# VSEImageEncoder

`VSEImageEncoder` is the image Encoder used to extract Visual Semantic Embeddings.
Taken from the results of [VSE++: Improving Visual-Semantic Embeddings with Hard Negatives](https://arxiv.org/abs/1707.05612).

This model extracts image feature embeddings that can be used in combination with a `VSETextEncoder` which in combination will
put images and its captions in nearby locations in the embedding space


## Encode image with the encoder:

Download model:

```shell
wget http://www.cs.toronto.edu/~faghri/vsepp/runs.tar && tar -xvf runs.tar
```

Encode an images:

```python
import numpy as np
from PIL import Image

from encoders.image.VSEImageEncoder import VSEImageEncoder

img_path = '1.jpg' # load images
img = Image.open(img_path)
encoder = VSEImageEncoder(path='runs/f30k_vse++_vggfull/model_best.pth.tar')
embedding = encoder.encode(image.unsqueeze(0).numpy())
```

