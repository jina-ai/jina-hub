# VSEImageEncoder

`VSEImageEncoder` is an image Encoder used to extract Visual Semantic Embeddings.
Taken from the results of [VSE++: Improving Visual-Semantic Embeddings with Hard Negatives](https://arxiv.org/abs/1707.05612).

This model extracts image feature embeddings that can be used in combination with a `VSETextEncoder` which in combination will
put images and its captions in nearby locations in the embedding space

## Usage

Users can use Pod images in several ways:

1. Run with docker:

```shell
docker run jinahub/pod.encoder.vseimageencoder:0.0.4-1.0.0
```

2. Run the Flow API:

```python
 from jina.flow import Flow

 f = (Flow()
     .add(name='vse_encoder', uses='docker://jinahub/pod.encoder.vseimageencoder:0.0.4-1.0.0', port_in=55555, port_out=55556))
```

3. Run with Jina CLI

```shell
 jina pod --uses docker://jinahub/pod.encoder.vseimageencoder:0.0.4-1.0.0 --port-in 55555 --port-out 55556
```

4. Conventional local usage with `uses` argument

```shell
jina pod --uses hub/example/vseimageencoder.yml --port-in 55555 --port-out 55556
```

5. Docker command, Specify the image name along with the version tag.
   The snippet below uses Jina version 1.0.1
   
```shell
 docker pull jinahub/pod.encoder.vseimageencoder:0.0.4-1.0.0
```

## Encode image with the encoder:

Download model:

```shell
wget http://www.cs.toronto.edu/~faghri/vsepp/runs.tar && tar -xvf runs.tar
```

Encode a image:

```python
import numpy as np
from PIL import Image

from encoders.image.VSEImageEncoder import VSEImageEncoder

img_path = '1.jpg' # load images
img = Image.open(img_path)
encoder = VSEImageEncoder(path='runs/f30k_vse++_vggfull/model_best.pth.tar')
embedding = encoder.encode(image.unsqueeze(0).numpy())
```

