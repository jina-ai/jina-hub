# TirgImageEncoder

**TIRG** model was original proposed in [Composing Text and Image for Image Retrieval - An Empirical Odyssey](https://arxiv.org/abs/1812.07119).
It has been used in Jina multimodal retrieval example.
The **TIRG** model uses `TirgImageEncoder` and encode images into embeddings.

As a part of **TIRG** model,
`TirgImageEncoder` encodes data from a ndarray, potentially BatchSize x (Channel x Height x Width) into a BatchSize x d ndarray.

Note: In the TIRG multimodal retrieval example, **TIRG** model takes `TirgMultimodalEncoder` and encode images together with it's associated texts into embeddings at query time.

## Usage

Users can use Pod images in several ways:

1. To use it in the index flow, you can use our pre-trained docker image with yaml configuration, The YAML configuration has been created in our [TIRG multimodal search example](https://github.com/jina-ai/examples/blob/master/multimodal-search-tirg/flow-index.yml):

```yaml
!Flow
version: '1'
pods:
  - ...
  - name: encode
    uses: docker://jinahub/pod.encoder.tirgimageencoder:0.0.4-1.0.1
    shards: $JINA_PARALLEL
    timeout_ready: 600000
    read_only: true
  - ...
```

2. Run with docker:

```shell
docker run jinahub/pod.encoder.tirgimageencoder:0.0.4-1.0.1
```

3. Run the Flow API:

```python
 from jina.flow import Flow
 f = (Flow()
     .add(name='tirg_image_encoder', uses='docker://jinahub/pod.encoder.tirgimageencoder:0.0.4-1.0.1', port_in=55555, port_out=55556))
```

4. Run with Jina CLI

```shell
 jina pod --uses docker://jinahub/pod.encoder.tirgimageencoder:0.0.4-1.0.1 --port-in 55555 --port-out 55556
```

5. Conventional local usage with `uses` argument

```shell
jina pod --uses hub/example/tirgimageencoder.yml --port-in 55555 --port-out 55556
```

6. Docker command, Specify the image name along with the version tag.
   The snippet below uses Jina version 1.0.1

```shell
 docker pull jinahub/pod.encoder.tirgimageencoder:0.0.4-1.0.1
```

## Encode with the encoder:

Download model and the text resources:

**Note:** The following code will download the pre-trained TIRG model on fashion-200k dataset.
The other pre-trained models can be found [here](https://github.com/google/tirg#pretrained-models).


```shell
wget -O checkpoint.pth https://storage.googleapis.com/image_retrieval_css/pretrained_models/checkpoint_fashion200k.pth
wget https://github.com/bwanglzu/tirg/raw/main/texts.pkl
```

Encode a sample of three images:

```python
import torch
import numpy as np
from PIL import Image

from encoders.image.TirgImageEncoder import TirgImageEncoder

img_paths = ['1.jpeg', '2.jpeg', '3.jpeg'] # load images
for img_name in img_paths:
    img = Image.open(img_path)
    imgs.append(img)
encoder = TirgImageEncoder(
    model_path='checkpoint.pth',
    texts_path='texts.pkl',
    channel_axis=1,
)
imgs = torch.stack(imgs).float()
embeddings = encoder.encode(imgs.numpy())
```
