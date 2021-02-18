# TirgImageEncoder

**Tirg** model was original proposed in [Composing Text and Image for Image Retrieval - An Empirical Odyssey](https://arxiv.org/abs/1812.07119).
It has been used in Jina multimodal retrieval example.

As a part of **TIRG** model,
`TirgImageEncoder` encodes data from a ndarray, potentially B x (Channel x Height x Width) into a B x d ndarray.

To use it in the index flow, you can use our pre-trained docker image:
```yaml
!Flow
version: '1'
pods:
  - name: read
    uses: pods/image-read.yml
    read_only: true
  - name: normalize
    uses: pods/normalize.yml
    read_only: true
  - name: encode
    uses: docker://jinahub/pod.encoder.tirgimageencoder:0.0.4-1.0.1
    shards: $JINA_PARALLEL
    timeout_ready: 600000
    read_only: true
  - name: vec_idx
    polling: any
    uses: pods/vec.yml
    shards: $JINA_SHARDS
    timeout_ready: 100000 # larger timeout as in query time will read all the data
  - name: image_kv
    polling: any
    uses: pods/doc.yml
    needs: [gateway]
    shards: $JINA_SHARDS
    timeout_ready: 100000 # larger timeout as in query time will read all the data
  - name: join_all
    uses: _merge
    needs: [image_kv, vec_idx]
    read_only: true
```

The YAML configuration has been created in our [TIRG multimodal search example]().

To encode a batch of images directly with the encoder:

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

