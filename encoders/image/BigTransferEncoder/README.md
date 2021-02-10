# BigTransferEncoder

BigTransferEncoder from [Big Transfer (BiT)](https://github.com/google-research/big_transfer) is Google's state-of-the-art transfer learning model for computer vision. This class uses pretrained BiT to encode data from an `ndarray`, potentially B x (Channel x Height x Width) into an `ndarray` of shape `B x D`. Internally, :class:`BigTransferEncoder` wraps the models from [here](https://storage.googleapis.com/bit_models/).
Additional links:
- [Tensorflow Blog](https://blog.tensorflow.org/2020/05/bigtransfer-bit-state-of-art-transfer-learning-computer-vision.html)
- [Google AI Blog](https://ai.googleblog.com/2020/05/open-sourcing-bit-exploring-large-scale.html)
- [BiT Research Paper](https://arxiv.org/abs/1912.11370)

The model is pre-trained on [ImageNet](http://www.image-net.org/) data


## Usage:

Initialise this Executor specifying parameters i.e.:

| `param_name`  | `param_remarks` |
| ------------- | ------------- |
| `model_path`  | the directory path of the model in the `SavedModel` format  |
| `model_name`  | includes `R50x1`, `R101x1`, `R50x3`, `R101x3`, `R152x4`  |
| `channel_axis`| axis id of the channel), etc.  |

The pretrained default path is the result of downloading the models in `download.sh`

### Snippets:

Initialise BigTransferEncoder:

`BigTransferEncoder(model_path='pretrained', channel_axis=1, metas=metas)`

Users can use Pod images in several ways:

- Run with Docker (`docker run`)
  - ```bash
    docker run jinahub/pod.encoder.bigtransferencoder:0.0.6-0.9.33 --port-in 55555 --port-out 55556
    ```
    
- Flow API
  - ```python
    from jina.flow import Flow

    f = (Flow()
        .add(name='my-encoder', image='jinahub/pod.encoder.bigtransferencoder:0.0.6-0.9.33', port_in=55555, port_out=55556)
        .add(name='my-indexer', uses='indexer.yml'))
    ```
    
- Jina CLI
  - ```bash
    jina pod --uses jinahub/pod.encoder.bigtransferencoder:0.0.6-0.9.33 --port-in 55555 --port-out 55556
    ```
    
- Conventional local usage with `uses` argument
  - ```bash
    jina pod --uses hub/example/bigtransferencoder.yml --port-in 55555 --port-out 55556
    ```
    
- Docker command

  - Specify the image name along with the version tag. The snippet below uses Jina version `0.9.20`

  - ```bash
    docker pull jinahub/pod.encoder.bigtransferencoder:0.0.6-0.9.33
    ```
   
 Note:
 
 One of the limitations with the Hub Executors currently is the tags - all Executor images should have the versions appended in the name i.e.
 if the version is `0.0.6-0.9.33`, the image name would be `jinahub/pod.encoder.bigtransferencoder:0.0.6-0.9.33`.
   
