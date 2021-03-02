# ImageTorchEncoder

`ImageTorchEncoder` encodes data from a ndarray, potentially BatchSize x (Channel x Height x Width) into a ndarray of `BatchSize * d`. Internally, :class:`ImageTorchEncoder` wraps the models from [`tensorflow.keras.applications`](https://keras.io/applications/)


## Usage:

Initialise this Executor specifying parameters i.e.:

| `param_name`  | `param_remarks` |
| ------------- | ------------- |
| `model_path`  | the directory path of the model in the `SavedModel` format  |
| `model_name`  | includes `resnet18`, `alexnet`, `squeezenet1_0`, `vgg16`, `densenet161`, `inception_v3` |
| `channel_axis`| axis of the color channel  |
| `pool_strategy` | strategy of pooling operation |

Datasets `resnet18`, `alexnet`, `squeezenet1_0`, `vgg16`, `densenet161`, `inception_v3` can be used for pretraining the model.
The pretrained default path is the result of downloading the models in `download.sh`

### Snippets:

Initialise ImageTorchEncoder:

`ImageTorchEncoder(model_path='pretrained', channel_axis=1, metas=metas, model_name=MobileNetV2)`

Users can use Pod images in several ways:

- Run with Docker (`docker run`)
  - ```bash
    docker run jinahub/pod.encoder.imagetorchencoder:0.0.8-1.0.4 --port-in 55555 --port-out 55556
    ```

- Flow API
  - ```python
    from jina.flow import Flow
    f = (Flow()
        .add(name='my_encoder', uses='jinahub/pod.encoder.imagetorchencoder:0.0.8-1.0.4', port_in=55555, port_out=55556)
    ```

- Jina CLI
  - ```bash
    jina pod --uses docker://jinahub/pod.encoder.imagetorchencoder:0.0.8-1.0.4 --port-in 55555 --port-out 55556
    ```

- Conventional local usage with `uses` argument
  - ```bash
    jina pod --uses hub/example/config.yml --port-in 55555 --port-out 55556
    ```

- Docker command

  - Specify the image name along with the version tag. The snippet below uses Jina version `1.0.4`

  - ```bash
    docker pull jinahub/pod.encoder.imagetorchencoder:0.0.8-1.0.4
    ```