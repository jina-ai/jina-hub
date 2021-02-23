# ImageKerasEncoder

`ImageKerasEncoder` encodes data from a ndarray, potentially BatchSize x (Channel x Height x Width) into a ndarray of `BatchSize * d`. Internally, :class:`ImageKerasEncoder` wraps the models from `tensorflow.keras.applications`. https://keras.io/applications/


## Usage:

Initialise this Executor specifying parameters i.e.:

| `param_name`  | `param_remarks` |
| ------------- | ------------- |
| `model_path`  | the directory path of the model in the `SavedModel` format  |
| `model_name`  | includes `MobileNetV2`, `DenseNet121`, `ResNet50`, `VGG16`, `Xception`, `InceptionV3` |
| `channel_axis`| axis of the color channel  |
| `pool_strategy` | strategy of pooling operation |

The model is pre-trained on [MobileNetV2] data
The pretrained default path is the result of downloading the models in `download.sh`

### Snippets:

Initialise ImageKerasEncoder:

`ImageKerasEncoder(model_path='pretrained', channel_axis=1, metas=metas, model_name=MobileNetV2)`

Users can use Pod images in several ways:

- Run with Docker (`docker run`)
  - ```bash
    docker run jinahub/pod.encoder.imagekerasencoder:0.0.8-0.9.28 --port-in 55555 --port-out 55556
    ```
    
- Flow API
  - ```python
    from jina.flow import Flow

    f = (Flow()
        .add(name='my-encoder', image='jinahub/pod.encoder.imagekerasencoder:0.0.8-0.9.28', port_in=55555, port_out=55556)
    ```
    
- Jina CLI
  - ```bash
    jina pod --uses jinahub/pod.encoder.imagekerasencoder:0.0.8-0.9.28 --port-in 55555 --port-out 55556
    ```
    
- Conventional local usage with `uses` argument
  - ```bash
    jina pod --uses hub/example/config.yml --port-in 55555 --port-out 55556
    ```
    
- Docker command

  - Specify the image name along with the version tag. The snippet below uses Jina version `0.9.20`

  - ```bash
    docker pull jinahub/pod.encoder.imagekerasencoder:0.0.8-0.9.28
    ```
