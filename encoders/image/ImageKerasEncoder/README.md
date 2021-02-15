# ImageKerasEncoder

`ImageKerasEncoder` encodes data from a ndarray, potentially B x (Channel x Height x Width) into a ndarray of `B x D`. Internally, :class:`ImageKerasEncoder` wraps the models from `tensorflow.keras.applications`. https://keras.io/applications/




## Usage:

Initialise this Executor specifying parameters.
The pretrained default path is the result of downloading the models in `download.sh`

### Snippets:

Initialise BigTransferEncoder:

`BigTransferEncoder(model_path='pretrained', channel_axis=1, metas=metas)`

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
        .add(name='my-indexer', uses='indexer.yml'))
    ```
    
- Jina CLI
  - ```bash
    jina pod --uses jinahub/pod.encoder.imagekerasencoder:0.0.8-0.9.28 --port-in 55555 --port-out 55556
    ```
    
- Conventional local usage with `uses` argument
  - ```bash
    jina pod --uses hub/example/imagekerasencoder.yml --port-in 55555 --port-out 55556
    ```
    
- Docker command

  - Specify the image name along with the version tag. The snippet below uses Jina version `0.9.20`

  - ```bash
    docker pull jinahub/pod.encoder.imagekerasencoder:0.0.8-0.9.28
    ```
   
 Note:
 
 One of the limitations with the Hub Executors currently is the tags - all Executor images should have the versions appended in the name i.e.
 if the version is `0.0.8-0.9.28`, the image name would be `jinahub/pod.encoder.imagekerasencoder:0.0.8-0.9.28`.
   
