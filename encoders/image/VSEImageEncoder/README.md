# VSEImageEncoder

`VSEImageEncoder` is the image Encoder used to extract Visual Semantic Embeddings. Taken from the results of
VSE++: Improving Visual-Semantic Embeddings with Hard Negatives (https://arxiv.org/abs/1707.05612). This model
extracts image feature embeddings that can be used in combination with a VSETextEncoder which in combination will
put images and its captions in nearby locations in the embedding space

@article
{
  faghri2018vse++,
  title={VSE++: Improving Visual-Semantic Embeddings with Hard Negatives},
  author={Faghri, Fartash and Fleet, David J and Kiros, Jamie Ryan and Fidler, Sanja},
  booktitle = {Proceedings of the British Machine Vision Conference ({BMVC})},
  url = {https://github.com/fartashf/vsepp},
  year={2018}
}

## Usage:

Initialise this Executor specifying parameters i.e.:

| `param_name`  | `param_remarks` |
| ------------- | ------------- |
| `model_path`  | the directory path of the model in the `SavedModel` format  |
| `model_name`  | name of the model to be trained  |
| `channel_axis`| axis id of the channel, etc.  |

The pretrained default path is the result of downloading the models in `download.sh`

### Snippets:

Initialise VSEImageEncoder:

`VSEImageEncoder(model_path='pretrained', channel_axis=1, metas=metas)`

Users can use Pod images in several ways:

- Run with Docker (`docker run`)
  - ```bash
    docker run jinahub/pod.encoder.vseimageencoder:0.0.6-0.9.33 --port-in 55555 --port-out 55556
    ```
    
- Flow API
  - ```python
    from jina.flow import Flow

    f = (Flow()
        .add(name='my-encoder', image='jinahub/pod.encoder.vseimageencoder:0.0.6-0.9.33', port_in=55555, port_out=55556)
    ```
    
- Jina CLI
  - ```bash
    jina pod --uses jinahub/pod.encoder.vseimageencoder:0.0.6-0.9.33 --port-in 55555 --port-out 55556
    ```
    
- Conventional local usage with `uses` argument
  - ```bash
    jina pod --uses hub/example/config.yml --port-in 55555 --port-out 55556
    ```
    
- Docker command

  - Specify the image name along with the version tag. The snippet below uses Jina version `0.9.20`

  - ```bash
    docker pull jinahub/pod.encoder.vseimageencoder:0.0.6-0.9.33
    ```
   
