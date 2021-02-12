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
        .add(name='my-indexer', uses='indexer.yml'))
    ```
    
- Jina CLI
  - ```bash
    jina pod --uses jinahub/pod.encoder.vseimageencoder:0.0.6-0.9.33 --port-in 55555 --port-out 55556
    ```
    
- Conventional local usage with `uses` argument
  - ```bash
    jina pod --uses hub/example/vseimageencoder.yml --port-in 55555 --port-out 55556
    ```
    
- Docker command

  - Specify the image name along with the version tag. The snippet below uses Jina version `0.9.20`

  - ```bash
    docker pull jinahub/pod.encoder.vseimageencoder:0.0.6-0.9.33
    ```
   
 Note:
 
 One of the limitations with the Hub Executors currently is the tags - all Executor images should have the versions appended in the name i.e.
 if the version is `0.0.6-0.9.33`, the image name would be `jinahub/pod.encoder.vseimageencoder:0.0.6-0.9.33`.
   
