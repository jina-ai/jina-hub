# CustomKerasImageEncoder

**`CustomKerasImageEncoder`** encodes ``Document`` content from a ndarray, potentially `BatchSize x (Channel x Height x Width)` into a ndarray of  `BatchSize x EmbeddingDimension`. 

Internally, `CustomKerasImageEncoder` wraps any custom tf.keras model not part of models from `tensorflow.keras.applications`:  https://www.tensorflow.org/api_docs/python/tf/keras/applications

## Usage

Users can use Pod images in several ways:

1. Run with docker:

```
docker run jinahub/pod.encoder.customkerasimageencoder:0.0.8-1.0.4
```

2. Run the Flow API:

```
 from jina.flow import Flow

 f = (Flow()
     .add(name='vse_encoder', uses='docker://jinahub/pod.encoder.customkerasimageencoder:0.0.8-1.0.4', port_in=55555, port_out=55556))
```

3. Run with Jina CLI

```
 jina pod --uses docker://jinahub/pod.encoder.customkerasimageencoder:0.0.8-1.0.4 --port-in 55555 --port-out 55556
```

4. Conventional local usage with `uses` argument

```
jina pod --uses custom_folder/customkerasimageencoder.yml --port-in 55555 --port-out 55556
```

5. Docker command, Specify the image name along with the version tag. The snippet below uses Jina version 1.0.1

```
 docker pull jinahub/pod.encoder.customkerasimageencoder:0.0.8-1.0.4
```
