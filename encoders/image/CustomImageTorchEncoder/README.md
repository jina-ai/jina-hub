# CustomImageTorchEncoder

**`CustomImageTorchEncoder`** encodes images using a custom encoder based on pytorch.

`CustomImageTorchEncoder` encodes ``Document`` content from a `np.ndarray` containing images and returns a `np.ndarray` with embeddings.

- Input shape: `BatchSize x (Channel x Height x Width)`

- Output shape: `BatchSize x EmbeddingDimension`

## Usage

Users can use Pod images in several ways:

1. Run with docker:

```
docker run jinahub/pod.encoder.customimagetorchencoder:0.0.11-1.0.4
```

2. Run the Flow API:

```
 from jina.flow import Flow

 f = (Flow()
     .add(name='custom_image_torch_encoder', uses='docker://jinahub/pod.encoder.customimagetorchencoder:0.0.11-1.0.4', port_in=55555, port_out=55556))
```

3. Run with Jina CLI

```
 jina pod --uses docker://jinahub/pod.encoder.customimagetorchencoder:0.0.11-1.0.4 --port-in 55555 --port-out 55556
```

4. Conventional local usage with `uses` argument

```
jina pod --uses custom_folder/customimagetorchencoder.yml --port-in 55555 --port-out 55556
```

5. Docker command, Specify the image name along with the version tag. The snippet below uses Jina version 1.0.4

```
 docker pull jinahub/pod.encoder.customimagetorchencoder:0.0.11-1.0.4
```
