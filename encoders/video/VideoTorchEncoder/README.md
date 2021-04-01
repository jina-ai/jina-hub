# VideoTorchEncoder

**`VideoTorchEncoder`** encodes data from a Numpy array containing batches of video captions to an embedding array.  Internally, VideoTorchEncoder` wraps the models from `torchvision.models`. https://pytorch.org/docs/stable/torchvision/models.html 

- Input shape:  `BatchSize x Time x (Channel x Height x Width)`

- Output shape:  `BatchSize x EmbeddingDimension` 




## Usage

Users can use Pod images in several ways:

1. Run with docker:

```
docker run jinahub/pod.encoder.videotorchencoder:0.0.9-1.0.4
```

2. Run the Flow API:

```
 from jina.flow import Flow

 f = (Flow()
    .add(name='videotorch_encoder', uses='docker://jinahub/pod.encoder.videotorchencoder:0.0.9-1.0.4', port_in=55555, port_out=55556))
```

3. Run with Jina CLI

```
 jina pod --uses docker://jinahub/pod.encoder.videotorchencoder:0.0.9-1.0.4 --port-in 55555 --port-out 55556
```

4 .Conventional local usage with `uses` argument

```
jina pod --uses custom_folder/videotorchencoder.yml --port-in 55555 --port-out 55556
```

5. Docker command, Specify the image name along with the version tag. The snippet below uses Jina version 1.0.1

```
 docker pull jinahub/pod.encoder.videotorchencoder:0.0.9-1.0.4
```



## Encode with the encoder:

The following example show show to generate output embeddings given an input `np.ndarray` of video.

    encoder = VideoTorchEncoder()
    test_data = np.random.rand(batch_size, num_frames, channel, input_dim, input_dim)
    encoded_data = encoder.encode(test_data)

