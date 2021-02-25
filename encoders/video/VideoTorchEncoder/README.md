# VideoTorchEncoder

**`VideoTorchEncoder`** encodes data from a Numpy array containing batches of video captions to an embedding array.  Internally, VideoTorchEncoder` wraps the models from `torchvision.models`. https://pytorch.org/docs/stable/torchvision/models.html 

- Input shape:  `BatchSize x Time x (Channel x Height x Width)`

- Output shape:  `BatchSize x EmbeddingDimension` 

    

## Encode with the encoder:

The following example show show to generate output embeddings given an input `np.ndarray` of video.

    encoder = VideoTorchEncoder()
    test_data = np.random.rand(batch_size, num_frames, channel, input_dim, input_dim)
    encoded_data = encoder.encode(test_data)
