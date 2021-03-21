# FaceNetEncoder

**FaceNetEncoder** is a class that wraps the image embedding functionality from the **FaceNet** model introduced in [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832).

The model implementation is taken from the [FaceNet-PyTorch implementation](https://github.com/timesler/facenet-pytorch) by timesler.

### Algorithm

`FaceNetEncoder` encodes a batch of images with faces given as an `np.ndarray` of floats and returns a `np.ndarray` of floats.

- Input shape: `BatchSize x (Height x Width x Channels)`

- Output shape: `BatchSize x 512`

`Channels` dimension can be changed (e.g. set `channel_axis` to 1 for channels first mode instead of channels last). The model supports only 3 channels.

## Usage

The following example shows how to generate output embeddings:
```python 
encoder = FaceNetEncoder()
embeddeding_batch_np = encoder.encode(images_batch)    
```
