# FaceNetEncoder

**FaceNetEncoder** is a class that wraps the image embedding functionality from the **FaceNet** model.

### Algorithm
* Face detector detects faces on the image
    * If multiple faces are detected the largest is selected (default heuristic)
    * If no faces are detected a dummy face is used (an array with zeroes)
* The face is encoded to an embedding


The model is taken from the FaceNet implementation by timesler (https://github.com/timesler/facenet-pytorch)

The original FaceNet model was introduced in [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832).

`FaceNetEncoder` encodes image batches given as an `np.ndarray` of floats and returns a `np.ndarray` of floats.

- Input shape: `BatchSize x (Height x Width x Channels)`

- Output shape: `BatchSize x EmbeddingDimension`
  
EmbeddingDimension is equal to 512.

`Channels` dimension can be changed (e.g. set `channel_axis` to 1 for channels first mode instead of channels last).
      

## Usage

The following example shows how to generate output embeddings given an input `np.ndarray` containing images.

```python
# Encoder embedding 
encoder = FaceNetEncoder()
embeddeding_batch_np = encoder.encode(batch_of_images)    
```

