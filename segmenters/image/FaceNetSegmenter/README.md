# FaceNetSegmenter

**FaceNetSegmenter** is a class that wraps the face detection functionality from the **FaceNet** model introduced in [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832).

The model implementation is taken from the [FaceNet-PyTorch implementation](https://github.com/timesler/facenet-pytorch) by timesler.

`FaceNetSegmenter` takes an image given as an `np.ndarray` of floats and returns a list of dictionaries that contain a cropped face in a `'blob'` key.  
A single face is detected by default. Use `keep_all=True` to get all faces from an image.

- Input shape: `(Height x Width x Channels)`

- Output: `List[Dict]`. Face shape: `Channels x ImageSize x ImageSize`

`Channels` input dimension can be changed (e.g. set `channel_axis` to 1 for channels-first mode instead of channels-last).

## Usage

The following example shows how to get cropped faces for a single image represented by `np.ndarray`.

```python
segmenter = FaceNetSegmenter()
result = segmenter.encode([image])
face = result[0]['blob']
```
