# BigTransferEncoder

`BigTransferEncoder` is Big Transfer (BiT) presented by Google (https://github.com/google-research/big_transfer), is a State-of-the-art transfer learning model for computer vision. This class uses pretrained BiT to encode data from an `ndarray`, potentially B x (Channel x Height x Width) into an `ndarray` of shape `B x D`. Internally, :class:`BigTransferEncoder` wraps the models from https://storage.googleapis.com/bit_models/.
Additional links:
- [Tensorflow Blog](https://blog.tensorflow.org/2020/05/bigtransfer-bit-state-of-art-transfer-learning-computer-vision.html)
- [Google AI Blog](https://ai.googleblog.com/2020/05/open-sourcing-bit-exploring-large-scale.html)
- [BiT Research Paper](https://arxiv.org/abs/1912.11370)

Usage:
Initialise this executor specifying parameters i.e. `model_path` (the directory path of the model in the `SavedModel` format), `model_name` (includes `R50x1`, `R101x1`, `R50x3`, `R101x3`, `R152x4`) `channel_axis` (axis id of the channel), etc.
`BigTransferEncoder(model_path='pretrained', channel_axis=1, metas=metas)`

