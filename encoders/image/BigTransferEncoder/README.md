# BigTransferEncoder

`BigTransferEncoder` is Big Transfer (BiT) presented by Google (https://github.com/google-research/big_transfer), is a State-of-the-art transfer learning model for computer vision. This class uses pretrained BiT to encode data from an `ndarray`, potentially B x (Channel x Height x Width) into an `ndarray` of shape `B x D`. Internally, :class:`BigTransferEncoder` wraps the models from https://storage.googleapis.com/bit_models/.
Additional links:
- [Tensorflow Blog](https://blog.tensorflow.org/2020/05/bigtransfer-bit-state-of-art-transfer-learning-computer-vision.html)
- [Google AI Blog](https://ai.googleblog.com/2020/05/open-sourcing-bit-exploring-large-scale.html)
- [BiT Research Paper](https://arxiv.org/abs/1912.11370)

