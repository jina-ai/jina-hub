# ImagePaddlehubEncoder

:class:`ImagePaddlehubEncoder` encodes `Document` content from a ndarray, potentially B x (Channel x Height x Width) into a ndarray of `B x D`. Internally, :class:`ImagePaddlehubEncoder` wraps the models from `paddlehub` : https://github.com/PaddlePaddle/PaddleHub