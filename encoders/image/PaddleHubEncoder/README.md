# PaddleHubEncoder

:class:`PaddleHubEncoder` encodes data from a ndarray, potentially B x (Channel x Height x Width) into a ndarray of `B x D`. Internally, :class:`PaddleHubEncoder` wraps the models from `paddlehub` : https://github.com/PaddlePaddle/PaddleHub