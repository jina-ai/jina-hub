# VideoPaddleEncoder

class:`VideoPaddleEncoder` encodes data from a ndarray, potentially B x T x (Channel x Height x Width) into a ndarray of `B x D`. Internally, :class:`VideoPaddleEncoder` wraps the models from `paddlehub`. https://github.com/PaddlePaddle/PaddleHub 